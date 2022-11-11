from torchseq.utils.functions import cos_sim, onehot
import torch
import torch.nn as nn

from math import e, floor, pow, exp

# https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb


class VectorQuantizerMultiHead(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
        num_heads=1,
        residual=False,
        ema=True,
        ema_schedule_steps=None,
        ema_first_only=False,
        code_offset=0,
        soft_em=True,
        warmup_steps=None,
        code_entropy_weight=0,
        hierarchical=False,
        hierarchical_balance_dims=False,
        hierarchical_greedy=False,
        transitions=False,
        transitions_bias=False,
        transitions_embed=False,
        transitions_log=False,
        relative_error=False,
        relative_error_cumulative=False,
        use_cosine_similarities=False,
        use_gumbel=False,
        gumbel_temp=1.0,
        temp_min=0.5,
        temp_schedule=False,
        temp_schedule_gamma=1000,
        use_straight_through=True,
        separate_output_embedding=False,
        use_code_classifier=False,
        additive=False,
        only_final=False,
        norm_loss_weight=None,
        projected_output=False,
        full_dim_input=False,
        init_decay_weight=1.0,
        init_embeds_xavier=False,
        init_delay_steps=None,
        init_dynamic_var=False,
        init_scale=0.5,
        head_dropout=None,
        head_dropout_keep_first=True,
        learnable_priors=False,
        init_sphere=False,
        soft_gumbel=False,
        kl_weight=None,
    ):

        # residual_head_range=(0, 0),
        super(VectorQuantizerMultiHead, self).__init__()

        if additive and not (use_code_classifier or separate_output_embedding or projected_output or full_dim_input):
            raise Exception(
                "If additive mode us used in VQ, the output embedding must be separate from the code prediction!"
            )

        if only_final and not (use_code_classifier or separate_output_embedding or projected_output or full_dim_input):
            raise Exception(
                "If only final embedding is to be returned in VQ, the output embedding must be separate from the code prediction!"
            )

        if full_dim_input and not (additive or only_final):
            raise Exception(
                "If using full dim as input in VQ, the output embedding must also be full dim (ie additive or only_final)!"
            )

        self._num_embeddings = num_embeddings
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim if full_dim_input else embedding_dim // self._num_heads

        self._ema = ema
        self._ema_schedule_steps = ema_schedule_steps
        self._ema_first_only = ema_first_only
        self._code_offset = code_offset

        self._soft_em = soft_em
        self._use_gumbel = use_gumbel
        self._soft_gumbel = soft_gumbel
        self._gumbel_temp = gumbel_temp
        self._temp_min = temp_min
        self._temp_schedule = temp_schedule
        self._temp_schedule_gamma = temp_schedule_gamma
        self._use_straight_through = use_straight_through
        self._use_code_classifier = use_code_classifier

        self._warmup_steps = warmup_steps
        self._hierarchical = hierarchical
        self._hierarchical_balance_dims = hierarchical_balance_dims
        self._hierarchical_greedy = hierarchical_greedy

        self._use_transitions = transitions
        self._transitions_bias = transitions_bias
        self._transitions_embed = transitions_embed
        self._transitions_log = transitions_log

        self._relative_error = relative_error
        self._relative_error_cumulative = relative_error_cumulative

        self._kl_weight = kl_weight

        self._cos_sim = use_cosine_similarities

        self._additive = additive
        self._only_final = only_final
        self._norm_loss_weight = norm_loss_weight
        self._full_dim_input = full_dim_input

        if head_dropout is not None and head_dropout > 0:
            self._head_dropout = torch.distributions.Bernoulli(1 - head_dropout)
        else:
            self._head_dropout = None
        self._head_dropout_keep_first = head_dropout_keep_first

        if hierarchical:

            if self._hierarchical_balance_dims:
                dim_weights = [2**x for x in range(self._num_heads)]
                total_dim = self._embedding_dim * self._num_heads
                self.dims = [floor(x * total_dim / sum(dim_weights)) for x in dim_weights]
                # Reset the smallest dim to account for rounding
                self.dims[0] = total_dim - sum(self.dims[1:])
            else:
                self.dims = [self._embedding_dim] * self._num_heads

            self._embedding = nn.ModuleList(
                [nn.Embedding(self._num_embeddings ** (1 + h), self.dims[h]) for h in range(num_heads)]
            )

            if self._ema:
                self._ema_w = nn.ParameterList(
                    [nn.Parameter(torch.Tensor(num_embeddings ** (1 + h), self.dims[h])) for h in range(num_heads)]
                )
        else:
            self.dims = [self._embedding_dim] * self._num_heads

            self._embedding = nn.ModuleList(
                [nn.Embedding(self._num_embeddings, self._embedding_dim) for _ in range(num_heads)]
            )
            if self._ema:
                self._ema_w = nn.ParameterList(
                    [nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim)) for _ in range(num_heads)]
                )
        for hix, embedding in enumerate(self._embedding):
            torch.nn.init.xavier_uniform_(
                embedding.weight.data, gain=6.0 * init_scale * init_decay_weight**hix
            ) if init_embeds_xavier else embedding.weight.data.normal_(std=init_scale * init_decay_weight**hix)
            if init_sphere:
                embedding.weight.data = (
                    embedding.weight.data
                    / torch.linalg.vector_norm(embedding.weight, dim=1, keepdim=True)
                    * init_scale
                )

        if separate_output_embedding:
            self._output_embedding = nn.ModuleList(
                [
                    nn.Embedding(
                        self._num_embeddings, self._embedding_dim if not (additive or only_final) else embedding_dim
                    )
                    for _ in range(num_heads)
                ]
            )
            for embedding in self._output_embedding:
                embedding.weight.data.normal_()
        else:
            self._output_embedding = None

        if projected_output:
            self._output_projection = nn.Linear(
                self._embedding_dim, self._embedding_dim if not (additive or only_final) else embedding_dim
            )
        else:
            self._output_projection = None

        if self._use_code_classifier:
            self._code_classifiers = nn.ModuleList([nn.Linear(d, num_embeddings, bias=False) for d in self.dims])

        if self._use_transitions:
            self._transitions = nn.ModuleList(
                [
                    nn.Linear(
                        d if self._transitions_embed else num_embeddings, num_embeddings, bias=self._transitions_bias
                    )
                    for d in self.dims[1:]
                ]
            )
            # for trans in self._transitions:
            #     nn.init.xavier_uniform_(trans.weight, gain=10)
            # self._transition = nn.Linear(num_embeddings, num_embeddings)

        if learnable_priors:
            self._learnable_priors = nn.ParameterList(
                [nn.Parameter(torch.zeros(num_embeddings)) for _ in range(num_heads)]
            )
        else:
            self._learnable_priors = None

        self._commitment_cost = commitment_cost

        self._decay = decay
        self._epsilon = epsilon
        self._residual = residual

        if self._ema:
            self._alpha = nn.Parameter(torch.Tensor(num_heads))

            for ix in range(self._num_heads):
                power = (1 + ix) if self._hierarchical else 1
                self.register_buffer("_ema_cluster_size" + str(ix), torch.zeros(num_embeddings**power))

                self._ema_w[ix].data.normal_()

        self._code_entropy_weight = code_entropy_weight

        self._init_decay_weight = init_decay_weight
        self._init_delay_steps = init_delay_steps
        self._init_dynamic_var = init_dynamic_var
        if self._init_delay_steps is not None:
            self.register_buffer("_init_cumsum", torch.zeros(embedding_dim))
            self.register_buffer("_init_cumsquared", torch.zeros(embedding_dim))
            self._init_samples = 0
            self._init_done = False
        else:
            self._init_done = True

        if code_entropy_weight > 0:
            cooccur_shape = [num_embeddings] * num_heads
            self.register_buffer("_code_cooccurrence", torch.zeros(*cooccur_shape))

    def encoding_to_logits(self, input, head_ix, prev_codes):
        pass

    def forward(self, inputs, global_step=None, forced_codes=None, head_mask=None):
        input_shape = inputs.shape

        quantized_list = []
        vq_codes = []
        all_probs = []

        loss = torch.zeros(input_shape[0]).to(inputs.device)

        if (
            self.training
            and not self._init_done
            and self._init_delay_steps is not None
            and global_step < self._init_delay_steps
        ):
            self._init_cumsum += inputs.squeeze(dim=1).sum(dim=0)
            self._init_cumsquared += (inputs**2).squeeze(dim=1).sum(dim=0)
            self._init_samples += input_shape[0]
        elif self.training and not self._init_done and global_step >= self._init_delay_steps:
            init_mean = self._init_cumsum / float(self._init_samples)
            init_var = (
                torch.sqrt(self._init_cumsquared / float(self._init_samples) - init_mean**2)
                if self._init_dynamic_var
                else torch.full_like(init_mean, 0.5)
            )
            for hix, embedding in enumerate(self._embedding):
                this_mean = init_mean if hix == 0 else torch.zeros_like(init_mean)
                self._embedding[hix].weight.data = torch.normal(
                    mean=this_mean.unsqueeze(0).expand(self._num_embeddings, -1),
                    std=init_var.unsqueeze(0).expand(self._num_embeddings, -1) * self._init_decay_weight**hix,
                )
            self._init_done = True

        all_distances = []
        for head_ix, embedding in enumerate(self._embedding):
            # this_input = flat_input[:, head_ix, :]
            head_begin, head_end = sum(self.dims[:head_ix]), sum(self.dims[: head_ix + 1])
            this_input = inputs[:, 0, :] if self._full_dim_input else inputs[:, 0, head_begin:head_end]

            # print(f"h{head_ix} input", torch.linalg.norm(this_input, dim=-1)[:4])
            # print(f"h{head_ix} embed", torch.linalg.norm(embedding.weight, dim=-1)[:4])

            distances = torch.zeros(input_shape[0], embedding.weight.shape[0]).to(this_input.device)
            if self._learnable_priors is not None:
                distances += self._learnable_priors[head_ix]

            # Calculate distances
            if self._relative_error and len(all_probs) > 0:
                if self._relative_error_cumulative:
                    resid_error = this_input
                    for hix in range(head_ix):
                        resid_error = resid_error - torch.matmul(
                            all_probs[hix], self._embedding[hix].weight  # .detach()
                        ).squeeze(1)
                else:
                    prev_embed = torch.matmul(
                        all_probs[head_ix - 1], self._embedding[head_ix - 1].weight  # .detach()
                    ).squeeze(1)

                    resid_error = this_input - prev_embed

                if self._cos_sim:
                    distances += cos_sim(resid_error, self._embedding[head_ix].weight)
                elif self._use_code_classifier:
                    distances += self._code_classifiers[head_ix](resid_error)
                else:
                    distances += -1.0 * (
                        torch.sum(resid_error**2, dim=1, keepdim=True)
                        + torch.sum(self._embedding[head_ix].weight ** 2, dim=1)
                        - 2 * torch.matmul(resid_error, self._embedding[head_ix].weight.t())
                    )
            else:

                if self._cos_sim:
                    distances += cos_sim(this_input, self._embedding[head_ix].weight)
                elif self._use_code_classifier:
                    distances += self._code_classifiers[head_ix](this_input)
                else:
                    distances += -1.0 * (
                        torch.sum(this_input**2, dim=1, keepdim=True)
                        + torch.sum(self._embedding[head_ix].weight ** 2, dim=1)
                        - 2 * torch.matmul(this_input, self._embedding[head_ix].weight.t())
                    )

            # Convert distances into log probs
            logits = distances.detach() if self._use_straight_through else distances

            if self._learnable_priors is not None:
                prior = torch.softmax(self._learnable_priors[head_ix], dim=-1)
                posterior = nn.functional.log_softmax(logits, dim=-1)
                kl_loss = torch.nn.KLDivLoss(reduction="none")
                loss += kl_loss(posterior, prior).sum(dim=-1)

            if self._kl_weight is not None:
                posterior = nn.functional.log_softmax(logits, dim=-1)
                prior = torch.ones_like(posterior).detach() / torch.ones_like(posterior).sum(-1, keepdim=True).detach()
                kl_loss = torch.nn.KLDivLoss(reduction="none")
                loss += kl_loss(posterior, prior).sum(dim=-1) * self._kl_weight

            if self._hierarchical and len(all_probs) > 0:
                # For the hierarchical case, weight the current probs by the prob of their parent node
                logits += torch.log(all_probs[-1] + 1e-10).squeeze(1).repeat_interleave(self._num_embeddings, dim=1)

            if self._use_transitions and len(all_probs) > 0:

                if self._transitions_embed:
                    prev_quantized = torch.matmul(all_probs[head_ix - 1], self._embedding[head_ix - 1].weight.detach())
                    trans_logits = self._transitions[head_ix - 1](prev_quantized).squeeze(1)
                elif self._transitions_log:
                    trans_logits = self._transitions[head_ix - 1](torch.log(all_probs[head_ix - 1] + 1e-10)).squeeze(1)
                else:
                    trans = self._transitions[head_ix - 1]
                    prev_prob = all_probs[head_ix - 1].squeeze(1)  # .detach()
                    trans_logits = trans(prev_prob)

                logits = logits + trans_logits

                all_distances.append(distances)

            if self._use_gumbel and self.training:

                # gumbel_sched_weight = 2 - 2 / (1 + exp(-float(global_step) / float(self._temp_schedule_gamma)))
                gumbel_sched_weight = exp(-float(global_step) / float(self._temp_schedule_gamma))
                gumbel_temp = (
                    max(self._gumbel_temp * gumbel_sched_weight, self._temp_min)
                    if self._temp_schedule
                    else self._gumbel_temp
                )
                probs = torch.nn.functional.gumbel_softmax(
                    logits, tau=gumbel_temp, hard=(not self._soft_gumbel), dim=-1
                )
            elif self._use_gumbel:
                indices = torch.argmax(logits, dim=-1)
                probs = onehot(indices, N=logits.shape[-1])
            else:
                probs = torch.softmax(logits, dim=-1)

            all_probs.append(probs.unsqueeze(1))

            # Use EMA to update the embedding vectors
            if (
                self.training
                and self._ema
                and (head_ix == 0 or not self._ema_first_only)
                and (self._ema_schedule_steps is None or global_step <= self._ema_schedule_steps)
            ):
                with torch.no_grad():

                    curr_decay = 1 - (1 - self._decay) * (
                        1
                        if self._ema_schedule_steps is None
                        else max(0, (1 - global_step / float(self._ema_schedule_steps)))
                    )

                    _ema_cluster_size = getattr(self, "_ema_cluster_size" + str(head_ix))
                    _ema_cluster_size = _ema_cluster_size * curr_decay + (1 - curr_decay) * torch.sum(probs, 0)

                    # Laplace smoothing of the cluster size
                    n = torch.sum(_ema_cluster_size.data)
                    _ema_cluster_size = (
                        (_ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
                    )
                    setattr(self, "_ema_cluster_size" + str(head_ix), _ema_cluster_size)

                    if self._relative_error and len(all_probs) > 1:
                        dw = torch.matmul(probs.t(), resid_error)
                    else:
                        dw = torch.matmul(probs.t(), this_input)
                    self._ema_w[head_ix] = nn.Parameter(self._ema_w[head_ix] * curr_decay + (1 - curr_decay) * dw)

                    self._embedding[head_ix].weight.data = nn.Parameter(
                        self._ema_w[head_ix] / _ema_cluster_size.unsqueeze(1)
                    ).detach()

        # torch.save({'distances': all_distances, 'transition_logits': transition_logits, 'transitions': self._transitions.state_dict()}, './vq_internals.pt')
        # exit()

        if forced_codes is not None:
            assert (
                forced_codes.shape[1] == self._num_heads
            ), "If forced_codes is supplied, it must be the same length as the number of quantizer heads! {:} vs {:}".format(
                forced_codes.shape[1], self._num_heads
            )
            vq_codes = forced_codes.unbind(dim=1)
        elif self._hierarchical and not self._hierarchical_greedy:
            # work backwards!
            for head_ix in reversed(range(self._num_heads)):
                this_probs = all_probs[head_ix]
                if len(vq_codes) > 0:
                    mask = torch.where(
                        torch.arange(this_probs.shape[-1]).to(this_probs.device) // self.dims[head_ix] == vq_codes[0],
                        1.0,
                        0.0,
                    )
                    # print(this_probs.shape, mask.shape)
                    this_probs *= mask.unsqueeze(1)

                vq_codes.insert(0, torch.argmax(this_probs, dim=-1))
        elif not isinstance(self._code_offset, int) or self._code_offset > 0:
            # print("code offset", self._code_offset)
            for head_ix in range(self._num_heads):
                this_offset = (
                    self._code_offset[head_ix] if not isinstance(self._code_offset, int) else self._code_offset
                )

                min_k = torch.topk(all_probs[head_ix], this_offset + 1, dim=1, largest=False).indices
                vq_codes.append(min_k[:, this_offset])
        else:
            vq_codes = [torch.argmax(probs, dim=-1).squeeze(1) for probs in all_probs]

        # Now that we have the codes, calculate their embeddings
        out_embeds = self._output_embedding if self._output_embedding is not None else self._embedding
        for head_ix, embedding in enumerate(out_embeds):
            # If soft training, use distribution
            if self.training and (self._soft_em or self._transitions or not self._use_straight_through):
                this_quantized = torch.matmul(
                    all_probs[head_ix],
                    embedding.weight.detach() if (self._ema and self._output_embedding is None) else embedding.weight,
                    # embedding.weight,
                )

            # otherwise use one hot
            else:
                this_quantized = embedding(vq_codes[head_ix].type(torch.LongTensor).to(inputs.device)).unsqueeze(1)

            if self._output_projection is not None:
                this_quantized = self._output_projection(this_quantized)
            quantized_list.append(this_quantized)

        if self._only_final:
            quantized = quantized_list[-1]
        else:
            quantized = torch.cat(quantized_list, dim=1)

            if head_mask is not None:
                # print('mask found')
                # print(head_mask)
                assert (
                    head_mask.shape[1] == self._num_heads
                ), "If head_mask is set, it must be the same length as the number of quantizer heads! {:} vs {:}".format(
                    head_mask.shape[1], self._num_heads
                )
                # print(vq_codes[0].shape)
                # print(quantized_list[0].shape)
                # print(head_mask.shape, quantized.shape)
                quantized = quantized * head_mask.unsqueeze(-1)

        if self._head_dropout is not None and self.training:
            mask = self._head_dropout.sample(sample_shape=(*quantized.shape[:-1], 1))
            if self._head_dropout_keep_first:
                mask[:, 0, :] = 1.0
            mask = torch.cumprod(mask, dim=1).to(quantized.device)
            quantized = quantized * mask

        if self._additive:
            quantized = torch.sum(quantized, dim=1)
        quantized = quantized.view(input_shape)

        # Losses
        if not self._ema:
            q_latent_loss = (
                nn.functional.mse_loss(quantized, inputs.detach(), reduction="none").mean(dim=-1).mean(dim=-1)
            )
        else:
            q_latent_loss = 0
        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs, reduction="none").mean(dim=-1).mean(dim=-1)

        loss = torch.zeros(input_shape[0]).to(inputs.device)

        # Straight Through Estimator
        if self._residual:
            # Calculate residual weightings
            resid_weight = torch.sigmoid(self._alpha).unsqueeze(-1)

            loss += self._commitment_cost * e_latent_loss + q_latent_loss

            alpha_loss = torch.sum(torch.square(resid_weight))
            loss += alpha_loss

            quantized = (
                inputs.view(-1, self._num_heads, self._embedding_dim) * resid_weight
                + quantized.detach().view(-1, self._num_heads, self._embedding_dim) * (1 - resid_weight)
            ).view(input_shape)
        elif self._use_straight_through:
            loss += self._commitment_cost * e_latent_loss + q_latent_loss

            if self._warmup_steps is not None and global_step is not None:
                w = min(global_step / self._warmup_steps, 1.0)
                quantized = inputs + (w * quantized - w * inputs).detach()
            else:
                quantized = inputs + (quantized - inputs).detach()

        out_embeds = self._output_embedding if self._output_embedding is not None else self._embedding
        if self._norm_loss_weight is not None and self._norm_loss_weight > 0:
            norm_loss = 0.0

            for hix in range(self._num_heads - 1):
                prev_norm = torch.min(torch.linalg.norm(out_embeds[hix].weight, dim=-1)) * self._norm_loss_weight
                this_norm = torch.linalg.norm(out_embeds[hix + 1].weight, dim=-1)

                # Hinge loss
                norm_loss += torch.where(
                    this_norm - prev_norm > 0.0, this_norm - prev_norm, torch.zeros_like(this_norm)
                ).mean()
            loss += norm_loss

        # print('vq: ', vq_codes)
        # print(quantized.shape)
        # print(quantized[0,0,:10])

        return loss, quantized, vq_codes
