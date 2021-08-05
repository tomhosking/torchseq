from torchseq.utils.functions import cos_sim, onehot
import torch
import torch.nn as nn

from math import e, floor, pow

# https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb


class VectorQuantizerMultiHead(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay,
        epsilon=1e-5,
        num_heads=1,
        residual=False,
        ema=True,
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
        use_cosine_similarities=False,
        use_gumbel=False,
        gumbel_temp=1.0,
        temp_schedule=False,
        use_straight_through=True,
        separate_output_embedding=False,
        use_code_classifier=False,
        additive=False,
        only_final=False,
        subtract_previous=False,
        norm_loss_weight=None,
    ):

        # residual_head_range=(0, 0),
        super(VectorQuantizerMultiHead, self).__init__()

        if additive and not (use_code_classifier or separate_output_embedding):
            raise Exception(
                "If additive mode us used in VQ, the output embedding must be separate from the code prediction!"
            )

        if only_final and not (use_code_classifier or separate_output_embedding):
            raise Exception(
                "If only final embedding is to be returned in VQ, the output embedding must be separate from the code prediction!"
            )

        self._num_embeddings = num_embeddings
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim // self._num_heads

        self._ema = ema
        self._code_offset = code_offset

        self._soft_em = soft_em
        self._use_gumbel = use_gumbel
        self._gumbel_temp = gumbel_temp
        self._temp_schedule = temp_schedule
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

        self._cos_sim = use_cosine_similarities

        self._additive = additive
        self._only_final = only_final
        self._norm_loss_weight = norm_loss_weight

        if hierarchical:

            if self._hierarchical_balance_dims:
                dim_weights = [2 ** x for x in range(self._num_heads)]
                total_dim = self._embedding_dim * self._num_heads
                self.dims = [floor(x * total_dim / sum(dim_weights)) for x in dim_weights]
                # Reset the smallest dim to account for rounding
                self.dims[0] = total_dim - sum(self.dims[1:])
            else:
                self.dims = [self._embedding_dim] * self._num_heads

            self._embedding = nn.ModuleList(
                [nn.Embedding(self._num_embeddings ** (1 + h), self.dims[h]) for h in range(num_heads)]
            )

            self._ema_w = nn.ParameterList(
                [nn.Parameter(torch.Tensor(num_embeddings ** (1 + h), self.dims[h])) for h in range(num_heads)]
            )
        else:
            self.dims = [self._embedding_dim] * self._num_heads

            self._embedding = nn.ModuleList(
                [nn.Embedding(self._num_embeddings, self._embedding_dim) for _ in range(num_heads)]
            )

            self._ema_w = nn.ParameterList(
                [nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim)) for _ in range(num_heads)]
            )
        for embedding in self._embedding:
            embedding.weight.data.normal_()

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

        self._commitment_cost = commitment_cost

        for ix in range(self._num_heads):
            power = (1 + ix) if self._hierarchical else 1
            self.register_buffer("_ema_cluster_size" + str(ix), torch.zeros(num_embeddings ** power))

            self._ema_w[ix].data.normal_()

        self._decay = decay
        self._epsilon = epsilon
        self._residual = residual
        self._alpha = nn.Parameter(torch.Tensor(num_heads))
        self._code_entropy_weight = code_entropy_weight

        if code_entropy_weight > 0:
            cooccur_shape = [num_embeddings] * num_heads
            self.register_buffer("_code_cooccurrence", torch.zeros(*cooccur_shape))

    def encoding_to_logits(self, input, head_ix, prev_codes):
        pass

    def forward(self, inputs, global_step=None, forced_codes=None):
        input_shape = inputs.shape

        quantized_list = []
        vq_codes = []
        all_probs = []

        all_distances = []
        for head_ix, embedding in enumerate(self._embedding):
            # this_input = flat_input[:, head_ix, :]
            head_begin, head_end = sum(self.dims[:head_ix]), sum(self.dims[: head_ix + 1])
            this_input = inputs[:, 0, head_begin:head_end]

            # Calculate distances
            if self._cos_sim:
                distances = cos_sim(this_input, self._embedding[head_ix].weight)
            elif self._use_code_classifier:
                distances = self._code_classifiers[head_ix](this_input)
            else:
                if self._relative_error and len(all_probs) > 0:
                    prev_embed = torch.matmul(
                        all_probs[head_ix - 1], self._embedding[head_ix - 1].weight.detach()
                    ).squeeze(1)

                    resid_error = this_input - prev_embed
                    distances = -1.0 * (
                        torch.sum(resid_error ** 2, dim=1, keepdim=True)
                        + torch.sum(self._embedding[head_ix].weight ** 2, dim=1)
                        - 2 * torch.matmul(resid_error, self._embedding[head_ix].weight.t())
                    )
                else:
                    distances = -1.0 * (
                        torch.sum(this_input ** 2, dim=1, keepdim=True)
                        + torch.sum(self._embedding[head_ix].weight ** 2, dim=1)
                        - 2 * torch.matmul(this_input, self._embedding[head_ix].weight.t())
                    )

            # Convert distances into log probs
            logits = distances.detach() if self._use_straight_through else distances
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
                gumbel_temp = (
                    self._gumbel_temp / pow(1.0 + global_step * 1.0, 0.25)
                    if self._temp_schedule
                    else self._gumbel_temp
                )
                probs = torch.nn.functional.gumbel_softmax(logits, tau=gumbel_temp, hard=True, dim=-1)
            elif self._use_gumbel:
                indices = torch.argmax(logits, dim=-1)
                probs = onehot(indices, N=logits.shape[-1])
            else:
                probs = torch.softmax(logits, dim=-1)

            all_probs.append(probs.unsqueeze(1))

            # Use EMA to update the embedding vectors
            if self.training and self._ema:
                with torch.no_grad():
                    _ema_cluster_size = getattr(self, "_ema_cluster_size" + str(head_ix))
                    _ema_cluster_size = _ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(probs, 0)

                    # Laplace smoothing of the cluster size
                    n = torch.sum(_ema_cluster_size.data)
                    _ema_cluster_size = (
                        (_ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
                    )
                    setattr(self, "_ema_cluster_size" + str(head_ix), _ema_cluster_size)

                    dw = torch.matmul(probs.t(), this_input)
                    self._ema_w[head_ix] = nn.Parameter(self._ema_w[head_ix] * self._decay + (1 - self._decay) * dw)

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
            print("code offset", self._code_offset)
            for head_ix in range(self._num_heads):
                this_offset = (
                    self._code_offset[head_ix] if not isinstance(self._code_offset, int) else self._code_offset
                )

                min_k = torch.topk(all_probs[head_ix], this_offset + 1, dim=1, largest=False).indices
                vq_codes.append(min_k[:, this_offset])
        else:
            vq_codes = [torch.argmax(probs, dim=-1) for probs in all_probs]

        # Now that we have the codes, calculate their embeddings
        out_embeds = self._output_embedding if self._output_embedding is not None else self._embedding
        for head_ix, embedding in enumerate(out_embeds):
            # If soft training, use distribution
            if self.training and (self._soft_em or self._transitions or not self._use_straight_through):
                this_quantized = torch.matmul(
                    all_probs[head_ix],
                    embedding.weight.detach() if self._ema else embedding.weight,
                )

            # otherwise use one hot
            else:
                this_quantized = embedding(vq_codes[head_ix].type(torch.LongTensor).to(inputs.device)).unsqueeze(1)

            quantized_list.append(this_quantized)

        if self._only_final:
            quantized = quantized_list[-1]
        else:
            quantized = torch.cat(quantized_list, dim=1)

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
                prev_norm = torch.linalg.matrix_norm(out_embeds[hix].weight) * self._norm_loss_weight
                this_norm = torch.linalg.matrix_norm(out_embeds[hix + 1].weight)

                # Hinge loss
                if this_norm > prev_norm:
                    norm_loss += this_norm - prev_norm
            loss += norm_loss

        return loss, quantized, vq_codes
