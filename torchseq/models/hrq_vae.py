from torchseq.utils.functions import cos_sim, onehot, initialize_truncated_normal_, initialize_polar_normal_
import torch
import torch.nn as nn

from sklearn.cluster import KMeans


from math import e, floor, pow, exp, sqrt

from torchseq.utils.logging import Logger


class HierarchicalRefinementQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        num_heads=3,
        code_offset=0,
        warmup_steps=None,
        use_cosine_similarities=False,
        gumbel_temp=2.0,
        temp_min=0.5,
        temp_schedule=True,
        temp_schedule_gamma=10000,
        temp_schedule_depth_factor=1,
        norm_loss_weight=None,
        norm_loss_scale=None,
        norm_loss_diff=False,
        init_decay_weight=0.5,
        init_embeds_xavier=True,
        init_embeds_truncnorm=False,
        init_embeds_uniform=False,
        init_embeds_polar=False,
        init_delay_steps=None,
        init_dynamic_var=False,
        init_scale=1.0,
        head_dropout=0.3,
        head_dropout_keep_first=False,
        head_dropout_schedule_gamma=None,
        head_dropout_min=None,
        learnable_priors=False,
        include_residual=False,
        residual_penalty=0.0,
        adaptive_depth=False,
        adaptive_cascade=True,
        adaptive_penalty_weight=0.0,
        residual_warmup_steps=0,
        residual_dropout_steps=0,
        kmeans_delay=0,
        kl_weight=0.0,
        pre_norm=False,
        post_linear=False,
        init_sphere=False,
        soft_gumbel=False,
        output_seq=False,
        output_cumsum=False,
        simple_norm=False,
        logits_warmup_steps=0,
        sqrt_distances=False,
        learned_cluster_variances=False,
        kl_warmup_steps=0,
        commitment_weight=0,
        freeze_embeddings=False,
        detach=False,
        noise_inputs=False,
        demean_inputs=False,
        pre_scale=False,
        post_scale=False,
        diversity_penalty_weight=None,
        force_positive_embeddings=False,
        debug={},
    ):
        super(HierarchicalRefinementQuantizer, self).__init__()

        if demean_inputs:
            raise Exception("Demean inputs not supported for HRQ! Use pythae:vqvae instead")

        self._detach = detach
        self._debug = debug

        if isinstance(num_embeddings, int):
            num_embeddings = [num_embeddings for _ in range(num_heads)]
        self._num_embeddings = num_embeddings
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim

        self._code_offset = code_offset

        self._gumbel_temp = gumbel_temp
        self._temp_min = temp_min
        self._temp_schedule = temp_schedule
        self._temp_schedule_gamma = temp_schedule_gamma
        self._temp_schedule_depth_factor = temp_schedule_depth_factor
        self._soft_gumbel = soft_gumbel
        self._noise_inputs = noise_inputs

        self._warmup_steps = warmup_steps

        self._cos_sim = use_cosine_similarities

        self._norm_loss_weight = norm_loss_weight
        self._norm_loss_scale = norm_loss_scale
        self._norm_loss_diff = norm_loss_diff
        self._commitment_weight = commitment_weight

        self._include_residual = include_residual
        self._residual_penalty = residual_penalty

        self._residual_warmup_steps = residual_warmup_steps
        self._residual_dropout_steps = residual_dropout_steps

        self._adaptive_depth = adaptive_depth
        self._adaptive_cascade = adaptive_cascade
        self._adaptive_penalty_weight = adaptive_penalty_weight

        self._kmeans_delay = kmeans_delay

        self._output_seq = output_seq
        self._output_cumsum = output_cumsum

        self._simple_norm = simple_norm
        self._pre_scale = pre_scale
        self._post_scale = post_scale

        self._logits_warmup_steps = logits_warmup_steps

        self._sqrt_distances = sqrt_distances
        self._force_positive_embeddings = force_positive_embeddings

        self._head_dropout = head_dropout
        self._head_dropout_keep_first = head_dropout_keep_first
        self._head_dropout_schedule_gamma = head_dropout_schedule_gamma
        self._head_dropout_min = head_dropout_min

        self._diversity_penalty_weight = diversity_penalty_weight

        self.dims = [self._embedding_dim] * self._num_heads

        self._embedding = nn.ModuleList(
            [
                nn.Embedding(
                    self._num_embeddings[hix], self._embedding_dim, padding_idx=(0 if self._adaptive_depth else None)
                )
                for hix in range(num_heads)
            ]
        )

        if freeze_embeddings:
            for p in self._embedding:
                p.weight.requires_grad = False

        if learned_cluster_variances:
            self._cluster_variances = nn.Parameter(torch.ones(num_heads, num_embeddings))
        else:
            self._cluster_variances = None

        init_ix = 1 if self._adaptive_depth else 0
        for hix, embedding in enumerate(self._embedding):
            if init_embeds_xavier:
                torch.nn.init.xavier_uniform_(
                    embedding.weight.data[init_ix:, :], gain=6.0 * init_scale * init_decay_weight**hix
                )
            elif init_embeds_truncnorm:
                initialize_truncated_normal_(
                    embedding.weight.data[init_ix:, :], std=init_scale * init_decay_weight**hix
                )  # / sqrt(self._embedding_dim)
            elif init_embeds_uniform:
                scale = init_scale * init_decay_weight**hix / sqrt(self._embedding_dim)
                embedding.weight.data[init_ix:, :].uniform_(-1.0 * scale, scale)
            elif init_embeds_polar:
                initialize_polar_normal_(
                    embedding.weight.data[init_ix:, :], scale=init_scale * init_decay_weight**hix
                )  # / sqrt(self._embedding_dim)
            else:
                embedding.weight.data[init_ix:, :].normal_(std=init_scale * init_decay_weight**hix)

        if init_sphere:
            for hix, embedding in enumerate(self._embedding):
                embedding.weight.data[init_ix:, :] = (
                    embedding.weight.data[init_ix:, :]
                    / torch.linalg.vector_norm(embedding.weight[init_ix:, :], dim=1, keepdim=True)
                    * init_scale
                    * init_decay_weight**hix
                )

        if self._adaptive_depth:
            for hix, embedding in enumerate(self._embedding):
                scale = 0.05 * init_scale * init_decay_weight**hix / sqrt(self._embedding_dim)
                embedding.weight.data[0, :].fill_(scale)

        if self._force_positive_embeddings:
            with torch.no_grad():
                for hix, embedding in enumerate(self._embedding):
                    embedding.weight.data = embedding.weight.data.abs()

        self._kl_weight = kl_weight
        self._kl_warmup_steps = kl_warmup_steps
        self._kl_loss = torch.nn.KLDivLoss(reduction="none")

        if learnable_priors:
            self._learnable_priors = nn.ParameterList(
                [nn.Parameter(torch.zeros(num_embeddings)) for _ in range(num_heads)]
            )
        else:
            self._learnable_priors = None

        self._init_decay_weight = init_decay_weight
        self._init_delay_steps = init_delay_steps
        self._init_dynamic_var = init_dynamic_var
        if self._init_delay_steps is not None:
            self.register_buffer("_init_cumsum", torch.zeros(embedding_dim))
            self.register_buffer("_init_cumsquared", torch.zeros(embedding_dim))
            self._init_samples = 0
            self._init_done = False
        elif self._kmeans_delay > 0:
            self.register_buffer("_kmeans_history", torch.zeros(1, embedding_dim))
            self._init_done = False
        else:
            self._init_done = True

        if pre_norm:
            self._pre_norm = nn.LayerNorm(embedding_dim, affine=False)
        else:
            self._pre_norm = None

        if post_linear:
            self._post_linear = nn.Linear(embedding_dim, embedding_dim)
        else:
            self._post_linear = None

        # def bwd_debug(mod, grad_in, grad_out):
        #     print(grad_in[0].shape)
        #     print(grad_out[1].shape)
        #     print(torch.linalg.norm(grad_in[0].squeeze(1), dim=1).abs().mean())
        #     print(torch.linalg.norm(grad_in[0].squeeze(1), dim=1).abs().mean())
        #     print(torch.linalg.norm(mod._embedding[0].weight.grad, dim=1).abs().mean())
        #     print(torch.linalg.norm(mod._embedding[1].weight.grad, dim=1).abs().mean())
        #     print(torch.linalg.norm(mod._embedding[2].weight.grad, dim=1).abs().mean())
        #     exit()
        # self.register_full_backward_hook(bwd_debug)

    def forward(self, inputs, global_step=None, forced_codes=None, head_mask=None, residual_mask=None):
        input_shape = inputs.shape

        quantized_list = []
        vq_codes = []
        all_probs = []

        loss = torch.zeros(input_shape[0]).to(inputs.device)

        this_input = inputs[:, 0, :]

        if self._pre_scale:
            this_input = this_input / sqrt(self._embedding_dim)

        if self._noise_inputs and self.training:
            noise = torch.empty_like(this_input).normal_() * 0.1

            this_input = this_input + noise

        if self._pre_norm is not None:
            this_input = self._pre_norm(this_input) / sqrt(self.embedding_dim)

        if self._simple_norm:
            this_input = this_input / torch.linalg.norm(this_input, dim=-1, keepdim=True) / sqrt(self.embedding_dim)

        if (
            self.training
            and not self._init_done
            and self._init_delay_steps is not None
            and global_step < self._init_delay_steps
        ):
            self._init_cumsum += inputs.squeeze(dim=1).sum(dim=0)
            self._init_cumsquared += (inputs**2).squeeze(dim=1).sum(dim=0)
            self._init_samples += input_shape[0]
        elif (
            self.training
            and not self._init_done
            and self._init_delay_steps is not None
            and global_step >= self._init_delay_steps
        ):
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

        if self.training and not self._init_done and self._kmeans_delay > 0 and global_step < self._kmeans_delay:
            # add to embedding history
            self._kmeans_history = torch.cat([self._kmeans_history.cpu(), inputs[:, 0, :].detach().cpu()], dim=0)
        elif (
            self.training
            and not self._init_done
            and self._kmeans_delay is not None
            and global_step >= self._kmeans_delay
        ):
            kmeans = KMeans(n_clusters=self._num_embeddings, random_state=0).fit(self._kmeans_history[1:, :].numpy())
            centroids = kmeans.cluster_centers_
            self._embedding[0].weight.data = torch.from_numpy(centroids).to(self._embedding[0].weight)
            self._init_done = True
            self._kmeans_history = None

        for head_ix, embedding in enumerate(self._embedding):
            if self._adaptive_depth:
                embedding = torch.cat(
                    [self._embedding[head_ix].weight[:1].detach(), self._embedding[head_ix].weight[1:]], dim=0
                )
            else:
                embedding = embedding.weight

            if self._force_positive_embeddings:
                embedding = embedding.abs()

            distances = torch.zeros(input_shape[0], embedding.shape[0]).to(this_input.device)
            if self._learnable_priors is not None and head_ix == 0:
                distances += self._learnable_priors[head_ix]

            resid_error = this_input
            # Calculate distances
            if len(all_probs) > 0:
                for hix in range(head_ix):
                    if self._adaptive_depth:
                        prev_embed = torch.cat(
                            [self._embedding[hix].weight[:1].detach(), self._embedding[hix].weight[1:]], dim=0
                        )
                    else:
                        prev_embed = self._embedding[hix].weight
                    if self._force_positive_embeddings:
                        prev_embed = prev_embed.abs()
                    resid_error = resid_error - torch.matmul(all_probs[hix], prev_embed).squeeze(1)  # .detach()

                # resid_error = resid_error.detach()

                if self._cos_sim:
                    distances += cos_sim(resid_error, embedding)
                else:
                    distances += -1.0 * (
                        torch.sum(resid_error**2, dim=1, keepdim=True)
                        + torch.sum(embedding**2, dim=1)
                        - 2 * torch.matmul(resid_error, embedding.t())
                    )
            else:
                if self._cos_sim:
                    distances += cos_sim(this_input, embedding)
                else:
                    embed = embedding.detach() if self._detach else embedding
                    distances += -1.0 * (
                        torch.sum(this_input**2, dim=1, keepdim=True)
                        + torch.sum(embed**2, dim=1)
                        - 2 * torch.matmul(this_input, embed.t())
                    )

            # Convert distances into log probs
            logits = -1.0 * torch.sqrt(-1.0 * distances) if self._sqrt_distances else distances

            if self._cluster_variances is not None:
                logits = logits / self._cluster_variances[head_ix, :]

            if self._logits_warmup_steps > 0:
                logits_weight = min(float(global_step) / float(self._logits_warmup_steps), 1.0)
                logits = logits * logits_weight

            if self.training:
                # gumbel_sched_weight = 2 - 2 / (1 + exp(-float(global_step) / float(self._temp_schedule_gamma)))
                gumbel_sched_weight = exp(
                    -float(global_step)
                    / float(self._temp_schedule_gamma * self._temp_schedule_depth_factor**head_ix)
                )
                gumbel_temp = (
                    max(self._gumbel_temp * gumbel_sched_weight, self._temp_min)
                    if self._temp_schedule
                    else self._gumbel_temp
                )
                probs = torch.nn.functional.gumbel_softmax(
                    logits, tau=gumbel_temp, hard=(not self._soft_gumbel), dim=-1
                )
            else:
                gumbel_temp = self._gumbel_temp
                indices = torch.argmax(logits, dim=-1)
                probs = onehot(indices, N=logits.shape[-1])

            # Prior should only be included (if at all) for first head - others are dependent on previous levels
            # posterior = torch.softmax(logits, dim=-1)
            posterior = torch.nn.functional.softmax(logits, dim=-1)
            # if head_ix == 0:
            #     print(posterior.max(dim=1).indices, posterior.min(dim=1).indices, posterior.mean(dim=1))

            if self._learnable_priors is not None and head_ix == 0:
                prior = (
                    torch.softmax(self._learnable_priors[head_ix], dim=-1).unsqueeze(0).expand(posterior.shape[0], -1)
                )
            else:
                prior = torch.ones_like(posterior).detach() / torch.ones_like(posterior).sum(-1, keepdim=True).detach()

            dev_str = "train" if self.training else "dev"

            kl_warmup_weight = (
                min(float(global_step) / float(self._kl_warmup_steps), 1.0) if self._kl_warmup_steps > 0 else 1.0
            )
            kl = self._kl_loss(nn.functional.log_softmax(logits, dim=-1), prior).sum(dim=-1)
            Logger().log_scalar(f"hrq_{dev_str}/{head_ix}/kl", kl.mean(), global_step)
            if self._kl_weight > 0:
                loss += kl * self._kl_weight * kl_warmup_weight

            Logger().log_scalar(f"hrq_{dev_str}/{head_ix}/temp", gumbel_temp, global_step)

            Logger().log_scalar(
                f"hrq_{dev_str}/{head_ix}/probs_ent",
                -1.0 * torch.sum(posterior * torch.log(posterior + 1e-10), dim=1).mean(),
                global_step,
            )
            Logger().log_scalar(
                f"hrq_{dev_str}/{head_ix}/probs_ent_batch",
                -1.0
                * torch.sum(
                    posterior
                    / torch.sum(posterior, dim=0)
                    * torch.log(posterior / torch.sum(posterior, dim=0) + 1e-10),
                    dim=0,
                ).mean(),
                global_step,
            )
            Logger().log_scalar(f"hrq_{dev_str}/{head_ix}/probs_min", torch.min(posterior), global_step)
            Logger().log_scalar(f"hrq_{dev_str}/{head_ix}/probs_max", torch.max(posterior), global_step)
            # Logger().log_histogram(f"hrq_{dev_str}/probs_hist_" + str(head_ix), probs.cpu().detach().tolist(), global_step)
            Logger().log_scalar(
                f"hrq_{dev_str}/{head_ix}/norm_input", torch.linalg.vector_norm(resid_error, dim=1).mean(), global_step
            )
            Logger().log_scalar(
                f"hrq_{dev_str}/{head_ix}/norm_meaninput",
                torch.linalg.vector_norm(resid_error.mean(dim=0)),
                global_step,
            )
            Logger().log_scalar(
                f"hrq_{dev_str}/{head_ix}/norm_embed",
                torch.linalg.vector_norm(embedding, dim=1).mean(),
                global_step,
            )

            all_probs.append(probs.unsqueeze(1))

        if forced_codes is not None:
            assert (
                forced_codes.shape[1] == self._num_heads
            ), "If forced_codes is supplied, it must be the same length as the number of quantizer heads! {:} vs {:}".format(
                forced_codes.shape[1], self._num_heads
            )
            vq_codes = forced_codes.unbind(dim=1)
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
        for head_ix, embedding in enumerate(self._embedding):
            # If soft training, use distribution
            if self.training:
                embedding = embedding.weight

                if self._force_positive_embeddings:
                    embedding = embedding.abs()

                if self._adaptive_depth:
                    embedding = torch.cat([embedding[:1].detach(), embedding[1:]], dim=0)
                # else:
                #     embedding = embedding.weight
                # # mask grad for the first index
                # quantized_null = torch.matmul(
                #     all_probs[head_ix][:, :, :1],
                #     embedding.weight[:1].detach(),
                # )
                # quantized_normal = torch.matmul(
                #     all_probs[head_ix][:, :, 1:],
                #     embedding.weight[1:],
                # )
                # this_quantized = quantized_null + quantized_normal
                # else:
                this_quantized = torch.matmul(
                    all_probs[head_ix],
                    embedding,
                )

            # otherwise use one hot
            else:
                this_quantized = embedding(vq_codes[head_ix].type(torch.LongTensor).to(inputs.device)).unsqueeze(1)
                if self._force_positive_embeddings:
                    this_quantized = this_quantized.abs()

            quantized_list.append(this_quantized)

        quantized = torch.cat(quantized_list, dim=1)

        # print(quantized.shape)
        # print(inputs.shape)

        if self._norm_loss_weight is not None:
            upper_norms = torch.linalg.vector_norm(quantized[:, :-1, :], dim=-1)
            lower_norms = torch.linalg.vector_norm(quantized[:, 1:, :], dim=-1)
            if self._norm_loss_diff:
                norm_loss = (
                    torch.max(lower_norms * self._norm_loss_scale - upper_norms, torch.zeros_like(lower_norms))
                ) ** 2
            else:
                norm_loss = (
                    torch.max(lower_norms / upper_norms * self._norm_loss_scale, torch.ones_like(lower_norms)) - 1.0
                ) ** 2

            loss += norm_loss.mean(dim=1) * self._norm_loss_weight
            Logger().log_scalar(f"hrq_{dev_str}/norm_loss", norm_loss.mean(), global_step)

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
            head_drop_weight = (
                exp(-float(global_step) / float(self._head_dropout_schedule_gamma))
                if self._head_dropout_schedule_gamma is not None
                else 1.0
            )
            head_drop_rate = (
                max(self._head_dropout * head_drop_weight, self._head_dropout_min)
                if self._head_dropout_min is not None
                else self._head_dropout * head_drop_weight
            )

            Logger().log_scalar(f"hrq_{dev_str}/head_drop_rate", head_drop_rate, global_step)

            drop_dist = torch.distributions.Bernoulli(1 - head_drop_rate)

            mask = drop_dist.sample(sample_shape=(*quantized.shape[:-1], 1))
            if self._head_dropout_keep_first:
                mask[:, 0, :] = 1.0
            mask = torch.cumprod(mask, dim=1).to(quantized.device)
            quantized = quantized * mask

        if self._adaptive_depth and self._adaptive_cascade:
            # propagate null codes down to lower levels
            mask = torch.where(torch.stack(vq_codes, dim=1) > 0, 1, 0)
            mask[:, 0] = 1
            mask = mask.cumprod(dim=1)

            # right shift the mask, to allow grad to propagate through the first zero index
            mask_codes = torch.cat([torch.ones_like(mask)[:, :1], mask[:, :-1]], dim=1)

            vq_codes = (torch.stack(vq_codes, dim=1) * mask_codes).unbind(dim=1)
            quantized = quantized * mask.unsqueeze(dim=2)

        if self._diversity_penalty_weight is not None:
            # partial_embeddings = torch.cumsum(quantized, dim=1)
            pairwise_diffs = torch.linalg.vector_norm(quantized.unsqueeze(0) - quantized.unsqueeze(1), -1).mean(-1) * (
                1 - torch.eye(quantized.shape[0], device=quantized.device).unsqueeze(-1)
            )

            norms = torch.linalg.vector_norm(quantized, dim=-1).mean(-1)
            loss += (norms - pairwise_diffs.mean().unsqueeze(0)) * self._diversity_penalty_weight

        # Stash the embeddings of all subpaths for use in training losses
        subpath_embeddings = torch.cumsum(quantized, dim=1)

        if self._output_cumsum:
            quantized = torch.cumsum(quantized, dim=1)
        elif not self._output_seq:
            quantized = torch.sum(quantized, dim=1)

        if self._include_residual:
            quantized += resid_error * (residual_mask if residual_mask is not None else 1.0)
            if self._residual_penalty > 0:
                loss += torch.linalg.norm(resid_error, dim=-1) * self._residual_penalty

        if not self._output_seq and not self._output_cumsum:
            quantized = quantized.view(input_shape)

        if self._residual_warmup_steps > 0 and self.training:
            residual_alpha = max(1.0 - 1.0 * float(global_step) / float(self._residual_warmup_steps), 0.0)
            quantized = (1 - residual_alpha) * quantized + residual_alpha * inputs

        if self._residual_dropout_steps > 0 and self.training:
            resid_drop_rate = max(1.0 - 1.0 * float(global_step) / float(self._residual_dropout_steps), 0.0)
            dropper = torch.distributions.Bernoulli(1 - resid_drop_rate)
            mask = dropper.sample(sample_shape=(quantized.shape)).to(quantized.device)
            quantized = quantized * mask + inputs * (1 - mask)

        if self._adaptive_depth and self._adaptive_penalty_weight > 0:
            # penalize non-zero code indices, after the first level
            adaptive_penalty = (
                torch.cat(all_probs, dim=1)[:, 1:, 1:].mean(dim=2).sum(dim=1) * self._adaptive_penalty_weight
            )
            loss += adaptive_penalty

        if self._simple_norm:
            quantized = quantized * self._simple_norm_weight + self._simple_norm_bias

        if self._post_linear is not None:
            quantized = self._post_linear(quantized)

        if self._post_scale:
            quantized = quantized * sqrt(self._embedding_dim)

        commitment_loss = nn.functional.mse_loss(this_input, quantized[:, 0, :], reduction="none").mean(dim=-1)

        Logger().log_scalar(f"hrq_{dev_str}/commitment_loss", commitment_loss.mean(), global_step)
        if self._commitment_weight > 0:
            loss += commitment_loss * self._commitment_weight

        Logger().log_scalar(
            f"hrq_{dev_str}/norm_output",
            torch.linalg.vector_norm(quantized[:, 0, :], dim=1).mean(),
            global_step,
        )

        if self._debug.get("bypass", False):
            if self._debug.get("round_digits", None) is not None:
                inputs = torch.round(inputs, decimals=self._debug.get("round_digits", None))
            return 0.0, inputs, vq_codes

        return loss, quantized, vq_codes, subpath_embeddings
