from typing import Optional

from torchseq.utils.functions import initialize_truncated_normal_

import math

import torch
import torch.nn as nn


class MultiHeadedPooling(nn.Module):
    dim_per_head: int
    model_dim: int
    model_dim_out: int
    head_count: int

    def __init__(
        self,
        head_count,
        model_dim,
        dropout=0.1,
        use_bilinear=False,
        use_final_linear=True,
        model_dim_out=None,
        use_layer_norm=False,
        query_token_ix=None,
    ):
        super(MultiHeadedPooling, self).__init__()

        assert model_dim % head_count == 0, "model_dim must be divisible by head_count!"
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.model_dim_out = model_dim_out if model_dim_out is not None else model_dim
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, head_count)
        if query_token_ix is not None:
            self.linear_query = nn.Linear(model_dim, model_dim)
        self.bilinear_keys = nn.Bilinear(model_dim, model_dim, head_count) if use_bilinear else None
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if use_final_linear:
            self.final_linear = nn.Linear(model_dim, model_dim_out)
        self.use_final_linear = use_final_linear

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.final_layer_norm = nn.LayerNorm(model_dim)

        self.query_token_ix = query_token_ix

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        query: Optional[torch.Tensor] = None,
        mask=None,
    ) -> torch.Tensor:
        batch_size = key.size(0)

        def shape(x: torch.Tensor, dim=self.dim_per_head) -> torch.Tensor:
            """projection"""
            return x.view(batch_size, -1, self.head_count, dim).transpose(1, 2)

        def unshape(x: torch.Tensor, dim=self.dim_per_head) -> torch.Tensor:
            """compute context"""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_count * dim)

        if self.query_token_ix is not None:
            # print(key.shape)
            query = key[:, self.query_token_ix, :].unsqueeze(1)
            keys = shape(
                torch.stack([x for i, x in enumerate(torch.unbind(key, dim=1)) if i != self.query_token_ix], dim=1)
            )
            # print(keys.shape, query.shape)
            scores = (shape(self.linear_query(query)) * keys).sum(-1)
            # print(scores.shape)
            value = torch.stack(
                [x for i, x in enumerate(torch.unbind(value, dim=1)) if i != self.query_token_ix], dim=1
            )
            mask = torch.stack([x for i, x in enumerate(torch.unbind(mask, dim=1)) if i != self.query_token_ix], dim=1)
            value = self.linear_values(value)
            # print(value.shape)
        else:
            scores = self.linear_keys(key) if query is None else self.bilinear_keys(key, query)
            value = self.linear_values(value)

        scores = shape(scores, 1).squeeze(-1)
        value = shape(value)
        # key_len = key.size(2)
        # query_len = query.size(2)
        #
        # scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -torch.inf)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.sum((drop_attn.unsqueeze(-1) * value), -2)
        context = unshape(context).squeeze(1)
        if self.use_final_linear:
            output = self.final_linear(context)
            return self.final_layer_norm(output) if self.use_layer_norm else output
        else:
            return self.final_layer_norm(context) if self.use_layer_norm else context


class MultiHeadPoolingFast(nn.Module):
    model_dim: int
    num_heads: int
    head_dim: int
    dropout_p: float

    query_is_learned: bool
    learned_query: nn.Parameter
    query_token_ix: Optional[int]

    def __init__(
        self, model_dim: int, num_heads: int, dropout_p: float = 0.1, query_token_ix: Optional[int] = None
    ) -> None:
        super(MultiHeadPoolingFast, self).__init__()

        assert model_dim % num_heads == 0, "model_dim must be divisible by head_count!"

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.key_proj = nn.Linear(self.model_dim, self.model_dim)
        self.value_proj = nn.Linear(self.model_dim, self.model_dim)

        self.dropout_p = dropout_p

        if query_token_ix is not None:
            self.query_is_learned = False
            self.query_token_ix = query_token_ix
            self.query_proj = nn.Linear(self.model_dim, self.model_dim)
        else:
            self.query_is_learned = True
            query_init = torch.zeros(model_dim)
            query_init.normal_()
            self.learned_query = nn.Parameter(query_init)

    def forward(self, sequence: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, model_dim = sequence.shape
        assert model_dim == self.model_dim, f"Received a sequence with dim = {model_dim}, expected = {self.model_dim}!"

        if self.query_is_learned:
            q = self.learned_query.view(1, self.model_dim).expand(bsz, -1)
            k = self.key_proj(sequence)
            v = self.value_proj(sequence)
        else:
            # Use one of the seq tokens as the query, and remove it from the keys/values
            q = sequence[:, self.query_token_ix, :].unsqueeze(1)
            sequence_dropped = torch.stack(
                [x for i, x in enumerate(torch.unbind(sequence, dim=1)) if i != self.query_token_ix], dim=1
            )
            key_padding_mask = torch.stack(
                [x for i, x in enumerate(torch.unbind(key_padding_mask, dim=1)) if i != self.query_token_ix], dim=1
            )

            seq_len = seq_len - 1
            k = self.key_proj(sequence_dropped)
            v = self.value_proj(sequence_dropped)
            q = self.query_proj(q)

        attn_mask = torch.zeros_like(key_padding_mask, dtype=torch.float).masked_fill(key_padding_mask, -torch.inf)
        attn_mask = attn_mask.reshape(bsz, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1).contiguous()

        q = q.reshape(bsz, self.num_heads, 1, self.head_dim).contiguous()
        k = k.reshape(bsz, self.num_heads, seq_len, self.head_dim).contiguous()
        v = v.reshape(bsz, self.num_heads, seq_len, self.head_dim).contiguous()

        if True:
            q = q.view(bsz, self.num_heads, 1, self.head_dim)
            k = k.view(bsz, self.num_heads, seq_len, self.head_dim)
            v = v.view(bsz, self.num_heads, seq_len, self.head_dim)
            attn_output = nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask, self.dropout_p, is_causal=False
            )

            # Drop the sequence dim and re concat the heads
            attn_output = attn_output.view(bsz, self.model_dim)

        else:
            q = q.view(bsz * self.num_heads, 1, self.head_dim)
            k = k.view(bsz * self.num_heads, seq_len, self.head_dim)
            v = v.view(bsz * self.num_heads, seq_len, self.head_dim)
            attn_mask = attn_mask.view(bsz * self.num_heads, 1, seq_len)

            B, Nt, E = q.shape
            q_scaled = q / math.sqrt(E)

            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)
            if self.dropout_p > 0.0:
                attn_output_weights = nn.functional.dropout(attn_output_weights, p=self.dropout_p)

            attn_output = torch.bmm(attn_output_weights, v)

            attn_output = attn_output.contiguous().view(bsz, self.model_dim)

            # attn_output = attn_output.view(1, bsz, attn_output.size(1))

            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, 1, seq_len)

            # print(attn_output_weights.isnan().any())

        return attn_output
