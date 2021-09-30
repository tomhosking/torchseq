import math

import torch
import torch.nn as nn


class MultiHeadedPooling(nn.Module):
    def __init__(
        self,
        head_count,
        model_dim,
        dropout=0.1,
        use_bilinear=False,
        use_final_linear=True,
        model_dim_out=None,
        use_layer_norm=False,
    ):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.model_dim_out = model_dim_out if model_dim_out is not None else model_dim
        super(MultiHeadedPooling, self).__init__()
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim, head_count)
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

    def forward(self, key, value, query=None, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x, dim=dim_per_head):
            """projection"""
            return x.view(batch_size, -1, head_count, dim).transpose(1, 2)

        def unshape(x, dim=dim_per_head):
            """compute context"""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim)

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
            scores = scores.masked_fill(mask, -1e18)

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
