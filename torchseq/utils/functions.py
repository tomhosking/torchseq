import torch
import torch.nn as nn
import numpy as np
import math


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output * 1.0


# FROM: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check

    orig_shape = logits.shape

    logits = logits.view(-1, orig_shape[-1])
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits.reshape(orig_shape)


# Return the cosine similarity between x, y
def cos_sim(x, y):
    # prod = x*y
    prod = torch.matmul(x, y.T)
    norm = torch.matmul(x.norm(dim=-1, keepdim=True), y.norm(dim=-1, keepdim=True).T)
    return prod / norm


def reparameterize_gaussian(mu, logvar, var_weight=1.0):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return mu + eps * std * var_weight


from contextlib import contextmanager


@contextmanager
def evaluating(net):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


def batchify(input, batch_size=1, shuffle=False):
    if shuffle:
        np.random.shuffle(input)
    batched = []
    for bix in range(math.ceil((1.0 * len(input)) / batch_size)):
        batch = input[bix * batch_size : (bix + 1) * batch_size]
        batched.append(batch)
    return list(enumerate(batched))
    # yield bix, batch


def initialize_truncated_normal_(tensor, mean=0, std=0.02):
    # size = tensor.shape
    # tmp = tensor.new_empty(size + (4,)).normal_()
    # valid = (tmp < 2) & (tmp > -2)
    # ind = valid.max(-1, keepdim=True)[1]
    # tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    # tensor.data.mul_(std).add_(mean)

    # the pytorch truncnorm init clips by value, not by sigma
    nn.init.trunc_normal_(tensor, mean, std, a=-2.0 * std, b=2.0 * std)


# https://benanne.github.io/2020/09/01/typicality-addendum.html
def initialize_polar_normal_(tensor, mean=0, scale=1):
    direction = tensor.new_empty(tensor.shape).normal_()
    direction /= torch.sqrt(torch.sum(direction**2, dim=-1, keepdim=True))
    distance = tensor.new_empty(tensor.shape[:-1]).normal_(mean=0, std=np.sqrt(tensor.shape[-1]) * scale).unsqueeze(-1)
    tensor.data.copy_(distance * direction + mean)


def to_device_unless_marked(device):
    def _to_device_unless_marked(obj):
        if not getattr(obj, "force_device", False):
            obj.to(device)

    return _to_device_unless_marked


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def _init_bert_params(module):
        def normal_(data):
            # with FSDP, module params will be on CUDA, so we cast them back to CPU
            # so that the RNG is consistent with and without FSDP
            data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

        if isinstance(module, nn.Linear):
            normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, nn.MultiheadAttention):
            if module.q_proj_weight is not None:
                normal_(module.q_proj_weight.data)
                normal_(module.k_proj_weight.data)
                normal_(module.v_proj_weight.data)
            else:
                normal_(module.in_proj_weight.data)

    module.apply(_init_bert_params)
