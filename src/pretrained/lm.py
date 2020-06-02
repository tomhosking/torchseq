# from flask import Flask, request, current_app
import json
import logging
import os

import torch
from transformers import GPT2LMHeadModel, GPT2Model, GPT2Tokenizer

from tqdm import tqdm

# TODO: config this
USE_CUDA = True


def ceiling_division(n, d):
    return -(n // -d)


class PretrainedLM:
    def __init__(self):

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Load pre-trained model (weights)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.eval()

        # If you have a GPU, put everything on cuda
        if USE_CUDA:
            self.model.to(torch.device("cuda"))

    def get_log_prob(self, sentences, silent=False):

        if len(sentences) > 32:
            log_probs = []
            for b in tqdm(range(ceiling_division(len(sentences), 32)), desc="LM log probs", disable=silent):
                start_ix = b * 32
                end_ix = min(len(sentences), (b + 1) * 32)
                log_probs.extend(self.get_seq_log_prob(self.get_batch(sentences[start_ix:end_ix])))
        else:
            log_probs = self.get_seq_log_prob(self.get_batch(sentences))

        return log_probs

    def get_batch(self, str_in):
        tok_unpadded = [self.tokenizer.encode(x) for x in str_in]
        max_len = max([len(x) for x in tok_unpadded])
        tok_batch = [x + [0 for i in range(max_len - len(x))] for x in tok_unpadded]
        mask_batch = [[1 for i in range(len(x))] + [0 for i in range(max_len - len(x))] for x in tok_unpadded]

        return tok_batch, mask_batch

    def get_seq_log_prob(self, batch):

        tokens, mask = batch

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(tokens)
        mask_tensor = torch.tensor(mask, dtype=torch.float)

        if USE_CUDA:
            tokens_tensor = tokens_tensor.to(torch.device("cuda"))
            mask_tensor = mask_tensor.to(torch.device("cuda"))

        # Predict all tokens
        with torch.no_grad():
            predictions, _ = self.model(tokens_tensor)
        all_probs = torch.softmax(predictions, -1)

        torch.cuda.empty_cache()

        # print(all_probs.size(), all_probs)
        # print(tokens_tensor.unsqueeze(-1).size(), tokens_tensor.unsqueeze(-1))

        probs = torch.gather(all_probs, 2, tokens_tensor.unsqueeze(-1)).squeeze(-1)

        log_probs = torch.log(probs)
        nll = -1 * torch.mean(log_probs * mask_tensor, -1)
        return nll.tolist()
