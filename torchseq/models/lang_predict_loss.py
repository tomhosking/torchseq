import torch
import torch.nn as nn

from torchseq.models.pooling import MultiHeadedPooling
from torchseq.utils.tokenizer import FAIRSEQ_LANGUAGE_CODES


class LangPredictLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        tgt_langs = ["en_XX", "fr_XX", "de_DE", "zh_CN", "es_XX", "ar_AR"]

        self.lang_code_to_ix = {v: tgt_langs.index(k) for k, v in FAIRSEQ_LANGUAGE_CODES.items() if k in tgt_langs}

        self.pooling = MultiHeadedPooling(
            1,
            config.encoder.embedding_dim,
            dropout=config.dropout,
            model_dim_out=config.encoder.embedding_dim,
            use_final_linear=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.encoder.embedding_dim, config.encoder.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.encoder.embedding_dim, config.encoder.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.encoder.embedding_dim, len(tgt_langs)),
        )

        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, encoding, memory, lang):

        pooled = self.pooling(encoding, encoding)

        logits = self.classifier(pooled)

        lang_ix_batch = torch.tensor([self.lang_code_to_ix[x.item()] for x in lang]).to(logits.device)

        lang_pred_loss = self.loss(logits, lang_ix_batch)

        return lang_pred_loss
