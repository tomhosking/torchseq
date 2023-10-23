from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


class PretrainedNliModel:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    pipe: pipeline

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("tomhosking/deberta-v3-base-debiased-nli")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "tomhosking/deberta-v3-base-debiased-nli"
        ).cuda()

        self.ENTAILMENT_LABEL = (
            self.model.config.label2id["ENTAILMENT"]
            if "ENTAILMENT" in self.model.config.label2id
            else self.model.config.label2id["entailment"]
        )

        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=0)

    def get_scores(
        self,
        premises: List[str],
        hypotheses: List[str],
        return_entailment_prob: bool = True,
        bsz: int = 64,
        progress: bool = False,
    ):
        dataset = ListDataset([{"text": p, "text_pair": h} for p, h in zip(premises, hypotheses)])

        outputs = [
            {x["label"]: x["score"] for x in res}
            for res in tqdm(
                self.pipe(dataset, batch_size=bsz, top_k=None, num_workers=2),
                disable=(not progress),
                total=len(dataset),
            )
        ]
        # outputs = [{x["label"]: x["score"] for x in res} for res in outputs]

        if return_entailment_prob:
            return [p["ENTAILMENT"] for p in outputs]
        else:
            return outputs
