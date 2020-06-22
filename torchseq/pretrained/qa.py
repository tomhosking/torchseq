import torch
import torch.nn.functional as F
from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
)

from collections import Counter, defaultdict
from tqdm import tqdm


def ceiling_division(n, d):
    return -(n // -d)


class PreTrainedQA:
    def __init__(self, device=None):

        self.device = torch.device("cuda") if device is None else device
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.model = BertForQuestionAnswering.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad"
        ).to(self.device)

        # self.tokenizer = DistilBertTokenizer().from_pretrained('distilbert-base-uncased')
        # self.model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

    def infer_single(self, question, text):
        input_ids = self.tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = self.model(
            torch.tensor([input_ids]).to(self.device), token_type_ids=torch.tensor([token_type_ids]).to(self.device)
        )
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        return " ".join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores) + 1])

    def infer_batch(self, question_list, context_list, silent=False):

        MAX_LEN = 384
        CHUNK_STRIDE = 128

        BATCH_SIZE = 2

        tokenized = self.tokenizer.batch_encode_plus(
            list(zip(question_list, context_list)),
            max_length=MAX_LEN,
            return_overflowing_tokens=True,
            stride=CHUNK_STRIDE,
            truncation_strategy="only_second",
            pad_to_max_length=False,
            return_offsets_mapping=True,
        )

        start_scores, end_scores = [], []
        for bix in tqdm(
            range(ceiling_division(len(tokenized["input_ids"]), BATCH_SIZE)), desc="QA model", disable=silent
        ):

            res = self.model(
                self.pad_batch(tokenized["input_ids"][bix * BATCH_SIZE : (bix + 1) * BATCH_SIZE]),
                token_type_ids=self.pad_batch(tokenized["token_type_ids"][bix * BATCH_SIZE : (bix + 1) * BATCH_SIZE]),
                attention_mask=self.pad_batch(tokenized["attention_mask"][bix * BATCH_SIZE : (bix + 1) * BATCH_SIZE]),
            )

            start_scores.extend(res[0].detach().cpu())
            end_scores.extend(res[1].detach().cpu())
            if bix % 16 == 0:
                torch.cuda.empty_cache()
        # all_tokens_batch = [self.Tokenizer().convert_ids_to_tokens(input_ids) for input_ids in tokenized["input_ids"]]

        all_answers = [
            self.extract_answer(
                context_list[tokenized["overflow_to_sample_mapping"][ix]],
                tokenized["offset_mapping"][ix],
                start_scores[ix],
                end_scores[ix],
                tokenized["token_type_ids"][ix],
            )
            for ix in range(len(start_scores))
        ]

        answer_candidates = [defaultdict(float) for _ in range(len(question_list))]

        for ix in range(len(start_scores)):
            answer_candidates[tokenized["overflow_to_sample_mapping"][ix]][all_answers[ix][0]] += all_answers[ix][1]

        answers = [sorted(x.items(), key=lambda kv: kv[1], reverse=True)[0][0] for x in answer_candidates]

        return answers

    def extract_answer(self, context, char_spans, start_scores, end_scores, type_ids):
        # If the answer is actually in the question... then this has failed!!

        s_ix, e_ix = torch.argmax(start_scores), torch.argmax(end_scores)

        if e_ix <= type_ids.index(1) - 1:
            return "", 0.0

        return context[char_spans[s_ix][0] : char_spans[e_ix][1]], (start_scores[s_ix] + end_scores[e_ix]).item()

    def pad_batch(self, batch):
        pad_id = self.tokenizer.pad_token_id
        max_len = max([len(x) for x in batch])
        padded_batch = [F.pad(torch.tensor(x), (0, max_len - len(x)), value=pad_id) for x in batch]
        return torch.stack(padded_batch, 0).to(self.device)


if __name__ == "__main__":
    instance = PreTrainedQA()

    print(
        instance.infer_batch(
            ["Who was Jim Henson?", "Who was a nice puppet?"],
            ["Jim Henson was a nice puppet", "Jim Henson was a nice puppet" + ". Creme puff was an old cat" * 50],
        )
    )
