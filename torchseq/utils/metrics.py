import re
import string
from collections import Counter

from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
from nltk.translate.meteor_score import single_meteor_score

# from torchseq.utils.bleu import compute_bleu
from torchseq.utils.sari import SARIsent

import sacrebleu


def tokenize(text):
    # return text.split(' ')
    sents = sent_tokenize(text)
    tokens = [tok.lower() for sent in sents for tok in TreebankWordTokenizer().tokenize(sent)]
    return tokens


# # takes a single untokenised string as input
# def bleu(gold, prediction, order=4):
#     return compute_bleu([[tokenize(gold)]], [tokenize(prediction)], smooth=False, max_order=order)[0]


# takes a list of untokenized strings as inputs
def bleu_corpus(golds, preds, order=4):
    return sacrebleu.corpus_bleu([p.lower() for p in preds], [[g.lower() for g in golds]]).score
    # return compute_bleu(
    #     [[tokenize(gold)] for gold in golds], [tokenize(pred) for pred in preds], smooth=False, max_order=order
    # )[0]


def ibleu_corpus(golds, preds, inputs, alpha=0.8):
    return alpha * bleu_corpus(golds, preds) - (1 - alpha) * bleu_corpus(preds, inputs)
    # return sum([alpha*bleu(golds[i], preds[i]) - (1-alpha)*bleu(golds[i], inputs[i]) for i in range(len(golds))])/len(golds)


def sari_corpus(golds, preds, inputs):
    return sum([SARIsent(i, p, g) for g, p, i in zip(golds, preds, inputs)]) / len(golds)


def meteor_corpus(golds, preds):
    return sum(
        [
            single_meteor_score(TreebankWordTokenizer().tokenize(g), TreebankWordTokenizer().tokenize(p))
            for g, p in zip(golds, preds)
        ]
    ) / len(golds)


def f1(gold, prediction):
    prediction_tokens = prediction.lower().split()
    ground_truth_tokens = gold.lower().split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
