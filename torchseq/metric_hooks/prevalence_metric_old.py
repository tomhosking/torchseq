# Modified version of the metric from https://github.com/cdmalon/opinion-prevalence
# Original is a CLI, this provides a more generic OO wrapper
# See https://arxiv.org/abs/2307.14305

import nltk.tokenize
from summac.model_summac import SummaCZS
from tqdm import tqdm


class PrevalenceMetric:
    threshold = 0.04

    def __init__(self):
        self.model = SummaCZS(
            granularity="document", model_name="mnli", bins="percentile", use_con=False, device="cuda"
        )

    def get_prevalence(
        self,
        reviews,
        generated_summaries,
        product_names=None,
        pbar=False,
        ignore_redundancy=False,
        summaries_are_sentences=False,
    ):
        threshold = 0.04

        if product_names is None:
            product_names = [""] * len(reviews)

        prevalences = []
        redundancies = []
        trivials = []
        for curr_reviews, summ, product_name in tqdm(
            zip(reviews, generated_summaries, product_names), disable=(not pbar), total=len(reviews)
        ):
            nsent = 0
            prevalence = 0
            redundancy = 0
            trivial_count = 0

            if not summaries_are_sentences:
                sents = nltk.tokenize.sent_tokenize(summ)
            else:
                sents = summ

            for i, generated in enumerate(sents):
                nsent = nsent + 1

                implied = 0
                tot = 0

                # trivial = "I bought {:}.".format(product_name)
                trivial = "I stayed at {:}.".format(product_name)
                if self.model.score([trivial], [generated])["scores"][0] > threshold:
                    # output = output + " " + generated + " (T)"
                    trivial_count += 1
                    # print('Triv: ', generated)
                    continue

                redundant = False
                for j in range(i):
                    if self.model.score([sents[j]], [generated])["scores"][0] > threshold:
                        # print("Redund: ", sents[j], generated)
                        redundant = True
                        redundancy += 1
                        break

                if redundant and not ignore_redundancy:
                    continue

                for original in curr_reviews:
                    tot = tot + 1
                    score = self.model.score([original], [generated])["scores"][0]
                    if score > threshold:
                        implied = implied + 1
                # print(implied/tot)

                prevalence = prevalence + (implied / tot)
                # output = output + " " + generated + " (" + str(implied) + ")"

            prevalence = prevalence / nsent

            prevalences.append(prevalence)
            redundancies.append(redundancy / nsent)
            trivials.append(trivial_count / nsent)

        return prevalences, redundancies, trivials
