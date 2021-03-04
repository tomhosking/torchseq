import json
import re
from collections import defaultdict

import numpy as np

# from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
# import spacy


def load_qa_dataset(path, dev=False, test=False, v2=False, filename=None):
    expected_version = "v2.0" if v2 else "1.1"
    if filename is not None:
        pass
    elif v2:
        filename = "train-v2.0.json" if not dev else "dev-v2.0.json"
    elif test and not dev:
        filename = "test-v1.1.json"
    else:
        filename = "train-v1.1.json" if not dev else "dev-v1.1.json"
    with open(path + filename) as dataset_file:
        dataset_json = json.load(dataset_file)
        if "version" in dataset_json and dataset_json["version"] != expected_version and filename is None:
            print("Expected SQuAD v-" + expected_version + ", but got dataset with v-" + str(dataset_json["version"]))
        dataset = dataset_json["data"]
        return dataset


def load_squad_triples(path, dev=False, test=False, v2=False, as_dict=False, ans_list=False, filename=None):
    raw_data = load_qa_dataset(path, dev=dev, test=test, v2=v2, filename=filename)
    triples = [] if not as_dict else {}
    for doc in raw_data:
        for para in doc["paragraphs"]:
            for qa in para["qas"]:
                id = qa["id"]
                # NOTE: this only takes the first answer per question! ToDo handle this more intelligently
                if ans_list:
                    ans_text = [a["text"] for a in qa["answers"]]
                    ans_pos = [int(a["answer_start"]) for a in qa["answers"]]
                else:
                    ans_count = defaultdict(int)
                    for ans in qa["answers"]:
                        ans_count[(ans["text"], int(ans["answer_start"]))] += 1

                    ans_text, ans_pos = sorted(ans_count.items(), reverse=True, key=lambda x: x[1])[0][0]
                if v2:
                    if qa["is_impossible"]:
                        el = (
                            para["context"],
                            qa["question"],
                            qa["plausible_answers"][0]["text"] if not dev else "",
                            int(qa["plausible_answers"][0]["answer_start"]) if not dev else None,
                            True,
                        )
                    else:
                        el = (
                            para["context"],
                            qa["question"],
                            qa["answers"][0]["text"],
                            int(qa["answers"][0]["answer_start"]),
                            False,
                        )
                else:
                    el = (para["context"], qa["question"], ans_text, ans_pos)
                if as_dict:
                    triples[id] = el
                else:
                    triples.append(el)
    return triples
