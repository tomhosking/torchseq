#!/usr/bin/python3

import numpy as np

from math import floor

filtered_pairs = []

included = 0
skipped = 0

np.random.seed(0)

TRAIN_FRACTION = 0.9

with open('./data/paranmt/para-nmt-50m.txt') as f:
    with open('./data/paranmt/paraphrases.train.txt', 'w') as f_train:
        with open('./data/paranmt/paraphrases.dev.txt', 'w') as f_dev:
            for line in f:
                cols = line.strip().split('\t')

                if line.strip() == '' or len(cols[0]) > 300 or len(cols[1]) > 300 or len(cols[0]) < 30 or len(cols[1]) < 30:
                    continue

                if float(cols[2]) < 0.9 and float(cols[2]) > 0.5:
                    if np.random.random() > TRAIN_FRACTION:
                        f_dev.write('{:}\t{:}\n'.format(cols[1].strip(), cols[0].strip()))
                    else:
                        f_train.write('{:}\t{:}\n'.format(cols[1].strip(), cols[0].strip()))
                    included += 1
                else:
                    skipped += 1

with open('./data/parabank/parabank-1.0-large-diverse/parabank.50m.tsv') as f:
    with open('./data/parabank/paraphrases.train.txt', 'w') as f_train:
        with open('./data/parabank/paraphrases.dev.txt', 'w') as f_dev:
            for line in f:
                cols = line.strip().split('\t')

                if line.strip() == '' or len(cols[0]) > 300 or len(cols[1]) > 300 or len(cols[0]) < 30 or len(cols[1]) < 30:
                    continue

                
                if np.random.random() > TRAIN_FRACTION:
                    f_dev.write('{:}\t{:}\n'.format(cols[1].strip(), cols[0].strip()))
                else:
                    f_train.write('{:}\t{:}\n'.format(cols[1].strip(), cols[0].strip()))
                included += 1


print(included, skipped)