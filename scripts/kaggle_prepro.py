#!/usr/bin/python3

import numpy as np

from math import floor
import csv

filtered_pairs = []

included = 0
skipped = 0

np.random.seed(0)

TRAIN_FRACTION = 0.9

with open('./data/kaggle/questions.csv') as f:
    with open('./data/kaggle/paraphrases.train.txt', 'w') as f_train:
        with open('./data/kaggle/paraphrases.dev.txt', 'w') as f_dev:
            csv_reader = csv.reader(f)
            for ix, cols in enumerate(csv_reader):
                if ix == 0:
                    continue

                # cols = line.split('\t')

                if len(cols[3]) > 500 or len(cols[4]) > 500:
                    continue

                if cols[5] == '1':
                    if np.random.random() > TRAIN_FRACTION:
                        f_dev.write('{:}\t{:}\n'.format(cols[3], cols[4]))
                    else:
                        f_train.write('{:}\t{:}\n'.format(cols[3], cols[4]))
                    included += 1
                else:
                    skipped += 1


print(included, skipped)