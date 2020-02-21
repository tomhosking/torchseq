#!/usr/bin/python3

import numpy as np

from math import floor
import csv

filtered_pairs = []

included = 0
skipped = 0

np.random.seed(0)

TRAIN_FRACTION = 0.9

with open('/mnt/ext/phd/data/kaggle-questionpairs/questions.csv') as f:
    with open('./data/kaggle/paraphrases.train.txt', 'w') as f_train:
        with open('./data/kaggle/paraphrases.dev.txt', 'w') as f_dev:
            csv_reader = csv.reader(f)
            for ix, cols in enumerate(csv_reader):
                if ix == 0:
                    continue

                # cols = line.split('\t')

                if len(cols[3]) > 500 or len(cols[4]) > 500:
                    continue

                # if  == '1':
                if np.random.random() > TRAIN_FRACTION and cols[5] == '1':
                    f_dev.write('{:}\t{:}\t{:}\n'.format(cols[3], cols[4], cols[5]))
                else:
                    f_train.write('{:}\t{:}\t{:}\n'.format(cols[3], cols[4], cols[5]))
                included += 1
                # else:
                #     skipped += 1


print(included, skipped)