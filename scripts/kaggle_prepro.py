#!/usr/bin/python3

import numpy as np

from math import floor
import csv

filtered_pairs = []

included = 0
skipped = 0

np.random.seed(0)

DEVTEST_FRACTION = 0.1

with open('/mnt/ext/phd/data/kaggle-questionpairs/questions.csv') as f, \
    open('./data/kaggle/paraphrases.train.txt', 'w') as f_train, \
    open('./data/kaggle/paraphrases.dev.txt', 'w') as f_dev, \
    open('./data/kaggle/paraphrases.test.txt', 'w') as f_test:
    csv_reader = csv.reader(f)
    for ix, cols in enumerate(csv_reader):
        if ix == 0:
            continue

        # cols = line.split('\t')

        if len(cols[3]) > 500 or len(cols[4]) > 500:
            continue

        if cols[5] == '1':
            rand = np.random.random() 
            # 0.8-0.9
            if rand >= (1-DEVTEST_FRACTION*2) and rand < (1-DEVTEST_FRACTION) and cols[5] == '1':
                f_dev.write('{:}\t{:}\n'.format(cols[3], cols[4]))
            # 0.9+
            elif rand >= (1-DEVTEST_FRACTION) and cols[5] == '1':
                f_test.write('{:}\t{:}\n'.format(cols[3], cols[4]))
            # Under 0.8
            else:
                f_train.write('{:}\t{:}\n'.format(cols[3], cols[4]))
            included += 1
        else:
            skipped += 1


print(included, skipped)