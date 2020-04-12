#!/usr/bin/python3

import numpy as np

from math import floor
import csv

filtered_pairs = []

included = 0
skipped = 0

np.random.seed(0)

DEVTEST_FRACTION = 0.1

qids_train = set()
qids_dev = set()
qids_test = set()

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

            qid1 = cols[1]
            qid2 = cols[2]

            if qid1 in qids_train or qid2 in qids_train:
                print('Forcing to train')
                f_train.write('{:}\t{:}\n'.format(cols[3], cols[4]))
                
            elif qid1 in qids_dev or qid2 in qids_dev:
                print('Forcing to dev')
                f_dev.write('{:}\t{:}\n'.format(cols[3], cols[4]))
            elif qid1 in qids_test or qid2 in qids_test:
                print('Forcing to test')
                f_test.write('{:}\t{:}\n'.format(cols[3], cols[4]))
            else:
                # 0.8-0.9
                if rand >= (1-DEVTEST_FRACTION*2) and rand < (1-DEVTEST_FRACTION) and cols[5] == '1':

                    f_dev.write('{:}\t{:}\n'.format(cols[3], cols[4]))
                    qids_dev.add(qid1)
                    qids_dev.add(qid2)
                # 0.9+
                elif rand >= (1-DEVTEST_FRACTION) and cols[5] == '1':
                    f_test.write('{:}\t{:}\n'.format(cols[3], cols[4]))
                    qids_test.add(qid1)
                    qids_test.add(qid2)
                # Under 0.8
                else:
                    f_train.write('{:}\t{:}\n'.format(cols[3], cols[4]))
                    qids_train.add(qid1)
                    qids_train.add(qid2)
            included += 1
        else:
            skipped += 1


print(included, skipped)