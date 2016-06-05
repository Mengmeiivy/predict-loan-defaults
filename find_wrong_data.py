import numpy as np
from sklearn.cluster import KMeans
import csv
import re
import string
from nltk.stem.lancaster import LancasterStemmer
import math
import heapq

routes = ['LoanStats3a_securev1.csv', 'LoanStats3b_securev1.csv']
training_data = []
training_label = []
training_desc = []
original_feature_map = {}
file = open('feature.txt', 'r')
useful_feature = file.readline().split('^')
uf_names = []
for f in useful_feature:
    uf_names.append(f.split(':')[0])

for route in routes:
    f = open(route, 'rt')
    try:
        reader = csv.reader(f)
        # first line is useless
        row = next(reader)
        original_feature = next(reader)
        for i in range(0, len(original_feature)):
            original_feature_map[original_feature[i]] = i
        feature_index = []
        for feature in useful_feature:
            feature_index.append(original_feature_map[feature.split(':')[0]])

        row = next(reader)
        while len(row) > 1 and row[0] != '':
            has_n_a = False
            for i in feature_index:
                if 'n/a' in row[i] or len(row[i]) == 0:
                    has_n_a = True
                    break
            if has_n_a:
                row = next(reader)
                continue
            try:
                if row[16] == 'Charged Off' or 'Fully Paid':
                    tl = -1 if row[16] == 'Charged Off' else 1
                    training_label.append(tl)
                    data = []
                    for index in feature_index:
                        data.append(row[index].strip())
                    training_data.append(data)
                    training_desc.append(row[19])
                    row = next(reader)
            except UnicodeDecodeError:
                print('UnicodeDecodeError')
    finally:
        f.close()

idx_mapping = []
idx_file = open('index_file.txt', 'r').readlines()
for i in idx_file:
    idx_mapping.append(int(i))

error_idx = []
err_file = open('error_idx', 'r').readlines()
for e in err_file:
    e = e.split()[0]
    error_idx.append(int(e))

desc = []
desc_file = open('desc_file', 'r').readlines()
for d in desc_file:
    desc.append(d)

for i in error_idx:
    e = idx_mapping[i]
    if training_label[e] == -1:
        print(i, training_data[e], training_label[e], desc[i])
