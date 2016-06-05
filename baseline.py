import numpy as np
from sklearn.cluster import KMeans
import csv
import re
import string
from nltk.stem.lancaster import LancasterStemmer
import math
import heapq

def preprocess(routes):
    info = []
    grade = []
    label = []

    for route in routes:
        f = open(route, 'rt')
        try:
            reader = csv.reader(f)
            # first line is useless
            row = next(reader)

            try:
                while len(row) > 1 and row[0] != '':
                    try:
                        g = row[8]
                        if g == 'A' or g == 'B' or g == 'C' or g == 'D' or g == 'E' or g == 'F':
                            if row[16] == 'Charged Off' or row[16] == 'Fully Paid':
                                tl = -1 if row[16] == 'Charged Off' else 1
                                grade.append(row[8])
                                label.append(tl)
                        row = next(reader)
                    except UnicodeDecodeError:
                        print('UnicodeDecodeError')
            except UnicodeDecodeError:
                print('UnicodeDecodeError')
        finally:
            f.close()

    print("======== Preprocessing ========")
    print("Length of Data Read:", len(grade),len(label))

    grade, label = balance_data(grade,label)

    info.append(grade)
    info.append(label)

    return info

def balance_data(training_data, training_label):
    # count 0 and 1 in res
    zs = []
    os = []

    for i in range(0, len(training_data)):
        if training_label[i] == -1:
            zs.append(i)
        else:
            os.append(i)
    zs = np.random.choice(zs, 7500)
    chosen_os = np.random.choice(os, len(zs))

    filtered_fv = []
    filtered_label = []
    for i in zs:
        filtered_fv.append(training_data[i])
        filtered_label.append(training_label[i])
    for i in chosen_os:
        filtered_fv.append(training_data[i])
        filtered_label.append(training_label[i])

    print("Length of Balanced Training Data:", len(filtered_fv))
    return filtered_fv, filtered_label

def execute():
    info = preprocess(['LoanStats3a_securev1.csv', 'LoanStats3b_securev1.csv'])
    grade = info[0]
    label = info[1]

    grade_level = ['A','B','C','D','E','F']
    accuracy = []
    for g in grade_level:
        correct = 0;
        print('============================================================================')
        for i in range(0,len(info[0])):
            if ord(grade[i]) <= ord(g):
                if label[i] == 1:
                    correct = correct + 1
            else:
                if label[i] == -1:
                    correct = correct + 1
        accuracy.append(correct/len(info[0]))
    print(accuracy)
execute()
