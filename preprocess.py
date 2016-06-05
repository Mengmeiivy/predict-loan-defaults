import numpy as np
from sklearn.cluster import KMeans
import csv
import re
import string
from nltk.stem.lancaster import LancasterStemmer
import math
import heapq


def preprocess(routes):
    np.random.seed(0)
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

    print("======== Preprocessing ========")
    print("Length of Data Read:", len(training_data))

    # Balancing Data and read in loan descriptions
    filtered_fv, filtered_label, desc_bad, desc_good, filtered_desc, filtered_idx = balance_data(training_data,training_label, training_desc)

    # preprocess loan descriptions
    desc_bad = desc_process(desc_bad)
    desc_good = desc_process(desc_good)

    # calculate term frequencies
    word = {} # will have 14923 unique words in it
    for i in range (len(desc_bad)):
        for j in range (len(desc_bad[i])):
            if desc_bad[i][j] not in word:
                word[desc_bad[i][j]] = {}
            word[desc_bad[i][j]]['LOCAL'] = 1
        for j in range (len(desc_bad[i])):
            word[desc_bad[i][j]]['BAD'] = word[desc_bad[i][j]].get('BAD', 0.0) + word[desc_bad[i][j]]['LOCAL']
            word[desc_bad[i][j]]['LOCAL'] = 0
    for i in range (len(desc_good)):
        for j in range (len(desc_good[i])):
            if desc_good[i][j] not in word:
                word[desc_good[i][j]] = {}
            word[desc_good[i][j]]['LOCAL'] = 1
        for j in range (len(desc_good[i])):
            word[desc_good[i][j]]['GOOD'] = word[desc_good[i][j]].get('GOOD', 0.0) + word[desc_good[i][j]]['LOCAL']
            word[desc_good[i][j]]['LOCAL'] = 0

    # calculate the gap and normalize
    gap = {}
    for key in word:
        gap[key] = abs(word[key].get('BAD', 1)-word[key].get('GOOD', 1))\
        /max(word[key].get('BAD', 1), word[key].get('GOOD', 1))

    # find the words with the largest gap
    wordlength = 19
    k_keys_sorted = heapq.nlargest(wordlength, gap, key=gap.get)


    # turning categorical features to binary
    features_binary = []
    non_ctgr = set()
    for f in useful_feature:
        f = re.split('[:,]+', f)
        if len(f) == 2:
            features_binary.append(f[0])
            non_ctgr.add(f[0])
        else:
            cur_f = f[0]
            for i in range(1, len(f)):
                features_binary.append(cur_f + "$" + f[i])

    grade_dict = {'A' : 0, 'B': 5, 'C': 15, 'D': 20, 'E':25, 'F': 30, 'G':35}
    fv_binary = []

    for d in filtered_fv:
        tmp_fv = []
        for f in features_binary:
            if f in non_ctgr:
                idx = uf_names.index(f)
                val = d[idx]
                # if f == 'sub_grade':
                #     v = d[idx][:1]
                #     grade = int(d[idx][1:])
                #     tmp_fv.append(grade_dict[v] + grade)
                if f == 'emp_length':
                    if '<' in val:
                        tmp_fv.append(0)
                    elif '+' in val:
                        tmp_fv.append(10)
                    else:
                        tmp_fv.append(val.split()[0])
                elif f == 'earliest_cr_line':
                    tmp_fv.append(int(val.split('-')[1]))
                elif '%' in val:
                    tmp_fv.append(float(val.split('%')[0]))
                elif '.' in val:
                    tmp_fv.append(float(val))
                else:
                    tmp_fv.append(int(val))

            else:
                rfv = f.split('$')
                f_key = rfv[0]
                idx = uf_names.index(f_key)
                f_val = rfv[1]
                if d[idx] == f_val:
                    tmp_fv.append(1)
                else:
                    tmp_fv.append(0)

        # add useful words into features
        # desc = []
        # desc.append(d[11])
        # desc = desc_process(desc)
        # for word in k_keys_sorted:
        #     if word in desc[0]:
        #         tmp_fv.append(1)
        #     else:
        #         tmp_fv.append(0)
        fv_binary.append(tmp_fv)

    is_conti = []
    for f in features_binary:
        if '$' in f:
            is_conti.append(0)
        else:
            is_conti.append(1)
    for f in k_keys_sorted:
        is_conti.append(0)
    # normalize
    maxes, mins = findminmax(fv_binary)
    fv_binary = norm(maxes, mins, fv_binary, is_conti)

    rand = np.random.permutation(len(fv_binary))
    idx_file = open('index_file.txt', 'w')
    tmp_fv = []
    tmp_label = []
    for i in rand:
        idx_file.write(str(filtered_idx[i]))
        idx_file.write('\n')
        tmp_fv.append(fv_binary[i])
        tmp_label.append(filtered_label[i])
    fv_binary = tmp_fv
    filtered_label = tmp_label

    label_file = open('label_file.txt', 'w')
    label_file.write(str(filtered_label[0]))
    for i in range(1, len(filtered_label)):
        label_file.write(",")
        label_file.write(str(filtered_label[i]))

    desc_file = open('desc_file', 'w')
    for i in rand:
        desc_file.write(filtered_desc[i])
        desc_file.write('\n')
    desc_file.close()

    processed_file = open('processed_file.txt', 'w')
    for f in fv_binary:
        processed_file.write(str(f[0]))
        for i in range(1, len(f)):
            processed_file.write(',')
            processed_file.write(str(f[i]))
        processed_file.write('\n')
    processed_file.close()

    processed_feature = open('processed_feature.txt', 'w')
    processed_feature.write(features_binary[0])
    for i in range(1, len(features_binary)):
        processed_feature.write('^')
        processed_feature.write(features_binary[i])
    # for i in range(0, 19):
    #     processed_feature.write('^')
    #     processed_feature.write("word")
    processed_feature.close()



def norm(maxes, mins, fv_binary, is_conti):
    for v in fv_binary:
        for i, vv in enumerate(v):
            if maxes[i] == 0 or is_conti[i] == 0:
                continue
            else:
                vv = np.float(vv)
                v[i] = np.divide(np.subtract(vv, mins[i]), np.subtract(maxes[i],mins[i]))

    return fv_binary

def findminmax(x):
    v = np.array(x).astype(np.float)
    maxes = np.amax(v, axis=0)
    mins = np.amin(v, axis=0)
    return maxes, mins

# preprocess loan description
def desc_process(desc):
    # remove HTML tags, punctuations, special characters, numbers, etc
    for i in range (len(desc)):
        desc[i] = re.sub(r'<br>|<br/>|Borrower added on', r'', desc[i])
        desc[i] = re.sub(r',|\.|/|-', r' ', desc[i])

    # remove stop words from the description and stem the remaining words
    st = LancasterStemmer()
    closed_class_stop_words = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed','i', 'im', "I'm",\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or'
                           ]
    for i in range (len(desc)):
        desc[i] = ''.join([x for x in desc[i] if x in string.ascii_letters + '\'- '])
        desc[i] = ' '.join(desc[i].split())
        temp_list = []
        string1 = desc[i].split()
        for j in range (len(string1)):
            if (string1[j] not in closed_class_stop_words):
                string1[j] = st.stem(string1[j])
                temp_list.append(string1[j])
        desc[i] = temp_list

    return desc


# Balancing Data
def balance_data(training_data, training_label, training_desc):
    # count 0 and 1 in res
    zs = []
    os = []
    desc_bad = []
    desc_good = []

    for i in range(0, len(training_data)):
        if training_label[i] == -1:
            zs.append(i)
        else:
            os.append(i)
    zs = np.random.choice(zs, 7500)
    chosen_os = np.random.choice(os, len(zs))

    filtered_idx = []
    filtered_fv = []
    filtered_label = []
    filtered_desc = []
    for i in zs:
        filtered_idx.append(i)
        filtered_fv.append(training_data[i])
        filtered_label.append(training_label[i])
        filtered_desc.append(training_desc[i])
        desc_bad.append(training_data[i][11])
    for i in chosen_os:
        filtered_idx.append(i)
        filtered_fv.append(training_data[i])
        filtered_label.append(training_label[i])
        filtered_desc.append(training_desc[i])
        desc_good.append(training_data[i][11])

    print("Length of Balanced Training Data:", len(filtered_fv))
    return filtered_fv, filtered_label, desc_bad, desc_good, filtered_desc, filtered_idx

def execute():
    info = preprocess(['LoanStats3a_securev1.csv', 'LoanStats3b_securev1.csv'])

'''============================== Main ===================================== '''
execute()
