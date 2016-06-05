from collections import OrderedDict
import re
from stemming.porter2 import stem
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import tree

def get_prob_desc():
	label_file = open('label_file.txt', 'r')
	desc_file = open('desc_file', 'r')
	labels = []
	label = label_file.readline().strip().split(',')
	for l in label:
		labels.append(int(l))
	desc_lines = desc_file.readlines()
	processed_desc, desc_length = process_desc(desc_lines)
	vocab_list = get_vocab_list(processed_desc)
	fv = get_fv(processed_desc, vocab_list, desc_length)
	# for i in range(0, len(fv)):
	# 	fv[i] = np.append(fv[i], desc_length[i])

	# probability
	logistic = linear_model.LogisticRegression(C = 0.1, penalty='l1',)
	# Can use nb or logistic
	#prob = logistic.fit(fv[:9600], labels[:9600]).predict_proba(fv)
	'''tr = tree.DecisionTreeClassifier(max_depth = 8)'''
	prob = logistic.fit(fv[:9600], labels[:9600]).predict_proba(fv)
	# length_log = linear_model.LogisticRegression()
	# length_prob = logistic.fit(desc_length, labels).predict_proba(desc_length)

	desc_prob_file = open('desc_prob_file.txt', 'w')
	for i in range(len(prob)):
		p = prob[i]
		desc_prob_file.write(str(p[1]))
		desc_prob_file.write(' ')
		# desc_prob_file.write(str(length_prob[i][0]))
		desc_prob_file.write('\n')
	desc_prob_file.close()

def get_fv(processed_desc, vocab_list, desc_length):
	vsize = len(vocab_list) + 1
	fv = np.zeros((len(processed_desc), vsize))

	for li, l in enumerate(processed_desc):
		if len(l) == 0:
			continue
		words = set(l.split())

		for idx, v in enumerate(vocab_list):
			if v in words:
				fv[li][idx] = 1
		fv[li][vsize - 1] = desc_length[li]
	return fv

def process_desc(lines):
	disgard_regs = ['<[^>]*>', 'borrower added on \d+/\d+/\d+', '\&gt;', '>', '\d{6} added on \d+/\d+/\d+', 'a','the','an','and','or','but','about','above','after','along','amid','among',\
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
                           'you','your','yours','me','my','mine','I','we','us','much','and/or']
	replace_regs_k = ['\$\d+[,.]?(\d+)?', '\d+[+,. ]?(\d+)?\%', '\%\d+[+,. ]?(\d+)?', '\d+', 'credit card', 'thank you', 'pay off', '\W', '\s\w\s', '\s+']
	replace_regs_v = ['dollarnumber', 'percentnumber', 'percentnumber', 'numbernumber', 'creditcard', 'thankyou', 'payoff', ' ', ' ', ' ']
	processed_desc = []
	desc_length = []
	max_length = 0
	for l in lines:
		l = l.strip().lower()
		if len(l) == 0:
			desc_length.append(0)
			processed_desc.append(l)
			continue
		for reg in disgard_regs:
			l = re.sub(r'%s' % reg, '', l)
		for i in range(0, len(replace_regs_v)):
			l = re.sub(r'%s' % replace_regs_k[i], replace_regs_v[i], l)
		l = l.strip()
		max_length = max(max_length, len(l.split()))
		desc_length.append(len(l.split()))
		processed_desc.append(l)
	for i in range(0, len(desc_length)):
	 	desc_length[i] = np.divide(desc_length[i] * 1.0, max_length)

	return processed_desc, desc_length

def get_vocab_list(lines):
	vo_list = dict()
	for l in lines:
		words = set(l.split())
		for w in words:
			w = stem(w)
			if w in vo_list.keys():
				vo_list[w] += 1
			else:
				vo_list[w] = 1
	vocab_tmp = []
	for w, c in vo_list.items():
		if c >= 0:
			vocab_tmp.append(w)

	'''print("dict size:")
	print(len(vo_list))
	print(vo_list)'''
	vocab = []
	for w in vocab_tmp:
		vocab.append(w)
	return vocab

get_prob_desc()
