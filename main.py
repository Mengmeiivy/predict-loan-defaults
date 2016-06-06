from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn import ensemble
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt

def read_file():
    training_labels = []
    feature_vectors = []
    feature_names = []
    info = []

    label_file = open('label_file.txt', 'r').readline().strip().split(',')
    for l in label_file:
        training_labels.append(int(l))

    fv_file = open('processed_file.txt', 'r').readlines()
    for l in fv_file:
        l = l.strip().split(',')
        tmp = []
        for v in l:
            if '.' in v:
                tmp.append(float(v))
            else:
                tmp.append(int(v))
        feature_vectors.append(tmp)

    desc_file = open('desc_prob_file.txt', 'r').readlines()
    for i in range(0, len(desc_file)):
      l = desc_file[i].strip().split()
      feature_vectors[i].append(float(l[0]))
      '''feature_vectors[i].append(float(l[1]))'''

    fn_file = open('processed_feature.txt', 'r').readline().split('^')
    for l in fn_file:
        feature_names.append(l)

    # split test_set and training_set
    train_set = []
    train_label = []
    validate_set = []
    validate_label = []
    test_set = []
    test_label = []

    train_length = len(feature_vectors)*0.8
    for i in range(0, len(feature_vectors)):
        if i < train_length:
            if i < int(train_length*0.8):
                train_set.append(feature_vectors[i])
                train_label.append(training_labels[i])
            else:
                validate_set.append(feature_vectors[i])
                validate_label.append(training_labels[i])
        else:
            test_set.append(feature_vectors[i])
            test_label.append(training_labels[i])

    print("Length of Train Set", len(train_set), "Length of Validate Set", len(validate_set), "Length of Test Set", len(test_set))
    info.append(train_set)
    info.append(train_label)
    info.append(validate_set)
    info.append(validate_label)
    info.append(test_set)
    info.append(test_label)
    return info

def feature_explore(weight):
    file = open('feature.txt', 'r')
    useful_feature = file.readline().split('^')
    uf_names = []
    uf_categories = []
    uf_combined = [];
    explore_dict = {};
    for f in useful_feature:
        uf_names.append(f.split(':')[0])
        uf_categories = f.split(':')[1].split(',')
        for u in uf_categories:
            uf_combined.append(str(f.split(':')[0]) + str('^') + str(u))

    for i in range(0, len(uf_combined)):
        explore_dict[uf_combined[i]] = weight[i]
        if i+2 <= len(uf_combined):
            explore_dict['description'] = weight[i+1]

    explore_file = open('explore_weight.txt', 'w')
    explore_dict = sorted(explore_dict.items(), key=lambda x: x[1])
    for e in explore_dict:
        explore_file.write(str(e))
        explore_file.write(str('\n'))
    explore_file.close()
    print('Explore Weight Done')

def measure(y_true, y_predict):
  TP = 0.0
  FP = 0.0
  TN = 0.0
  FN = 0.0
  error = 0.0

  error_idx = open('error_idx', 'w')

  for i in range(len(y_predict)):

    if y_true[i] != y_predict[i]:
      error_idx.write("%s %s %s\n" % (i + 9600, y_true[i], y_predict[i]))

    if y_true[i] == 1 and y_predict[i] == 1:
      TP += 1
    if y_true[i] == 1 and y_predict[i] == -1:
      FN += 1
      error += 1
    if y_true[i] == -1 and y_predict[i] == 1:
      FP += 1
      error += 1
    if y_true[i] == -1 and y_predict[i] == -1:
      TN += 1

  sensitivity = TP / (TP + FN)
  specificity = TN / (TN + FP)
  error = error / len(y_predict)
  accuracy = 1 - error
  PPV = TP / (TP + FP)

  return accuracy, sensitivity, specificity, PPV

'''============================== Main ===================================== '''
info = read_file()

'''gnb = GaussianNB()
y_predict = gnb.fit(info[0],info[1]).predict(info[2])
accuracy, sensitivity, specificity, PPV = measure (info[3], y_predict)
print ('Accuracy, sensitivity, specificity, PPV:', accuracy, sensitivity, specificity, PPV)'''

'''clf = svm.SVC(kernel='linear')
clf.fit(info[0],info[1])
weight = clf.coef_
feature_explore(weight[0])
print(clf.score(info[2],info[3]))'''
'''y_predict = tree.DecisionTreeClassifier(min_samples_leaf = 100).fit(info[0], info[1]).predict(info[2])
accuracy, sensitivity, specificity, PPV = measure (info[3], y_predict)
print ('Accuracy, sensitivity, specificity, PPV:', accuracy, sensitivity, specificity, PPV)'''

'''clf = svm.SVC()
clf.fit(info[0],info[1])
y_predict = clf.predict(info[2])
accuracy, sensitivity, specificity, PPV = measure (info[3], y_predict)
print ('Accuracy, sensitivity, specificity, PPV:', accuracy, sensitivity, specificity, PPV)'''

# clf = joblib.load('classfier.pkl') 

"""
rf = ensemble.RandomForestClassifier(n_estimators = 400, max_depth = 30)
rf.fit(info[0]+info[2],info[1]+info[3])

classes = ["Default", "Fully Paid"]
file = open("processed_feature.txt", "r")
for line in file:
  features = line.split("^")
features.append("loan description")
for a_tree in rf.estimators_:
  with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(rf, out_file = f, max_depth = 5, feature_names = features, class_names = classes, filled = True, rounded = True)


#weight = rf.fit(info[0]+info[2],info[1]+info[3]).feature_importances_
#feature_explore(weight)
y_predict = rf.predict(info[4])
accuracy, sensitivity, specificity, PPV = measure (info[5], y_predict)
print ('Accuracy, sensitivity, specificity, PPV:', accuracy, sensitivity, specificity, PPV)


"""
clf = tree.DecisionTreeClassifier(min_samples_leaf = 600)
clf.fit(info[0], info[1])
#classes = ["Default", "Fully Paid"]
#file = open("processed_feature.txt", "r")
#for line in file:
#features = line.split("^")
#features.append("loan description")

#with open("tree.dot", 'w') as f:
#f = tree.export_graphviz(clf, out_file = f, max_depth = 3, feature_names = features, class_names = classes, filled = True, rounded = True)

y_predict = clf.predict(info[2])
accuracy, sensitivity, specificity, PPV = measure (info[3], y_predict)
print ('Accuracy, sensitivity, specificity, PPV:', accuracy, sensitivity, specificity, PPV)



'''logistic = linear_model.LogisticRegression(C = 6000)
#y_predict_prob = logistic.fit(info[0],info[1]).predict_proba(info[2])
y_predict = logistic.fit(info[0],info[1]).predict(info[2])
accuracy, sensitivity, specificity, PPV = measure (info[3], y_predict)
print(logistic.coef_)
print ('Accuracy, sensitivity, specificity, PPV:', accuracy, sensitivity, specificity, PPV)'''
