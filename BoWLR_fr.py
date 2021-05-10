import pprs
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy
from collections import OrderedDict
import logging
from collections import Counter
from gensim.models.wrappers import FastText


def import_data(path):
    dataset_x = []
    dataset_y = []
    # supplement reader

    reader = pd.read_csv(path, sep='\t')
    '''

    # medicine reader
    reader = pd.read_csv(
        path, names=['class', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'text'])
    '''

    for line_id, line in enumerate(reader['tweet']):
        dataset_x.append(line)
        dataset_y.append(reader['class'][line_id])

    return dataset_x, dataset_y


def clean_data(data):
    all_words = []
    for line_id, line in enumerate(data):
        print(line_id)
        words = pprs.string2words(
            line, remove_stopwords=True, remove_url=True, tolower=True, remove_rt=True, stem=True)
        all_words.append(' '.join(words))
    return all_words


def build_dict(sentences):

    print('Building dictionary..')
    all_words = []
    for words in sentences:
        all_words.extend(words)
    words_counter = Counter(all_words)
    sorted_counter = words_counter.most_common()
    word_dict = dict()
    for idx, tup in enumerate(sorted_counter):
        word, _ = tup
        word_dict[word] = idx + 2  # leave 0 and 1 (UNK)

    print(len(all_words), ' total words ', len(words_counter), ' unique words')
    return word_dict


def tokenize_data(data):
    logging.info('tokenize data')
    all_words = []
    for line_id, line in enumerate(data):
        print(line_id)
        words = pprs.string2words(line, remove_stopwords=False, stem=False)
        all_words.append(words)
    return all_words


def Logistic_regression(traindata, y_train_num, testdata, y_test_num):
    LR = LogisticRegression(C=1000, random_state=0)
    LR.fit(traindata, y_train_num)
    y_pred = LR.predict(testdata)
    print("=========Logistic Regression==========")
    print(confusion_matrix(y_test_num, y_pred))
    print(accuracy_score(y_test_num, y_pred))
    print(precision_score(y_test_num, y_pred))
    print(recall_score(y_test_num, y_pred))
    print(f1_score(y_test_num, y_pred))
    print(roc_auc_score(y_test_num, y_pred))


def SVM(traindata, y_train_num, testdata, y_test_num):
    SVM = LinearSVC()
    SVM.fit(traindata, y_train_num)
    y_pred = SVM.predict(testdata)
    print("=========SVM==========")
    print(confusion_matrix(y_test_num, y_pred))
    print(accuracy_score(y_test_num, y_pred))
    print(precision_score(y_test_num, y_pred))
    print(recall_score(y_test_num, y_pred))
    print(f1_score(y_test_num, y_pred))
    print(roc_auc_score(y_test_num, y_pred))
    return y_pred


def DecisionTree(traindata, y_train_num, testdata, y_test_num):
    tree = DecisionTreeClassifier(
        criterion='entropy', max_depth=30)
    tree.fit(traindata, y_train_num)
    y_pred = tree.predict(testdata)
    print("=========Decision Tree=======")
    print(confusion_matrix(y_test_num, y_pred))
    print(accuracy_score(y_test_num, y_pred))
    print(precision_score(y_test_num, y_pred))
    print(recall_score(y_test_num, y_pred))
    print(f1_score(y_test_num, y_pred))
    print(roc_auc_score(y_test_num, y_pred))
    return y_pred


def RandomForest(traindata, y_train_num, testdata, y_test_num):
    tree = RandomForestClassifier(max_depth=2, random_state=0)
    tree.fit(traindata, y_train_num)
    y_pred = tree.predict(testdata)
    print("=========random forest=======")
    print(confusion_matrix(y_test_num, y_pred))
    print(accuracy_score(y_test_num, y_pred))
    print(precision_score(y_test_num, y_pred))
    print(recall_score(y_test_num, y_pred))
    print(f1_score(y_test_num, y_pred))
    print(roc_auc_score(y_test_num, y_pred))
    return y_pred


'''
with open('sharedTaskCorpus') as f:
     data=[]
     for line_id,line in enumerate(f):
         data.append(line)

train=tokenize_data(data)
corpus = build_dict(train)
f = open('corpus_dict.pkl', 'wb')
pickle.dump(corpus, f, -1)
f.close()
'''
words = []
with open('sharedTaskCorpus_fr') as f:
    for line in f:
        words.append(line.strip())


X_train, y_train = import_data('task2_fr_training.tsv')
X_test, y_test = import_data('task2_fr_validation.tsv')

vectorizer = CountVectorizer()
train_data_features = vectorizer.fit(words).transform(X_train)
feature_metrics_train = train_data_features.toarray()
test_data_features = vectorizer.fit(words).transform(X_test)
feature_metrics_test = test_data_features.toarray()
Logistic_regression(feature_metrics_train, y_train,
                    feature_metrics_test, y_test)


'''
print("the features' shape:")
print(train_data_features.shape)
print(test_data_features.shape)

from sklearn.model_selection import cross_val_score
LR = LogisticRegression(C=1000, random_state=0)
scores = cross_val_score(LR, feature_metrics_train, y_train, cv=10)
print(scores)
print(scores.mean())
'''
