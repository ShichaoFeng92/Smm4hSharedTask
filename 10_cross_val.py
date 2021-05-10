import os
import string
import pandas as pd
import re
import string
exclude = set(string.punctuation)
exclude.remove("'")
from gensim import corpora
from nltk.tokenize import TweetTokenizer
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.wrappers import Wordrank
import numpy as np
import pickle
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.regularizers import l2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
import scipy
from keras import backend as K
from keras_self_attention import SeqSelfAttention
from keras.callbacks import EarlyStopping, ModelCheckpoint


def remove_p(s):
    for ch in exclude:
        s.replace(ch, '')
    return s


def remove_link_text(text):
    """Attempts to match and remove hyperlink text"""
    text = re.sub(r"\S*https?://\S*", "", text)
    text = re.sub(r"\S*http?://\S*", "", text)

    return text


def remove_RT_MT(text):
    """Removes all hanging instances of 'RT' and 'MT'. NOTE: Expects lower case"""
    text = re.sub(r" rt ", " ", text)
    text = re.sub(r"^rt ", " ", text)
    text = re.sub(r" rt$", " ", text)

    text = re.sub(r" mt ", " ", text)
    text = re.sub(r"^mt ", " ", text)
    text = re.sub(r" mt$", " ", text)

    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)
    return text


def preprocess(s):
    if s is None:
        s = ' '
    s = str(s)
    s = s.lower()
    s = s.encode('ascii', errors='ignore').strip().decode('utf-8')
    # remove \n and \t
    s = s.replace("\n", " ")
    s = s.replace("\t", " ")
    s = remove_link_text(s)
    s = remove_RT_MT(s)
    s = ''.join(remove_p(ch) for ch in s)
    s = s.strip()
    return s


def grab_data(sentences, dictionary):

    seqs = [None] * len(sentences)
    for idx, words in enumerate(sentences):
        # words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs


def cosine_distance_wordembedding_method(model, s1, s2):
    if (len(s1) < 1) | len(s2 < 1):
        return 1
    vector_1 = np.mean([model[word] for word in s1], axis=0)
    vector_2 = np.mean([model[word] for word in s2], axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    return cosine


def tagcounter(pred_y, y_test_num):
    tp_count = 0.
    tn_count = 0.
    fp_count = 0.
    fn_count = 0.
    total_no = 0.
    total_yes = 0.

    test_pred_tags = [t[0] for t in pred_y]
    testY_tags = y_test_num

    for i, p in enumerate(test_pred_tags):
        if testY_tags[i] == 0:
            total_no += 1
        if testY_tags[i] == 1:
            total_yes += 1
        if p == 0:
            if testY_tags[i] == 0:
                tn_count += 1
            else:
                fn_count += 1
        else:
            if testY_tags[i] == 1:
                tp_count += 1
            else:
                fp_count += 1

    return testY_tags, test_pred_tags, tp_count, fp_count, fn_count, tn_count
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def main():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    y_train_num = []
    y_test_num = []
    classYes = 0
    classNo = 0
    testYes = 0
    testNo = 0
# To replace the word vector space
    model = FastText.load_fasttext_format(
        'fasttext_52m.bin')
    wv = model.wv

    temp = []
    wordlist = []

    dictionary = dict()

    for word, model in wv.vocab.items():
        wordlist.append(word)

    temp.append(wordlist)
    dictionary = corpora.Dictionary(temp)
    dictionary = dictionary.token2id

    for word, index in dictionary.items():
        dictionary[word] = dictionary[word] + 2

    train_collect = []
    reader = pd.read_csv('task2_en_training.tsv', sep='\t')
    for line_id, line in enumerate(reader['tweet']):
        sentence = preprocess(line) + ' '
        train_collect.append(TweetTokenizer().tokenize(sentence))
        y_train.append(reader['class'][line_id])

    reader = pd.read_csv('task2_en_validation.tsv', sep='\t')
    for line_id, line in enumerate(reader['tweet']):
        sentence = preprocess(line) + ' '
        train_collect.append(TweetTokenizer().tokenize(sentence))
        y_train.append(reader['class'][line_id])

    X_train = grab_data(train_collect, dictionary)

    test_collect = []
    reader = pd.read_csv('task2_en_test_participant.tsv', sep='\t')
    for line_id, line in enumerate(reader['tweet']):
        sentence = preprocess(line) + ' '
        test_collect.append(TweetTokenizer().tokenize(sentence))
        #y_test.append(reader['class'][line_id])

    X_test = grab_data(test_collect, dictionary)
    

    for line_id, line in enumerate(y_train):
        if line == 1:
            classYes = classYes + 1
            y_train_num.append(1)
        else:
            classNo = classNo + 1
            y_train_num.append(0)

    for line_id, line in enumerate(y_test):
        if line == 1:
            y_test_num.append(1)
            testYes = testYes + 1
        else:
            y_test_num.append(0)
            testNo = testNo + 1

    print(classYes)
    print(classNo)
    print(testYes)
    print(testNo)

    max_words = 50
    nb_epochs = 6
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)
    sm = SMOTE(random_state=42)
    #X_train, y_train = sm.fit_sample(X_train, y_train)
    #ros = RandomOverSampler(random_state=0)
    #X_train, y_train = ros.fit_resample(X_train, y_train)
    word_vector = dict()
    for word, index in dictionary.items():
        word_vector[word] = wv[word]
    n_symbols = len(dictionary) + 1
    embedding_weights = np.zeros((n_symbols + 1, 200))
    for word, index in dictionary.items():
        embedding_weights[index, :] = word_vector[word]

    class Histories(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.testacc = []
            self.acc = []

        def on_train_end(self, logs={}):
            return

        def on_epoch_begin(self, epoch, logs={}):
            return

        def on_epoch_end(self, epoch, logs={}):
            self.acc.append(logs.get("acc"))
            pred_y = self.model.predict_classes(X_test)
            testY_tags, test_pred_tags, tp_count, fp_count, fn_count, tn_count = tagcounter(
                pred_y)
            accuracy = accuracy_score(testY_tags, test_pred_tags)
            self.testacc.append(accuracy)
            return

        def on_batch_begin(self, batch, logs={}):
            return

        def on_batch_end(self, batch, logs={}):
            return

    def tagcounter(pred_y):
        tp_count = 0.
        tn_count = 0.
        fp_count = 0.
        fn_count = 0.
        total_no = 0.
        total_yes = 0.

        test_pred_tags = [t[0] for t in pred_y]
        testY_tags = y_test_num

        for i, p in enumerate(test_pred_tags):
            if testY_tags[i] == 0:
                total_no += 1
            if testY_tags[i] == 1:
                total_yes += 1
            if p == 0:
                if testY_tags[i] == 0:
                    tn_count += 1
                else:
                    fn_count += 1
            else:
                if testY_tags[i] == 1:
                    tp_count += 1
                else:
                    fp_count += 1

        return testY_tags, test_pred_tags, tp_count, fp_count, fn_count, tn_count

    model = Sequential()
    model.add(Embedding(output_dim=200, input_dim=n_symbols + 1,
                        mask_zero=False, input_length=max_words, weights=[embedding_weights]))

    class_weight = {0: classYes, 1: classNo}
# set LSTM input dropout and recurrent dropout

    model.add(Bidirectional(
        LSTM(128, W_regularizer=l2(1e-4))))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=[f1_m])

    ckpt = ModelCheckpoint('model.ckpt', monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min',save_weights_only=True)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=1)
    model.fit(X_train, y_train_num, nb_epoch=nb_epochs, validation_split=0.1,
              batch_size=256, callbacks=[ckpt, early])
    # print(histories.testacc)
    # testacc = histories.testacc
    # acc = histories.acc

    # Final evaluation of the model
    y_score = model.predict(X_test)
    pred_y = model.predict_classes(X_test)
    y_prob = model.predict_proba(X_test)

    with open('task2_prediction_original.csv','w') as f:
        for i, p in enumerate(pred_y):
            f.write(str(p[0]))
            f.write('\n')
    
    
    testY_tags, test_pred_tags, tp_count, fp_count, fn_count, tn_count = tagcounter(
        pred_y)
    print('accuracy score: %f' % accuracy_score(testY_tags, test_pred_tags))
    print('precision: %f' % precision_score(testY_tags, test_pred_tags))
    print('recall: %f' % recall_score(testY_tags, test_pred_tags))
    print('f1 score: %f' % f1_score(testY_tags, test_pred_tags))
    print('roc area score: %f' % roc_auc_score(testY_tags, test_pred_tags))

    d = {'predicted yes': pd.Series([tp_count, fp_count], index=[
        'yes', 'no']), 'predicted no': pd.Series([fn_count, tn_count], index=['yes', 'no'])}
    df = pd.DataFrame(d, columns=['predicted yes', 'predicted no'])
    print('Confusion Matrix: ')
    print(df)
    print('------------------------------------')


if __name__ == '__main__':
    main()
