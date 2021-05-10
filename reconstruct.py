import os
from pprs import string2words
import string
import pandas as pd
import re
import string
import enchant
from gensim.models.phrases import Phraser, Phrases
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
exclude = set(string.punctuation)
exclude.remove("'")
from gensim import corpora
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.wrappers import Wordrank
import numpy as np
import pickle
import pandas as pd
import nltk

stemmer = PorterStemmer()
# remove ' character from exclude set because we'd like to keep track of
# phrases like "I'm, someone's"
exclude = set(string.punctuation)
exclude.remove("'")
stopwords_list = stopwords.words('english')


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


def preprocess(s, remove_stopwords=True, remove_url=True, remove_digits=True,
               tolower=True, remove_rt=True, stem=True):
    if s is None:
        return []
    s = str(s)
    s = s.encode('ascii', errors='ignore').strip().decode('utf-8')
    if tolower:
        s = s.lower()

    if remove_url:
        s = remove_link_text(s)
    if remove_rt:
        s = remove_RT_MT(s)

    words = s.strip().split()

    pos = nltk.pos_tag(words)
    words = [w[0] for w in pos if ((w[1] == 'NN') | (w[1] == 'NNS'))]

    s = ' '.join(words)
    s = ''.join(remove_p(ch) for ch in s)

    words = s.strip().split()
    if remove_digits:
        words = [w for w in words if not w.isdigit()]
    if remove_stopwords:
        words = [w for w in words if w not in stopwords_list]
    if stem:
        words = [stemmer.stem(w) for w in words]

    if len(words) == 0:
        return False
    else:
        return True


# with open('52m_NoURLENNoDupPuncOE.txt', 'a+') as f:
'''
fw = open('task1_negative.tsv', 'w')
fw.write('tweet_id\ttweet\tclass\n')
f = open('task1_training.tsv')
for line_id, line in enumerate(f):
    if line_id == 0:
        continue
    tup = line.strip().split('\t')
    if int(tup[4]) == 0:
        if preprocess(tup[2]):
            fw.write(tup[0] + '\t' + tup[2] + '\t' + tup[4] + '\n')

f.close()
fw.close()

fw = open('task1_positive.tsv', 'w')
fw.write('tweet_id\ttweet\tclass\n')
f = open('task1_training.tsv')
for line_id, line in enumerate(f):
    if line_id == 0:
        continue
    tup = line.strip().split('\t')
    if int(tup[4]) == 1:
        fw.write(tup[0] + '\t' + tup[2] + '\t' + tup[4] + '\n')

f.close()
fw.close()

fw = open('intake_sample.txt', 'w')
fw.write('tweet_id\ttweet\tclass\n')
f = open('intake.txt')
for line_id, line in enumerate(f):
    tup = line.strip().split('\t')
    fw.write(tup[0] + '\t' + tup[4] + '\t' + '1' + '\n')

fw = open('medical_sample.txt', 'w')
fw.write('tweet_id\ttweet\tclass\n')
f = open('medical.txt')
for line_id, line in enumerate(f):
    tup = line.strip().split('\t')
    fw.write(tup[1] + '\t' + tup[2] + '\t' + '1' + '\n')
'''
'''
pos_one = pd.read_csv('task1_positive.tsv', sep='\t')
pos_two = pd.read_csv('intake_sample.txt', sep='\t')
pos_three = pd.read_csv('medical_sample.txt', sep='\t')
negative = pd.read_csv('task1_negative.tsv', sep='\t')
validation = pd.read_csv('task1_validation.tsv', sep='\t')
#negative = negative.sample(frac=1, random_state=1)
newdata = negative.append(pos_one, ignore_index=True)
newdata = newdata.append(pos_two, ignore_index=True)
newdata = newdata.append(pos_three, ignore_index=True)
newdata = newdata.append(validation, ignore_index=True)
newdata = newdata.sample(frac=1, random_state=1)
newdata.to_csv('reconstruct_task1.tsv', sep='\t', index=False)
'''
training = pd.read_csv('task2_en_training.tsv', sep='\t')
additional = pd.read_csv('additional_task2_en.csv')
validation = pd.read_csv('task2_en_validation.tsv', sep='\t')
newdata = training.append(additional, ignore_index=True)
newdata = newdata.append(validation, ignore_index=True)
newdata.to_csv('reconstruct_task2.tsv', sep='\t', index=False)

