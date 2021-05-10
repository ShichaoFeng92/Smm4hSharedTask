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
    return ' '.join(words)


# with open('52m_NoURLENNoDupPuncOE.txt', 'a+') as f:
'''
with open('sharedTaskCorpus', 'w') as f:
    reader = pd.read_csv('task1_training.tsv', sep='\t')
    for line in reader['tweet']:
        f.write(line + '\n')
        f.write('\n')
    reader = pd.read_csv('task1_validation.tsv', sep='\t')
    for line in reader['tweet']:
        f.write(line + '\n')
        f.write('\n')
    reader = pd.read_csv('task2_en_training.tsv', sep='\t')
    for line in reader['tweet']:
        f.write(line + '\n')
        f.write('\n')
    reader = pd.read_csv('task2_en_validation.tsv', sep='\t')
    for line in reader['tweet']:
        f.write(line + '\n')
        f.write('\n')
'''
with open('sharedTaskCorpus_fr', 'w') as f:
    reader = pd.read_csv('task2_fr_training.tsv', sep='\t')
    for line in reader['tweet']:
        f.write(line + '\n')
    reader = pd.read_csv('task2_fr_validation.tsv', sep='\t')
    for line in reader['tweet']:
        f.write(line + '\n')
