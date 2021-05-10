from gensim.models.wrappers import FastText
import numpy as np
from nltk.tokenize import TweetTokenizer
import pandas as pd
import re
import string
import scipy
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
exclude = set(string.punctuation)
exclude.remove("'")
tknzr = TweetTokenizer()
stopwords_list = stopwords.words('english')

def remove_p(ch):
    if ch in exclude:
        ch = ' '
    return ch


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
    return text


def preprocess(s, remove_stopwords=False, remove_url=True, remove_digits=True,
               tolower=True, remove_rt=True, stem=False):
    if s is None:
        return []
    s = str(s)
    s = s.encode('ascii', errors='ignore').strip().decode('utf-8')
    if tolower:
        s = s.lower()
    # remove \n and \t
    s = s.replace("\n", " ")
    s = s.replace("\t", " ")
    # remove punctuation
    s = ''.join(remove_p(ch) for ch in s)
    # s = ''.join(ch for ch in s if ch not in exclude)
    if remove_url:
        s = remove_link_text(s)
    if remove_rt:
        s = remove_RT_MT(s)

    words = tknzr.tokenize(s)

    if stem:
        words = [stemmer.stem(w) for w in words]

    return ' '.join(words)

def cosine_distance_wordembedding_method(model, s1, s2):
    if (len(s1) < 1) | (len(s2) < 1):
        return 1
    UNK=[]
    for word in s1:
        if word not in model:
            UNK.append(word)
    for word in s2:
        if word not in model:
            UNK.append(word)

    vector_1 = np.mean([model[word] for word in s1 if word not in UNK], axis=0)
    vector_2 = np.mean([model[word] for word in s2 if word not in UNK], axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    return cosine


model = FastText.load_fasttext_format('fasttext_52m.bin')
negative_collect = []
reader = pd.read_csv('task1_training.tsv', sep='\t')
print(len(reader))
for line_id, line in enumerate(reader['tweet']):
    if reader['class'][line_id]==0:
        sentence = preprocess(line)
        negative_collect.append(TweetTokenizer().tokenize(sentence))

cos_matrix=np.zeros((len(negative_collect),len(negative_collect)))
for i in range(len(negative_collect)):
    for j in range(len(negative_collect)):
        cos_matrix[i][j]=cosine_distance_wordembedding_method(model,negative_collect[i],negative_collect[j])
        print(str(i)+' '+str(j))

dt=pd.DataFrame(cos_matrix)
dt.to_csv('cos_matrix.csv')
