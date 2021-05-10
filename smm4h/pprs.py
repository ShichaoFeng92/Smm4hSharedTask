from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import re
import enchant
from gensim.models.phrases import Phraser, Phrases

stemmer = PorterStemmer()
# remove ' character from exclude set because we'd like to keep track of
# phrases like "I'm, someone's"
exclude = set(string.punctuation)
exclude.remove("'")
tknzr = TweetTokenizer()
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

    return text


def string2words(s, remove_stopwords=True, remove_url=True, remove_digits=True,
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
    #bigram = Phrases.load('bigramPhrases_52m')
    #trigram = Phrases.load('trigramPhrases_52m')
    #words = trigram[bigram[words]]

    if remove_digits:
        words = [w for w in words if not w.isdigit()]
    if remove_stopwords:
        words = [w for w in words if w not in stopwords_list]
    if stem:
        words = [stemmer.stem(w) for w in words]

# remove non-english words
    d = enchant.Dict("en_US")
    words = [w for w in words if d.check(w)]

    return words


# def pre_process_string(s, tolower=True):
#     s = unicode(s).encode('ascii', 'ignore').strip()
#     if tolower:
#         s = s.lower()
#     # remove \n and \t
#     s = s.replace("\n", " ")
#     s = s.replace("\t", " ")
#     return s


def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.
       Same elements will be added
    '''
    z = x.copy()
    z.update(y)
    return z
