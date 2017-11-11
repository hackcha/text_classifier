"""
数据预处理工具：
功能：
    1.英文分词 Tokenizer
    2.构建字典 create_vocabulary    load_vocab
    3.pad_sequnces
    4.load_embedding
    5.split (stratification)
    6.data2dict 使用类别标签为key，相应类别下的数据列表作为value
    7.pickle_load、pikle_dump
"""

import json
import numpy as np
import re
# from tensorflow import gfile
# import tensorflow as tf
import os

data_dir = './data/'#'D:/dataset/ed/'#
glove_file = data_dir + 'glove.6B.100d.txt'
use_glove300 = True
glove_file300 = data_dir + 'glove.840B.300d.txt'

vocab_file = data_dir + 'vocab.txt'

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
_DIGIT_RE = re.compile(r"\d")
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

nb_vocab = 40000
MAX_LEN = 100

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]
from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer
class Tokenizer(object):
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
        self.tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

    def tokenize(self, sentence):
        terms = self.tokenizer.tokenize(sentence)
        res = [self.stemmer.stem(term) for term in terms]
        return res


def save_vocab(vocab_list, vocabulary_path):
    with open(vocabulary_path, mode="w", encoding='utf-8') as vocab_file:
        for w in vocab_list:
            vocab_file.write(w + "\n")

def _create_vocabulary( texts, max_vocab_size, normalize_digits = True ):
    tokenizer = Tokenizer()
    vocab = {}
    counter = 0
    for line in texts:
        counter += 1
        if counter % 100000 == 0:
            print("  processing text %d" % counter)
        entries = line.split('\t')
        tokens = tokenizer.tokenize(line.lower() )
        for w in tokens:
            if w.startswith('http://') or w.startswith('https://') \
                    or w.startswith('@') or w.startswith('#'):
                continue
            word = _DIGIT_RE.sub("0", w) if normalize_digits else w
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocab_size:
        vocab_list = vocab_list[:max_vocab_size]
    return vocab_list

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      normalize_digits=True):
    tokenizer = Tokenizer()
    # if not os.path.exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with open(data_path, mode="r") as f:
        counter = 0
        for line in f:
            counter += 1
            if counter % 100000 == 0:
                print("  processing line %d" % counter)
            entries = line.split('\t')
            if len(entries)<2:
                continue
            # tokens = tokenizer(entries[1]) if tokenizer else basic_tokenizer(line)
            tokens = tokenizer.tokenize( entries[1].lower() )
            for w in tokens:
                if w.startswith('http://') or w.startswith('https://')\
                    or w.startswith('@') or w.startswith('#'):
                    continue
                word = _DIGIT_RE.sub("0", w) if normalize_digits else w
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with open(vocabulary_path, mode="w") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")


def load_vocab(vocab_file = vocab_file):
    vocab = {}
    vocab_res = {}
    vid = 0
    with open(vocab_file, mode="r",encoding='utf-8') as vocab_file:
        for w in vocab_file:
            w = w.strip()
            vocab[w] = vid
            vocab_res[vid] = w
            vid+=1
    return vocab, vocab_res


import pickle
def pickle_dump(obj, fn ):
    with open(fn,'wb') as f:
        pickle.dump(obj, f)

def pickle_load(fn):
    obj = None
    with open(fn,'rb') as f:
        obj = pickle.load(f)
    return obj


def convert2idlist( texts, vocab, normalize_digits=True ):
    terms_list = list()
    tokenizer = Tokenizer( )
    for line in texts:
        # entries = line.lower().split('\t')
        terms = tokenizer.tokenize( line.lower( ) )
        term_ids = []
        for term in terms:
            term = _DIGIT_RE.sub("0", term) if normalize_digits else term
            if term in vocab:
                term_ids.append(vocab[term])
            else:
                if term.startswith('http://') or term.startswith('https://')\
                    or term.startswith('@') or term.startswith('#'):
                    continue
                print('skip unknown term', term)
        terms_list.append(term_ids)
    return terms_list


def pad_data(terms_list , max_len = MAX_LEN, pad_pre = True):
    if max_len is None:
        max_len = 0
        for terms in terms_list:
            if len(terms) > max_len:
                max_len = len(terms)
    new_terms_list = []
    for terms in terms_list:
        pad_len = max_len-len(terms)
        if pad_len > 0:
            if pad_pre:
                new_terms = [PAD_ID]*pad_len + terms
            else:
                new_terms = terms + [PAD_ID]*pad_len
        else:
            new_terms = terms[-max_len:]
        new_terms_list.append(new_terms)
    return new_terms_list


def load_embedding300( vocab_dict, glove_file300, include_stem=True ):
    #word 2 id; id to word
    # dump_path = 'data/embed300.pkl'
    # if os.path.exists( dump_path ):
    #     matrix = pickle.load( open( dump_path, 'rb' ) )
    # else:
    embeddings_index = dict( )
    embed_dim = 300
    with open(glove_file300, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if len(values) > 300+1:
                word = ' '.join(values[:-300])
                print(word)
            try:
                coefs = np.asarray(values[-300:], dtype='float32')
                embeddings_index[word] = coefs
            except:
                print('error:',line)
    nb_vocab = len(vocab_dict)
    matrix = np.zeros( (nb_vocab, embed_dim) )
    nb_in_glove = 0
    stemmer = SnowballStemmer('english')
    include_set = set()
    for word, vec in embeddings_index.items():
        if word in vocab_dict:
            id = vocab_dict[word]
            matrix[id] = vec
            nb_in_glove += 1
            include_set.add( word )
    if include_stem:
        for word, vec in embeddings_index.items():
            stem_word = stemmer.stem(word)
            if stem_word != word and (stem_word not in include_set) and stem_word in vocab_dict:
                id = vocab_dict[stem_word]
                matrix[id] = vec
                nb_in_glove += 1
                include_set.add(stem_word)
    print('number of words in glove embedding: {}/{}'.format(nb_in_glove, len(vocab_dict)))
    # pickle.dump( matrix, open(dump_path,'wb'))
    return matrix

#load embedding
def load_embedding( vocab_dict, glove_file, include_stem=True ):
    #word 2 id; id to word
    # vocab_dict, _ = load_vocab(vocab_file)
    embeddings_index = dict()
    embed_dim = 100
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    nb_vocab = len(vocab_dict)
    matrix = np.zeros( (nb_vocab, embed_dim) )
    nb_in_glove = 0
    stemmer = SnowballStemmer('english')
    include_set = set()
    for word, vec in embeddings_index.items():
        if word in vocab_dict:
            id = vocab_dict[word]
            matrix[id] = vec
            nb_in_glove += 1
            include_set.add( word )
    if include_stem:
        for word, vec in embeddings_index.items():
            stem_word = stemmer.stem( word )
            if stem_word != word and (stem_word not in include_set) and stem_word in vocab_dict:
                id = vocab_dict[stem_word]
                matrix[id] = vec
                nb_in_glove += 1
                include_set.add( stem_word )
    print('number of words in glove embedding: {}/{}'.format(nb_in_glove, len(vocab_dict)))
    return matrix

def load_stop_words( path = 'data/stopwords.txt'):
    stopwords = set()
    stemmer = SnowballStemmer('english')
    with open( path, 'r',encoding='utf-8') as fin:
        for line in fin:
            word = line.strip( )
            stopwords.add( word )
            stem_word = stemmer.stem( word )
            stopwords.add( stem_word )
    return stopwords


def sepData(x, y, percent=0.8):
    n = len(y)
    num_train = int(percent * n)
    x_tn = x[0:num_train]
    y_tn = y[0:num_train]
    x_ts = x[num_train:]
    y_ts = y[num_train:]
    return x_tn, y_tn, x_ts, y_ts


if __name__=='__main__':
    pass