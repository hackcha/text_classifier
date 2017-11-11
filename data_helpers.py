import string
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical


# def load_ag_data():
#     train = pd.read_csv('data/ag_news_csv/train.csv', header=None)
#     train = train.dropna()
#
#     x_train = train[1] + train[2]
#     x_train = np.array(x_train)
#
#     y_train = train[0] - 1
#     y_train = to_categorical(y_train)
#
#     test = pd.read_csv('data/ag_news_csv/test.csv', header=None)
#     x_test = test[1] + test[2]
#     x_test = np.array(x_test)
#
#     y_test = test[0] - 1
#     y_test = to_categorical(y_test)
#     #print(x_test)
#     return (x_train, y_train), (x_test, y_test)


def load_data_file( filename , txt_cols ):
    train = pd.read_csv(filename, header=None)
    train = train.dropna()
    for txt_col in txt_cols:
        if txt_col == txt_cols[0]:
            x_train = train[txt_col]
        else:
            x_train = x_train+ ' '+ train[txt_col]
    x_train = np.array(x_train)
    y_train = train[0] - 1
    y_train = to_categorical(y_train)
    return x_train, y_train

def load_data( dir , txt_cols ):
    x_train, y_train = load_data_file(dir+'train.csv', txt_cols )
    x_test, y_test = load_data_file(dir+'test.csv', txt_cols )
    return x_train, y_train, x_test, y_test

def mini_batch_generator(x, y, vocab, vocab_size, vocab_check, maxlen,
                         batch_size=128):
    while 1:
        for i in range(0, len(x), batch_size):
            end_idx = i+batch_size
            if end_idx > x.shape[0]:
                end_idx = x.shape[0]
            # print( '{} {}'.format(i , end_idx) )
            x_sample = x[i:end_idx]
            y_sample = y[i:end_idx]

            input_data = encode_data(x_sample, maxlen, vocab, vocab_size,
                                     vocab_check)
            # print(x_sample.shape)
            # print(y_sample.shape)
            # for dix, xi in enumerate(input_data):
            #     yield (xi, y_sample[dix])
            yield (input_data, y_sample)


def encode_data(x, maxlen, vocab, vocab_size, check):
    #Iterate over the loaded data and create a matrix of size maxlen x vocabsize
    #In this case that will be 1014x69. This is then placed in a 3D matrix of size
    #data_samples x maxlen x vocab_size. Each character is encoded into a one-hot
    #array. Chars not in the vocab are encoded into an all zero vector.

    input_data = np.zeros((len(x), maxlen, vocab_size))
    for dix, sent in enumerate(x):
        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))
        chars = list(sent.lower().replace(' ', ''))
        for c in chars:
            if counter >= maxlen:
                pass
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        input_data[dix, :, :] = sent_array

    return input_data

def encode_twt(x, max_sents, maxlen, vocab, vocab_size, check):
    input_data = np.zeros( (len(x), max_sents, maxlen, vocab_size))
    for i in range(len(x)):
        for dix, sent in enumerate(x[i]):
            counter = 0
            sent_array = np.zeros((maxlen, vocab_size))
            chars = list(sent.lower().replace(' ', ''))
            for c in chars:
                if counter >= maxlen:
                    pass
                else:
                    char_array = np.zeros(vocab_size, dtype=np.int)
                    if c in check:
                        ix = vocab[c]
                        char_array[ix] = 1
                    sent_array[counter, :] = char_array
                    counter += 1
            input_data[i, dix, :, :] = sent_array
    return input_data

def shuffle_matrix(x, y):
    stacked = np.hstack((np.matrix(x).T, y))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    yi = np.array(stacked[:, 1:])
    return xi, yi


def create_vocab_set():
    #This alphabet is 69 chars vs. 70 reported in the paper since they include two
    # '-' characters. See https://github.com/zhangxiangxiao/Crepe#issues.
    alphabet = (list(string.ascii_lowercase) + list(string.digits) +
                list(string.punctuation) + ['\n'])
    vocab_size = len(alphabet)
    check = set(alphabet)
    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t
    return vocab, reverse_vocab, vocab_size, check

