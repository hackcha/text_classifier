import logging, pickle, csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
import data_utils
from nltk import tokenize
import string

data_dir = 'data/'
embed_dir = '/home/cgd/code/edgan/data/'
glove_file = embed_dir+'glove.840B.300d.txt'#'glove.6B.100d.txt'



def mini_batch_generator(x, y, vocab, vocab_size, vocab_check, maxlen,
                         batch_size=128):
    while 1:
        indices = np.arange(x.shape[0])
        np.random.shuffle( indices )
        x = x[indices]
        y = y[indices]
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

def load_data( dir , txt_cols ):
    x_tn, y_tn = load_csv(dir+'train.csv', txt_cols )
    x_ts, y_ts = load_csv(dir+'test.csv', txt_cols )
    x_tn = np.array(x_tn)
    y_tn = to_categorical(y_tn)
    x_ts = np.array( x_ts )
    y_ts = to_categorical( y_ts )
    return x_tn, y_tn, x_ts, y_ts

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




def get_max_sent_len( texts ):
    MAX_LEN = 10
    for text in texts:
        sents = tokenize.sent_tokenize(text)
        for sent in sents:
            tokens = text_to_word_sequence(sent)
            if len(tokens) > MAX_LEN:
                MAX_LEN = len(tokens)
    return MAX_LEN

def padding_texts_sent( texts , word_index , max_len , max_sent = 8):
    nb_words = len(word_index)
    data = np.zeros( (len(texts), max_sent, max_len), np.int32 )
    #padding and truncate, both are 'pre' by default.
    i = 0
    for text in texts:
        sents = tokenize.sent_tokenize( text )
        sj = len(sents)-1
        si = max_sent-1
        while sj >= 0 and si >=0:
            sent = sents[sj]
            tokens = text_to_word_sequence(sent)
            wj = len(tokens) - 1
            j = max_len - 1
            while wj >= 0:
                w = tokens[wj]
                index = word_index.get(w)
                wj = wj - 1
                if index is not None:
                    if j >= 0:
                        data[i, si, j] = index
                        j -= 1
            sj = sj - 1
            si = si - 1
        i += 1
    return data


def load_csv(filename , txt_cols, max_num = -1 ):
    y = list()
    texts = list()
    num =0
    with open(filename,encoding='utf-8') as csv_file:
        csv_file.readline()
        spamreader = csv.reader(csv_file, delimiter=',', quotechar='\"')
        for row in spamreader:
            y.append( int(row[0])-1)
            text = ''
            for txt_col in txt_cols:
                text = text + row[txt_col]+' '
            texts.append( text )
            num+=1
            if max_num > 0 and num >= max_num:
                break
    return texts, y



def cache_data(name, txt_cols, sep_sent, dir, MAX_SENT = 8 ):
    vocab_file = data_dir+'{}_vocab.txt'.format(name)
    train_file = dir + 'train.csv'
    test_file = dir + 'test.csv'
    txt_train, y_train = load_csv(train_file, txt_cols)
    txt_test, y_test = load_csv(test_file, txt_cols)
    txt_all = []
    txt_all.extend( txt_train)
    txt_all.extend( txt_test )
    vocab_list = data_utils._create_vocabulary(txt_all, max_vocab_size=50000)
    data_utils.save_vocab( vocab_list, vocab_file )
    vocab_dict,dict_res = data_utils.load_vocab( vocab_file )
    embed_mat = data_utils.load_embedding300( vocab_dict, glove_file)
    if sep_sent:
        MAX_LEN = get_max_sent_len( txt_train )
        x_tn = padding_texts_sent(txt_train, vocab_dict,MAX_LEN, MAX_SENT )
    else:
        x_tn = data_utils.convert2idlist(txt_train, vocab_dict)
        MAX_LEN = 10
        for x in x_tn:
            if len(x)>MAX_LEN:
                MAX_LEN = len(x)
        if MAX_LEN > 1000:
            print('max len: ',MAX_LEN)
            MAX_LEN = 1000
        x_tn = data_utils.pad_data(x_tn, MAX_LEN, pad_pre=True)
    x_tn = np.array( x_tn )
    y_tn = np.array( y_train )
    if sep_sent:
        x_ts = padding_texts_sent(txt_test, vocab_dict,MAX_LEN, MAX_SENT )
    else:
        x_ts = data_utils.convert2idlist(txt_test, vocab_dict)
        x_ts = data_utils.pad_data(x_ts, MAX_LEN, pad_pre=True)
    x_ts = np.array( x_ts )
    y_ts = np.array( y_test )
    data_dict = {'MAX_LEN':MAX_LEN,'x_tn':x_tn, 'y_tn':y_tn, 'x_ts':x_ts, 'y_ts':y_ts,'embed_mat':embed_mat}
    if sep_sent:
        filename = './data/'+name+'_sent.pkl'
    else:
        filename = './data/'+name+'_word.pkl'
    data_utils.pickle_dump( data_dict, filename )


def load_data_cache(name, sep_sent = True , sample = False):
    if sep_sent:
        filename = './data/'+name+'_sent.pkl'
    else:
        filename = './data/'+name+'_word.pkl'
    r = data_utils.pickle_load( filename )
    x_tn = r['x_tn']
    y_tn = r['y_tn']
    x_ts = r['x_ts']
    y_ts = r['y_ts']
    embedding_matrix = r['embed_mat']
    y_tn = to_categorical(y_tn)
    y_ts = to_categorical(y_ts)
    indices = np.arange(y_tn.shape[0])
    np.random.shuffle(indices)
    x_tn = x_tn[indices]
    y_tn = y_tn[indices]
    if sample:
        sample_num = min( 500000, y_tn.shape[0] )
        # indices = np.arange( y_train.shape[0] )
        # np.random.shuffle( indices )
        # indices = indices[0:sample_num]
        x_tn = x_tn[0:sample_num]
        y_tn = y_tn[0:sample_num]
        sample_num = min(100000, y_ts.shape[0] )
        indices = np.arange( y_ts.shape[0] )
        np.random.shuffle( indices )
        indices = indices[0:sample_num ]
        y_ts = y_ts[indices]
        x_ts = x_ts[indices]
    return x_tn, y_tn, x_ts, y_ts, embedding_matrix

def convert_amazon(sep_sent = False ):
    cache_data('amazon', [1, 2], sep_sent, './data/amazon_review_polarity_csv/')

def load_amazon(sep_sent = False ):
    return load_data_cache( 'amazon', sep_sent )

def convert_dbpedia(sep_sent = False ):
    cache_data('dbpedia', [1,2] ,sep_sent ,'./data/dbpedia_csv/')

def load_dbpedia(sep_sent = False ):
    return load_data_cache( 'dbpedia', sep_sent )

def convert_yahoo(sep_sent=False):
    cache_data('yahoo', [1,2,3], sep_sent, './data/yahoo_answers_csv/')

def load_yahoo(sep_sent=True):
    return load_data_cache('yahoo', sep_sent)

def convert_yelp(sep_sent=False):
    cache_data('yelp', [1], sep_sent, './data/yelp_review_polarity_csv/')
    # cache_data('yelp', [1], sep_sent, './data/yelp_review_full_csv/')
def load_yelp(sep_sent=False):
    return load_data_cache('yelp', sep_sent)

def convert_ag(sep_sent=False):
    cache_data('ag', [1,2] , sep_sent, './data/ag_news_csv/')

def load_ag(sep_sent=False):
    return load_data_cache('ag', sep_sent)

if __name__ == '__main__':
    logging.basicConfig( format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO )
    convert_ag( False )
    # convert_ag( True )
    logging.info('done.')