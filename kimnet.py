import numpy as np
import pickle
from collections import defaultdict
import re
import sys
import os
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2
from keras.layers import Embedding, concatenate
from keras.layers import Dense, Input, Flatten,Bidirectional,GRU
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout,TimeDistributed
from keras.optimizers import Adagrad
from keras.models import Model
import aux_data, data_utils

batch_size = 50
EMBEDDING_DIM = 100
def kimNet(embed_mat, MAX_LEN, num_cls, filter_sz1 = 100, use_dynamic_embed=True):
    filter_lens = [3, 4, 5]
    embed = Embedding(embed_mat.shape[0],
                                embed_mat.shape[1],
                                weights=[embed_mat],
                                input_length=MAX_LEN,
                                trainable=False)
    embed_dynamic = Embedding(embed_mat.shape[0],
                      embed_mat.shape[1],
                      weights=[embed_mat],
                      input_length=MAX_LEN,
                      trainable=True)
    input = Input(shape=(MAX_LEN,), dtype='int32')
    seq1 = embed(input)
    if use_dynamic_embed:
        seq2 = embed_dynamic(input)
        seq = concatenate( [seq1,seq2] )
    else:
        seq = seq1
    convs = []
    for fsl in filter_lens:
        l_conv = Conv1D(nb_filter=filter_sz1, filter_length=fsl, activation='relu')(seq)
        l_pool = GlobalAveragePooling1D( )(l_conv)
        convs.append(l_pool)
    l_merge = Merge(mode='concat', concat_axis=1,name='sent_conv')(convs)
    dropout = Dropout(0.5)( l_merge )
    preds = Dense(num_cls, activation='softmax', name='sent_dense',kernel_regularizer=l2(3.0) )(dropout)
    model = Model(input, preds)
    opt = Adagrad(0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
    return model

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    set_session(sess)
    x_tn, y_tn, x_ts, y_ts, embedding_matrix = aux_data.load_ag( sep_sent= False)
    x_tn, y_tn, x_val, y_val = data_utils.sepData(x_tn, y_tn)
    model = kimNet(embedding_matrix, x_tn.shape[1], y_tn.shape[1] )
    print("compile done.")
    model.summary()
    model.fit(x_tn, y_tn, validation_data=(x_val, y_val),
              nb_epoch=20, batch_size=50)
    score, acc = model.evaluate(x_ts, y_ts, batch_size=512)
    print('acc: %.3f'% acc )
