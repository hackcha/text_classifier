import numpy as np
import pandas as pd
from collections import defaultdict
import re
import sys
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.engine.topology import Layer, InputSpec
from attention import AttLayer
import aux_data
import data_utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def RNNModel(embed_mat, MAX_LEN, num_cls, rnn_sz = 100 ):
    embed = Embedding(embed_mat.shape[0],
                      embed_mat.shape[1],
                      weights=[embed_mat],
                      input_length=MAX_LEN,
                      trainable=False)
    sequence_input = Input(shape=(MAX_LEN,), dtype='int32')
    embedded_sequences = embed(sequence_input)
    l_lstm = Bidirectional(LSTM(rnn_sz))(embedded_sequences)
    preds = Dense(num_cls, activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['acc'])
    return model


def RNNAtt(embed_mat, MAX_LEN, num_cls, rnn_sz = 100 ):
    embed = Embedding(embed_mat.shape[0],
                      embed_mat.shape[1],
                      weights=[embed_mat],
                      input_length=MAX_LEN,
                      trainable=False)
    sequence_input = Input(shape=(MAX_LEN,), dtype='int32')
    embedded_sequences = embed(sequence_input)
    l_lstm = Bidirectional(LSTM(rnn_sz, return_sequences=True) )(embedded_sequences)
    z = AttLayer()( l_lstm )
    preds = Dense(num_cls, activation='softmax')(z)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['acc'])
    return model

def eval( att = True ):
    x_tn, y_tn, x_ts, y_ts, embedding_matrix = aux_data.load_ag(sep_sent=False)
    x_tn, y_tn, x_val, y_val = data_utils.sepData(x_tn, y_tn)
    if att:
        model = RNNAtt(embedding_matrix, x_tn.shape[1], y_tn.shape[1])
    else:
        model = RNNModel(embedding_matrix, x_tn.shape[1], y_tn.shape[1])
    print("compile done.")
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=1)
    model.fit(x_tn, y_tn, validation_data=(x_val, y_val),callbacks=[early_stopping],
              epochs=20, batch_size=50)
    score, acc = model.evaluate(x_ts, y_ts, batch_size=512)
    print('acc: %.3f' % acc)

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    set_session(sess)
    # print('LSTM without attention.')
    # eval( False )
    print('LSTM with attention.')
    eval( True )