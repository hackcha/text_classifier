import numpy as np
import pandas as pd
from collections import defaultdict
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import Embedding,Masking
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.optimizers import SGD, RMSprop, Adagrad
from attention import AttLayer
import aux_data, data_utils

def LSTMAtt( ):
    pass

def HAN( embed_mat, MAX_LEN, MAX_SENTS, num_cls, gru_sz1 = 100, gru_sz2 = 100):
    embedding_layer = Embedding(embed_mat.shape[0] ,
                                embed_mat.shape[1],
                                weights=[embed_mat],
                                input_length=MAX_LEN,
                                mask_zero=True,
                                trainable=True)

    sentence_input = Input(shape=(MAX_LEN,), dtype='int32')
    embedded_sequences = embedding_layer( sentence_input )#sentence_input)
    l_lstm = Bidirectional(LSTM(gru_sz1, return_sequences=True))(embedded_sequences)
    l_att = AttLayer( name='att_1')(l_lstm)
    sentEncoder = Model(sentence_input, l_att)
    review_input = Input(shape=(MAX_SENTS,MAX_LEN), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    review_encoder = Masking(mask_value=0.)(review_encoder )
    l_lstm_sent = Bidirectional(LSTM(gru_sz2, return_sequences=True))(review_encoder)
    l_att_sent = AttLayer(name='att_2')(l_lstm_sent)
    preds = Dense(num_cls, activation='softmax',name='twt_softmax')(l_att_sent)
    model = Model(review_input, preds)
    opt = Adagrad(lr=0.1)#, clipvalue=5.0
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

    x_tn, y_tn, x_ts, y_ts, embedding_matrix = aux_data.load_ag(sep_sent=True)
    x_tn, y_tn, x_val, y_val = data_utils.sepData(x_tn, y_tn)
    model = HAN(embedding_matrix, x_tn.shape[2],x_tn.shape[1], y_tn.shape[1])
    print("compile done.")
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=1)
    model.fit(x_tn, y_tn, validation_data=(x_val, y_val),callbacks=[early_stopping],
              epochs=20, batch_size=50)
    validate = False
    score, acc = model.evaluate(x_ts, y_ts, batch_size=512)
    print('acc: %.3f' % acc)