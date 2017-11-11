'''Trains an SRU model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- Increase depth to obtain similar performance to LSTM
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM
from keras.datasets import imdb
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping
from sru import SRU
import data_utils
import aux_data

def eval_imdb():
    max_features = 20000
    maxlen = 80  # cut texts after this number of words (among top max_features most common words)
    batch_size = 128

    depth = 1

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    ip = Input(shape=(maxlen,))
    embed = Embedding(max_features, 128)(ip)

    prev_input = embed
    hidden_states = []

    if depth > 1:
        for i in range(depth - 1):
            h, h_final, c_final = SRU(128, dropout=0.0, recurrent_dropout=0.0,
                                      return_sequences=True, return_state=True,
                                      unroll=True)(prev_input)
            prev_input = h
            hidden_states.append(c_final)

    outputs = SRU(128, dropout=0.0, recurrent_dropout=0.0, unroll=True)(prev_input)
    outputs = Dense(1, activation='sigmoid')(outputs)

    model = Model(ip, outputs)
    model.summary()

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=100,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

def SRUModel( embed_mat, MAX_LEN, num_cls, sru_sz = 128  ):
    ip = Input(shape=(MAX_LEN,))
    embed = Embedding(embed_mat.shape[0],
                      embed_mat.shape[1],
                      weights=[embed_mat],
                      input_length=MAX_LEN,
                      trainable=False)

    prev_input = embed(ip)
    hidden_states = []
    depth = 2
    if depth > 1:
        for i in range(depth - 1):
            h, h_final, c_final = SRU(sru_sz, dropout=0.0, recurrent_dropout=0.0,
                                      return_sequences=True, return_state=True,
                                      unroll=True)(prev_input)
            prev_input = h
            hidden_states.append(c_final)
    outputs = SRU(sru_sz, dropout=0.0, recurrent_dropout=0.0, unroll=True)(prev_input)
    outputs = Dense(num_cls, activation='softmax')(outputs)
    model = Model(ip, outputs)
    model.summary()
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    set_session(sess)
    x_tn, y_tn, x_ts, y_ts, embedding_matrix = aux_data.load_ag(sep_sent=False)
    x_tn, y_tn, x_val, y_val = data_utils.sepData(x_tn, y_tn)
    model = SRUModel(embedding_matrix, x_tn.shape[1], y_tn.shape[1])
    print("compile done.")
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=1)
    model.fit(x_tn, y_tn, validation_data=(x_val, y_val),
              epochs=20, callbacks=[early_stopping], batch_size=50)
    score, acc = model.evaluate(x_ts, y_ts, batch_size=512)
    print('acc: %.3f' % acc)