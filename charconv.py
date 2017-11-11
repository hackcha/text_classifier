from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import SGD,Adadelta,Adagrad, Adam
from keras.layers import Input, Dense,BatchNormalization
from keras.layers import Dropout, Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Sequential
import data_utils
import aux_data
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import math

def charConv(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filters,
          cat_output):
    init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    #Define what the input shape looks like
    inputs = Input(shape=(maxlen, vocab_size), name='input', dtype='float32')

    #All the convolutional layers...
    conv = Convolution1D(nb_filter=nb_filters[0], filter_length=filter_kernels[0],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size), kernel_initializer=init,
                         bias_initializer=init)(inputs)
    conv = MaxPooling1D(pool_size=3)(conv)

    conv1 = Convolution1D(nb_filter=nb_filters[1], filter_length=filter_kernels[1],
                          border_mode='valid', activation='relu', kernel_initializer=init,
                         bias_initializer=init)(conv)
    conv1 = MaxPooling1D(pool_size=3)(conv1)

    conv2 = Convolution1D(nb_filter=nb_filters[2], filter_length=filter_kernels[2],
                          border_mode='valid', activation='relu', kernel_initializer=init,
                         bias_initializer=init)(conv1)

    conv3 = Convolution1D(nb_filter=nb_filters[3], filter_length=filter_kernels[3],
                          border_mode='valid', activation='relu', kernel_initializer=init,
                         bias_initializer=init)(conv2)

    conv4 = Convolution1D(nb_filter=nb_filters[4], filter_length=filter_kernels[4],
                          border_mode='valid', activation='relu', kernel_initializer=init,
                         bias_initializer=init)(conv3)
    #
    conv5 = Convolution1D(nb_filter=nb_filters[5], filter_length=filter_kernels[5],
                          border_mode='valid', activation='relu', kernel_initializer=init,
                         bias_initializer=init)(conv4)
    conv5 = MaxPooling1D(pool_size=3)(conv5)
    conv5 = Flatten()(conv5)
    #Two dense layers with dropout of .5
    use_bn = True
    if not use_bn:
        z = Dropout(0.5)(Dense(dense_outputs, activation='relu', kernel_initializer=init,
                         bias_initializer=init)(conv5))
        z = Dropout(0.5)(Dense(dense_outputs, activation='relu', kernel_initializer=init,
                         bias_initializer=init)(z))
    else:
        z = BatchNormalization()(Dense(dense_outputs, activation='relu', kernel_initializer=init,
                         bias_initializer=init)(conv5))
        z = BatchNormalization()(Dense(dense_outputs, activation='relu', kernel_initializer=init,
                         bias_initializer=init)(z))
    #Output dense layer with softmax activation
    pred = Dense(cat_output, activation='softmax', name='output', kernel_initializer=init,
                         bias_initializer=init)(z)
    model = Model(input=inputs, output=pred)
    model.summary( )
    #原始的优化算法为using momentum 0.9 and initial step size 0.01 which is halved every 3 epoches for 10 times
    # opt = SGD(0.05, momentum=0.9, decay= 0.999 )#clipvalue = 5.0
    opt = Adam( )
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    return model

def get_default_model(num_cls):
    maxlen = 1014
    dense_outputs = 1024
    # Conv layer kernel size
    nb_filters = [256,256,256,256,256,256]#[32,32,64,64,128,128 ]
    filter_kernels = [7, 7, 3, 3, 3, 3]
    print('Creating vocab...')
    _, _, vocab_size, _ = aux_data.create_vocab_set( )
    print('Build model...')
    m = charConv(filter_kernels, dense_outputs, maxlen, vocab_size,
                 nb_filters, num_cls)
    return m


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    set_session(sess)

    subset = None
    data_dir = './data/ag_news_csv/'
    maxlen = 1014
    x_tn, y_tn, x_ts, y_ts = aux_data.load_data(data_dir, [1,2])
    x_tn, y_tn, x_val, y_val = data_utils.sepData(x_tn, y_tn)
    vocab, reverse_vocab, vocab_size, check = aux_data.create_vocab_set()
    model = get_default_model(num_cls= y_tn.shape[1])
    batch_size = 128
    tn_size = y_tn.shape[0]
    batches = aux_data.mini_batch_generator(x_tn, y_tn, vocab, vocab_size,
                                                check, maxlen,
                                                batch_size=batch_size)
    val_batches = aux_data.mini_batch_generator(x_val ,y_val, vocab,
                                                    vocab_size, check , maxlen,
                                                    batch_size=batch_size )
    test_batches = aux_data.mini_batch_generator(x_ts, y_ts, vocab,
                                                     vocab_size, check, maxlen,
                                                     batch_size=batch_size)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    save_best = ModelCheckpoint('charconv_best.h5',save_best_only=True)
    model.fit_generator(batches , steps_per_epoch = math.ceil(tn_size/batch_size) ,
                        validation_data=val_batches, validation_steps= math.ceil(y_val.shape[0]/batch_size),
                        callbacks=[early_stopping, save_best],epochs=50)
    score, acc = model.evaluate_generator(test_batches, steps = math.ceil(y_ts.shape[0]/batch_size) )
    print('acc: %.3f' % acc)
    model.save_weights('charconv.h5')