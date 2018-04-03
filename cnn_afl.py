from __future__ import print_function
import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
import os
import numpy as np
import glob
import ipdb

seed_list = glob.glob('/home/dongdong/Project/afl/afl_out/fuzzer01/queue/*')
bitmap_list = glob.glob('/home/dongdong/Project/afl/afl_out/fuzzer01/bitmap/*')
seed_list.sort()
bitmap_list.sort()

#rand_index = np.arange(10778)
#np.random.shuffle(rand_index)
with open('rand_index_list', 'rb') as fp:
    rand_index = pickle.load(fp)
MAX_FILE_SIZE = 8561
MAX_BITMAP_SIZE = 65536

def generate_training_data(lb,ub):
    seed = np.zeros((ub-lb,MAX_FILE_SIZE))
    bitmap = np.zeros((ub-lb,MAX_BITMAP_SIZE))
    for i in range(lb,ub):
        tmp = open(seed_list[rand_index[i]],'r').read()
        ln = len(tmp)
        if ln < MAX_FILE_SIZE:
            tmp = tmp + (MAX_FILE_SIZE - ln) * '\0'
        seed[i-lb] = [ord(j) for j in list(tmp)]

    for i in range(lb,ub):
        tmp = open(bitmap_list[rand_index[i]],'r').read().split('\n')[:-1]
        tmp = [int(j) for j in tmp]
        for j in tmp:
            bitmap[i-lb][j] = 1
    return seed,bitmap

batch_size = 32
num_classes = 65536
epochs = 100
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cnn_afl_model.h5'

model = Sequential()
model.add(Conv1D(256, 5, strides=3,padding='valid',input_shape=(MAX_FILE_SIZE,1)))
model.add(Activation('relu'))
#model.add(Conv1D(256, 5))
#model.add(Activation('relu'))
#model.add(MaxPooling1D())
#model.add(Dropout(0.25))

#model.add(Conv1D(256, 5, strides=2, padding='valid'))
#model.add(Activation('relu'))
model.add(Conv1D(128, 7, strides=3))
model.add(Activation('relu'))
#model.add(MaxPooling1D())
#model.add(Dropout(0.25))

model.add(Conv1D(64, 9, strides=3))
model.add(Activation('relu'))
#model.add(MaxPooling1D())
#model.add(Dropout(0.25))

#model.add(Conv1D(128, 5, strides=2, padding='valid'))
#model.add(Activation('relu'))
#model.add(Conv1D(32, 5, strides=3))
#model.add(Activation('relu'))
#model.add(MaxPooling1D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#opt = keras.optimizers.adam(lr=0.0001, decay=1e-6)

import keras.backend as K
import tensorflow as tf

def accur(y_true,y_pred):
    #TODO:use dot product/ sum
    pred = tf.ceil(tf.subtract(y_pred,np.full((65536,),0.5)))
    summ = tf.reduce_sum(y_true)
    comp = np.full((65536,),2)
    ret = tf.reduce_sum(tf.cast(tf.equal(tf.add(y_true, pred),comp),tf.float32))/summ
    return ret
# Let's train the model using RMSprop
model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=[accur])

model.summary()
def train_generate(batch_size):
    while 1:
        for i in range(0,9701,batch_size):
            # create numpy arrays of input data
            # and labels, from each line in the file
            if (i+batch_size) > 9701:
                x,y=generate_training_data(i,9701)
                x = x.reshape((x.shape[0],x.shape[1],1)).astype('float32')/255
                #x = x_train[i:10000]
                #y = y_train[i:10000]
            else:
                x,y=generate_training_data(i,i+batch_size)
                x = x.reshape((x.shape[0],x.shape[1],1)).astype('float32')/255
                #x = x_train[i:i+batch_size]
                #y = y_train[i:i+batch_size]
            yield (x,y)

def test_generate(batch_size):
    while 1:
        for i in range(0,len(seed_list)-9701,batch_size):
            # create numpy arrays of input data
            # and labels, from each line in the file
            if (i+batch_size) > len(seed_list)-9701:
                x,y=generate_training_data(i+9701,len(seed_list))
                x = x.reshape((x.shape[0],x.shape[1],1)).astype('float32')/255
                #x = x_train[i+10000:]
                #y = y_train[i+10000:]
            else:
                x,y=generate_training_data(i+9701,i+batch_size + 9701)
                x = x.reshape((x.shape[0],x.shape[1],1)).astype('float32')/255
                #x = x_train[i+10000:i+batch_size+10000]
                #y = y_train[i+10000:i+batch_size+10000]
            yield (x,y)

filepath = 'best_weights_new.hdf5'
callback = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accur', verbose=2, save_best_only=True, save_weights_only=True, mode='max', period=1)

model.fit_generator(train_generate(32),
          steps_per_epoch = (9701/32 + 1),
          epochs=100,
          verbose=2,
          validation_data=test_generate(64),validation_steps=((len(seed_list)-9701)/64+1),shuffle=True,callbacks=[callback])
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model.save_weights("final_model_new.h5")
