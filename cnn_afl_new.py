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
seed_list = glob.glob('/home/dongdong/Project/objdump/afl_out/seeds/*')
bitmap_list = glob.glob('/home/dongdong/Project/objdump/afl_out/bitmaps/*')
seed_list.sort()
bitmap_list.sort()
test_list = glob.glob('/home/dongdong/Project/objdump/afl_out/test_seeds/*')
test_bitmap_list = glob.glob('/home/dongdong/Project/objdump/afl_out/test_bitmaps/*')
test_list.sort()
test_bitmap_list.sort()
import random
import time
random.seed(time.time())
#rand_index = np.arange(4601)
#np.random.shuffle(rand_index)
with open('new_int_sort','rb') as f:
    label_sort = f.read().split('\n')[:-1]
label_sort = [int(f.split(' ')[-1]) for f in label_sort]
with open('rand_index_list', 'rb') as fp:
    rand_index = pickle.load(fp)
#with open('rand_index_list', 'wb') as fp:
#    pickle.dump(rand_index,fp)

with open('new_int','rb') as f:
    label = f.read().split('\n')[:-1]
label = [int(f) for f in label]
label_sort_index = [label.index(f) for f in label_sort]
MAX_FILE_SIZE = 9959
MAX_BITMAP_SIZE = len(label)
SPLIT_RATIO = 4601

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
        file_name = "/home/dongdong/Project/objdump/afl_out/convert_bitmaps/" +                     bitmap_list[rand_index[i]].split('/')[-1] + ".npy"
        bitmap[i-lb] = np.load(file_name)
    return seed,bitmap

def generate_testing_data(lb,ub):
    seed = np.zeros((ub-lb,MAX_FILE_SIZE))
    bitmap = np.zeros((ub-lb,MAX_BITMAP_SIZE))
    for i in range(lb,ub):
        tmp = open(test_list[i],'r').read()
        ln = len(tmp)
        if ln < MAX_FILE_SIZE:
            tmp = tmp + (MAX_FILE_SIZE - ln) * '\0'
        seed[i-lb] = [ord(j) for j in list(tmp)]

    for i in range(lb,ub):
        file_name = "/home/dongdong/Project/objdump/afl_out/convert_test_bitmaps/" +                     test_list[i].split('/')[-1] + ".npy"
        bitmap[i-lb] = np.load(file_name)
    return seed,bitmap

batch_size = 32
num_classes = MAX_BITMAP_SIZE
epochs = 50
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
model.add(Dropout(0.25))

#model.add(Conv1D(128, 5, strides=2, padding='valid'))
#model.add(Activation('relu'))
#model.add(Conv1D(32, 5, strides=3))
#model.add(Activation('relu'))
#model.add(MaxPooling1D())

model.add(Flatten())
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.adam(lr=0.0001, decay=1e-6)

import keras.backend as K
import tensorflow as tf

def accur(y_true,y_pred):
    #TODO:use dot product/ sum
    pred = tf.ceil(tf.subtract(y_pred,np.full((MAX_BITMAP_SIZE,),0.5)))
    summ = tf.reduce_sum(y_true)
    comp = np.full((MAX_BITMAP_SIZE,),2)
    ret = tf.reduce_sum(tf.cast(tf.equal(tf.add(y_true, pred),comp),tf.float32))/summ
    return ret
# Let's train the model using RMSprop
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=[accur])

model.summary()
def train_generate(batch_size):
    while 1:
        for i in range(0,SPLIT_RATIO,batch_size):
            # create numpy arrays of input data
            # and labels, from each line in the file
            if (i+batch_size) > SPLIT_RATIO:
                x,y=generate_training_data(i,SPLIT_RATIO)
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
        for i in range(0,len(test_list),batch_size):
            # create numpy arrays of input data
            # and labels, from each line in the file
            if (i+batch_size) > len(test_list):
                x,y=generate_testing_data(i,len(test_list))
                x = x.reshape((x.shape[0],x.shape[1],1)).astype('float32')/255
                #x = x_train[i+10000:]
                #y = y_train[i+10000:]
            else:
                x,y=generate_testing_data(i,i+batch_size)
                x = x.reshape((x.shape[0],x.shape[1],1)).astype('float32')/255
                #x = x_train[i+10000:i+batch_size+10000]
                #y = y_train[i+10000:i+batch_size+10000]
            yield (x,y)

def gen_adv(f,fl,model,layer_list):
    adv_list = []
    loss = layer_list[-2][1].output[:,f]
    grads = K.gradients(loss,model.input)[0]
    grads /= (K.sqrt(K.sum(K.square(grads))))
    iterate = K.function([model.input], [loss, grads])
    #ll = random.sample(xrange(9701),4)
    #fl = [483]
    #ll = [int(e) for e in random_input_map[str(f)]]
    ll=4
    fl= random.sample(xrange(SPLIT_RATIO),4)
    for index in range(ll):
        x_ori,y=generate_training_data(fl[index],fl[index]+1)
        x = x_ori.reshape((x_ori.shape[0],x_ori.shape[1],1)).astype('float32')/255
        loss_value, grads_value = iterate([x])
        idx = np.argsort(np.absolute(grads_value),axis=1)[:,-20:,:].reshape((20,))
        val = np.sign(grads_value[0][idx])
        adv_list.append((idx,fl[index],val))
    return adv_list

def gen_adv1(f,fl,model,layer_list):
    adv_list = []
    loss = layer_list[-2][1].output[:,f]
    grads = K.gradients(loss,model.input)[0]
    grads /= (K.sqrt(K.sum(K.square(grads))))
    iterate = K.function([model.input], [loss, grads])
    #ll = random.sample(xrange(9701),4)
    #fl = [483]
    #ll = [int(e) for e in random_input_map[str(f)]]
    ll=2
    fl= random.sample(xrange(SPLIT_RATIO),2)
    for index in range(ll):
        x_ori,y=generate_training_data(fl[index],fl[index]+1)
        x = x_ori.reshape((x_ori.shape[0],x_ori.shape[1],1)).astype('float32')/255
        loss_value, grads_value = iterate([x])
        idx = np.argsort(np.absolute(grads_value),axis=1)[:,-1024:,:].reshape((1024,))
        val = np.sign(grads_value[0][idx])
        adv_list.append((idx,fl[index],val))
    return adv_list
import os
def gen_mutate():
    tmp_list = []
    #with open('target','rb') as f:
    #    interested_indice = pickle.load(f)
    #interested_indice = open('middle','r').read().split('\n')[:-1]
    interested_indice = np.random.choice(label_sort_index,125)
    model.load_weights('final_model_reduce.h5')
    layer_list = [(layer.name, layer) for layer in model.layers]
    cnt = 111111
    folder_cnt = 0
    for idxx in range(len(interested_indice[:])):
        #if not folder_cnt % 5:
        #    folder_name = str(folder_cnt)
        #    os.makedirs(folder_name)
        print("number of feature "+str(idxx))
        #index = int(interested_indice[idxx].split(':')[-1])
        #fl = "id:" + interested_indice[idxx].split(':')[-2]
        index = int(interested_indice[idxx])
        fl = 0
        adv_list = gen_adv(index,fl,model,layer_list)
        #print (index, adv_list)
        for ele in adv_list:
            print(ele[0])
            for el in ele[0]:
                tmp_list.append(el)

        for i in range(len(adv_list)):
            tmp = open(seed_list[rand_index[adv_list[i][1]]],'r').read()
            ln = len(tmp)
            max_loc = np.amax(adv_list[i][0])
            max_loc = MAX_FILE_SIZE
            if ln < max_loc:
                tmp = tmp + (max_loc - ln + 1) * '\0'
            '''
            for idx in adv_list[i][0]:
                for j in range(256):
                    tmp_str = list(tmp)
                    tmp_str[idx] = chr(j)
                    tmp_str = "".join(tmp_str)
                    # input_output_location_value
                    with open("/home/dongdong/Project/objdump/adv_gen_seeds/"+folder_name+"/id:"+str(cnt)+","+str(adv_list[i][1])+    "_"+str(index)+"_"+str(idx)+"_"+str(j), 'w') as f:
                        f.write(tmp_str)
                    cnt = cnt + 1
            '''
        folder_cnt+=1
    from collections import Counter
    c = Counter(tmp_list)
    print (len(c))
    print(c)
    ipdb.set_trace()

def gen_mutate1():
    tmp_list = []
    #with open('target','rb') as f:
    #    interested_indice = pickle.load(f)
    #interested_indice = open('middle','r').read().split('\n')[:-1]
    interested_indice = np.random.choice(label_sort_index,125)
    model.load_weights('final_model_reduce.h5')
    layer_list = [(layer.name, layer) for layer in model.layers]
    cnt = 111111
    folder_cnt = 0
    for idxx in range(len(interested_indice[:])):
        if not folder_cnt % 5:
            folder_name = str(folder_cnt)
            os.makedirs("adv_gen_seeds/"+folder_name)
        print("number of feature "+str(idxx))
        #index = int(interested_indice[idxx].split(':')[-1])
        #fl = "id:" + interested_indice[idxx].split(':')[-2]
        index = int(interested_indice[idxx])
        fl = 0
        adv_list = gen_adv1(index,fl,model,layer_list)
        #print (index, adv_list)
        #for ele in adv_list:
        #    print(ele[0])
        #    for el in ele[0]:
        #        tmp_list.append(el)

        for i in range(len(adv_list)):
            tmp = open(seed_list[rand_index[adv_list[i][1]]],'r').read()
            ln = len(tmp)
            max_loc = np.amax(adv_list[i][0])
            max_loc = MAX_FILE_SIZE
            if ln < max_loc:
                tmp = tmp + (max_loc - ln + 1) * '\0'
            lll = [0,2,4,8,16,32,64,128,256,512,1024]
            for xx in range(10):
                grad_vv = adv_list[i][-1][-int(lll[xx+1]):-int(lll[xx])]
                grad_idxx = adv_list[i][0][-int(lll[xx+1]):-int(lll[xx])]
                for idx in range(255):
                    grad = grad_vv*(idx+1)
                    tmp_str = list(tmp)
                    for grad_idx,val in enumerate(grad_idxx):
                        tmp_str[val] = chr(np.clip(ord(tmp_str[val])+grad[grad_idx],0,255))
                    tmp_str = "".join(tmp_str)
                    with open("/home/dongdong/Project/objdump/adv_gen_seeds/"+folder_name+"/id:"+str(cnt)+","+str(adv_list[i][1])+"_"+    str(index)+"_"+str(idx)+"_", 'w') as f:
                        f.write(tmp_str)
                    cnt = cnt + 1
                for idx in range(255):
                    grad = grad_vv*(-(idx+1))
                    tmp_str = list(tmp)
                    for grad_idx,val in enumerate(grad_idxx):
                        tmp_str[val] = chr(np.clip(ord(tmp_str[val])+grad[grad_idx],0,255))
                    tmp_str = "".join(tmp_str)
                    with open("/home/dongdong/Project/objdump/adv_gen_seeds/"+folder_name+"/id:"+str(cnt)+","+str(adv_list[i][1])+"_"+    str(index)+"_"+str(idx)+"_", 'w') as f:
                        f.write(tmp_str)
                    cnt = cnt + 1
        folder_cnt+=1
    #from collections import Counter
    #c = Counter(tmp_list)
    #print (len(c))
    #print(c)

def train():

    model.fit_generator(train_generate(32),
              steps_per_epoch = (SPLIT_RATIO/32 + 1),
              epochs=70,
              verbose=1,
              validation_data=test_generate(64),validation_steps=((len(test_list))/64+1),shuffle=True)#,callbacks=[callback])
    # Save model and weights
    model.save_weights("final_model_reduce.h5")

gen_mutate()
#train()
