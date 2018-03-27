# coding: utf-8

# In[171]:

# Ido Michael
import tensorflow as tf
import os, struct
import numpy as np
import matplotlib.pyplot as plt
import ParsePowerEDFA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import sys
import configparser
import random


print(tf.__version__)
# In case we need to average results of 5 different debug files and plot them on a graph.
# ParsePowerEDFA.getTestFiles()

# Average files by name and then write collected results into a csv file.
# [testdb, tr2, tr4, tr6, tr8, tr1, mse_tr, mae_tr] = ParsePowerEDFA.averageResults("100%")
# ParsePowerEDFA.plot_to_matrix(tr2, tr4, tr6, tr8, tr1, mse_tr, mae_tr)

# 20%
# [testdb, val2, val4, val6, val8, val1] = ParsePowerEDFA.averageResults([
#     "./TestPar29.ini140-debug.log",
#     "./TestPar29.ini84-debug.log",
#     "./TestPar29.ini150-debug.log"
# ])

# [testdb, val2, val4, val6, val8, val1] = ParsePowerEDFA.averageResults(["./test/TestPar25.ini-smaller53-debug.log", "./test/TestPar25.ini-smaller103-debug.log", "./test/TestPar25.ini-smaller25-debug.log", "./test/TestPar25.ini-smaller37-debug.log", "./test/TestPar25.ini-smaller30-debug.log"])
# ParsePowerEDFA.plotGraph(val2, val4, val6, val8, val1)

# file='Data/sameroute_3initial.txt'
# file='Data/train_dataset_3to5initial_1new_2017-02-03-16-58-03.txt'
# [counter,on_channels,input_power,new_channel,power_excursion]=ParseData.ParseDataFile(file)


import imp

imp.reload(ParsePowerEDFA)

# Set config file and parse it to the different variables.
config = configparser.ConfigParser()
config_file = sys.argv[1]
config.read(config_file)

configuration = "DEFAULT"
in_format = config.get(configuration, 'input')
out_format = config.get(configuration, 'output')
norm_in = config.get(configuration, 'normalize_in')
norm_out = config.get(configuration, 'normalize_out')
num_neurons = config.get(configuration, 'layers').replace("\n","")
num_neurons = list(map(int, num_neurons.split(",")))
layers = int(len(num_neurons))
act_functions = config.get(configuration, 'functions').replace("\n","")
act_functions = list(map(str, act_functions.split(",")))
num_iterations = int(config.get(configuration, 'iterations'))
learning_rate = float(config.get(configuration, 'epsilon'))
decrease_rate = float(config.get(configuration, 'decrease_rate'))
out_channel = int(config.get(configuration, 'output_shape'))
scale = int(config.get(configuration, 'scaler'))
scale_in = int(config.get(configuration, 'scaleInput'))
only_on = config.get(configuration, 'onlyOn')
train_percentage = float(config.get(configuration, 'trainPercent'))
test_percentage = float(config.get(configuration, 'testPercent'))
isLoging = config.get(configuration, 'debugFile')
data_percentage = float(config.get(configuration, 'dataPercentage'))
file1 = config.get(configuration, 'dataFile')
chan_num = int(config.get(configuration, 'activeChannel'))

[counter, input_dict, output_dict] = ParsePowerEDFA.ParseDataFileSingleChannel(file1)

# Set a single channel out of the dictionary
input_power = input_dict[chan_num]
output_power = output_dict[chan_num]

# Save print outputs into a log file, always works with most updated file at a time.
if(isLoging == "on"):
    old_stdout = sys.stdout
    try:
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        wantedLogFiles = [filename for filename in files if filename.startswith(config_file)]
        wantedLogFiles = [file for file in wantedLogFiles if file.replace(config_file, "")]

        # Randomize logs name for multiThreading
        if len(wantedLogFiles) != 0:
            numIdentifier = random.randint(2, 150)
            log_file = open(config_file + str(numIdentifier) + "-debug.log", 'w')
        else:
            numIdentifier = random.randint(2, 150)
            log_file = open(config_file + str(numIdentifier) + "-debug.log", 'w')
    except:
        # There was an exception
        print("exception opening a file - log file")
        log_file = open("default.log", 'w')

    sys.stdout = log_file

# In[180]:
# print parsed config file values.
print("Config file: %s " % str(config_file))
print("Channel number: %s " % chan_num)
print("Input Parameter file: %s " % str(file1))
print("Input Format: %s" % str(in_format))
print("Output Format: %s" % str(out_format))
print("Input Normalize: %s" % str(norm_in))
print("Output Normalize: %s" % str(norm_out))
print("Number of layers: %s" % str(layers))
print("Number of neurons per layer: %s" % str(num_neurons))
print("Activation functions for each layer: %s" % str(act_functions))
print("Number of iterations of loop: %d" % num_iterations)
print("Learning rate: %f" % learning_rate)
print("Output Channel: %d" % out_channel)
print("Scale: %d" % scale)
print("Scale in: %d" % scale_in)
print("On or All Channels: %s" % str(only_on))
print("Training percentage: %f" % train_percentage)
print("Testing percentage: %f" % test_percentage)
print("Debug to external file: %s" % str(isLoging))
print("Data percentage: %f" % data_percentage)
print("Config file name: %s" % str(file1))
print("Active channel number: %d" % chan_num)
print("Those are the active channels:")
print(output_dict.keys())



# In[10]:

def find_max_values(data):
    a, b = data.shape
    x_max = [0] * b
    for i in range(0, b):
        x_max[i] = max([item[i] for item in data])
    return x_max


# In[11]:

def find_min_values(data):
    a, b = data.shape
    x_min = [0] * b
    for i in range(0, b):
        x_min[i] = min([item[i] for item in data])
    return x_min


# In[12]:

def find_mean(data):
    a, b = data.shape
    mean_data = [0] * b
    for i in range(0, b):
        mean_data[i] = np.mean([item[i] for item in data])
    return mean_data


# In[13]:

def normalize_input(values, min_values, max_values, normal_factor, scale):
    temp = values.copy()
    a, b = temp.shape
    for i in range(0, a):
        for j in range(0, b):
            if (max_values[j] == min_values[j]):
                temp[i][j] = 0
            else:
                temp[i][j] = (temp[i][j] - normal_factor[j]) / (max_values[j] - min_values[j]) * scale
    return temp


# In[14]:

def denormalize_input(values, min_values, max_values, normal_factor, out_format, scale):
    temp = values.copy()
    a, b = temp.shape
    for i in range(0, a):
        for j in range(0, b):
            temp[i][j] = (temp[i][j] / scale) * (max_values[j] - min_values[j]) + normal_factor[j]
    if out_format == 'dB':
        return temp
    if out_format == 'dec':
        return dec_to_dB(temp)


# Convert values from decimals to db (inputs are automatically converted to decimals).
def dec_to_dB(values):
    temp = values.copy()
    a, b = temp.shape
    for i in range(0, a):
        for j in range(0, b):
            if temp[i][j] > 0:
                temp[i][j] = 10 * math.log10(temp[i][j])
            else:
                temp[i][j] = -1000
    return temp


# In[16]:

def ErrorMethod(methodtype, vector_on, power_diff):
    # does calculations on individual inputs, need to creat function that does it on all of the data
    if methodtype == 'Absolute':
        out = [sum(abs(power_diff[i, :])) / (sum((vector_on[i, :] != 0))) for i in range(vector_on.shape[0])]
    if methodtype == 'Square':
        out = [sum(np.square(power_diff[i, :])) / (sum((vector_on[i, :] != 0))) for i in range(vector_on.shape[0])]
    if methodtype == 'Max':
        out = [max(abs(power_diff[i, :])) for i in range(vector_on.shape[0])]
    return out


# Set input/outputs to correct measurment of db/decimals
if in_format == 'dB':
    in_power_unit = dec_to_dB(input_power)
if in_format == 'dec':
    in_power_unit = input_power
if out_format == 'dB':
    out_power_unit = dec_to_dB(output_power)
if out_format == 'dec':
    out_power_unit = output_power

# In[125]:

# print input array
in_power_unit.shape

# Shuffle data for randomness
n = in_power_unit.shape[0]
indices = np.arange(n)
random.shuffle(indices)
in_power_unit = in_power_unit[indices]
out_power_unit = out_power_unit[indices]

def printInputs():
    a, b = in_power_unit.shape
    for i in range(0, a):
        print("input")
        print(in_power_unit[i])
        print("output")
        print(out_power_unit[i])

def get_channel(values, channel):
    a = values.shape
    out_power = [0] * a[0]
    for i in range(0, a[0]):
        out_power[i] = values[i][channel - 1]
    out_power = np.asarray(out_power).astype(np.float32)
    d = out_power.reshape((a, 1))
    return d


# only_on=1
if only_on == 'on':
    out_power_unit = out_power_unit[input_power[:, out_channel - 1] > 0]
    in_power_unit = in_power_unit[input_power[:, out_channel - 1] > 0]

# In[149]:

data_size =  math.floor(len(out_power_unit) * data_percentage)
print("Data size: %d" % data_size)


# In[131]:

# out_power_channel=get_channel(out_power_unit,out_channel)


# In[132]:

def create_weight(shape):
    # creates and initializes a weight matrix of the specified size
    return tf.Variable(tf.truncated_normal(shape, stddev=.1, dtype=tf.float32))


def create_bias(shape):
    # creates and initializes a bias term of the specified size
    return tf.Variable(tf.constant(.1, shape=shape, dtype=tf.float32))


def create_perm(data):
    out = np.random.permuation(data.shape[0])
    return out


# In[133]:

def all_nn_computations(X, weights_0, biases_0, weights_1, biases_1):
    Z_0 = tf.nn.relu(tf.matmul(X, weights_0) + biases_0)
    logits = tf.matmul(Z_0, weights_1) + biases_1
    return logits


# In[134]:

def all_nn_computations2(X, weights, biases):
    return tf.matmul(X, weights) + biases


# In[135]:

def all_nn_computations3(X, weights_0, biases_0, weights_1, biases_1, weights_2, biases_2):
    Z_0 = (tf.matmul(X, weights_0) + biases_0)
    Z_1 = (tf.matmul(Z_0, weights_1) + biases_1)
    logits = tf.matmul(Z_1, weights_2) + biases_2
    return logits


# In[136]:

def all_nn_computations5(X, weights_0, biases_0, weights_1, biases_1, weights_2, biases_2, weights_3, biases_3,
                         weights_4, biases_4):
    Z_0 = tf.nn.tanh(tf.matmul(X, weights_0) + biases_0)
    Z_1 = tf.nn.tanh(tf.matmul(Z_0, weights_1) + biases_1)
    Z_2 = tf.nn.tanh(tf.matmul(Z_1, weights_2) + biases_2)
    Z_3 = (tf.matmul(Z_2, weights_3) + biases_3)
    logits = tf.matmul(Z_3, weights_4) + biases_4

    return logits


# In[137]:

def all_nn_computations5relu(X, weights_0, biases_0, weights_1, biases_1, weights_2, biases_2, weights_3, biases_3,
                             weights_4, biases_4):
    Z_0 = tf.nn.relu(tf.matmul(X, weights_0) + biases_0)
    Z_1 = tf.nn.relu(tf.matmul(Z_0, weights_1) + biases_1)
    Z_2 = tf.nn.relu(tf.matmul(Z_1, weights_2) + biases_2)
    Z_3 = tf.nn.relu(tf.matmul(Z_2, weights_3) + biases_3)
    logits = tf.matmul(Z_3, weights_4) + biases_4

    return logits



counter2 = np.arange(1, data_size + 1)

# In[158]:

# Number of observations in each group by size, split data to training, testing and validation.
n_test_cases = math.floor(len(out_power_unit) * test_percentage) #72000*.1   #200
n_training_cases = math.floor(data_size * train_percentage)  # 72000*.8 #500
n_validation_cases = math.floor(data_size * (1 - test_percentage - train_percentage))
n_subset_train = n_training_cases * 1
total_data = n_test_cases + n_validation_cases + n_training_cases

input_power_train = in_power_unit[0: n_training_cases].astype(np.float32)
output_power_train = out_power_unit[0: n_training_cases].astype(np.float32)

# input_power_train = in_power_unit[counter2[:] <= n_subset_train].astype(np.float32)
# output_power_train = out_power_unit[counter2[:] <= n_subset_train].astype(np.float32)

input_power_test=in_power_unit[n_training_cases: n_training_cases+n_test_cases].astype(np.float32)
output_power_test=out_power_unit[n_training_cases: n_training_cases+n_test_cases].astype(np.float32)

# input_power_test=in_power_unit[np.logical_and(counter2[:]>n_training_cases,counter2[:]<=n_training_cases+n_test_cases)].astype(np.float32)
# output_power_test=out_power_unit[np.logical_and(counter2[:]>n_training_cases,counter2[:]<=n_training_cases+n_test_cases)].astype(np.float32)

input_power_val=in_power_unit[n_training_cases+n_test_cases:total_data].astype(np.float32)
output_power_val=out_power_unit[n_training_cases+n_test_cases:total_data].astype(np.float32)

# input_power_val=in_power_unit[np.logical_and(counter2[:]>(n_training_cases+n_test_cases),counter2[:]<=data_size)].astype(np.float32)
# output_power_val=out_power_unit[np.logical_and(counter2[:]>(n_training_cases+n_test_cases),counter2[:]<=data_size)].astype(np.float32)


print("Shape of X_train: ", input_power_train.shape, "\t Shape of Y_train: ", output_power_train.shape)
print("Shape of X_test: ", input_power_test.shape, "\t Shape of Y_test: ", output_power_test.shape)
print("Shape of X_val: ", input_power_val.shape, "\t Shape of Y_val: ", output_power_val.shape)

# Collect different mean/max/min values for inputs
max_values_in = find_max_values(input_power_train)
min_values_in = find_min_values(input_power_train)
mean_values_in = find_mean(input_power_train)
if norm_in == 'no':
    X_train = input_power_train
    X_test = input_power_test
    X_val = input_power_val
elif norm_in == 'mu':
    X_train = normalize_input(input_power_train, min_values_in, max_values_in, mean_values_in, scale_in)
    X_test = normalize_input(input_power_test, min_values_in, max_values_in, mean_values_in, scale_in)
    X_val = normalize_input(input_power_val, min_values_in, max_values_in, mean_values_in, scale_in)
elif norm_in == 'min':
    X_train = normalize_input(input_power_train, min_values_in, max_values_in, min_values_in, scale_in)
    X_test = normalize_input(input_power_test, min_values_in, max_values_in, min_values_in, scale_in)
    X_val = normalize_input(input_power_val, min_values_in, max_values_in, min_values_in, scale_in)

# Collect different mean/max/min values for outputs
max_values_out = find_max_values(output_power_train)
min_values_out = find_min_values(output_power_train)
mean_values_out = find_mean(output_power_train)
if norm_out == 'no':
    Y_train = output_power_train
    Y_test = output_power_test
    Y_val = output_power_val
elif norm_out == 'mu':

    Y_train = normalize_input(output_power_train, min_values_out, max_values_out, mean_values_out, scale)
    Y_test = normalize_input(output_power_test, min_values_out, max_values_out, mean_values_out, scale)
    Y_val = normalize_input(output_power_val, min_values_out, max_values_out, mean_values_out, scale)
elif norm_out == 'min':
    Y_train = normalize_input(output_power_train, min_values_out, max_values_out, min_values_out, scale)
    Y_test = normalize_input(output_power_test, min_values_out, max_values_out, min_values_out, scale)
    Y_val = normalize_input(output_power_val, min_values_out, max_values_out, min_values_out, scale)

# Different measures for success
# error for each individual
def accuracy(predictions, truth):
    # Return % of correctly classified images
    # return (100.0 * np.sum(np.argmax(predictions, axis=1) == np.squeeze(labels)) / predictions.shape[0])
    return (np.sum((predictions - truth) ** 2)) / predictions.shape[0]


# how many you get correct
def PredictionAccuracy(predictions, truth, set_num):
    mi = int(min(set_num))
    mx = int(max(set_num))
    # print(type(mi))
    correct = 0
    for i in range(mi, mx + 1):
        # print(i)
        temp1 = truth[set_num == i]
        np.where(temp1 == min(temp1))
        temp2 = predictions[set_num == i]
        mi_truth = np.where(temp1 == min(temp1))
        mi_truth = np.array(mi_truth)
        # print(temp1)
        mi_predict = np.where(temp2 == min(temp2))
        mi_predict = np.array(mi_predict)
        # print(temp2)
        # print(mi_truth)
        # print(mi_predict)
        if np.any(mi_truth[0] == mi_predict[0]):
            correct = correct + 1
    return 1.0 * correct / (mx - mi + 1)


def PredictionDifference(predictions, truth, set_num):
    mi = int(min(set_num))
    mx = int(max(set_num))
    # print(type(mi))
    error_total = 0
    for i in range(mi, mx + 1):
        # print(i)
        temp1 = truth[set_num == i]
        # np.where(temp1==min(temp1))
        temp2 = predictions[set_num == i]
        # mi_truth=np.where(temp1==min(temp1))
        # mi_truth=np.array(mi_truth)
        # print(temp1)
        mi_predict = np.where(temp2 == min(temp2))
        mi_predict = np.array(mi_predict)
        min_value_actual = min(temp1)
        min_value_predicted = temp1[mi_predict[0][0]]
        err = min_value_predicted - min_value_actual
        # print(temp2)
        # print(mi_truth)
        # print(mi_predict)
        error_total = error_total + err
    return 1.0 * error_total / (mx - mi + 1)


def PredictionEpsilonAccuracy(predictions, truth, set_num, epsilon):
    # if channel chosen is within epsilon of the best channel, it is considered correct
    mi = int(min(set_num))
    mx = int(max(set_num))
    # print(type(mi))
    correct = 0
    for i in range(mi, mx + 1):
        # print(i)
        temp1 = truth[set_num == i]
        # np.where(temp1==min(temp1))
        temp2 = predictions[set_num == i]
        mi_predict = np.where(temp2 == min(temp2))
        mi_predict = np.array(mi_predict)
        min_value_actual = min(temp1)
        min_value_predicted = temp1[mi_predict[0][0]]
        err = min_value_predicted - min_value_actual
        if err <= epsilon:
            correct = correct + 1
    return 1.0 * correct / (mx - mi + 1)


def plot_weights(weights):
    plt.figure()
    for j in range(num_classes):
        # Create and choose subplot
        ax = plt.subplot(1, num_classes, j + 1)
        # Obtain the weights corresponding to class j
        weights_j = weights[:, j]
        # Reshape
        weights_reshaped = np.reshape(weights_j, (28, 28))
        # Plot
        ax.imshow(weights_reshaped, cmap=plt.get_cmap('Greys'))
        plt.axis('off')
        plt.title('digit #' + str(j), fontsize=7.0)
    plt.show()


# In[65]:
# if some entry has a big error between predicted and actual output.
def checkErrors(pred, true):
    a, b = pred.shape
    pred_vector = []
    true_vector = []
    index = []
    for i in range(0, a):
        for j in range(0, b):
            if abs(pred[i][j] - true[i][j]) > 500:
                index.append(i)
                index.append(j)
                pred_vector.append(float(pred[i][j]))
                true_vector.append(float(true[i][j]))
    return index, pred_vector, true_vector


# In[66]:

def next_minibatch(X_, Y_, batch_size):
    # Create a vector with batch_size random integers
    perm = np.random.permutation(X_.shape[0])
    perm = perm[:batch_size]
    # Generate the minibatch
    X_batch = X_[perm, :]
    Y_batch = Y_[perm, :]
    # Return the images and the labels
    return X_batch, Y_batch


# In[67]:

def Accuracy(pred, true, error):
    a, b = pred.shape
    count = 0
    hit = 0
    for i in range(0, a):
            if true[i] != -1000:
                count += 1
                if abs(pred[i] - true[i]) < error:
                    hit += 1

    return hit / count * 1.0

def layer_calculations(X, weights, biases, act_function):
    if act_function == 'tanh':
        Z = tf.nn.tanh(tf.matmul(X, weights) + biases)
    if act_function == 'linear':
        Z = (tf.matmul(X, weights) + biases)
    if act_function == 'relu':
        Z = tf.nn.relu(tf.matmul(X, weights) + biases)
    if act_function == 'sigmoid':
        Z = tf.nn.sigmoid(tf.matmul(X, weights) + biases)
    return Z


def all_nn_computationsloop(X, layers, weights, biases, activations):
    Z = X
    for i in range(0, layers):
        Z = layer_calculations(Z, weights[i], biases[i], activations[i])

    return Z



def print_data(data, limit):
    a, b = data.shape
    mi = min(a, limit)
    for i in range(0, mi):
        print(data[i])


#  Create a Tensorflow graph for multinomial logistic regression
num_pixels = X_train.shape[1]
print(num_pixels)
# num_classes = 90
num_classes = 1
batch_size = 60
beta = 0
graph_MLR = tf.Graph()

with graph_MLR.as_default():
    # (a) Input data
    #     Load the training, validation and test data into constants that are
    #     attached to the graph
    tf_train_data = tf.placeholder(tf.float32, shape=(batch_size, num_pixels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_classes))
    tf_test_data = tf.constant(X_test)
    tf_val_data = tf.constant(X_val)

    # (b) Variables
    #     Indicate the parameters that we need to infer
    network_weights = {}
    network_biases = {}
    for i in range(0, layers):
        if i == 0:
            network_weights[0] = create_weight([num_pixels, num_neurons[i]])
            network_biases[0] = create_bias([num_neurons[i]])
        elif i < (layers - 1):
            network_weights[i] = create_weight([num_neurons[i - 1], num_neurons[i]])
            network_biases[i] = create_bias([num_neurons[i]])
        else:
            network_weights[i] = create_weight([num_neurons[i - 1], num_classes])
            network_biases[i] = create_bias([num_classes])
    # num_pixels = input layer, num_neurons = inner layers, Num_classes = output layer


    # (c) Computations
    #     Indicate the computations that we want to perform with the variables and data
    train_logits = all_nn_computationsloop(tf_train_data, layers, network_weights, network_biases, act_functions)
    # loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(train_logits, tf_train_labels) )


    # Mean squared error
    # cost = tf.reduce_sum(tf.pow(train_logits-tf_train_labels, 2))/(2*num_pixels)
    # bias=tf.reduce_mean(tf.sub(tf.transpose(train_logits),tf_train_labels))
    print(train_logits)
    print(tf_train_labels)
    # loss = tf.reduce_mean(tf.square(tf.sub(tf.transpose(train_logits),tf_train_labels)))
    # loss = tf.reduce_mean(tf.mul(tf.sub((train_logits),tf_train_labels),(tf.sub((train_logits),tf_train_labels))))
    loss = tf.reduce_mean(tf.square(tf.subtract((train_logits),
                                                tf_train_labels)))
    print(loss)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(learning_rate, global_step,10000, decrease_rate, staircase=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)

    train_prediction = (train_logits)
    val_prediction = all_nn_computationsloop(tf_val_data, layers, network_weights, network_biases, act_functions)
    test_prediction = all_nn_computationsloop(tf_test_data, layers, network_weights, network_biases, act_functions)

# Run NN
max_iterations = num_iterations  # 100001#500001

with tf.Session(graph=graph_MLR) as session:
    # 1. Initialize the weights and biases. This is a one-time operation
    # tf.global_variables_initializer().run()
    tf.global_variables_initializer().run()
    # tf.initialize_all_variables().run()
    print('Initialized')
    tr_predict_hist = []
    tr_score_hist = []
    val_predict_hist = []
    val_score_hist = []
    val_error_hist = []
    val_error_epsilon_pt1_hist = []
    val_error_epsilon_pt2_hist = []
    val_error_epsilon_pt3_hist = []
    for step in range(max_iterations):
        X_batch, Y_batch = next_minibatch(X_train, Y_train, batch_size)

        # feed_dict = { tf_train_data   : X_batch,
        #              tf_train_labels : np.squeeze(Y_batch)}
        feed_dict = {tf_train_data: X_batch,
                     tf_train_labels: Y_batch}
        # Run the computations. We tell .run() that we want to run the optimizer,
        # Craig: May want to run optimizer by itself most time, dont need to calculate train prediction every step
        # print("boo")
        _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)

        # Report every 10000 iterations
        if (step % 200 == 0):
            # Print the loss
            # predictions=train_prediction.eval()

            print('Loss at step %d: %f' % (step, l))
            print("learning rate %d: %f" % (step, lr))
            # Obtain and print the accuracy on the training set
            tr_predict = mean_squared_error(predictions, Y_batch)
            tr_predict_hist = np.append(tr_predict_hist, tr_predict)

            print('  +Training MSE: %f' % tr_predict)
            print('  +Training MAE: %f' % mean_absolute_error(predictions, Y_batch))
            if norm_in != 'no':
                tr_pred_dB = denormalize_input(predictions, min_values_out, max_values_out, min_values_out, out_format,
                                               scale)
                Y_tr_dB = denormalize_input(Y_batch, min_values_out, max_values_out, min_values_out, out_format, scale)
                print('  +Training MSE: %f' % mean_squared_error(tr_pred_dB, Y_tr_dB))
                print('  +Training MAE: %f' % mean_absolute_error(tr_pred_dB, Y_tr_dB))
            print('Predictions')
            print_data(predictions, 20)
            print('      ')
            print('Batch')
            print_data(Y_batch, 20)
            print('      ')
            if norm_in != 'no':
                print('Tr Predictions dB')
                print_data(tr_pred_dB, 20)
                print('      ')
                print('Tr_dB')
                print_data(Y_tr_dB, 20)

            # print('  +Training Prediction: %f' % tr_score)
            # Obtain and print the accuracy on the validation set
            val_pred_value = val_prediction.eval()
            val_predict = mean_squared_error(val_pred_value, Y_val)
            val_predict_hist = np.append(val_predict_hist, val_predict)

            print('  +Validation MSE: %f' % mean_squared_error(val_prediction.eval(), Y_val))
            print('  +Validation MAE: %f' % mean_absolute_error(val_prediction.eval(), Y_val))
            if norm_out != 'no':
                pred_dB = denormalize_input(val_prediction.eval(), min_values_out, max_values_out, min_values_out,
                                            out_format, scale)
                Y_Val_dB = denormalize_input(Y_val, min_values_out, max_values_out, min_values_out, out_format, scale)
                print('  +Validation MSE: %f' % mean_squared_error(pred_dB, Y_Val_dB))
                print('  +Validation MAE: %f' % mean_absolute_error(pred_dB, Y_Val_dB))

            print('Val Predictions')
            print_data(val_prediction.eval(), 20)
            print('      ')
            print('Val Data')
            print_data(Y_val, 20)
            print('      ')
            if norm_out != 'no':
                print('Predictions dB')
                print_data(pred_dB, 20)
                print('      ')
                print('Val_dB')
                print_data(Y_Val_dB, 20)
                print('Validation: Percent of non zero Power predictions within .2,.4,.6 dB')
                print(Accuracy(pred_dB,Y_Val_dB,.2))
                print(Accuracy(pred_dB,Y_Val_dB,.4))
                print(Accuracy(pred_dB,Y_Val_dB,.6))
                print(Accuracy(pred_dB, Y_Val_dB, .8))
                print(Accuracy(pred_dB, Y_Val_dB, 1))
                print('Training: Percent of non zero Power predictions within .2,.4,.6 dB')
                print(Accuracy(tr_pred_dB, Y_tr_dB, .2))
                print(Accuracy(tr_pred_dB, Y_tr_dB, .4))
                print(Accuracy(tr_pred_dB, Y_tr_dB, .6))
                print(Accuracy(tr_pred_dB, Y_tr_dB, .8))
                print(Accuracy(tr_pred_dB, Y_tr_dB, 1))

                # There's no testing defined yet.
                if step != 0:
                    ind, pred_vector, true_vector = checkErrors(pred_dB, Y_Val_dB)
                    print("Number of Test Cases 0 Power")
                    for i in range(0,len(Y_Val_dB[0])):
                        print(int(sum(Y_Val_dB[i] == -1000)))
                    print("Number of cases that were 0 power that we had predicted power")
                    print(len(ind)/2)
                    print("Number of Test Cases that had non 0 power")
                    print(ind)
                    # print(pred_vector)
                    # print(true_vector)
    # 3. Accuracy on the test set
    test_pred = test_prediction.eval()
    if norm_out != 'no':
        test_pred_dB = denormalize_input(test_pred, min_values_out, max_values_out, min_values_out, out_format, scale)
        Y_test_dB = denormalize_input(Y_test, min_values_out, max_values_out, min_values_out, out_format, scale)
    print('Test Data Pred')
    print_data(test_pred, 200)
    print('   ')
    print('Test Data')
    print_data(Y_test, 200)
    if norm_out != 'no':
        print('Test Data Pred dB')
        print_data(test_pred_dB, 200)
        print('   ')
        print('Test Data dB')
        print_data(Y_test_dB, 200)
    print('Test MSE: %f' % mean_squared_error(test_pred, Y_test))
    print('Test MAE: %f' % mean_absolute_error(test_pred, Y_test))
    if norm_out != 'no':
        print('  +TEST MSE: %f' % mean_squared_error(test_pred_dB, Y_test_dB))
        print('  +TEST MAE: %f' % mean_absolute_error(test_pred_dB, Y_test_dB))

# Print accuracy vals for each size
print('Test: Percent of non zero Power predictions within .2,.4,.6 dB')
print(Accuracy(test_pred_dB, Y_test_dB, .2))
print(Accuracy(test_pred_dB, Y_test_dB, .4))
print(Accuracy(test_pred_dB, Y_test_dB, .6))
print(Accuracy(test_pred_dB, Y_test_dB, .8))
print(Accuracy(test_pred_dB, Y_test_dB, 1))

# In[57]:

print("Large Test Errors")
ind, pred_vector, true_vector = checkErrors(test_pred_dB, Y_test_dB)
print(ind)
print(pred_vector)
print(true_vector)

print("Training RMSE")
print(tr_predict_hist)

# In[110]:

print("validation RMSE")
print(val_predict_hist)

# Set the stdout back to console instead of debug file.
if(isLoging == "on"):
    sys.stdout = old_stdout
    log_file.close()
