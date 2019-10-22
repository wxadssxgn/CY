# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

# read data
# shape of DATA (:, 3)
# the first column is TIME
# the second is COLUMN_C, the third is COLUMN_D
csvFile = open('BP_prediction_raw_data.csv', 'r')
reader = csv.reader(csvFile)
data = []
for item in reader:
    data.append(item)
csvFile.close()
data = np.array(data)
data_time = data[:, 0]
data_time = data_time[:, np.newaxis]
data_C = data[:, 2]
data_C = data_C[:, np.newaxis]
data_D = data[:, 3]
data_D = data_D[:, np.newaxis]
data = np.concatenate((data_time, data_C, data_D), axis=1)
# Some data is missing, so I chose the complete example
data = data[2: 9538, :]

# Take 60 points as a training set and 5 points as a test set
i = 3
data_temp = np.expand_dims(data[i, :], axis=0)
for n in range(1, 65):
    data_temp = np.concatenate((data_temp, np.expand_dims(data[i + n*60, :], axis=0)), axis=0)
data = data_temp[:, 1:]
# shape: (65, 2) type: float
data = data.astype(float)

# Hyper Parameters
BATCH = 1
TIME_STEP = 1
INPUT_SIZE = 60
OUTPUT_SIZE = 5
NUM_UNITS = 10
LR = 0.1

# tensorflow placeholders
# tf_x: input
# tf_y: output
tf_x = tf.placeholder(tf.float32, [BATCH, TIME_STEP, INPUT_SIZE])
tf_y = tf.placeholder(tf.float32, [BATCH, TIME_STEP, OUTPUT_SIZE])

lstm_model = tf.nn.rnn_cell.LSTMCell(num_units=NUM_UNITS)

# The first state is initialized to ZERO 
init_s = lstm_model.zero_state(batch_size=1, dtype=tf.float32)
outputs, final_s = tf.nn.dynamic_rnn(
    lstm_model,
    tf_x,
    initial_state=init_s, # the initial hidden state
    time_major=False, # BATCH is the first dimension
)

outs2D = tf.reshape(outputs, [-1, NUM_UNITS])
net_outs2D = tf.layers.dense(outs2D, INPUT_SIZE)
outs = tf.reshape(net_outs2D, [-1, TIME_STEP, INPUT_SIZE])
loss_func = tf.losses.mean_squared_error(labels=tf_y, predictions=outs)
train_option = tf.train.AdamOptimizer(LR).minimize(loss_func)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1000):
    x = data[: 60, 0]
    x = x[np.newaxis, np.newaxis, :].astype(np.float32)
    y = data[60:, 0]
    y = y[np.newaxis, np.newaxis, :].astype(np.float32)
    if 'final_s_' not in globals():
        feed_dict = {tf_x: x, tf_y: y}
    else:
        feed_dict = {tf_x: x, tf_y: y, init_s: final_s_}
    _, pred_, final_s_ = sess.run([train_option, outs, final_s], feed_dict)
    print(step)
