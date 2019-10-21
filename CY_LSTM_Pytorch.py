# -*- coding: utf-8 -*-

# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch
from torch import nn, optim
from torch.autograd import Variable

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

INPUT_SIZE = 60
BATCH = 1
OUTPUT_SIZE = 5
LR = 0.1

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=10,
            num_layers=1,
            batch_first=True,
            ) 
        self.out = nn.Linear(10, OUTPUT_SIZE)

    def forward(self, x, h_state):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

rnn = LSTM()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()
h_state = None

loss_tmp = []
for step in range(1000):
    x = data[: 60, 0]
    x = x[np.newaxis, np.newaxis, :]
    x = torch.from_numpy(x).float()
    y = data[60:, 0]
    y = y[np.newaxis, np.newaxis, :]
    y = torch.from_numpy(y).float()
    prediction = rnn(x, h_state) 
    loss = loss_func(prediction, y)
    loss_tmp.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(loss_tmp)
plt.show()
