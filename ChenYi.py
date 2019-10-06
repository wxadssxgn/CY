#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv

#------------------------------------------------------------------
#读取csv数据，保存到data里
#data是numpy.array格式的数组
csvFile = open('samples.csv', 'r')
reader = csv.reader(csvFile)
data = []
for item in reader:
    data.append(item)
csvFile.close()
data = np.array(data)
data = data[2: ]

#------------------------------------------------------------------
#将头和尾两个不完整的周期去掉
#index0是头部不完整周期最后一个点的位置
#index2是尾部不完整周期第一个点的位置
elapse = np.array(list(map(float, data[:, 0])))
abp = np.array(list(map(float, data[:, 4])))
pleth = np.array(list(map(float, data[:, 6])))

tmp0 = pleth[0: 300]
min0 = np.min(tmp0)
index0 = np.argmin(tmp0)

elapse = elapse[index0: ]
abp = abp[index0: ]
pleth = pleth[index0: ]

tmp1 = pleth[29700: ]
min1 = np.min(tmp1)
index1 = np.argmin(tmp1)
index2 = len(tmp1) - index1

#elapse，abp，pleth为最终原始数据，且均是numpy.array格式的数组
elapse = elapse[0: -index2 + 1]
abp = abp[0: -index2 + 1]
pleth = pleth[0: -index2 + 1]

#------------------------------------------------------------------
#定义两个函数分别搜寻pleth极大值和极小值的位置index和对应的数值ans
#搜寻极大值的函数窗长为40，搜寻极小值的函数窗长为24
#考虑到你的原始数据没经过滤波等操作，故并不是完全意义上的平滑函数，故我在书写极值搜索函数时，直接试出最合适的窗长 -。-
#一般情况下都是将函数作平滑处理，然后自适应搜索 -。-
#index和ans都是numpy.array格式的数组
#这样搜寻出的极小值数目应该比极大值数目少一，并且不包含一头一尾两个极小值，后面会添加上
def localmin(x):
    ans = []
    index = []
    for i in range(20, len(x) - 20):
        if x[i] < x[i - 1] and x[i] < x[i - 2] and x[i] < x[i - 3] and x[i] < x[i - 4] and x[i] < x[i - 5]:
            if x[i] < x[i - 6] and x[i] < x[i - 7] and x[i] < x[i - 8] and x[i] < x[i - 9] and x[i] < x[i - 10]:
                if x[i] < x[i - 11] and x[i] < x[i - 12] and x[i] < x[i - 13] and x[i] < x[i - 14] and x[i] < x[i - 15]:
                    if x[i] < x[i - 16] and x[i] < x[i - 17] and x[i] < x[i - 18] and x[i] < x[i - 19] and x[i] < x[i - 20]:
                        if x[i] <= x[i + 1] and x[i] <= x[i + 2] and x[i] <= x[i + 3] and \
                        x[i] <= x[i + 4] and x[i] <= x[i + 5]:
                            if x[i] <= x[i + 6] and x[i] <= x[i + 7] and x[i] <= x[i + 8] and x[i] <= x[i + 9] and x[i] <= x[i + 10]:
                                if x[i] <= x[i + 11] and x[i] <= x[i + 12] and x[i] <= x[i + 13] \
                                and x[i] <= x[i + 14] and x[i] <= x[i + 15]:
                                    if x[i] <= x[i + 16] and x[i] <= x[i + 17] and x[i] <= x[i + 18] \
                                    and x[i] <= x[i + 19] and x[i] <= x[i + 20]:
                                        ans.append(x[i])
                                        index.append(i)
    return ans, index

def localmax(x):
    ans = []
    index = []
    for i in range(12, len(x) - 12):
        if x[i] > x[i - 1] and x[i] > x[i - 2] and x[i] > x[i - 3] and x[i] > x[i - 4] and x[i] > x[i - 5]:
            if x[i] > x[i - 6] and x[i] > x[i - 7] and x[i] > x[i - 8] \
            and x[i] > x[i - 9] and x[i] > x[i - 10]:
                if x[i] > x[i - 11] and x[i] > x[i - 12]:
                    if x[i] >= x[i + 1] and x[i] >= x[i + 2] and x[i] >= x[i + 3] and x[i] >= x[i + 4] and x[i] >= x[i + 5]:
                        if x[i] >= x[i + 6] and x[i] >= x[i + 7] and x[i] >= x[i + 8] \
                        and x[i] >= x[i + 9] and x[i] >= x[i + 10]:
                            if x[i] >= x[i + 11] and x[i] >= x[i + 12]:
                                ans.append(x[i])
                                index.append(i)
    return ans, index

localmin_value, localmin_index = localmin(pleth)
localmax_value, localmax_index = localmax(pleth)

#把第一个极小值和最后一个极小值添加进去
localmin_index = np.append(np.array(0), localmin_index)
localmin_index = np.append(localmin_index, np.array(len(pleth) - 1))
localmin_value = np.append(pleth[0], localmin_value)
localmin_value = np.append(localmin_value, pleth[-1])

#------------------------------------------------------------------
tmpindex = np.hstack((localmin_index, localmax_index))
tmpindex = sorted(tmpindex)

tmparray = np.array([])
for i in range(len(tmpindex)):
    tmparray = np.append(tmparray, pleth[tmpindex[i]])

halfAB_value = np.array([])
halfAB_elapse_value = np.array([])
features = np.zeros(shape=[int((len(tmpindex) - 1) / 4), 1, 14])

for i in range(int((len(tmpindex) - 1) / 4)):
    #h1,h2,h3,h4,t1,t2,t3,t4,t均是你需求书上定义的
    h1 = pleth[[tmpindex[i * 4 + 1]]] - pleth[[tmpindex[i * 4]]]
    h2 = pleth[[tmpindex[i * 4 + 2]]] - pleth[[tmpindex[i * 4]]]
    h3 = pleth[[tmpindex[i * 4 + 3]]] - pleth[[tmpindex[i * 4]]]
    h4 = pleth[[tmpindex[i * 4 + 3]]] - pleth[[tmpindex[i * 4 + 2]]]
    t1 = elapse[[tmpindex[i * 4 + 1]]] - elapse[[tmpindex[i * 4]]]
    t2 = elapse[[tmpindex[i * 4 + 2]]] - elapse[[tmpindex[i * 4]]]
    t3 = elapse[[tmpindex[i * 4 + 3]]] - elapse[[tmpindex[i * 4]]]
    t = elapse[[tmpindex[i * 4 + 4]]] - elapse[[tmpindex[i * 4]]]
    #h为0.5倍的h1
    h = h1 / 2 + pleth[[tmpindex[i * 4]]]
    plethAB = pleth[tmpindex[i * 4]: tmpindex[i * 4 + 1]]
    elapseAB = elapse[tmpindex[i * 4]: tmpindex[i * 4 + 1]]
    error = np.array([])
    for j in range(len(plethAB)):
        error = np.append(error, np.power(plethAB[j] - h, 2))
    errorindex = np.argmin(error)
    halfAB_value = np.append(halfAB_value, plethAB[errorindex])
    halfAB_elapse_value = np.append(halfAB_elapse_value, elapseAB[errorindex])
    t4 = elapseAB[errorindex] - elapse[[tmpindex[i * 4]]]
    abp_tmp = np.concatenate((abp[[tmpindex[i * 4]]],  abp[[tmpindex[i * 4 + 1]]], abp[[tmpindex[i * 4 + 2]]], \
                                                                                     abp[[tmpindex[i * 4 + 3]]], abp[[tmpindex[i * 4 + 4]]]), axis=0)
    features[i] = np.concatenate((h1, h2, h3, h4, t1, t2, t3, t4, t, abp_tmp), axis=0)
    
#------------------------------------------------------------------
#在交互模式下输入featuresplot()可以看提取出来的极值图
localmin_elapse_value = np.array([])
localmax_elapse_value = np.array([])

for p in range(len(localmin_index)):
    localmin_elapse_value = np.append(localmin_elapse_value, elapse[localmin_index[p]])

for q in range(len(localmax_index)):
    localmax_elapse_value = np.append(localmax_elapse_value, elapse[localmax_index[q]])

def featuresplot():
    plt.plot(elapse, pleth)
    plt.plot(localmin_elapse_value, localmin_value, 'g^', localmax_elapse_value, localmax_value, 'r^', \
             halfAB_elapse_value, halfAB_value, 'b^')
    plt.show()

featuresplot()
# #------------------------------------------------------------------
# #下面开始写神经网络部分
# #如果你只需要上面的features，请把下面代码全部屏蔽了
# #------------------------------------------------------------------
# #该部分定义神经网络结构
# import torch
# from torch import nn, optim
# from torch.autograd import Variable
# import datetime as dt

# #定义一个多层感知网络
# #我定义的是两个隐层，可以随便修改
# class MLP(nn.Module):
#     def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
#         super(MLP, self).__init__()
#         #in_dim是输入数据的维度
#         #out_dim是输出数据的维度
#         #n_hidden是隐层个数
#         #torch.nn.Dropout(0.0)表示不使用dropout操作
#         #nn.ReLU表示使用relu激活函数
#         self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), torch.nn.Dropout(0.0), nn.ReLU(inplace=True))
#         self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), torch.nn.Dropout(0.0), nn.ReLU(inplace=True))
#         self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

#     def forward(self, temp):
#         #前向传播
#         temp = self.layer1(temp)
#         temp = self.layer2(temp)
#         temp = self.layer3(temp)
#         return temp

# #------------------------------------------------------------------
# #该部分为网络训练部分
# #数据太少，故不考虑批量训练
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor#定义数据类型
# model = MLP(9, 200, 100, 5)#定义一个9输入，5输出，2隐层，每个隐层100个神经元的多层感知神经网络
# train_data = features[0: 80]
# test_data = features[80: ]
# criterion = nn.MSELoss(reduction='sum')#定义损失函数
# optimizer = optim.Adam(model.parameters())
# for epoch in range(100):
#     time = dt.datetime.now().isoformat()
#     for i in range(len(train_data)):
#         model_in = torch.from_numpy(train_data[i, 0, 0: 9])
#         model_label = torch.from_numpy(train_data[i, 0, 9: ])
#         x = Variable(model_in.type(dtype), requires_grad=False)#定义网络输入变量
#         y = Variable(model_label.type(dtype), requires_grad=False)#定义网络输出变量
#         model_out = model(x)
#         loss = criterion(model_out, y)#计算误差
#         #下面是反向传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     #每轮epoch后打印损失值
#     if epoch % 1 == 0:
#         print(time, epoch, loss.data)
