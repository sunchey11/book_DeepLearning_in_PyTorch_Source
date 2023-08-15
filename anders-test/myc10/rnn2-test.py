# 训练模型
# 导入程序所需要的程序包

#PyTorch用的包
import os
import torch
import torch.nn as nn
import torch.optim

from collections import Counter #搜集器，可以让统计词频更简单

#绘图、计算用的程序包
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

debug = False
def d_print(s):
    if(debug):
        print(s)

# 2. 定义SimpleRNN

# https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
# 实现一个简单的RNN模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1):
        # 定义
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 一个embedding层
        self.embedding = nn.Embedding(input_size, hidden_size)
        # PyTorch的RNN层，batch_first标志可以让输入的张量的第一个维度表示batch指标
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first = True)
        # 输出的全链接层
        self.fc = nn.Linear(hidden_size, output_size)
        # 最后的logsoftmax层
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):
        # 运算过程
        # 先进行embedding层的计算，它可以把一个数值先转化为one-hot向量，再把这个向量转化为一个hidden_size维的向量
        # input的尺寸为：batch_size, num_step, data_dim
        x = self.embedding(input)
        # 从输入到隐含层的计算
        # x的尺寸为：batch_size, num_step, hidden_size
        output, hidden = self.rnn(x, hidden)
        # 从输出output中取出最后一个时间步的数值，注意output输出包含了所有时间步的结果,
        # output输出尺寸为：batch_size, num_step, hidden_size
        output = output[:,-1,:]
        # output尺寸为：batch_size, hidden_size
        # 喂入最后一层全链接网络
        output = self.fc(output)
        # output尺寸为：batch_size, output_size
        # softmax函数
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        # 对隐含单元的初始化
        # 注意尺寸是： layer_size, batch_size, hidden_size
        return torch.zeros(self.num_layers, 1, self.hidden_size)


file_dir = os.path.split(os.path.abspath(__file__))[0]
data_path = os.path.join(file_dir, 'rnn.mdl')

rnn:SimpleRNN = torch.load(data_path)
# 生成一个最简化的RNN，输入size为4，可能值为0,1,2,3，输出size为3，可能值为0,1,2
# rnn = SimpleRNN(input_size = 4, hidden_size = 2, output_size = 3)

for n in range(20):
    
    inputs = [0] * n + [1] * n
    inputs.insert(0, 3)
    inputs.append(2)
    outstring = ''
    targets = ''
    diff = 0
    hiddens = []
    hidden = rnn.initHidden()
    
    for t in range(len(inputs) - 1):
        x = torch.tensor([inputs[t]], dtype = torch.long).unsqueeze(0)
        # x尺寸：batch_size = 1, time_steps = 1, data_dimension = 1
        y = torch.tensor([inputs[t + 1]], dtype = torch.long)
        # y尺寸：batch_size = 1, data_dimension = 1
        d_print(x.shape)
        output, hidden = rnn(x, hidden)
        d_print(output.shape)
        # output尺寸：batch_size, output_size = 3
        # hidden尺寸：layer_size =1, batch_size=1, hidden_size
        hiddens.append(hidden.data.numpy()[0][0])
        #mm = torch.multinomial(output.view(-1).exp())
        # max返回最大的概率和最大的索引
        max_val = torch.max(output, 1)
        d_print(max_val)
        mm = max_val[1][0]
        outstring += str(mm.data.numpy())
        targets += str(y.data.numpy()[0])
        d_print(mm.eq(y))
        diff += 1 - mm.eq(y).data.numpy()[0]
    # 打印出每一个生成的字符串和目标字符串
    print(outstring)
    print(targets)
    print('Diff:{}'.format(diff))