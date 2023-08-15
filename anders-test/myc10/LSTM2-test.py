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

# 一个手动实现的LSTM模型，除了初始化隐含但愿部分，所有代码基本与SimpleRNN相同

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1):
        super(SimpleLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 一个embedding层
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 隐含层内部的相互链接
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):
        
        # 先进行embedding层的计算，它可以把一个
        # x的尺寸：batch_size, len_seq, input_size
        x = self.embedding(input)
        # x的尺寸：batch_size, len_seq, hidden_size
        # 从输入到隐含层的计算
        output, hidden = self.lstm(x, hidden)
        # output的尺寸：batch_size, len_seq, hidden_size
        # hidden: (layer_size, batch_size, hidden_size),(layer_size, batch_size,hidden_size)
        output = output[:,-1,:]
        # output的尺寸：batch_size, hidden_size
        output = self.fc(output)
        # output的尺寸：batch_size, output_size
        # softmax函数
        output = self.softmax(output)
        return output, hidden
 
    def initHidden(self):
        # 对隐含单元的初始化
        # 注意尺寸是： layer_size, batch_size, hidden_size
        # 对隐单元的初始化
        # 对引单元输出的初始化，全0.
        # 注意hidden和cell的维度都是layers,batch_size,hidden_size
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        # 对隐单元内部的状态cell的初始化，全0
        cell = torch.zeros(self.num_layers, 1, self.hidden_size)
        return (hidden, cell)
    

file_dir = os.path.split(os.path.abspath(__file__))[0]
data_path = os.path.join(file_dir, 'lstm.mdl')

lstm:SimpleLSTM = torch.load(data_path)
# lstm = SimpleLSTM(input_size = 4, hidden_size = 1, output_size = 3, num_layers = 1)

# 让n取0到50，看SimpleLSTM是否能够成功预测下一个字符
for n in range(50):
    
    inputs = [0] * n + [1] * n
    inputs.insert(0, 3)
    inputs.append(2)
    outstring = ''
    targets = ''
    diff = 0
    hiddens = []
    hidden = lstm.initHidden()
    for t in range(len(inputs) - 1):
        x = torch.tensor([inputs[t]], dtype = torch.long).unsqueeze(0)
        y = torch.tensor([inputs[t + 1]], dtype = torch.long)
        output, hidden = lstm(x, hidden)
        
        mm = torch.max(output, 1)[1][0]
        outstring += str(mm.data.numpy())
        targets += str(y.data.numpy()[0])

        diff += 1 - mm.eq(y).data.numpy()[0]
    print(n)
    print(outstring)
    print(targets)
    print('Diff:{}'.format(diff))