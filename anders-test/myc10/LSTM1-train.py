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



# 1.准备数据,字符串的生成
train_set = []
valid_set = []

# 生成的样本数量
samples = 2000

# 训练样本中n的最大值
sz = 10
# 定义不同n的权重，我们按照10:6:4:3:1:1...来配置字符串生成中的n=1,2,3,4,5,...
probability = 1.0 * np.array([10, 6, 4, 3, 1, 1, 1, 1, 1, 1])
# 保证n的最大值为sz
probability = probability[ : sz]
# 归一化，将权重变成概率
probability = probability / sum(probability)

# 开始生成samples这么多个样本
for m in range(samples):
    # 对于每一个生成的字符串，随机选择一个n，n被选择的权重被记录在probability中
    n = np.random.choice(range(1, sz + 1), p = probability)
    # 生成这个字符串，用list的形式完成记录
    inputs = [0] * n + [1] * n
    # 在最前面插入3表示起始字符，2插入尾端表示终止字符
    inputs.insert(0, 3)
    inputs.append(2)
    train_set.append(inputs) #将生成的字符串加入到train_set训练集中
    
# 再生成samples/10的校验样本
for m in range(samples // 10):
    n = np.random.choice(range(1, sz + 1), p = probability)
    inputs = [0] * n + [1] * n
    inputs.insert(0, 3)
    inputs.append(2)
    valid_set.append(inputs)

# 再生成若干n超大的校验样本
for m in range(2):
    n = sz + m
    inputs = [0] * n + [1] * n
    inputs.insert(0, 3)
    inputs.append(2)
    valid_set.append(inputs)
np.random.shuffle(valid_set)


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
    
lstm = SimpleLSTM(input_size = 4, hidden_size = 1, output_size = 3, num_layers = 1)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr = 0.001)

num_epoch = 50
results = []

# 开始训练循环
for epoch in range(num_epoch):
    train_loss = 0
    np.random.shuffle(train_set)
    # 开始所有训练数据的循环
    for i, seq in enumerate(train_set):
        loss = 0
        hidden = lstm.initHidden()
        # 开始每一个字符的循环
        for t in range(len(seq) - 1):
            x = torch.tensor([seq[t]], dtype = torch.long).unsqueeze(0)
            # x的尺寸：batch_size, len_seq, hidden_size
            y = torch.tensor([seq[t + 1]], dtype = torch.long)
            # y的尺寸：batch_size, data_dimension
            output, hidden = lstm(x, hidden)
            # output的尺寸：batch_size, data_dimension
            # hidden: (layer_size, batch_size, hidden_size),(layer_size, batch_size,hidden_size)
            loss += criterion(output, y)
        loss = 1.0 * loss / len(seq)
        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()
        train_loss += loss
        if i > 0 and i % 500 == 0:
            print('第{}轮, 第{}个，训练Loss:{:.2f}'.format(epoch,
                                                    i,
                                                    train_loss.data.numpy() / i
                                                   ))
            
            
    # 在校验集上跑结果
    valid_loss = 0
    errors = 0
    show_out = ''
    for i, seq in enumerate(valid_set):
        loss = 0
        outstring = ''
        targets = ''
        diff = 0
        hidden = lstm.initHidden()
        for t in range(len(seq) - 1):
            x = torch.tensor([seq[t]], dtype = torch.long).unsqueeze(0)
            # x的尺寸：batch_size, len_seq, hidden_size
            y = torch.tensor([seq[t + 1]], dtype = torch.long)
            # y的尺寸：batch_size, data_dimension
            output, hidden = lstm(x, hidden)
            # output的尺寸：batch_size, data_dimension
            # hidden: (layer_size, batch_size, hidden_size),(layer_size, batch_size,hidden_size)
            mm = torch.max(output, 1)[1][0]
            outstring += str(mm.data.numpy())
            targets += str(y.data.numpy()[0])
            loss += criterion(output, y)
            
            diff += 1 - mm.eq(y).data.numpy()[0]
        loss = 1.0 * loss / len(seq)
        valid_loss += loss
        errors += diff
        if np.random.rand() < 0.1:
            show_out = outstring + '\n' + targets
    print(output[0][2].data.numpy())
    print('第{}轮, 训练Loss:{:.2f}, 校验Loss:{:.2f}, 错误率:{:.2f}'.format(epoch, 
                                                               train_loss.data.numpy() / len(train_set),
                                                               valid_loss.data.numpy() / len(valid_set),
                                                               1.0 * errors / len(valid_set)
                                                              ))
    print(show_out)
    results.append([train_loss.data.numpy() / len(train_set), 
                    valid_loss.data.numpy() / len(train_set),
                   1.0 * errors / len(valid_set)
                   ])



file_dir = os.path.split(os.path.abspath(__file__))[0]
data_path = os.path.join(file_dir, 'lstm.mdl')

torch.save(lstm, data_path)