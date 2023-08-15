# 此代码演示程序里用到的一些基本的api，用来学习这些api的用法
# 导入程序所需要的程序包

#PyTorch用的包
import torch
import torch.nn as nn
import torch.optim

from collections import Counter #搜集器，可以让统计词频更简单

#绘图、计算用的程序包
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

# 切片的用法
# 简单的，取最后一行
t = torch.randn(2, 3)
print(t)
# -1代表取最后一行元素
t = t[-1,:]
print(t)
print(t.shape)

# 简单的，取最后一列
t = torch.randn(2, 3)
print(t)
# -1代表取最后一列元素
t = t[:,-1]
print(t)
print(t.shape)

# 中间有个-1是啥意思
t = torch.randn(2, 3, 4)
print(t)
print(t.shape)
# 中间那个维度，取最后一条
t = t[:,-1,:]
print(t)
print(t.shape)

# 下面的程序用到的情形
t = torch.randn(1, 1, 2)
print(t)
print(t.shape)
# 中间那个维度，取最后一条
t = t[:,-1,:]
print(t)
print(t.shape)

debug = True
def d_print(s):
    if(debug):
        print(s)

# 1.准备数据
train_set = []
valid_set = []

# 生成的样本数量
samples = 2000

# 训练样本中n的最大值
sz = 10
# 定义不同n的权重，我们按照10:6:4:3:1:1...来配置字符串生成中的n=1,2,3,4,5,...
probability = 1.0 * np.array([10, 6, 4, 3, 1, 1, 1, 1, 1, 1])
print(probability)
# 保证n的最大值为sz
probability = probability[ : sz]
print(probability)
# 归一化，将权重变成概率
probability = probability / sum(probability)
print(probability)
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
        d_print(input.shape)
        x = self.embedding(input)
        d_print(x.shape)
        d_print(x)
        # 从输入到隐含层的计算
        # x的尺寸为：batch_size, num_step, hidden_size
        output, hidden = self.rnn(x, hidden)
        d_print(output.shape)
        # 从输出output中取出最后一个时间步的数值，注意output输出包含了所有时间步的结果,
        # output输出尺寸为：batch_size, num_step, hidden_size
        output = output[:,-1,:]
        d_print(output.shape)
        # output尺寸为：batch_size, hidden_size
        # 喂入最后一层全链接网络
        output = self.fc(output)
        d_print(output.shape)
        # output尺寸为：batch_size, output_size
        # softmax函数
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        # 对隐含单元的初始化
        # 注意尺寸是： layer_size, batch_size, hidden_size
        return torch.zeros(self.num_layers, 1, self.hidden_size)

# 3. 使用SimpleRNN

# seq = train_set[0]
seq = [3, 0, 0, 1, 1, 2]

# 生成一个最简化的RNN，输入size为4，可能值为0,1,2,3，输出size为3，可能值为0,1,2
rnn = SimpleRNN(input_size = 4, hidden_size = 2, output_size = 3)
criterion = torch.nn.NLLLoss() #交叉熵损失函数
optimizer = torch.optim.Adam(rnn.parameters(), lr = 0.001) #Adam优化算法


loss = 0
hidden = rnn.initHidden()  # 初始化隐含神经元
d_print(seq)
print(rnn.embedding.weight)
# 对每一个序列的所有字符进行循环
for t in range(len(seq) - 1):
    # 当前字符作为输入，下一个字符作为标签
    e = seq[t]
    d_print(e)
    x = torch.LongTensor([e]).unsqueeze(0)
    d_print(x.shape)
    # x尺寸：batch_size = 1, time_steps = 1, data_dimension = 1
    y = torch.LongTensor([seq[t + 1]])
    d_print(y.shape)
    # y尺寸：batch_size = 1, data_dimension = 1
    output, hidden = rnn(x, hidden) # RNN输出
    d_print(y.shape)
    # output尺寸：batch_size, output_size = 26
    # hidden尺寸：layer_size =1, batch_size=1, hidden_size
    loss += criterion(output, y) # 计算损失函数
# embedding不会变
print(rnn.embedding.weight)
loss = 1.0 * loss / len(seq) # 计算每字符的损失数值
optimizer.zero_grad() # 清空梯度
loss.backward() # 反向传播，设置retain_variables
optimizer.step() # 一步梯度下降