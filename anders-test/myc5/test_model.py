# 测试模型的识别能力
# 1.加载模型digit_model_all.pth
# 2.显示前两个图形，7和2的图片
# 3.识别这两个图形，并print出结果，完全正确。
import torch
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import os 
# 定义超参数 
image_size = 28  #图像的总尺寸28*28
num_classes = 10  #标签的种类数

#定义卷积神经网络：4和8为人为指定的两个卷积层的厚度（feature map的数量）
depth = [4, 8]
class ConvNet(nn.Module):
    def __init__(self):
        # 该函数在创建一个ConvNet对象的时候，即调用如下语句：net=ConvNet()，就会被调用
        # 首先调用父类相应的构造函数
        super(ConvNet, self).__init__()
        
        # 其次构造ConvNet需要用到的各个神经模块。
        '''注意，定义组件并没有真正搭建这些组件，只是把基本建筑砖块先找好'''
        #定义一个卷积层，输入通道为1，输出通道为4，窗口大小为5，padding为2
        self.conv1 = nn.Conv2d(1, 4, 5, padding = 2) 
        #定义一个Pooling层，一个窗口为2*2的pooling运算
        self.pool = nn.MaxPool2d(2, 2) 
        #第二层卷积，输入通道为depth[0]=4, 
        #输出通道为depth[1]=8，窗口为5，padding为2
        self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding = 2) 
        #一个线性连接层，输入尺寸为最后一层立方体的平铺，输出层512个节点
        # 最后一层立方体为8层，大小为7*7，为392。
        layer_input = image_size // 4 * image_size // 4 * depth[1]
        self.fc1 = nn.Linear(layer_input , 512) 
                                                            
        self.fc2 = nn.Linear(512, num_classes) #最后一层线性分类单元，输入为512，输出为要做分类的类别数

    def forward(self, x):
        #该函数完成神经网络真正的前向运算，我们会在这里把各个组件进行实际的拼装
        #x的尺寸：(batch_size, image_channels, image_width, image_height)
        
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        #第一层卷积，激活函数用ReLu，为了防止过拟合
        x = F.relu(x) 
        # print(x.shape) #[64, 4, 28, 28]
        #x的尺寸：(batch_size, num_filters, image_width, image_height)
        #第二层pooling，将图片变小
        x = self.pool(x) #[64, 4, 14, 14]
        # print(x.shape) 
        #x的尺寸：(batch_size, depth[0], image_width/2, image_height/2)
        #第三层又是卷积，窗口为5，输入输出通道分别为depth[0]=4, depth[1]=8
        x=self.conv2(x)
        # [64, 4, 14, 14]
        # print(x.shape) 
        x = F.relu(x) 
        #x的尺寸：(batch_size, depth[1], image_width/2, image_height/2)
        x = self.pool(x) #第四层pooling，将图片缩小到原大小的1/4
        #x的尺寸：(batch_size, depth[1], image_width/4, image_height/4)
        # print(x.shape) #[64, 8, 7, 7]
        # 将立体的特征图Tensor，压成一个一维的向量
        # view这个函数可以将一个tensor按指定的方式重新排布。
        # 下面这个命令就是要让x按照batch_size * (image_size//4)^2*depth[1]的方式来排布向量
        x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])
        # print(x.shape) #[64, 392]
        #x的尺寸：(batch_size, depth[1]*image_width/4*image_height/4)
        x = self.fc1(x)
        # print(x.shape) #[64, 512]
        x = F.relu(x) #第五层为全链接，ReLu激活函数
        # print(x.shape)
        #x的尺寸：(batch_size, 512)
        x = F.dropout(x, training=self.training) #以默认为0.5的概率对这一层进行dropout操作，为了防止过拟合
        # print(x.shape)#[64, 512]
        x = self.fc2(x) #全链接
        # print(x.shape) #[64, 10]
        #x的尺寸：(batch_size, num_classes)
        # print(x)

        #输出层为log_softmax，即概率对数值log(p(x))。采用log_softmax可以使得后面的交叉熵计算更快
        x = F.log_softmax(x, dim = 0) 
        # print(x[0])
        # print(x.shape)
        return x
    
    def retrieve_features(self, x):
        #该函数专门用于提取卷积神经网络的特征图的功能，返回feature_map1, feature_map2为前两层卷积层的特征图
        feature_map1 = F.relu(self.conv1(x)) #完成第一层卷积
        x = self.pool(feature_map1)  # 完成第一层pooling
        feature_map2 = F.relu(self.conv2(x)) #第二层卷积，两层特征图都存储到了feature_map1, feature_map2中
        return (feature_map1, feature_map2)
    

file_dir = os.path.split(os.path.abspath(__file__))[0]
data_path = os.path.join(file_dir, "digit_model_all.pth")


# 第二种文件，直接读
net = torch.load(data_path)
net.eval()
#随便从测试集中读入一张图片，并检验模型的分类结果，并绘制出来
# 加载测试数据集
test_dataset = dsets.MNIST(root='D:\\pytorch_data\\mnist\\data', 
                           train=False, 
                           transform=transforms.ToTensor())

# 必须两个图片一起预测才行，不知道为啥
idx = 0
muteimg = test_dataset[idx][0].numpy()
muteimg1 = test_dataset[idx+1][0].numpy()
plt.imshow(muteimg[0,...])
plt.show()
plt.imshow(muteimg1[0,...])
plt.show()
# plt.imshow(muteimg1[0,...])
print(test_dataset[idx][0].shape)
print('标签是：',test_dataset[idx][1])
p_pic = [muteimg,muteimg1]
p_pic = torch.tensor(p_pic)
print(p_pic.shape)
print(p_pic)
v = net(p_pic)
# v里面放了两个元素的数组，每个元素有10个数字（不是概率）
print(v.shape)
value = torch.max(v.data, 1)
# value[0]里面放的数字的最大值，value[1]里面放的是最大值的index
print(value[0])
# 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
pred = value[1] 
# 索引即预测结果
print(pred)

# 第二种获取结果的方式,获取最大值的索引，这种更简单
y2 = v.argmax(1) 
print(y2)
print("finished")