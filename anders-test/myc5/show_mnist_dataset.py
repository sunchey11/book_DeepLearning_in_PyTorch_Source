# 此程序显示mnist数据集中的图片，全部是手写数字的图片，有60000个
# 测试数据有10000条
# 导入所需要的包，请保证torchvision已经在你的环境中安装好.
# 在Windows需要单独安装torchvision包，在命令行运行pip install torchvision即可
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

# MNIST数据是属于torchvision包自带的数据，所以可以直接调用。
# 在调用自己的数据的时候，我们可以用torchvision.datasets.ImageFolder或者torch.utils.data.TensorDataset来加载
train_dataset = dsets.MNIST(root='D:\\pytorch_data\\mnist\\data',  #文件存放路径
                            train=True,   #提取训练集
                            transform=transforms.ToTensor(),  #将图像转化为Tensor，在加载数据的时候，就可以对图像做预处理
                            download=True) #当找不到文件的时候，自动下载

# 加载测试数据集
test_dataset = dsets.MNIST(root='D:\\pytorch_data\\mnist\\data', 
                           train=False, 
                           download=True,
                           transform=transforms.ToTensor())

test_len = len(test_dataset)
# 测试数据有10000条
print(test_len)
labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
total_len = len(train_dataset)
print(total_len)
for i in range(1, cols * rows + 1):
    # sample_idx = torch.randint(total_len, size=(1,)).item()
    sample_idx = 0
    # img, label = train_dataset[sample_idx]
    img, label = test_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    print(img.shape)
    # squeeze去掉维度为1的维度,本来shape=[1,32,32]
    # 调用后变为[32,32]
    img = img.squeeze()
    print(img.shape)
    plt.imshow(img, cmap="gray")
plt.show()

