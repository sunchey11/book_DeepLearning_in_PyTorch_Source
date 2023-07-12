# 原文链接：https://blog.csdn.net/ITnanbei/article/details/118639874
# 目前还跑不通
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,TensorDataset



# 定义超参数
BATCH_SIZE = 16  # 每批次处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否用GPU或者CPU
EPOCHS = 10  # 训练的数据集的轮次


"""构建pipeline,对图像处理"""
pipeline = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换成tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 正则化 第一个参数是均值，第二个参数是标准差 这些系数都是数据集提供方计算好的数据
])
 
"""下载数据集 """
 
train_set = datasets.MNIST('D:\\pytorch_data\\mnist\\data', train=True, download=True, transform=pipeline)
 
test_set = datasets.MNIST('D:\\pytorch_data\\mnist\\data', train=False, download=True, transform=pipeline)
 
 
"""加载数据"""
train_loader = DataLoader(train_set, shuffle=True)
 
test_loader = DataLoader(test_set, shuffle=True)

""" 制作训练集数据和标签"""
number_x_train = np.zeros((30000, 1, 100, 100), dtype="float64")
coordinate_y_train = np.zeros((30000, 1, 4), dtype="float64")
number_y_train = np.zeros((30000, 1, 20), dtype="float64")
 
j = 0
for i, (data, label) in enumerate(train_loader):
    data = data.squeeze(axis=0)
    data = data.squeeze(axis=0)
    data = np.array(data)
    data[0, :], data[27, :], data[:, 0], data[:, 27] = 1, 1, 1, 1
    if i % 2 == 0:
        """背景板"""
        blank = np.zeros((100, 100))
        """临时存放标签和坐标 """
        multi_label = np.zeros((1, 20))
        multi_coordinate = np.array([[0, 0, 0, 0]])
        """只能生成在左半边的坐标"""
        dx = np.random.randint(0, 70)
        dy = np.random.randint(0, 20)
        multi_coordinate[0][0] = dx
        multi_coordinate[0][1] = dy
        """采用类似独热编码的方式存储 对应序号的数字为1 其余为0"""
        multi_label[0,int(label.numpy().item())] = 1
    else:
        """只能生成在右半边的坐标"""
        dx = np.random.randint(0, 70)
        dy = np.random.randint(50, 70)
        multi_coordinate[0][2] = dx
        multi_coordinate[0][3] = dy
        multi_label[0,int(label.numpy().item())+10] = 1
    """将数字放在背景板上"""
    blank[dx:dx + 28, dy:dy + 28] = blank[dx:dx + 28, dy:dy + 28] + data
    if i % 2 == 1:
        number_x_train[j, :, :, :] = blank
        number_y_train[j, :] = multi_label
        coordinate_y_train[j, :] = multi_coordinate
        # plt.imshow(number_x_train[j].squeeze(0),cmap='binary')
        # plt.show()
        j += 1
 
number_X_train = torch.from_numpy(number_x_train).float()
coordinate_Y_train = torch.from_numpy(coordinate_y_train).float()
number_Y_train = torch.from_numpy(number_y_train).float()
 
 
train_data = TensorDataset(number_X_train, number_Y_train)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

"""识别数字模型"""
class Multi_Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3,padding=1,stride=2)  # 1: 灰度图片的通道 10： 输出通道 3：kernel大小3*3
        self.conv2 = nn.Conv2d(10, 20, 3)  # 10: 输入通道 20： 输出通道 3：kernel大小3*3
        self.conv3 = nn.Conv2d(20, 40, 3)  # 10: 输入通道 20： 输出通道 3：kernel大小3*3
        self.fc1 = nn.Linear(3240, 500)  # 3240: 输入通道 500： 输出通道
        self.fc2 = nn.Linear(500, 20)  # 500:输入通道 20:输出通道
 
    def forward(self, x):  #  x 为  batch size *1 *100 *100   
        input_size = x.size(0)  # batch size
        x = self.conv1(x)  # 卷积   输入：batch *1 *100 *100 ,输出 ： batch*10*50*50
        x = F.relu(x)  # 激活函数 输出 ： batch*10*46*46
        x = F.max_pool2d(x, 2, 2)  # 池化层 f=2 s=2 输入 ：batch*10*50*50 输出 ：batch*10*25*25
 
        x = self.conv2(x)  # 卷积   输入：batch *10 *25 *25 ,输出 ： batch*20*23*23
        x = F.relu(x)
        x =F.max_pool2d(x,2,2)  # 卷积   输入：batch *20 *23 *23 ,输出 ： batch*20*11*11
 
        x =self.conv3(x) # 卷积   输入：batch *20 *11 *11 ,输出 ： batch*40*9*9
        x = F.relu(x)
        x = x.view(input_size, -1) 
        x = self.fc1(x)  # 输入 ：batch * 3240 输出：batch * 500
        x = F.relu(x)  
        x = self.fc2(x)  # 输入 batch *500   输出 ：batch *20
        return x
model = Multi_Digit().to(DEVICE)  # 创建模型部署到设备上 （CPU或者GPU）
 
optimizer = optim.Adam(model.parameters())  # 定义一个优化器 ，将模型的参数作为输入，优化参数
 
def train_model(model, device, train_loader, optimizer, epoch):  # train_loader为训练的数据
    # 数字识别模型训练
    model.train()  #启用batch normalization(批标准化)和drop out,model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
    for batch_index, (data, label) in enumerate(train_loader):  # data 为图片 label 为标签
        # 部署到DEVICE上去
        data, label = data.to(device), label.to(device)
        label = label.squeeze(axis=1)
        # 梯度初始化为 0
        optimizer.zero_grad() # 即将 optimizer中的weight置零
        # 训练后的结果
        output = model(data)
        # 计算损失
        MSE = nn.MSELoss()
        loss = MSE(output, label)
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        if batch_index % 10000 == 0:
            print("Train Epoch :{} \t Loss :{:.6f}".format(epoch, loss.item()))
 
 
# 定义数字识别测试方法
def test_model(model, device, test_loader):
    # 模型验证
    model.eval() 
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    n1,n2=0,0
    with torch.no_grad():  # 不会计算梯度，也不会反向传播
        for data, label in test_loader:
            # 部署到设备上
            data, label = data.to(device), label.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            label =label.squeeze(axis=1)
            MSE = nn.MSELoss()
            test_loss += MSE(output, label).item()
            # 累计正确率
            correct += np.round(output).eq(label.view_as(output)).sum().item()/20  
        test_loss /= len(test_loader.dataset)
        print("Test --- Average loss : {:.4f}  Accuracy:{:.4f}".format(test_loss,100.0 * correct / len(test_loader.dataset)))
 
#
for epoch in range(1, EPOCHS + 1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)


"""定位模型"""
class Multi_Number_Coordinate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1, stride=2)  # 1: 灰度图片的通道 10： 输出通道 3：kernel大小3*3
        self.conv2 = nn.Conv2d(10, 20, 3)  # 10: 输入通道 20： 输出通道 3：kernel大小3*3
        self.conv3 = nn.Conv2d(20, 40, 3)  # 10: 输入通道 20： 输出通道 3：kernel大小3*3
        self.fc1 = nn.Linear(3240, 500)  # 8820: 输入通道 500： 输出通道
        self.fc2 = nn.Linear(500, 4)  # 500:输入通道 4:输出通道
 
    def forward(self, x):  # 前向传播  x 为  batch size *1 *100 *100  1是通道（灰色）
        input_size = x.size(0)  # batch size
        x = self.conv1(x)  # 卷积   输入：batch *1 *100 *100 ,输出 ： batch*10*50*50
        x = F.relu(x)  # 激活函数 输出 ： batch*10*46*46
        x = F.max_pool2d(x, 2, 2)  # 池化层 f=2 s=2 输入 ：batch*10*50*50 输出 ：batch*10*25*25
 
        x = self.conv2(x)  # 卷积   输入：batch *10 *25 *25 ,输出 ： batch*20*23*23
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)  # 卷积   输入：batch *20 *23 *23 ,输出 ： batch*20*11*11
 
        x = self.conv3(x)  # 卷积   输入：batch *20 *11 *11 ,输出 ： batch*40*9*9
        x = F.relu(x)
        x = x.view(input_size, -1)  # 转化成一维 ， -1 激动计算维度
 
        x = self.fc1(x)  # 输入 ：batch * 8820 输出：batch * 500
        x = F.relu(x)  # 保持shape不变
        x = self.fc2(x)  # 输入 batch *500   输出 ：batch *10
        return x