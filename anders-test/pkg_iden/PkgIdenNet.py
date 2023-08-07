# 识别药品包装的类

from img_utils import d_print
import torch
import torch.nn as nn
import torch.nn.functional as F

class PkgIdenNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 11, padding=5)

        pool_size = 5
        self.pool1 = nn.MaxPool2d(pool_size, pool_size)
        
        self.conv2 = nn.Conv2d(6, 16, 11, padding=5)
        self.pool2 = nn.MaxPool2d(pool_size, pool_size)
        # 这里调整大小
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 16*24*32=12280
        self.fc1 = nn.Linear(9408, 1228)
        self.fc1_x = nn.Linear(1228, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        d_print(x.shape) # torch.Size([1, 3, 600, 800])
        # conv后的图片大小为：源宽-窗口宽+1
        # 28 - 5 +1 = 24
        # 如果想大小不变，应设置padding
        # 层数为输出层数
        x = self.conv1(x)
        d_print(x.shape) # torch.Size([1, 6, 600, 800])
        # relu只是将负值变为0，所以shape不变
        x = F.relu(x)
        d_print(x.shape) # torch.Size([1, 6, 600, 800])
        # 大小变为size/pool_size,层数不变
        x = self.pool1(x)
        d_print(x.shape) # torch.Size([1, 6, 120, 160])

        x = self.conv2(x)
        d_print(x.shape) # torch.Size([1, 16, 120, 160])
        x = F.relu(x)
        d_print(x.shape) # torch.Size([1, 16, 120, 160])
        # 大小变为size/pool_size,层数不变
        x = self.pool2(x)
        d_print(x.shape) # torch.Size([1, 16, 24, 32])

        # 在第一个维度打平了，16*24*32=12280
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        d_print(x.shape) # torch.Size([1, 12280])

        x = self.fc1(x)
        d_print(x.shape) # torch.Size([4, 1228])
        x = F.relu(x)
        d_print(x.shape) # torch.Size([4, 120])

        x = self.fc1_x(x)
        d_print(x.shape) # torch.Size([4, 120])
        x = F.relu(x)
        d_print(x.shape) # torch.Size([4, 120])

        x = self.fc2(x)
        d_print(x.shape) # torch.Size([4, 84])
        x = F.relu(x)
        d_print(x.shape) # torch.Size([4, 84])
        x = self.fc3(x) # torch.Size([4, 10])
        d_print(x.shape)
        return x
