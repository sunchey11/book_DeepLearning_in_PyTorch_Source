# 识别汉字的类

import torch
import torch.nn as nn
import torch.nn.functional as F

img_width = 128
img_height = 128
debug = False
def d_print(s):
    if(debug):
        print(s)
class FontIdenNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 11, padding=5)

        pool_size = 2
        self.pool1 = nn.MaxPool2d(pool_size, pool_size)
        
        self.conv2 = nn.Conv2d(6, 16, 11, padding=5)
        self.pool2 = nn.MaxPool2d(pool_size, pool_size)
        # 这里调整大小
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 16*24*32=12280
        w = img_width/pool_size/pool_size #150
        h = img_height/pool_size/pool_size #150
        # 16384
        pix = 16*w*h
        # print(type(pix))
        self.fc1 = nn.Linear(int(pix), 12000)
        self.fc2 = nn.Linear(12000, 10000)

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

        x = self.fc2(x)
        d_print(x.shape) # torch.Size([4, 84])
        x = F.relu(x)
        d_print(x.shape) # torch.Size([4, 84])
        return x
