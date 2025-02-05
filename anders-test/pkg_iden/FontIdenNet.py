# 识别汉字的类

import torch
import torch.nn as nn
import torch.nn.functional as F

img_width = 64
img_height = 64
debug = False
def d_print(s):
    if(debug):
        print(s)
class FontIdenNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3,1, padding=1)

        self.bn1 = nn.BatchNorm2d(64)

        pool_size = 2
        self.pool1 = nn.MaxPool2d(pool_size, pool_size, padding=1)
        
        self.conv2 = nn.Conv2d(64, 128, 3,1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(pool_size, pool_size, padding=1)


        self.conv3 = nn.Conv2d(128, 256, 3,1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(pool_size, pool_size, padding=1)


        self.conv4 = nn.Conv2d(256, 512, 3,1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(pool_size, pool_size, padding=1)

        w = img_width/pool_size/pool_size #150
        h = img_height/pool_size/pool_size #150
        # 12800
        # pix = 16*w*h
        # print(type(pix))
        self.fc1 = nn.Linear(12800, 10000)

    def forward(self, x):
        d_print(x.shape) # torch.Size([1, 3, 600, 800])
        # conv后的图片大小为：源宽-窗口宽+1
        # 28 - 5 +1 = 24
        # 如果想大小不变，应设置padding
        # 层数为输出层数
        x = self.conv1(x)
        d_print(x.shape) # torch.Size([1, 6, 600, 800])
        x = self.bn1(x)
        d_print(x.shape)
        # relu只是将负值变为0，所以shape不变
        x = F.relu(x)
        d_print(x.shape) # torch.Size([1, 6, 600, 800])
        # 大小变为size/pool_size,层数不变
        x = self.pool1(x)
        d_print(x.shape) # torch.Size([1, 6, 120, 160])

        x = self.conv2(x)
        d_print(x.shape) # torch.Size([1, 16, 120, 160])
        x = self.bn2(x)
        d_print(x.shape)
        x = F.relu(x)
        d_print(x.shape) # torch.Size([1, 16, 120, 160])
        # 大小变为size/pool_size,层数不变
        x = self.pool2(x)
        d_print(x.shape) # torch.Size([1, 16, 24, 32])

        x = self.conv3(x)
        d_print(x.shape) # torch.Size([1, 16, 120, 160])
        x = self.bn3(x)
        d_print(x.shape)
        x = F.relu(x)
        d_print(x.shape) # torch.Size([1, 16, 120, 160])
        # 大小变为size/pool_size,层数不变
        x = self.pool3(x)
        d_print(x.shape) # torch.Size([1, 16, 24, 32])

        x = self.conv4(x)
        d_print(x.shape) # torch.Size([1, 16, 120, 160])
        x = self.bn4(x)
        d_print(x.shape)
        x = F.relu(x)
        d_print(x.shape) # torch.Size([1, 16, 120, 160])
        # 大小变为size/pool_size,层数不变
        x = self.pool4(x)
        d_print(x.shape) # torch.Size([1, 16, 24, 32])

        

        # 在第一个维度打平了，16*24*32=12280
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        d_print(x.shape) # torch.Size([1, 12280])

        x = self.fc1(x)
        d_print(x.shape) # torch.Size([4, 1228])
        

        return x
