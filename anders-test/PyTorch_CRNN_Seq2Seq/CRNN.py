import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init


import torch.utils.data as data_utils
import numpy as np
import torch
import os

debug = True
def d_print(s):
    if(debug):
        print(s)
class CRNN(nn.Module):

    def __init__(self):
        super(CRNN, self).__init__()

        self.num_classes = 10 + 1
        self.image_H = 28

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.in1 = nn.InstanceNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3))
        self.in2 = nn.InstanceNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)
        self.in3 = nn.InstanceNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.in4 = nn.InstanceNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=(3,3))
        self.in5 = nn.InstanceNorm2d(64)

        self.conv6 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=2)
        self.in6 = nn.InstanceNorm2d(64)

        self.postconv_height = 3
        self.postconv_width = 31

        self.gru_input_size = self.postconv_height * 64
        self.gru_hidden_size = 128
        self.gru_num_layers = 2
        self.gru_h = None
        self.gru_cell = None

        self.gru = nn.GRU(self.gru_input_size, self.gru_hidden_size, self.gru_num_layers, batch_first = True, bidirectional = True)

        self.fc = nn.Linear(self.gru_hidden_size * 2, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0] #batch_size=64
        d_print(x.shape) # torch.Size([64, 1, 28, 140])
        out = self.conv1(x)
        out = F.leaky_relu(out)
        out = self.in1(out)
        d_print(out.shape) #torch.Size([64, 32, 26, 138])

        out = self.conv2(out)
        out = F.leaky_relu(out)
        out = self.in2(out)
        d_print(out.shape) # torch.Size([64, 32, 26, 138])

        out = self.conv3(out)
        # 这里变小了，因为stride=2
        d_print(out.shape) # torch.Size([64, 32, 11, 67])
        out = F.leaky_relu(out) 
        out = self.in3(out)
        d_print(out.shape) # # torch.Size([64, 32, 11, 67])

        out = self.conv4(out)
        out = F.leaky_relu(out)
        out = self.in4(out)
        d_print(out.shape) # torch.Size([64, 64, 9, 65])

        out = self.conv5(out)
        out = F.leaky_relu(out)
        out = self.in5(out)
        d_print(out.shape) # torch.Size([64, 64, 7, 63])

        out = self.conv6(out)
        # 这里变小了，因为stride=2
        d_print(out.shape) # torch.Size([64, 64, 3, 31])
        out = F.leaky_relu(out)
        out = self.in6(out)
        d_print(out.shape) # torch.Size([64, 64, 3, 31])

 
        out = out.permute(0, 3, 2, 1)
        d_print(out.shape) # torch.Size([64, 31, 3, 64])
        out = out.reshape(batch_size, -1, self.gru_input_size)
        d_print(out.shape) # torch.Size([64, 31, 192])
        out, gru_h = self.gru(out, self.gru_h)
        d_print(out.shape) # torch.Size([64, 31, 256])
        self.gru_h = gru_h.detach()
        out = torch.stack([F.log_softmax(self.fc(out[i])) for i in range(out.shape[0])])
        d_print(out.shape) # torch.Size([64, 31, 11])
        return out

    def reset_hidden(self, batch_size):
        h = torch.zeros(self.gru_num_layers * 2, batch_size, self.gru_hidden_size)
        self.gru_h = Variable(h)
