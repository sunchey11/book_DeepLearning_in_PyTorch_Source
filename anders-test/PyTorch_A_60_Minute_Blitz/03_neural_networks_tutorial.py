import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print(x.shape) # torch.Size([3, 1, 32, 32])

        # conv后的图片大小为：源宽-窗口宽+1
        # 32 - 5 +1 = 28
        # 如果想大小不变，应设置padding
        # 层数为输出层数
        x = self.conv1(x)
        print(x.shape) # torch.Size([3, 6, 28, 28])
        # relu只是将负值变为0，所以shape不变
        x = F.relu(x)
        print(x.shape) # torch.Size([3, 6, 28, 28])
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(x, (2, 2))
        # 大小变为size/pool_size,层数不变
        print(x.shape) # torch.Size([3, 6, 14, 14])
        x = self.conv2(x)
        print(x.shape) # torch.Size([3, 16, 10, 10])
        x = F.relu(x)
        print(x.shape)
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(x, 2)
        print(x.shape) # torch.Size([3, 16, 5, 5])
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        # 在第一个维度打平了，16*5*5=400
        print(x.shape) # torch.Size([3, 400])
        x = self.fc1(x)
        print(x.shape) # torch.Size([3, 120])
        x = F.relu(x)
        print(x.shape)

        x = self.fc2(x)
        print(x.shape) # torch.Size([3, 84])
        x = F.relu(x)
        print(x.shape) # torch.Size([3, 84])
        x = self.fc3(x)
        print(x.shape) # torch.Size([3, 10])
        return x


net = Net()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

# 3张图片，一个层次(黑白)
input = torch.randn(3, 1, 32, 32)
print(input.shape)
# print(input)
out = net(input)
print(out)

net.zero_grad()
# backward带参数是啥意思
out.backward(torch.randn(3, 10))

output = net(input)
# input是一个数组，shape=[1, 1, 32, 32],表示一张图片，图片32*32， 颜色深度为1
# output是预测结果，shape=[1,10],这个图片属于10各分类的概率
# 因为预测了3条，这里应该改为3条，但是我不想改了
target = torch.randn(10)  # a dummy target, for example
# target为实际分类，shape=[1,10]
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
print('finished')