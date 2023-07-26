# 训练
# 从这里拷贝的代码\PyTorch_A_60_Minute_Blitz\04_cifar10_tutorial_train.py
# 因为准确率不高，所以换成数字识别，看看效果如何
# 将book_DeepLearning_in_PyTorch_Source\anders-test\myc5\minst_convnet.py
# 中的相关内容拷贝过来
import torch
import torchvision
import torchvision.transforms as transforms
import os

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.MNIST(root='D:\\pytorch_data\\mnist\\data', train=True,
                                        download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='D:\\pytorch_data\\mnist\\data', train=False,
                                       download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)



import torch.nn as nn
import torch.nn.functional as F
debug = False
def d_print(s):
    if(debug):
        print(s)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 这里调整大小
        # self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1 = nn.Conv2d(1, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 这里调整大小
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        d_print(x.shape) # torch.Size([4, 1, 28, 28])
        # conv后的图片大小为：源宽-窗口宽+1
        # 28 - 5 +1 = 24
        # 如果想大小不变，应设置padding
        # 层数为输出层数
        x = self.conv1(x)
        d_print(x.shape) # torch.Size([4, 6, 24, 24])
        # relu只是将负值变为0，所以shape不变
        x = F.relu(x)
        d_print(x.shape) # torch.Size([4, 6, 24, 24])
        # 大小变为size/pool_size,层数不变
        x = self.pool(x)
        d_print(x.shape) # torch.Size([4, 6, 12, 12])

        x = self.conv2(x)
        d_print(x.shape) # torch.Size([4, 16, 8, 8])
        x = F.relu(x)
        d_print(x.shape) # torch.Size([4, 16, 8, 8])
        # 大小变为size/pool_size,层数不变
        x = self.pool(x)
        d_print(x.shape) # torch.Size([4, 16, 4, 4])

        # 在第一个维度打平了，16*4*4=256
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        d_print(x.shape) # torch.Size([4, 256])

        x = self.fc1(x)
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


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        d_print(inputs.shape) # torch.Size([4, 1, 28, 28])
        d_print(labels.shape) # torch.Size([4])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        d_print(outputs.shape) # torch.Size([4, 10])
        d_print(labels.shape) # torch.Size([4])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')


file_dir = os.path.split(os.path.abspath(__file__))[0]
PATH = os.path.join(file_dir, './my_digit.pth')

torch.save(net.state_dict(), PATH)