# 测试模型,准确率在xx
# 从这里拷贝的代码\PyTorch_A_60_Minute_Blitz\04_cifar10_tutorial_test.py
# 因为准确率不高，所以换成数字识别，看看效果如何
# 将book_DeepLearning_in_PyTorch_Source\anders-test\myc5\minst_convnet.py
# 中的相关内容拷贝过来


"""
epoch = 2, loss = 0.063, size of cifar_net.pth = 177k
Accuracy of the network on the 10000 test images: 98 %
Accuracy for class: 0     is 99.7 %
Accuracy for class: 1     is 99.4 %
Accuracy for class: 2     is 98.7 %
Accuracy for class: 3     is 98.7 %
Accuracy for class: 4     is 99.2 %
Accuracy for class: 5     is 99.0 %
Accuracy for class: 6     is 98.6 %
Accuracy for class: 7     is 98.7 %
Accuracy for class: 8     is 97.8 %
Accuracy for class: 9     is 95.8 %
"""

"""
epoch = 5, loss = 0.032, size of cifar_net.pth = 177k
Accuracy of the network on the 10000 test images: 98 %
Accuracy for class: 0     is 99.6 %
Accuracy for class: 1     is 99.8 %
Accuracy for class: 2     is 98.7 %
Accuracy for class: 3     is 99.7 %
Accuracy for class: 4     is 99.1 %
Accuracy for class: 5     is 98.1 %
Accuracy for class: 6     is 97.9 %
Accuracy for class: 7     is 98.4 %
Accuracy for class: 8     is 98.8 %
Accuracy for class: 9     is 97.3 %

"""
"""
epoch = 20, loss = 0.010, size of cifar_net.pth = 177k
Accuracy of the network on the 10000 test images: 98 %
Accuracy for class: 0     is 99.7 %
Accuracy for class: 1     is 99.9 %
Accuracy for class: 2     is 99.6 %
Accuracy for class: 3     is 99.4 %
Accuracy for class: 4     is 99.1 %
Accuracy for class: 5     is 98.0 %
Accuracy for class: 6     is 96.6 %
Accuracy for class: 7     is 99.2 %
Accuracy for class: 8     is 98.9 %
Accuracy for class: 9     is 97.8 %
"""

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



classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')


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

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    print(npimg.shape)
    npimg = np.transpose(npimg, (1, 2, 0))
    print(npimg.shape)
    plt.imshow(npimg)
    plt.show()




dataiter = iter(testloader)
images, labels = next(dataiter)
d_print(images.shape) # torch.Size([4, 3, 32, 32])
d_print(labels.shape) # torch.Size([4])


# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

file_dir = os.path.split(os.path.abspath(__file__))[0]
PATH = os.path.join(file_dir, './my_digit.pth')

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)
print(outputs.shape) # torch.Size([4, 10])

_, predicted = torch.max(outputs, 1)
print(type(_))
print(_.shape)
print(_)
print(predicted.shape)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')