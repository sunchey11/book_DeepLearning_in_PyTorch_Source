# 测试模型,准确率在60%左右，一直上不去，不知道为什么

# 
"""
epoch = 2, size of cifar_net.pth = 246k
Accuracy of the network on the 10000 test images: 52 %
Accuracy for class: plane is 51.0 %
Accuracy for class: car   is 91.6 %
Accuracy for class: bird  is 31.6 %
Accuracy for class: cat   is 31.0 %
Accuracy for class: deer  is 50.8 %
Accuracy for class: dog   is 38.3 %
Accuracy for class: frog  is 61.9 %
Accuracy for class: horse is 63.2 %
Accuracy for class: ship  is 73.5 %
Accuracy for class: truck is 32.5 %
"""

"""
epoch = 10, loss = 0.875 size of cifar_net.pth = 246k
Accuracy of the network on the 10000 test images: 60 %
Accuracy for class: plane is 66.2 %
Accuracy for class: car   is 77.1 %
Accuracy for class: bird  is 46.7 %
Accuracy for class: cat   is 39.6 %
Accuracy for class: deer  is 56.2 %
Accuracy for class: dog   is 51.3 %
Accuracy for class: frog  is 71.4 %
Accuracy for class: horse is 65.3 %
Accuracy for class: ship  is 71.6 %
Accuracy for class: truck is 63.6 %
"""

"""
epoch = 20, loss = 0.685, size of cifar_net.pth = 246k
Accuracy of the network on the 10000 test images: 61 %
Accuracy for class: plane is 60.0 %
Accuracy for class: car   is 89.1 %
Accuracy for class: bird  is 56.8 %
Accuracy for class: cat   is 52.4 %
Accuracy for class: deer  is 48.3 %
Accuracy for class: dog   is 38.1 %
Accuracy for class: frog  is 72.6 %
Accuracy for class: horse is 64.3 %
Accuracy for class: ship  is 71.4 %
Accuracy for class: truck is 63.9 %
"""

"""
epoch = 50, loss = 0.650, size of cifar_net.pth = 246k
Accuracy of the network on the 10000 test images: 58 %
Accuracy for class: plane is 67.0 %
Accuracy for class: car   is 72.2 %
Accuracy for class: bird  is 46.2 %
Accuracy for class: cat   is 40.6 %
Accuracy for class: deer  is 51.7 %
Accuracy for class: dog   is 41.7 %
Accuracy for class: frog  is 65.6 %
Accuracy for class: horse is 64.6 %
Accuracy for class: ship  is 71.3 %
Accuracy for class: truck is 66.5 %
"""
import torch
import torchvision
import torchvision.transforms as transforms
import os

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='D:\\pytorch_data\\cifar10\\data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='D:\\pytorch_data\\cifar10\\data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import torch.nn as nn
import torch.nn.functional as F


debug = False
def d_print(s):
    if(debug):
        print(s)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        d_print(x.shape) # torch.Size([4, 3, 32, 32])
        # conv后的图片大小为：源宽-窗口宽+1
        # 32 - 5 +1 = 28
        # 如果想大小不变，应设置padding
        # 层数为输出层数
        x = self.conv1(x)
        d_print(x.shape) # torch.Size([4, 6, 28, 28])
        # relu只是将负值变为0，所以shape不变
        x = F.relu(x)
        d_print(x.shape) # torch.Size([4, 6, 28, 28])
        # 大小变为size/pool_size,层数不变
        x = self.pool(x)
        d_print(x.shape) # torch.Size([4, 6, 14, 14])

        x = self.conv2(x)
        d_print(x.shape) # torch.Size([4, 16, 10, 10])
        x = F.relu(x)
        d_print(x.shape) # torch.Size([4, 16, 10, 10])
        # 大小变为size/pool_size,层数不变
        x = self.pool(x)
        d_print(x.shape) # torch.Size([4, 16, 5, 5])

        # 在第一个维度打平了，16*5*5=400
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        d_print(x.shape) # torch.Size([4, 400])

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
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




dataiter = iter(testloader)
images, labels = next(dataiter)
d_print(images.shape) # torch.Size([4, 3, 32, 32])
d_print(labels.shape) # torch.Size([4])


# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

file_dir = os.path.split(os.path.abspath(__file__))[0]
PATH = os.path.join(file_dir, './cifar_net.pth')

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