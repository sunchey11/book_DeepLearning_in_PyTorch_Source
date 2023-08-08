# 训练
# 识别药品包装，数据在data目录下
from img_utils import d_print
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import time

file_dir = os.path.split(__file__)[0]
print(file_dir)
data_path = os.path.join(file_dir, "../pkg_iden/data")
debug_dir = os.path.join(file_dir, "./debug_dir")
pad_width = 2
# 图像的大小为224*224
image_size = 224
# https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'),
                                    transforms.Compose([
                                        transforms.RandomResizedCrop(image_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
                                    )

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 10, shuffle = False, num_workers=0)
print(train_dataset.classes)
print(len(train_dataset))


import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
net = models.resnet18(pretrained=True)
# 读取最后线性层的输入单元数，这是前面各层卷积提取到的特征数量
num_ftrs = net.fc.in_features

print(type(net.fc)) # <class 'torch.nn.modules.linear.Linear'>
# 重新定义一个全新的线性层，它的输出为2，原本是1000
net.fc = nn.Linear(num_ftrs, 20)




PATH = os.path.join(file_dir, './pkg_iden_resnet18.pth')
# if os.path.exists(PATH):
#     net.load_state_dict(torch.load(PATH))

net = net.to(device)
import torch.optim as optim



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

start=time.time()

net.train(True)
for epoch in range(500):  # loop over the dataset multiple times
    
    running_loss = 0.0
    total = 0
    for i, data in enumerate(trainloader, 0):
        total +=1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        d_print(inputs.shape) # torch.Size([1, 3, 600, 800])
        d_print(labels.shape) # torch.Size([1])
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # label是个数字
        d_print(type(labels[0]))
        d_print(labels[0].item())
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        d_print(outputs.shape) # torch.Size([4, 10])
        d_print(labels.shape) # torch.Size([4])
        d_print(outputs) 
        d_print(labels) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
       
    print(f'[{epoch + 1}, {epoch + 1:5d}] loss: {running_loss / total:.6f}')

end=time.time()
# 760秒
print('程序运行时间为: %s Seconds'%(end-start))
print('Finished Training')
print(train_dataset.classes)

file_dir = os.path.split(os.path.abspath(__file__))[0]


torch.save(net.state_dict(), PATH)