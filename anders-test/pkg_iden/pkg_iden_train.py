# 训练
# 识别药品包装，数据在data目录下
from PkgIdenNet import d_print,PkgIdenNet,img_width,img_height
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms


file_dir = os.path.split(__file__)[0]
print(file_dir)
data_path = os.path.join(file_dir, "./data")

# https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'),
                                     transforms.Compose([
                                        
                                        transforms.Resize((img_height,img_width)),
                                        transforms.ToTensor(),
                                        
                                    ])
                                    )

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = False, num_workers=0)
print(train_dataset.classes)


import torch.nn as nn


net = PkgIdenNet()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(160):  # loop over the dataset multiple times

    running_loss = 0.0
    total = 0
    for i, data in enumerate(trainloader, 0):
        total +=1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        d_print(inputs.shape) # torch.Size([1, 3, 600, 800])
        d_print(labels.shape) # torch.Size([1])
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
    if epoch % 20 == 19:    # print every 2000 mini-batches
        print(f'[{epoch + 1}, {epoch + 1:5d}] loss: {running_loss / total:.3f}')

print('Finished Training')
print(train_dataset.classes)

file_dir = os.path.split(os.path.abspath(__file__))[0]
PATH = os.path.join(file_dir, './pkg_iden.pth')

torch.save(net.state_dict(), PATH)