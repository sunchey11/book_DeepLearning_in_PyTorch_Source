# 训练
# 识别药品包装，数据在data目录下
from FontIdenNet import FontIdenNet,img_height,img_width,d_print
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms


chinese_path = "D:\\pytorch_data\\font_to_png"


# https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
train_dataset = datasets.ImageFolder(os.path.join(chinese_path, 'train_gbk'),
                                     transforms.Compose([
                                        
                                        transforms.Resize((img_height,img_width)),
                                        transforms.ToTensor(),
                                        
                                    ])
                                    )

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 10, shuffle = False, num_workers=0)
print(train_dataset.classes)


import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
net = FontIdenNet()

file_dir = os.path.split(os.path.abspath(__file__))[0]
PATH = os.path.join(file_dir, './font_iden_gbk.pth')
if os.path.exists(PATH):
    net.load_state_dict(torch.load(PATH))
net = net.to(device)
import torch.optim as optim
import time
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
start=time.time()


for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    total = 0
    li = 0
    for i, data in enumerate(trainloader, 0):
        total +=1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
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
        li = loss.item()
        running_loss += li
        

        if i % 100 == 0:    # print every 2000 mini-batches
            print(f'[{i}] avg loss: {running_loss / total:.3f}')
            print('shot',i,'loss:',li)
            running_loss = 0.0
            total = 0

    print(epoch,'loss:',li)

end=time.time()
# 20个汉字
# 时间为batch size = 1,cpu 10次516秒左右
# 时间为batch size = 10,cpu 10次81秒左右
# 时间为batch size = 10,gpu 10次40秒左右
# 时间为batch size = 10,gpu 100次380秒左右,loss到可用范围,asul 68s
# 时间为batch size = 100,gpu 100次142秒左右,但是loss不正常

# 2500个汉字
# 时间为batch size = 10,gpu 两小时干了10次，没干完，干完100次得20小时

# gbk 6000多汉字，batch size = 100,gpu 100次6446秒,8823秒
# [600] loss: 3.297
# aba 600 loss: 2.763105869293213
# 99 loss: 7.163599014282227

# [600] avg loss: 3.297
# shot 600 loss: 2.763104200363159
# 99 loss: 7.163599014282227
print('程序运行时间为: %s Seconds'%(end-start))

print('Finished Training')
print(train_dataset.classes)


torch.save(net.state_dict(), PATH)
