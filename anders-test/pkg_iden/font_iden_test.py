# 测试模型,准确率在xx


import time
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torchvision import datasets, models, transforms
from FontIdenNet import d_print,FontIdenNet,img_width,img_height

chinese_path = "D:\\pytorch_data\\font_to_png"

# https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
test_dataset = datasets.ImageFolder(os.path.join(chinese_path, 'test_gbk'),
                                     transforms.Compose([
                                        transforms.Resize((img_height,img_width)),
                                        transforms.ToTensor(),
                                    ])
                                    )
train_dataset = datasets.ImageFolder(os.path.join(chinese_path, 'train_gbk'),
                                     transforms.Compose([
                                        
                                        transforms.Resize((img_height,img_width)),
                                        transforms.ToTensor(),
                                        
                                    ])
                                    )

t1 = test_dataset[0]

print(len(t1))
print(t1[0]) #这是图片数据
print(t1[1]) #这是一个整数，即label的index

# 所有的文件
print(type(test_dataset.imgs))
print(len(test_dataset.imgs))
print(type(test_dataset.imgs[0]))


batch_size = 1
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers=0)
print(test_dataset.classes)

# ['39', '620', 'aoli', 'eber', 'fengshi', 'kouzhao', 'kushen', 'lianhua', 'ningjiao', 'nut', 'shangtong', 'yikang', 'zhuangyao', 'zhuodu']
classes = train_dataset.classes

import torch.nn as nn
import torch.nn.functional as F



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
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

file_dir = os.path.split(os.path.abspath(__file__))[0]
PATH = os.path.join(file_dir, './font_iden_gbk.pth')

net = FontIdenNet()
net.load_state_dict(torch.load(PATH))


start=time.time()
outputs = net(images)
end=time.time()
# 时间为0.05秒左右
print('程序运行时间为: %s Seconds'%(end-start))

print(outputs.shape) # torch.Size([4, 10])
print(outputs)
# 将数据转为0到1之间的概率，总和为1
softmax = nn.Softmax(dim=1)

outputs = softmax(outputs)
print(outputs)
print(outputs[0].sum())


_, predicted = torch.max(outputs, 1)

print(type(_))
print(_.shape)
# _里面放的是概率，如果能预测，则概率>0.99
# 如果不能预测，则概率是0.6779
print(_)
print(predicted.shape)
print(predicted)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(batch_size)))


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        start=time.time()
        outputs = net(images)
        end=time.time()
        # 时间为0.05秒左右
        # print('程序运行时间为: %s Seconds'%(end-start))

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