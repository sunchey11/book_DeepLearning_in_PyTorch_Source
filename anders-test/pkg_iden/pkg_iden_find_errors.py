# 测试模型,找出不能正确识别的图片




import PIL
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torchvision import datasets, models, transforms
from PkgIdenNet import d_print,PkgIdenNet,img_width,img_height

file_dir = os.path.split(__file__)[0]
print(file_dir)
data_path = os.path.join(file_dir, "./data")

# https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
test_dataset = datasets.ImageFolder(os.path.join(data_path, 'test2'),
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
print(test_dataset.imgs[5])




batch_size = 1
print(test_dataset.classes)

# ['39', '620', 'aoli', 'eber', 'fengshi', 'kouzhao', 'kushen', 'lianhua', 'ningjiao', 'nut', 'shangtong', 'yikang', 'zhuangyao', 'zhuodu']
classes = test_dataset.classes

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




file_dir = os.path.split(os.path.abspath(__file__))[0]
PATH = os.path.join(file_dir, './pkg_iden2.pth')

net = PkgIdenNet()
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    t1 = test_dataset[0]

    # print(len(t1)) #数组两个元素
    # print(t1[0]) #这是图片数据
    # print(t1[1]) #这是一个整数，即label的index

    # 所有的文件列表，包含文件名，和label index
    # print(type(test_dataset.imgs))
    # print(len(test_dataset.imgs))
    # print(type(test_dataset.imgs[0]))
    # print(test_dataset.imgs[5])

    for i in range(len(test_dataset)):
        data = test_dataset[i]
        images, labels = data

        # 将tensor变成一个单元素数组
        images = torch.stack([images])
        # 将整数变成一个单元素数组
        labels = torch.tensor([labels])
        # calculate outputs by running images through the network
        outputs = net(images)
        
        # print(outputs.shape) 
        # print(outputs)
        # 将数据转为0到1之间的概率，总和为1
        softmax = nn.Softmax(dim=1)

        outputs = softmax(outputs)
        # print(outputs)
        # print(outputs[0].sum()) # 概率和应为1


        _, predicted = torch.max(outputs, 1)

        # print(type(_)) #<class 'torch.Tensor'>
        # print(_.shape)
        # _里面放的是概率，如果能预测，则概率>0.99
        # 如果不能预测，则概率是0.6779
        # print(_)
        # print(predicted.shape)
        # print(predicted)
        print('Predicted: ', classes[predicted[0]])

        if _[0]>0.9 and predicted[0] == labels[0]:
            correct += 1
            fn = test_dataset.imgs[i]
            print("ok", fn)
            print("index", i)
            print("rate",_[0])
            print("should be", labels[0])
            print("but it is", predicted[0])
            images = torchvision.utils.make_grid(images)
            imshow(images)
        else:

            # fn = test_dataset.imgs[i]
            # print("error", fn)
            # print("index", i)
            # print("rate",_[0])
            # print("should be", labels[0])
            # print("but it is", predicted[0])
            # images = torchvision.utils.make_grid(images)
            # imshow(images)
            pass
        total += 1


print(f'Accuracy of the network on the test images: {100 * correct // total} %')

