# 原文
# https://pytorch.org/vision/stable/auto_examples/plot_scripted_tensor_transforms.html
# https://github.com/pytorch/vision/tree/main/gallery
# 从这个记事本拷贝出来的例子
# anders-test/vision-examples-gallery/plot_scripted_tensor_transforms.ipynb
# 演示使用ResNet18，识别两只狗的品种
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision import datasets, models, transforms
import os
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

# 数据存储总路径
file_dir = os.path.split(os.path.abspath(__file__))[0]
dog1_path = os.path.join(file_dir, "assets/dog1.jpg")
dog2_path = os.path.join(file_dir, "assets/dog2.jpg")
# dog1_path = os.path.join(file_dir, "assets/ant1.jpg")
# dog2_path = os.path.join(file_dir, "assets/2ants.jpg")
label_path = os.path.join(file_dir, "assets/imagenet_class_index.json")
def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
dog1 = read_image(dog1_path)
dog2 = read_image(dog2_path)
show([dog1,dog2])

# 这段就是为了transforms一下
weights = ResNet18_Weights.DEFAULT
transforms = weights.transforms()

# 加不加eval，结果不一样，为什么？
net = models.resnet18(pretrained=True).eval()
# net = models.resnet18(pretrained=True)
print(type(dog1)) # <class 'torch.Tensor'>
print(dog1.shape) #[3, 500, 500]

# transforms本来是在后面的，但是我给的两个图片大小不一样，所以先transforms
# 图片大小就一样了
[dog1] = transforms(torch.stack([dog1]))
# 图片大小变成了224*224，并且颜色变了
[dog2] = transforms(torch.stack([dog2]))
print(type(dog1)) # <class 'torch.Tensor'>
print(dog1.shape) #[3, 224, 224]

# stack函数将两个tensor组成一个新的tensor
# https://pytorch.org/docs/stable/generated/torch.stack.html
batch = torch.stack([dog1, dog2])
print(type(batch)) #<class 'torch.Tensor'>
print(batch.shape) #[2, 3, 224, 224]


x = batch

show(x)
y1 = net(x)
print(type(y1))
print(y1.shape) #[2, 1000]

# 第一种获取结果的方法，顺带获取概率
# 将数据转为0到1之间的概率，概率总和为1
softmax = nn.Softmax(dim=1)
probability = softmax(y1)
# print(probability)
# 这里面放的每一种情况的概率
print(probability.shape) # torch.Size([2, 1000])
print(probability[0].sum()) # 概率和应为1


_, prob_value = torch.max(probability, 1)
# _存放的概率最大的值
print(_.shape) #torch.Size([2])
# 存放对应的lable的index
print(prob_value.shape) #torch.Size([2])
# 显示概率的值
print(_)
print(prob_value)


# 第二种获取结果的方法，不需要概率
# 获取了lable 的 index
res = y1.argmax(dim=1)
print(res.shape) # torch.Size([2])
print(res)

with open(label_path) as labels_file:
    labels = json.load(labels_file)

print(type(labels))
print(len(labels))
for i in res:
    print(i)
    print(labels[str(i.item())])

print('finished!')