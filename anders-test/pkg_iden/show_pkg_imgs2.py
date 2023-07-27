# 显示目录下的图片,transform是个函数
import torch
from torch.utils.data import Dataset

from torchvision.transforms import ToTensor
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import os


def showImgs_make_grid(trainloader, p_classes):
    # get some random training images
    dataiter = iter(trainloader)
    # 为啥会是随机的，不是第一个， shuffle控制是不是随机的
    images, labels = next(dataiter)
    # 维度和batch_size有关，batch_size为4，显示4个图片
    # print(images)
    # print(labels)
    
    # show images
    print(images.shape)
    # 将多个图片，合成为一个网格图片
    x = torchvision.utils.make_grid(images)
    print(x.shape)
    img = x
    npimg = img.numpy()
    print("npimg.shape",npimg.shape)
    # 轴变换 
    npimg = np.transpose(npimg, (1, 2, 0))
    print(npimg.shape)
    plt.imshow(npimg)
    plt.show()
    # print labels
    batch_size = images.shape[0]
    print(' '.join(f'{p_classes[labels[j]]:5s}' for j in range(batch_size)))
file_dir = os.path.split(__file__)[0]
print(file_dir)
data_path = os.path.join(file_dir, "./data")
def func1(img):
    print(type(img)) # <class 'PIL.Image.Image'>
    r = transforms.Resize((600,800))
    img = r(img)

    print(type(img))# <class 'PIL.Image.Image'>
    tt = transforms.ToTensor()
    img = tt(img)

    print(type(img))
    print(img.shape)
    print(img)
    return img
# https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'),
                                     func1
                                    )
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 3, shuffle = False, num_workers=0)
print("classes",train_dataset.classes)
showImgs_make_grid(train_loader, train_dataset.classes)
