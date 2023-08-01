# 原文
# https://pytorch.org/vision/stable/auto_examples/plot_scripted_tensor_transforms.html
# 从这个记事本拷贝出来的例子
# anders-test/vision-examples-gallery/plot_scripted_tensor_transforms.ipynb
# 演示图像识别两只狗的品种
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
from PIL import Image


# 数据存储总路径
file_dir = os.path.split(os.path.abspath(__file__))[0]
# dog1_path = os.path.join(file_dir, "assets/dog1.jpg")
# dog2_path = os.path.join(file_dir, "assets/dog2.jpg")
dog1_path = os.path.join(file_dir, "assets/ant1.jpg")
dog2_path = os.path.join(file_dir, "assets/2ants.jpg")
def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
dog1 = read_image(dog1_path)
print(type(dog1))
print(dog1.shape) #[3, 500, 500]

show([dog1])

# resize = transforms.Resize((224,224), interpolation=Image.BICUBIC) 
# resize = transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BILINEAR) 
# resize = transforms.Resize((224,224), interpolation=transforms.InterpolationMode.NEAREST) 
# resize = transforms.Resize((224,224), interpolation=transforms.InterpolationMode.NEAREST_EXACT) 
resize = transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BICUBIC) 

rdog1 = resize(dog1)
print(type(rdog1))
print(rdog1.shape) #[3, 500, 500]
show([rdog1])
arr = np.asarray(rdog1) 
print(arr.shape)
arr = np.transpose(arr, (1, 2, 0))
print(arr.shape)
print(arr[10][10])
rimg = Image.fromarray(arr,mode="RGB")
p = rimg.getpixel((10, 10))
print(p)
rimg.save(os.path.join(file_dir, "resize.png"))
print('finished!')