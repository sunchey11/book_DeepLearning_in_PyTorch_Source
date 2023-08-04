# 笔记在 anders-test/my-transforms-study
# 根据plot_transforms.py修改的，将transform后的文件保存在磁盘上
# 方便看大小
"""
==========================
Illustration of transforms
==========================

This example illustrates the various transforms available in :ref:`the
torchvision.transforms module <transforms>`.
"""

# sphinx_gallery_thumbnail_path = "../../gallery/assets/transforms_thumbnail.png"

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
import os
import torchvision.transforms as transforms

file_dir = os.path.split(os.path.abspath(__file__))[0]
assets_path = os.path.join(file_dir, 'assets')


plt.rcParams["savefig.bbox"] = 'tight'
# orig_img = Image.open(os.path.join(assets_path, 'astronaut.jpg'))
orig_img = Image.open(os.path.join(assets_path, 'person1.jpg'))
print(type(orig_img).__base__)
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)

outputs_path = os.path.join(file_dir, 'outputs')
def plot(imgs,transform_name):
    for i in range(len(imgs)):
        img = imgs[i]
        filename = os.path.join(outputs_path, transform_name+"-"+str(i)+".png")
        img.save(filename)




####################################
# Resize
# ------
# The :class:`~torchvision.transforms.Resize` transform
# (see also :func:`~torchvision.transforms.functional.resize`)
# resizes an image.
def resize_sample():
    print(orig_img.size) #宽,高
    # 但是下面的参数是高，宽, 所以最后一个图片比例失调
    resized_imgs = [T.Resize(size=size)(orig_img) for size in ((30,60), 50, 100, orig_img.size)]
    plot(resized_imgs,"resize")
# resize_sample()

def randomResizedCrop():
    ####################################
    # RandomResizedCrop
    # ~~~~~~~~~~~~~~~~~
    # The :class:`~torchvision.transforms.RandomResizedCrop` transform
    # (see also :func:`~torchvision.transforms.functional.resized_crop`)
    # crops an image at a random location, and then resizes the crop to a given
    # size.
    # 随机截取一块，最后resize到 (400,400)
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomResizedCrop.html#torchvision.transforms.RandomResizedCrop
    # scale好像是面积相对于原图的比例
    # ration是纵横比，即高/宽的比例的最小值和最大值
    # 证据1. 将ratio设置为(1,1),则宽高比为1，截出来是正方形，然后resize为正方形，看到宽高等比缩放
    # 证据1. 将ratio设置为(0.5,0.5),则高宽比为0.5，然后resize为(800,400)，看到宽高等比缩放
    resize_cropper = T.RandomResizedCrop(size=(800, 400),scale=(0.5, 1.0),ratio=(0.5,0.5))
    resized_crops = [resize_cropper(orig_img) for _ in range(4)]
    plot(resized_crops,"randomResizedCrop")

randomResizedCrop()

def randomRotation():
    ####################################
    # RandomRotation
    # ~~~~~~~~~~~~~~
    # The :class:`~torchvision.transforms.RandomRotation` transform
    # (see also :func:`~torchvision.transforms.functional.rotate`)
    # rotates an image with random angle.
    rotater = T.RandomRotation(degrees=(0, 180))
    rotated_imgs = [rotater(orig_img) for _ in range(4)]
    plot(rotated_imgs,"randomRotation")
# randomRotation()

def randomHorizontalFlip():
    # RandomHorizontalFlip
    # ~~~~~~~~~~~~~~~~~~~~
    # The :class:`~torchvision.transforms.RandomHorizontalFlip` transform
    # (see also :func:`~torchvision.transforms.functional.hflip`)
    # performs horizontal flip of an image, with a given probability.
    hflipper = T.RandomHorizontalFlip(p=0.5)
    transformed_imgs = [hflipper(orig_img) for _ in range(4)]
    plot(transformed_imgs, "randomHorizontalFlip")
# randomHorizontalFlip()



# 看看人家的，都是这样搞得
# https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/
image_transforms = { 
    'train': transforms.Compose([
        # scale参数我不知道啥意思，截取面积与原面积的比值：（最小值，最大值）
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        # degree为+15 ,-15
        transforms.RandomRotation(degrees=15),
        # 随机就两种情况，翻转和不翻转
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}