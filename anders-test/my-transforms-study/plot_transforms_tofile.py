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

# randomResizedCrop()

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

def randomPerspective():
    # RandomPerspective
    # ~~~~~~~~~~~~~~~~~
    # The :class:`~torchvision.transforms.RandomPerspective` transform
    # (see also :func:`~torchvision.transforms.functional.perspective`)
    # performs random perspective transform on an image.
    # p为被变化的概率（0到1），默认为0.5
    # fill为空白处的颜色
    # distortion_scale ，变化程度。0：没变化，1：扭曲的不成样子，0.2稍稍变化
    perspective_transformer = T.RandomPerspective(distortion_scale=0.6, p=0.3,fill =(255,0,0))
    perspective_imgs = [perspective_transformer(orig_img) for _ in range(4)]
    plot(perspective_imgs, "randomPerspective")

# randomPerspective()

def colorJitter():
    # ColorJitter
    # ~~~~~~~~~~~
    # The :class:`~torchvision.transforms.ColorJitter` transform
    # randomly changes the brightness, saturation, and other properties of an image.
    # brightness:亮度，0为全黑，0.5为半黑，1为不变，5已经非常亮，看不清了
    # contrast:对比度，0为看不见,0.5灰蒙蒙，1为不变，2对比很强烈，有点失真了
    # saturation：饱和度,0为纯黑白,0.5颜色淡，灰蒙蒙，1不变，2颜色鲜艳
    # hue:颜色 -0.5 <= min <= max <= 0.5
    # -0.5很奇诡的颜色，脸变蓝了
    # -0.2脸变红，衣服变绿
    # -0.1有点正常了
    # 0不变
    # 0.2脸变绿
    # 0.5很不正常的颜色
    jitter = T.ColorJitter(brightness=(0.8,1.2), contrast = (0.8,1.2),saturation = (0.8,1.2), hue=(0,0))
    # jitter = T.ColorJitter(brightness=.5, hue=.3)
    jitted_imgs = [jitter(orig_img) for _ in range(40)]
    plot(jitted_imgs,"colorJitter")
colorJitter()