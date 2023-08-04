from img_utils import save_tensor_to_img
import torchvision.transforms as transforms
import os
from PIL import Image
from torchvision.transforms.functional import rotate
import torch
from torch import Tensor

def rotate90(img):
    # print(img.size)
    img =  rotate(img,90,expand=True)
    # print(img.size)
    return img
def dummy(img):
    return img
class MainBodyGetter():
    def __init__(self,height,width,pad_width, debug_dir):
        self.height = height
        self.width = width
        self.debug_dir = debug_dir
        self.c = 0
        # 宽度的边框
        self.pad_width = pad_width
    def transformImg(self, img:Image):
        self.c = self.c+1
        com_list = []
        # 1.先让图片横过来
        img_height = img.height
        img_width = img.width
        if img.height > img.width:
            com_list.append(rotate90)
            img_height = img.width
            img_width = img.height
        else:
            com_list.append(dummy)
        # 此时width应该大于height
        # 2.按照比例，重新计算height,并进行缩放
        height_to = img_height * self.width/img_width
        height_to = round(height_to)
        com_list.append(
            transforms.Resize((height_to, self.width))
        )
        # 3.给图片加上边框
        com_list.append(
            transforms.Pad((self.pad_width,self.pad_width,self.pad_width, self.pad_width))
        )

        org_img = img
        # print(type(img))
        for i in range(len(com_list)):
            t = com_list[i]
            img = t(img)
            # print(type(img))
            # dir = os.path.join(self.debug_dir,"main"+str(i))
            # if not os.path.exists(dir):
            #     os.makedirs(dir)
            # filename = str(self.c)+".jpg"
            # img.save(os.path.join(dir,filename))
        return img
    
class ChangeShape():
    
    def __init__(self,height,width, debug_dir):
        self.height = height
        self.width = width
        self.debug_dir = debug_dir
        self.c = 0
    def rotate90_align(self):
        def func(img):
            img = rotate(img,90,expand=True)
            a = self.align(img,self.height,self.width)
            img = a(img)
            return img
        return func
    # 将一个图片变形，返回多个图片的列表
    def change(self,img_tensor):
        # 最终的结果，用来存放Tensor
        img_list = []
        # 存放转换列表
        com_list = []
        # 0.原始图片
        com_list.append(self.align(img_tensor,self.width,self.height))
        # 1.转90度
        com_list.append(self.rotate90_align())
        # 2.转180度
        # 3.转270度
        for i in range(len(com_list)):
            t = com_list[i]
            # print(type(img_tensor))
            img = t(img_tensor)
            # print(type(img))
            img_list.append(torch.stack([img]))

            # save tensor to imgfile
            dir = os.path.join(self.debug_dir,str(i))
            save_tensor_to_img(img, dir, str(self.c)+".jpg")
        return img_list
    # 将图片调整为指定大小
    def align(self,img:Tensor,width_to,height_to):
        bottom_pad = height_to - img.shape[1]
        right_pad = width_to - img.shape[2]
        # https://pytorch.org/vision/stable/generated/torchvision.transforms.Pad.html
        pad = transforms.Pad((0, 0, right_pad, bottom_pad))
        return pad
    

