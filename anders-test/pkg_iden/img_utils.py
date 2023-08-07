from torch import Tensor
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
from Constants import img_width,img_height

def save_tensor_to_img(img:Tensor,dir,filename):

    print(img.shape) #[3, 500, 500]
    arr = np.asarray(img) 
    print(arr.shape)
    arr = np.transpose(arr, (1, 2, 0))
    print(arr.shape)
    print(arr[10][10])
    # arr里面放的小数,需要转成1-255
    arr255 = arr*255
    # 类型一定要正确
    arr255 = arr255.astype(np.uint8)

    rimg = Image.fromarray(arr255,mode="RGB")
    # p = rimg.getpixel((10, 10))
    # print(p)
    if not os.path.exists(dir):
        os.makedirs(dir)
    rimg.save(os.path.join(dir, filename))

center_crop_ratio = 0.875
# 看看人家的，都是这样搞得
# https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/
image_transforms = { 
    'train': transforms.Compose([
        # scale参数为截取面积与原面积的比值：（最小值，最大值）
        transforms.RandomResizedCrop(size=(img_height,img_width), scale=(0.8, 1.0)),
        # degree为+15 ,-15
        transforms.RandomRotation(degrees=15),
        # 随机就两种情况，翻转和不翻转
        transforms.RandomHorizontalFlip(),
        # p为被变化的概率（0到1），默认为0.5
        # fill为空白处的颜色
        # distortion_scale ，变化程度。0：没变化，1：扭曲的不成样子，0.2稍稍变化
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.ColorJitter(brightness=(0.9,1.1), contrast = (0.9,1.1),saturation = (0.9,1.1), hue=(0,0)),
        transforms.CenterCrop(size=(img_height*center_crop_ratio,img_width*center_crop_ratio)),
        transforms.ToTensor(),
        
    ]),
    # valid没有用，没有改，保持原样
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=(img_height,img_width)),
        transforms.CenterCrop(size=(img_height*center_crop_ratio,img_width*center_crop_ratio)),
        transforms.ToTensor()
        
    ])
}

debug = False
def d_print(s):
    if(debug):
        print(s)