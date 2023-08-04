from torch import Tensor
from PIL import Image
import numpy as np
import os

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