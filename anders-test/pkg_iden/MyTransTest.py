from MyTrans import MainBodyGetter
from PkgIdenNetC5 import d_print,PkgIdenNet,img_width,img_height
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import time

file_dir = os.path.split(__file__)[0]
print(file_dir)
data_path = os.path.join(file_dir, "./data")
debug_dir = os.path.join(file_dir, "./debug_dir")

ts = MainBodyGetter(img_height, img_width, debug_dir)
def mytrans(img):
    img = ts.transformImg(img)
    return img
# https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'),
                                     transforms.Compose([
                                        mytrans,
                                        transforms.ToTensor(),
                                    ])
                                    )
for i in range(len(train_dataset)):
    t = train_dataset[i]
    print(t)
print("finished")