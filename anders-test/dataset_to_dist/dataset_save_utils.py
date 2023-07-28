import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
"""将一个dataset保存到硬盘"""
def save_dataset(dataset, classes, path):
    for i in range(len(dataset)):
        img, label = dataset[i]
        cname = str(label)
        if classes is not None:
            cname = classes[label]
        
        file_dir = os.path.join(path, cname)

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        filename = os.path.join(file_dir, str(i)+'.png')
        img.save(filename)
        if i % 1000 ==0:
            print(str(i)+"/"+str(len(dataset))," saved")
    print("all "+str(len(dataset))," saved")
save_root = "D:\\pytorch_data\\to_png"