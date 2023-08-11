from dataset_save_utils import save_dataset,save_root
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

training_data = datasets.MNIST(root='D:\\pytorch_data\\mnist\\data',  #文件存放路径
                            train=True,   #提取训练集
                            
                            download=True) #当找不到文件的时候，自动下载

# 加载测试数据集
test_dataset = datasets.MNIST(root='D:\\pytorch_data\\mnist\\data', 
                           train=False, 
                           download=True,
                           )


labels_map = training_data.classes


save_dataset(training_data, labels_map, os.path.join(save_root,"MNIST","train"))
save_dataset(test_dataset, labels_map, os.path.join(save_root,"MNIST","test"))
print('finished')