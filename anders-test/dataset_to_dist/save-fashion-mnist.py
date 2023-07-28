from dataset_save_utils import save_dataset,save_root
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
training_data = datasets.FashionMNIST(
    root="D:\\pytorch_data\\fashion-mnist\\data",
    train=True,
    download=True,
)

test_data = datasets.FashionMNIST(
    root="D:\\pytorch_data\\fashion-mnist\\data",
    train=False,
    download=True,
)

labels_map = training_data.classes


save_dataset(training_data, labels_map, os.path.join(save_root,"FashionMNIST","train"))
save_dataset(test_data, labels_map, os.path.join(save_root,"FashionMNIST","test"))
print('finished')