from dataset_save_utils import save_dataset, save_root
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os

training_data = datasets.CIFAR10(root='D:\\pytorch_data\\cifar10\\data',
                                 train=True,
                                 download=True)

test_data = datasets.CIFAR10(
    root='D:\\pytorch_data\\cifar10\\data',
    train=False,
    download=True,
)

labels_map = training_data.classes

save_dataset(training_data, labels_map,
             os.path.join(save_root, "cifar10", "train"))
save_dataset(test_data, labels_map, os.path.join(save_root, "cifar10", "test"))
print('finished')