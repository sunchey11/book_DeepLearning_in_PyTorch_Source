from dataset_save_utils import save_dataset, save_root
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
# 报下面这个错，没解决
# The archive ILSVRC2012_devkit_t12.tar.gz is not present in the root directory or is corrupted. You need to download it externally and place it in D:\pytorch_data\image_net\data.

training_data = datasets.ImageNet(root='D:\\pytorch_data\\image_net\\data',
                                 train=True,
                                 download=True,
                                 split = "val"
                                 )

test_data = datasets.ImageNet(
    root='D:\\pytorch_data\\image_net\\data',
    train=False,
    download=True,
)

labels_map = training_data.classes

save_dataset(training_data, labels_map,
             os.path.join(save_root, "image_net", "train"))
save_dataset(test_data, labels_map, os.path.join(save_root, "image_net", "test"))
print('finished')