from dataset_save_utils import save_dataset, save_root
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os

import scipy.io as io
training_data = datasets.Flowers102(root='D:\\pytorch_data\\flowers102\\data',
                                 split="train",
                                 download=True)

test_data = datasets.Flowers102(
    root='D:\\pytorch_data\\flowers102\\data',
    split="test",
    download=True,
)
# matr = io.loadmat('D:\\pytorch_data\\flowers102\\data\\flowers-102\\imagelabels.mat') 
# print(matr)
# matr2 = io.loadmat('D:\\pytorch_data\\flowers102\\data\\flowers-102\\setid.mat') 
# print(matr)
# labels_map = matr.labels

save_dataset(training_data, None,
             os.path.join(save_root, "flowers102", "train"))
save_dataset(test_data, None, os.path.join(save_root, "flowers102", "test"))
print('finished')