import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
# 根据这个网址改编的
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
training_data = datasets.CIFAR10(
    root="D:\\pytorch_data\\cifar10\\data",
    train=True,
    download=True,
    transform=ToTensor()
)


test_data = datasets.CIFAR10(
    root="D:\\pytorch_data\\cifar10\\data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: 'plane',
    1: 'car',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    print(img.shape)
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    print(npimg.shape)

    plt.imshow(npimg, cmap="gray")
plt.show()
print("ok1")