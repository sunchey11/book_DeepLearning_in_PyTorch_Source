import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
training_data = datasets.FashionMNIST(
    root="D:\\pytorch_data\\fashion-mnist\\data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="D:\\pytorch_data\\fashion-mnist\\data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    # print(img.shape)
    # squeeze去掉维度为1的维度,本来shape=[1,32,32]
    # 调用后变为[32,32]
    img = img.squeeze()
    # print(img)
    # print(img.shape)
    plt.imshow(img, cmap="gray")
plt.show()