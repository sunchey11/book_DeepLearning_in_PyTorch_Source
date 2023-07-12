
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision import datasets, models, transforms
import os
from torchvision.models import resnet18, ResNet18_Weights


# 数据存储总路径
file_dir = os.path.split(os.path.abspath(__file__))[0]
dog1_path = os.path.join(file_dir, "assets/dog1.jpg")
dog2_path = os.path.join(file_dir, "assets/dog2.jpg")
label_path = os.path.join(file_dir, "assets/imagenet_class_index.json")

dog1 = read_image(dog1_path)
dog2 = read_image(dog2_path)


weights = ResNet18_Weights.DEFAULT
transforms = weights.transforms()

# 加不加eval，结果不一样，为什么？
net = models.resnet18(pretrained=True).eval()
# net = models.resnet18(pretrained=True)
batch = torch.stack([dog1, dog2])

x = transforms(batch)
y1 = net(x)
res = y1.argmax(dim=1)


with open(label_path) as labels_file:
    labels = json.load(labels_file)

print(type(labels))
for i in res:
    print(i)
    print(labels[str(i.item())])

print('finished!')