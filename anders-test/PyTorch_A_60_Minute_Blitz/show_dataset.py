from multiprocessing import freeze_support
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
# 将4个图片做成一个网格显示
# torchvision.utils.make_grid
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='D:\\pytorch_data\\cifar10\\data', train=True,
                                        download=True, transform=transform)
# shuffle控制是不是随机的
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    print(npimg.shape)
    # 轴变换 
    npimg = np.transpose(npimg, (1, 2, 0))
    print(npimg.shape)
    plt.imshow(npimg)
    plt.show()

# get some random training images
dataiter = iter(trainloader)
# 为啥会是随机的，不是第一个， shuffle控制是不是随机的
images, labels = next(dataiter)
# 维度和batch_size有关，batch_size为4，显示4个图片
print(images)
print(labels)
img,label = trainset[0]
print(img)
print(label)

# show images
print(images.shape)
# 将多个图片，合成为一个网格图片
x = torchvision.utils.make_grid(images)
print(x.shape)
imshow(x)
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))