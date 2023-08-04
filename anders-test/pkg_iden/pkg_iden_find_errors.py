# 测试模型,找出不能正确识别的图片,
# 将不能识别的图片保存到D:\GitHub\book_DeepLearning_in_PyTorch_Source\anders-test\pkg_iden\data\test2_errors\dangshen-targets



import time
import PIL
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torchvision import datasets, models, transforms
from PkgIdenNetC5 import d_print,PkgIdenNet,img_width,img_height
from PIL import Image
from rembg import remove,new_session

file_dir = os.path.split(__file__)[0]
print(file_dir)
data_path = os.path.join(file_dir, "./data")

session = new_session("u2netp")
def remove_bg(img):
    print(type(img))
    print(img.mode)
    img = remove(img, session=session)
    print(type(img))
    print(img.mode)
    img = img.convert("RGB")
    print(type(img))
    print(img.mode)
    return img

train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'),
                                     transforms.Compose([
                                        
                                        transforms.Resize((img_height,img_width)),
                                        transforms.ToTensor(),
                                        
                                    ])
                                    )
# https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
test_dataset = datasets.ImageFolder(os.path.join(data_path, 'test'),
                                     transforms.Compose([
                                        # remove_bg,
                                        transforms.Resize((img_height,img_width)),
                                        transforms.ToTensor(),
                                        
                                    ])
                                    )
t1 = test_dataset[0]

print(len(t1))
print(t1[0]) #这是图片数据
print(t1[1]) #这是一个整数，即label的index

# 所有的文件
print(type(test_dataset.imgs))
print(len(test_dataset.imgs))
print(type(test_dataset.imgs[0]))




batch_size = 1
print(test_dataset.classes)


# ['39', '620', 'aoli', 'eber', 'fengshi', 'kouzhao', 'kushen', 'lianhua', 'ningjiao', 'nut', 'shangtong', 'yikang', 'zhuangyao', 'zhuodu']
test_classes = test_dataset.classes
train_classes = train_dataset.classes

import torch.nn as nn
import torch.nn.functional as F



import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    print(npimg.shape)
    npimg = np.transpose(npimg, (1, 2, 0))
    print(npimg.shape)
    plt.imshow(npimg)
    plt.show()




file_dir = os.path.split(os.path.abspath(__file__))[0]
PATH = os.path.join(file_dir, './pkg_iden.pth')
error_path = os.path.join(file_dir, 'data',"test3_errors")
net = PkgIdenNet()
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    t1 = test_dataset[0]

    # print(len(t1)) #数组两个元素
    # print(t1[0]) #这是图片数据
    # print(t1[1]) #这是一个整数，即label的index

    # 所有的文件列表，包含文件名，和label index
    # print(type(test_dataset.imgs))
    # print(len(test_dataset.imgs))
    # print(type(test_dataset.imgs[0]))
    # print(test_dataset.imgs[5])

    for i in range(len(test_dataset)):
        data = test_dataset[i]
        img, labels = data

        # 将tensor变成一个单元素数组
        images = torch.stack([img])
        # 将整数变成一个单元素数组
        labels = torch.tensor([labels])
        # calculate outputs by running images through the network

        
        start=time.time()
 
        outputs = net(images)
        end=time.time()
        # 时间为0.05秒
        print('程序运行时间为: %s Seconds'%(end-start))
        
        print(outputs.shape) 
        print(outputs)
        # 将数据转为0到1之间的概率，总和为1
        softmax = nn.Softmax(dim=1)

        outputs = softmax(outputs)
        # print(outputs)
        # print(outputs[0].sum()) # 概率和应为1


        _, predicted = torch.max(outputs, 1)

        # print(type(_)) #<class 'torch.Tensor'>
        # print(_.shape)
        # _里面放的是概率，如果能预测，则概率>0.99
        # 如果不能预测，则概率是0.6779
        # print(_)
        print(predicted.shape)
        print(predicted)
        print('Predicted: ', predicted[0])
        real_lable_text = test_classes[labels[0]]
        pred_lable_text = train_classes[predicted[0]]

        if _[0]>0.9 and real_lable_text == pred_lable_text:
            correct += 1
            fn = test_dataset.imgs[i]
            print("ok", fn)
            print("index", i)
            print("rate",_[0])
            print("\n")
            images = torchvision.utils.make_grid(images)
            # imshow(images)
        else:

            fn = test_dataset.imgs[i]
            print("error", fn)
            print("index", i)
            print("rate",_[0])
            print("should be", real_lable_text)
            print("but it is", pred_lable_text)
            print("\n")
            images = torchvision.utils.make_grid(images)
            # imshow(images)

            # 保存到硬盘上

            print(img.shape) #[3, 500, 500]
            arr = np.asarray(img) 
            print(arr.shape)
            arr = np.transpose(arr, (1, 2, 0))
            print(arr.shape)
            print(arr[10][10])

            # arr里面放的小数
            arr255 = arr*255
            # 类型一定要正确
            arr255 = arr255.astype(np.uint8)
            print(arr255[10][10])

            rimg = Image.fromarray(arr255,mode="RGB")
            print(type(fn))
            p = rimg.getpixel((10, 10))
            print(p)
            temp_arr = os.path.split(fn[0])
            file_name = temp_arr[len(temp_arr)-1]
            cat_dir = os.path.join(error_path,real_lable_text)
            if not os.path.exists(cat_dir):
                os.makedirs(cat_dir)
            rimg.save(os.path.join(cat_dir, file_name))
        total += 1


print(f'Accuracy of the network on the test images: {100 * correct // total} %')

