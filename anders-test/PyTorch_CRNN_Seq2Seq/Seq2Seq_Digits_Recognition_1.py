# https://codingvision.net/pytorch-crnn-seq2seq-digits-recognition-ctc
# https://colab.research.google.com/drive/1ZS8Mk7v75O9baOxfexKS5s-Ykafv1nWf#scrollTo=sXSjoltzYTV0
#@title Dataset Generator
#@markdown Generates **random sequences** of equal length using digits from the **EMNIST** dataset while also performing basic augmentation.
#@markdown Saves results in **dataset/** directory: *data_\<seq_len\>_\<num_seq\>.npy* (images) and *labels_\<seq_len\>_\<num_seq\>.npy* (labels/GTs).
import torch
from torch import Tensor
from torchvision import datasets, transforms
import numpy as np
import os
from PIL import Image
debug = True
def d_print(s):
    if(debug):
        print(s)
# 保存灰度图片，img.shape=(28, 140)
def save_grey_tensor_to_img(img:Tensor,dir,filename):

    d_print(img.shape) #(28, 140)
    arr = np.asarray(img) 
    print(arr.shape)
    # arr = np.transpose(arr, (1, 2, 0))
    print(arr.shape)
    print(arr[10][10])
    # arr里面放的小数,需要转成1-255
    arr255 = arr*255
    # 类型一定要正确
    arr255 = arr255.astype(np.uint8)

    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray
    # L (8-bit pixels, grayscale)
    rimg = Image.fromarray(arr255,mode="L")
    # p = rimg.getpixel((10, 10))
    # print(p)
    if not os.path.exists(dir):
        os.makedirs(dir)
    rimg.save(os.path.join(dir, filename))

file_dir = os.path.split(os.path.abspath(__file__))[0]
output_path = os.path.join(file_dir, 'dataset')

def saveImg(imgTensor: Tensor,labels: Tensor, index):
  #  imgTensor.shape = (28, 140),值为像素值(0,1)之间
  # lables.shape = (5,),为对应的5个图片的数字
  l = labels.tolist()
  l = [str(l[j]) for j in range(len(l))]
  file_name = '_'.join(l)
  img_dir = os.path.join(output_path, 'imgs')
  save_grey_tensor_to_img(imgTensor, img_dir, str(index)+"__"+file_name+".png")
# 生成图片数据和labels数据，保存到dataset目录下
digits_per_sequence = 5
n_of_sequences = 10000





emnist_dataset = datasets.EMNIST('D:\\pytorch_data\\emnist\\data', split="digits", train=True, download=True)

# 类型为list,里面放的是tensor，shape 为（28,140),表示的是一个黑白图片，
# 但是奇怪的是，像素值为0到1/255之间
dataset_sequences = []
# 类型为list,里面放的是tensor， shape为[5],表示的是5个数字
dataset_labels = []
dataset_len = len(emnist_dataset.data)
print(dataset_len) #240000条数据
for i in range(n_of_sequences):
  random_indices = np.random.randint(dataset_len, size=(digits_per_sequence))
  d_print(random_indices) #一维数组，里面有5个随机数
  random_digits_images = emnist_dataset.data[random_indices]
  # random_digits_images放了5个图片，每个图片大小为28*28
  d_print(random_digits_images.shape) # 5,28,28
  transformed_random_digits_images = []
  
  for img in random_digits_images:
    # 转换为'PIL.Image.Image'
    _temp_img = transforms.ToPILImage()(img)
    d_print(type(_temp_img)) # 转换为'PIL.Image.Image'

    # fix EMNIST's transposed images
    _temp_img = transforms.functional.rotate(_temp_img, -90, fill=(0,))
    _temp_img = transforms.functional.hflip(_temp_img)

    # basic augmentation on each EMNIST digit
    _temp_img = transforms.RandomAffine(degrees=10, translate=(0.2, 0.15), scale=(0.8, 1.1))(_temp_img)
    _temp_img = transforms.ToTensor()(_temp_img).numpy()
    #temp_img此时为tensor数组，shape=1,28,28
    transformed_random_digits_images.append(_temp_img)

  random_digits_images = np.array(transformed_random_digits_images)
  d_print(random_digits_images.shape) #(5, 1, 28, 28)
  
  random_digits_labels = emnist_dataset.targets[random_indices]
  d_print(random_digits_labels.shape) #一维数组，有5个元素

  temp_r = random_digits_images.reshape(digits_per_sequence, 28, 28)
  d_print(temp_r.shape) #(5, 28, 28)
  random_sequence = np.hstack(temp_r)
  d_print(random_sequence.shape) #(28, 140)

  temp_l = random_digits_labels.reshape(digits_per_sequence, 1)
  d_print(temp_l.shape)#[5, 1]
  random_labels = np.hstack(temp_l)

  d_print(random_labels.shape) #（5,)

  # 保存图片文件，方便浏览
  saveImg(random_sequence,random_labels, i)

  random_sequence_255 = random_sequence / 255
  d_print(random_sequence_255.shape)
  d_print(random_sequence_255)
  dataset_sequences.append(random_sequence_255)
  dataset_labels.append(random_labels)

if not os.path.exists(output_path): 
  os.makedirs(output_path)

dataset_data = np.array(dataset_sequences)
dataset_labels = np.array(dataset_labels)

data_path = output_path + "data_" + str(digits_per_sequence) + "_" + str(n_of_sequences) + ".npy"
np.save(data_path, dataset_data)

label_path = output_path + "labels_" + str(digits_per_sequence) + "_" + str(n_of_sequences) + ".npy"
np.save(label_path, dataset_labels)

