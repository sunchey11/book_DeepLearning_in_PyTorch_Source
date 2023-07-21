# https://codingvision.net/pytorch-crnn-seq2seq-digits-recognition-ctc
# https://colab.research.google.com/drive/1ZS8Mk7v75O9baOxfexKS5s-Ykafv1nWf#scrollTo=sXSjoltzYTV0
#@title Dataset Generator
#@markdown Generates **random sequences** of equal length using digits from the **EMNIST** dataset while also performing basic augmentation.
#@markdown Saves results in **dataset/** directory: *data_\<seq_len\>_\<num_seq\>.npy* (images) and *labels_\<seq_len\>_\<num_seq\>.npy* (labels/GTs).
import torch
from torchvision import datasets, transforms
import numpy as np
import os

digits_per_sequence = 5
n_of_sequences = 10000
output_path = './dataset/'

emnist_dataset = datasets.EMNIST('D:\\pytorch_data\\emnist\\data', split="digits", train=True, download=True)

dataset_sequences = []
dataset_labels = []
dataset_len = len(emnist_dataset.data)
print(dataset_len) #240000条数据
for i in range(n_of_sequences):
  random_indices = np.random.randint(dataset_len, size=(digits_per_sequence))
  print(random_indices) #一维数组，里面有5个随机数
  random_digits_images = emnist_dataset.data[random_indices]
  # random_digits_images放了5个图片，每个图片大小为28*28
  print(random_digits_images.shape) # 5,28,28
  transformed_random_digits_images = []
  
  for img in random_digits_images:
    # 转换为'PIL.Image.Image'
    _temp_img = transforms.ToPILImage()(img)
    print(type(_temp_img)) # 转换为'PIL.Image.Image'

    # fix EMNIST's transposed images
    _temp_img = transforms.functional.rotate(_temp_img, -90, fill=(0,))
    _temp_img = transforms.functional.hflip(_temp_img)

    # basic augmentation on each EMNIST digit
    _temp_img = transforms.RandomAffine(degrees=10, translate=(0.2, 0.15), scale=(0.8, 1.1))(_temp_img)
    _temp_img = transforms.ToTensor()(_temp_img).numpy()
    #temp_img此时为tensor数组，shape=1,28,28
    transformed_random_digits_images.append(_temp_img)

  random_digits_images = np.array(transformed_random_digits_images)
  print(random_digits_images.shape) #(5, 1, 28, 28)
  
  random_digits_labels = emnist_dataset.targets[random_indices]
  print(random_digits_labels.shape) #一维数组，有5个元素

  temp_r = random_digits_images.reshape(digits_per_sequence, 28, 28)
  print(temp_r.shape) #(5, 28, 28)
  random_sequence = np.hstack(temp_r)
  print(random_sequence.shape) #(28, 140)

  temp_l = random_digits_labels.reshape(digits_per_sequence, 1)
  print(temp_l.shape)#[5, 1]
  random_labels = np.hstack(temp_l)

  print(random_labels.shape) #（5,)

  dataset_sequences.append(random_sequence / 255)
  dataset_labels.append(random_labels)

if not os.path.exists(output_path): 
  os.makedirs(output_path)

dataset_data = np.array(dataset_sequences)
dataset_labels = np.array(dataset_labels)

data_path = output_path + "data_" + str(digits_per_sequence) + "_" + str(n_of_sequences) + ".npy"
np.save(data_path, dataset_data)

label_path = output_path + "labels_" + str(digits_per_sequence) + "_" + str(n_of_sequences) + ".npy"
np.save(label_path, dataset_labels)

