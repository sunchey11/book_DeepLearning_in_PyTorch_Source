# https://codingvision.net/pytorch-crnn-seq2seq-digits-recognition-ctc
# https://colab.research.google.com/drive/1ZS8Mk7v75O9baOxfexKS5s-Ykafv1nWf#scrollTo=sXSjoltzYTV0
#@title Dataset Generator
#@markdown Generates **random sequences** of equal length using digits from the **EMNIST** dataset while also performing basic augmentation.
#@markdown Saves results in **dataset/** directory: *data_\<seq_len\>_\<num_seq\>.npy* (images) and *labels_\<seq_len\>_\<num_seq\>.npy* (labels/GTs).
import torch
from torchvision import datasets, transforms
import numpy as np
import os

# ---2
#@title Dataset Loader
#@markdown Loads the generated dataset and creates 2 **DataLoader** instances (for training and testing)

import torch.utils.data as data_utils
import numpy as np
import torch

data_path = './dataset/data_5_10000.npy'
labels_path = './dataset/labels_5_10000.npy'

data = np.load(data_path)
print(type(data))
data = torch.Tensor(data)
print(data.shape)  #[10000,28,140]
labels = torch.IntTensor(np.load(labels_path).astype(int))
print(labels.shape)  #[10000,5]
seq_dataset = data_utils.TensorDataset(data, labels)
train_set, test_set = torch.utils.data.random_split(seq_dataset, [int(len(seq_dataset)*0.8), int(len(seq_dataset)*0.2)])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32*2, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)


#@title Debug: Show Dataset Images
import matplotlib.pyplot as plt

# 显示10张图片
number_of_printed_imgs = 10

for batch_id, (x_test, y_test) in enumerate(train_loader):
  # x_test是一个batch的数据，shape=[64,28,140]
  # y_test是一个batch的数据，shape=[64,5]
  for j in range(len(x_test)):
    # 每一个图片都是5个数字组成的
    plt.imshow(x_test[j], cmap='gray')
    plt.show()

    print(y_test[j])
    number_of_printed_imgs -= 1

    if number_of_printed_imgs <= 0:
      break

  if number_of_printed_imgs <= 0:
      break
