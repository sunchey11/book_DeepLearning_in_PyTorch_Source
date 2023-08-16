# https://codingvision.net/pytorch-crnn-seq2seq-digits-recognition-ctc
# https://colab.research.google.com/drive/1ZS8Mk7v75O9baOxfexKS5s-Ykafv1nWf#scrollTo=sXSjoltzYTV0
# 测试不要五个图片，是否能预测
#@title CRNN Model
#@markdown Applies a LeNet-5 based features extraction using strided convolutions instead of maxpooling.

# https://codingvision.net/pytorch-crnn-seq2seq-digits-recognition-ctc
# https://colab.research.google.com/drive/1ZS8Mk7v75O9baOxfexKS5s-Ykafv1nWf#scrollTo=sXSjoltzYTV0
#@title Dataset Generator
#@markdown Generates **random sequences** of equal length using digits from the **EMNIST** dataset while also performing basic augmentation.
#@markdown Saves results in **dataset/** directory: *data_\<seq_len\>_\<num_seq\>.npy* (images) and *labels_\<seq_len\>_\<num_seq\>.npy* (labels/GTs).
import torch
from torchvision import datasets, transforms
import numpy as np
import os

digits_per_sequence = 1
n_of_sequences = 1000

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


dataset_data = np.array(dataset_sequences)
dataset_labels = np.array(dataset_labels)










import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init


import torch.utils.data as data_utils
import numpy as np
import torch
import matplotlib.pyplot as plt
from CRNN import CRNN,d_print

data = torch.Tensor(dataset_data)
print(data.shape)  #[10000,28,140]
labels = torch.IntTensor(dataset_labels)
print(labels.shape)  #[10000,5]
seq_dataset = data_utils.TensorDataset(data, labels)
test_set = seq_dataset

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)


# 显示10张图片
number_of_printed_imgs = 10

for batch_id, (x_test, y_test) in enumerate(test_loader):
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



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

# crnn = CRNN().to(device)
crnn = CRNN()

file_dir = os.path.split(os.path.abspath(__file__))[0]
file_path = os.path.join(file_dir, "seq2seq_digits_recognition.pth")



crnn.load_state_dict(torch.load(file_path))

criterion = nn.CTCLoss(blank=10, reduction='mean', zero_infinity=True)
optimizer = torch.optim.Adam(crnn.parameters(), lr=0.001)

BLANK_LABEL = 10
MAX_EPOCHS = 1

#@title Training & Testing


from itertools import groupby

def test():
  correct = 0
  total = 0
  num_batches = 0

  total_loss = 0
  for batch_id, (x_test, y_test) in enumerate(test_loader):
    # 部署到DEVICE上去
    # x_test, y_test = x_test.to(device), y_test.to(device)

    print(x_test.shape) #[1, 28, 140]
    print(y_test.shape) #[1, 5]
    batch_size = x_test.shape[0]
    crnn.reset_hidden(batch_size)

    x_test = x_test.view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])
    print(x_test.shape) # [1, 1, 28, 140]

    y_pred = crnn(x_test)
    print(y_pred.shape) #[1, 31, 11]

    y_pred = y_pred.permute(1, 0, 2)
    print(y_pred.shape) #[31, 1, 11]

    input_lengths = torch.IntTensor(batch_size).fill_(crnn.postconv_width)
    target_lengths = torch.IntTensor([len(t) for t in y_test])

    # loss = criterion(y_pred, y_test, input_lengths, target_lengths)

    # total_loss += loss.detach().numpy()

    _, max_index = torch.max(y_pred, dim=2)
    # print(_)
    # print(max_index)

    for i in range(batch_size):
      raw_prediction = list(max_index[:, i].numpy())

      prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != BLANK_LABEL])

      if len(prediction) == len(y_test[i]) and torch.all(prediction.eq(y_test[i])):
        correct += 1
      total += 1
    num_batches += 1

  ratio = correct / total
  print('TEST correct: ', correct, '/', total, ' P:', ratio)

  return total_loss / num_batches


list_training_loss = []
list_testing_loss = []

for epoch in range(MAX_EPOCHS):

  testing_loss = test()

  list_testing_loss.append(testing_loss)

  if epoch == 5:
    print('training loss', list_training_loss)
    print('testing loss', list_testing_loss)
    break

print('finished')