# https://codingvision.net/pytorch-crnn-seq2seq-digits-recognition-ctc
# https://colab.research.google.com/drive/1ZS8Mk7v75O9baOxfexKS5s-Ykafv1nWf#scrollTo=sXSjoltzYTV0
# 定义模型，训练模型，测试模型
#@title CRNN Model
#@markdown Applies a LeNet-5 based features extraction using strided convolutions instead of maxpooling.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init


import torch.utils.data as data_utils
import numpy as np
import torch
import os
from CRNN import CRNN,d_print


file_dir = os.path.split(os.path.abspath(__file__))[0]
dataset_path = os.path.join(file_dir, 'dataset')

data_path = os.path.join(dataset_path, 'data_5_10000.npy')
labels_path = os.path.join(dataset_path, 'labels_5_10000.npy')


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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

# crnn = CRNN().to(device)
crnn = CRNN()
criterion = nn.CTCLoss(blank=10, reduction='mean', zero_infinity=True)
optimizer = torch.optim.Adam(crnn.parameters(), lr=0.001)

BLANK_LABEL = 10
MAX_EPOCHS = 10

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
    batch_size = x_test.shape[0]
    crnn.reset_hidden(batch_size)

    x_test = x_test.view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])

    y_pred = crnn(x_test)
    y_pred = y_pred.permute(1, 0, 2)

    input_lengths = torch.IntTensor(batch_size).fill_(crnn.postconv_width)
    target_lengths = torch.IntTensor([len(t) for t in y_test])

    loss = criterion(y_pred, y_test, input_lengths, target_lengths)

    total_loss += loss.detach().numpy()

    _, max_index = torch.max(y_pred, dim=2)

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

def train():

  correct = 0
  total = 0

  total_loss = 0
  num_batches = 0

  for batch_id, (x_train, y_train) in enumerate(train_loader):

    # 部署到DEVICE上去
    # x_train, y_train = x_train.to(device), y_train.to(device)
    batch_size = x_train.shape[0]
    crnn.reset_hidden(batch_size)

    d_print(x_train.shape) # torch.Size([64, 28, 140])
    x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
    d_print(x_train.shape) # torch.Size([64, 1, 28, 140])

    optimizer.zero_grad()

    y_pred = crnn(x_train)
    d_print(y_pred.shape) # torch.Size([64, 31, 11])

    y_pred = y_pred.permute(1, 0, 2)
    d_print(y_pred.shape)  # torch.Size([31, 64, 11])

    input_lengths = torch.IntTensor(batch_size).fill_(crnn.postconv_width)
    d_print(input_lengths.shape) # torch.Size([64])
    target_lengths = torch.IntTensor([len(t) for t in y_train])
    d_print(target_lengths.shape)

    # input_length里全是31，target_lengths里全是5
    loss = criterion(y_pred, y_train, input_lengths, target_lengths)
    total_loss += loss.detach().numpy()

    loss.backward()
    optimizer.step()

    _, max_index = torch.max(y_pred, dim=2)

    for i in range(batch_size):
      raw_prediction = list(max_index[:, i].numpy())

      prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != BLANK_LABEL])

      if len(prediction) == len(y_train[i]) and torch.all(prediction.eq(y_train[i])):
        correct += 1
      total += 1

    num_batches += 1

  ratio = correct / total
  print('TRAIN correct: ', correct, '/', total, ' P:', ratio)

  return total_loss / num_batches

list_training_loss = []
list_testing_loss = []

for epoch in range(MAX_EPOCHS):

  training_loss = train()
  testing_loss = test()

  list_training_loss.append(training_loss)
  list_testing_loss.append(testing_loss)

  if epoch == 5:
    print('training loss', list_training_loss)
    print('testing loss', list_testing_loss)
    break

model_path = os.path.join(file_dir, 'seq2seq_digits_recognition.pth')

torch.save(crnn.state_dict(), model_path)