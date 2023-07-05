from C3BikeNeuralNetwork import C3BikeNeuralNetwork
import numpy as np
import pandas as pd #读取csv文件的库
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import os 

# 此文件所在的路径
print(os.path.abspath(__file__))
print(os.path.split(os.path.abspath(__file__)))
file_dir = os.path.split(os.path.abspath(__file__))[0]
data_path = os.path.join(file_dir,'../', "Bike-Sharing-Dataset/hour.csv")


#读取数据到内存中，rides为一个dataframe对象
rides = pd.read_csv(data_path)
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
quant_features = ['cnt', 'temp', 'hum', 'windspeed']
target_fields = ['cnt', 'casual', 'registered']
nn:C3BikeNeuralNetwork = C3BikeNeuralNetwork(rides,dummy_fields,quant_features,target_fields,16875)
nn.prepareData()
nn.train()
# nn.showLosses()
nn.showTest()
values = nn.predictData([])
print(values)
print(type(values))
