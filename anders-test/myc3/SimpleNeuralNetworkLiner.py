
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # 读取csv文件的库
from SimpleNeuralNetwork import SimpleNeuralNetwork
import torch
import torch.optim as optim
import os 

"""使用SimpleNeuralNetwork预测线性数据"""
class SimpleNeuralNetworkLiner(SimpleNeuralNetwork):
    """Represent aspects of a car, specific to electric vehicles."""
    
    def __init__(self, data_frame:pd.DataFrame, dummy_fields,fields_to_drop, quant_features,target_fields, train_dataCnt):
        """
        调用父类构造器
        """
        super().__init__( data_frame, dummy_fields,fields_to_drop, quant_features,target_fields, train_dataCnt)
    def labelX(self, ax):
        pass

# 测试
file_dir = os.path.split(os.path.abspath(__file__))[0]
data_path = os.path.join(file_dir, "liner-data1.csv")

rides = pd.read_csv(data_path)

print(rides.shape)
rowCount = rides.shape[0]

dummy_fields = []
fields_to_drop = []
quant_features = ['x','y']
target_fields = ['y']
nn:SimpleNeuralNetworkLiner = SimpleNeuralNetworkLiner(rides,dummy_fields,fields_to_drop,quant_features,target_fields,150)
nn.prepareData()
nn.train()
# nn.showLosses()
# nn.showTest()
rides2 = pd.read_csv(data_path)
nn.analyseData(rides2)
values = nn.predictData([[150],[160]])
print(values)
print(type(values))