
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # 读取csv文件的库
from SimpleNeuralNetwork import SimpleNeuralNetwork
import torch
import torch.optim as optim
import os 

"""使用SimpleNeuralNetwork预测上班时间"""
class SimpleNeuralNetworkWorkTime(SimpleNeuralNetwork):
    """Represent aspects of a car, specific to electric vehicles."""
    
    def __init__(self, data_frame:pd.DataFrame, dummy_fields,fields_to_drop, quant_features,target_fields, train_dataCnt):
        """
        调用父类构造器
        """
        super().__init__( data_frame, dummy_fields,fields_to_drop, quant_features,target_fields, train_dataCnt)
    def labelX(self, ax):
        pass
    def debugData(self,orgData:pd.DataFrame, real_predict:np.ndarray, real_targets:np.ndarray):
        orgArray:np.ndarray = orgData.values
        print(orgArray.shape)
        print(real_predict.shape)
        print(real_targets.shape)
        i = 10
        print(orgArray[i])
        print(real_predict[i])
        print(real_targets[i])
        print("ok")

# 测试
file_dir = os.path.split(os.path.abspath(__file__))[0]
data_path = os.path.join(file_dir, "worktime1.csv")

rides = pd.read_csv(data_path)

print(rides.shape)
rowCount = rides.shape[0]

dummy_fields = ['weekday']
fields_to_drop = ['weekday','instant']
quant_features = ['workhour']
target_fields = ['workhour']
nn:SimpleNeuralNetworkWorkTime = SimpleNeuralNetworkWorkTime(rides,dummy_fields,fields_to_drop,quant_features,target_fields,30)
nn.prepareData()
nn.train()
# nn.showLosses()
# nn.showTest()
data_path = os.path.join(file_dir, "worktime2.csv")
rides2 = pd.read_csv(data_path)
nn.analyseData(rides2)
values = nn.predictData([[1,1]])
print(values)
print(type(values))