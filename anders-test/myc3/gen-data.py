# 此程序用来生成数据，生成的数据保存到一个csv文件中
import pandas as pd
import torch  #导入torch包
# 生成多少行数据
data_num=200
x = torch.linspace(1, data_num, data_num).type(torch.FloatTensor) #linspace可以生成0-100之间的均匀的100个数字
rand = torch.randn(data_num) * 10 #随机生成100个满足标准正态分布的随机数，均值为0，方差为1.将这个数字乘以10，标准方差变为10
y = 2*x+100 + rand #将x和rand相加，得到伪造的标签数据y。所以(x,y)应能近似地落在y=x这条直线上

d = {'x': x, 'y': y}
df = pd.DataFrame(data=d)
data_path = "./liner-data1.csv"
df.to_csv(data_path, index=False)

