# 此程序显示./liner-data1.csv的数据到图上
import pandas as pd
import matplotlib.pyplot as plt

data_path = "./liner-data1.csv"

# 读取csv数据，并作图
csv_datas = pd.read_csv(data_path)
csv_x = csv_datas['x']
csv_y = csv_datas['y']
print(type(csv_x))

# 绘制一个图形，展示曲线长的样子
plt.figure(figsize = (10, 7)) #设定绘图窗口大小
plt.plot(csv_x, csv_y, 'o-') # 绘制原始数据
plt.xlabel('X') #更改坐标轴标注
plt.ylabel('Y') #更改坐标轴标注
plt.show()