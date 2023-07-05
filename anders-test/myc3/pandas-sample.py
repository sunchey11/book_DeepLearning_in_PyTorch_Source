import pandas as pd
data1 = ["小明", 20] # 以list 存储每一行的数据
data2 = ["小雨", 21]
data3 = ["小花", 22]
header = ["姓名", "年龄"]
df = pd.DataFrame([data1, data2, data3], columns=header) # 组成一个csv
print(df)
df.to_csv("./data111.csv", index=False)