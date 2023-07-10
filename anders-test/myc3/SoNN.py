
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # 读取csv文件的库
from SimpleNeuralNetwork import SimpleNeuralNetwork
import torch
import torch.optim as optim
import os 
import pymysql


class SoNN(SimpleNeuralNetwork):
    """
    预测so表的数据,预测每天每个小时
    https://blog.csdn.net/li123456hkho/article/details/120491217

    测试数据通过此sql插入:
    create table nn_so_sample
    as
    select WARID,GOODSID ,so_hour as so_time, 
        DATE_FORMAT(STR_TO_DATE(so_hour, "%Y-%m-%d %H"),'%Y') so_year,
        QUARTER(STR_TO_DATE(so_hour, "%Y-%m-%d %H")) so_quarter,
        DATE_FORMAT(STR_TO_DATE(so_hour, "%Y-%m-%d %H"),'%m') so_month,
        DATE_FORMAT(STR_TO_DATE(so_hour, "%Y-%m-%d %H"),'%d') so_month_day,
        DATE_FORMAT(STR_TO_DATE(so_hour, "%Y-%m-%d %H"),'%w') so_weekday,
        DATE_FORMAT(STR_TO_DATE(so_hour, "%Y-%m-%d %H"),'%H') so_hour24,
        SOCNT,SOQTY 
    from wms_so_agg wsa;
        

    """    
    
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

# 返回一个 Connection 对象
db_conn = pymysql.connect(
    host='localhost',
    port=13306,
    user='root',
    password='Lrh13022958534',
    database='ebeit_demo',
    charset='utf8'
    )
print(db_conn)


# 执行sql操作
sql = """
    select so_time,so_quarter,so_month,so_month_day,so_weekday,so_hour24,socnt
    from nn_so_sample wsa
    where warid=%s
    and GOODSID =%s
    """
rides = pd.read_sql(sql,con=db_conn,params=[1,24229],index_col='so_time')
print(rides)


print(rides.shape)

dummy_fields = ['so_quarter','so_month','so_month_day','so_weekday','so_hour24']
fields_to_drop = ['so_quarter','so_month','so_month_day','so_weekday','so_hour24']
quant_features = ['socnt']
target_fields = ['socnt']
nn:SoNN = SoNN(rides,dummy_fields,fields_to_drop,quant_features,target_fields,12000)
nn.prepareData()
nn.train()
nn.showLosses()
# nn.showTest()

rides2 = pd.read_sql(sql
                    #  +' limit 0,1000'
                     ,con=db_conn,params=[1,24229],index_col='so_time')
nn.analyseData(rides2)
print('ok')
# values = nn.predictData([[1,1]])
# print(values)
# print(type(values))