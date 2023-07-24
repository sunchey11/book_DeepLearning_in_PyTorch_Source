""" 将c3_bike_predictor.py变成类的方式
    测试类：SimpleNeuralNetworkMain

"""
import numpy as np
import pandas as pd #读取csv文件的库
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import os 

"""对神经网络的封装"""
class SimpleNeuralNetwork:
    
    data_frame: pd.DataFrame
    features: pd.DataFrame
    targets: pd.DataFrame
    test_features: pd.DataFrame
    test_targets: pd.DataFrame
    """
    data_frame:数据
    dummy_fields:编码字段，需要扩展为独热编码
    fields_to_drop:不需要的字段，应包含dummy_fields
    quant_features:需要标准化处理的字段，应为数值字段
    target_fields:需要预测的字段
    train_dataCnt:训练数据的数量是多少
    """
    def __init__(self, data_frame:pd.DataFrame, dummy_fields,fields_to_drop, quant_features,target_fields, train_dataCnt):
        """构造神经网络"""
        self.data_frame = data_frame
        self.dummy_fields = dummy_fields
        self.quant_features = quant_features
        self.target_fields = target_fields
        self.train_dataCnt = train_dataCnt
        self.train_data =[]
        self.test_data =[]
        self.fields_to_drop = fields_to_drop
        
    """预处理数据."""
    def prepareData(self):
        # 测试__file__是主程序的路径，还是此文件的路径
        # 是此文件的路径
        print(os.path.abspath(__file__))
        rides = self.data_frame
        # a. 对于类型变量的处理
        #对于类型变量的特殊处理
        # season=1,2,3,4, weathersi=1,2,3, mnth= 1,2,...,12, hr=0,1, ...,23, weekday=0,1,...,6
        # 经过下面的处理后，将会多出若干特征，例如，对于season变量就会有 season_1, season_2, season_3, season_4
        # 这四种不同的特征。
        for each in self.dummy_fields:
            #利用pandas对象，我们可以很方便地将一个类型变量属性进行one-hot编码，变成多个属性
            dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
            rides = pd.concat([rides, dummies], axis=1)
        #看看数据长什么样子
        print(rides.head())
        fields_to_drop = self.fields_to_drop
        self.data = rides.drop(fields_to_drop, axis=1)
        print("热编码后的数据，共59列")
        print(self.data.head())


        data:pd.DataFrame = self.data
        # b. 对于数值类型变量进行标准化
        # 调整所有的特征，标准化处理
        quant_features = self.quant_features
        #quant_features = ['temp', 'hum', 'windspeed']

        # 我们将每一个变量的均值和方差都存储到scaled_features变量中。
        scaled_features = {}
        for each in quant_features:
            mean, std = data[each].mean(), data[each].std()
            scaled_features[each] = [mean, std]
            data.loc[:, each] = (data[each] - mean)/std
        self.scaled_features = scaled_features
        # c. 将数据集进行分割

        # 将所有的数据集分为测试集和训练集，我们以后21天数据一共21*24个数据点作为测试集，其它是训练集
        train_data:pd.DataFrame = data[:self.train_dataCnt]
        self.test_data = data[self.train_dataCnt:]
        
        print('训练数据：',len(train_data),'测试数据：',len(self.test_data))

        #目标列
        target_fields = self.target_fields
        self.features, self.targets = train_data.drop(target_fields, axis=1), train_data[target_fields]
        self.test_features, self.test_targets = self.test_data.drop(target_fields, axis=1), self.test_data[target_fields]
        self.features_columns=[]
        for i in range(0,self.features.columns.size):
            c = self.features.columns[i]
            self.features_columns.append(c)

    def train(self):
        # 将数据从pandas dataframe转换为numpy
        X = self.features.values
        print('X:')
        print(X)
        Y = self.targets[self.target_fields].values
        Y = Y.astype(float)
        # Y是一个一维数组
        print(Y)
        # 将它变成2维，和X同结构
        # Y = np.reshape(Y, [len(Y),1])
        print(Y)
        losses = []

        print(self.features.head())

        # 调用PyTorch现成的函数，构建序列化的神经网络
        # 定义神经网络架构，features.shape[1]个输入层单元，10个隐含层，1个输出层
        input_size = self.features.shape[1]
        hidden_size = 60
        output_size = self.targets.shape[1]
        batch_size = 128
        # 神经网络
        neu = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_size, output_size),
        )
        # 损失函数
        cost = torch.nn.MSELoss()
        # PyTorch还自带了优化器来自动实现优化算法
        optimizer = torch.optim.SGD(neu.parameters(), lr = 0.01)

        # 神经网络训练循环
        losses = []
        for i in range(150000):
            # 每128个样本点被划分为一个撮，在循环的时候一批一批地读取
            batch_loss = []
            # start和end分别是提取一个batch数据的起始和终止下标
            for start in range(0, len(X), batch_size):
                end = start + batch_size if start + batch_size < len(X) else len(X)
                xx = torch.tensor(X[start:end], dtype = torch.float, requires_grad = True)
                yy = torch.tensor(Y[start:end], dtype = torch.float, requires_grad = True)
                predict = neu(xx)
                # predict.shape=[128,1]
                # yy.shape=[128,1]
                # 两个参数结构一样
                loss = cost(predict, yy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.data.numpy())
            
            # 每隔100步输出一下损失值（loss）
            if i % 100==0:
                losses.append(np.mean(batch_loss))
                print(i, np.mean(batch_loss))
        self.losses = losses
        self.neu = neu
    def showLosses(self):
        # 打印输出损失值
        fig = plt.figure(figsize=(10, 7))
        plt.plot(np.arange(len(self.losses))*100,self.losses, 'o-')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.show()

    def showTest(self):
        neu = self.neu
        # 3. 测试神经网络
        # 用训练好的神经网络在测试集上进行预测
        targets = self.test_targets[self.target_fields] #读取测试集的cnt数值
        # targets = targets.values.reshape([len(targets),1]) #将数据转换成合适的tensor形式
        # 转换成numpy数组
        targets = targets.values
        targets = targets.astype(float) #保证数据为实数

        # 测试样本的x
        x = torch.tensor(self.test_features.values, dtype = torch.float, requires_grad = True)
        # 测试样本的y
        y = torch.tensor(targets, dtype = torch.float, requires_grad = True)

        print(x[:10])
        # 用神经网络进行预测
        # 预测得到的y
        predict = neu(x)
        predict = predict.data.numpy()
        mean, std = self.scaled_features[self.target_fields[0]]
        print((predict * std + mean)[:10])


        # 将后21天的预测数据与真实数据画在一起并比较
        # 横坐标轴是不同的日期，纵坐标轴是预测或者真实数据的值
        fig, ax = plt.subplots(figsize = (10, 7))

        mean, std = self.scaled_features[self.target_fields[0]]
        # plot没有x参数，表示x从0开始，步长为1
        ax.plot(predict * std + mean, label='Prediction', linestyle = '--')
        ax.plot(targets * std + mean, label='Data', linestyle = '-')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel(self.target_fields[0])
        self.labelX(ax)
        print('aaa')
        plt.show()
    """ 预测数据
    x 是一个二维数组。x表示行的数组，每一行的数据是用数组表达"""
    def predictData(self,x:np.array):
        x = torch.tensor(x, dtype = torch.float)
        for i in range(0,len(self.features_columns)):
            column_name = self.features_columns[i]
            if column_name in self.scaled_features:
                m,s = self.scaled_features[column_name]
                x[:,i] = (x[:,i] - m)/s
        
        print(x)
        # neu 的参数必须是 Tensor, not numpy.ndarray
        predict = self.neu(x)

        mean, std = self.scaled_features[self.target_fields[0]]
        predict = predict * std + mean
        # predict也是个tensor类型
        predict = predict.data.numpy()
        return predict
    def labelX(self,ax):
        # 对横坐标轴进行标注
        rides = self.data_frame
        s = rides.loc[self.test_data.index]['dteday']
        print(s)
        dates = pd.to_datetime(rides.loc[self.test_data.index]['dteday'])
        dates = dates.apply(lambda d: d.strftime('%b %d'))
        arr1 = np.arange(len(dates))
        # 从12条数据开始，每24条取一次
        arr2 = arr1[12::24]
        # 设置刻度
        ax.set_xticks(arr2)
        # 设置刻度的显示文本，与arr2对应
        _ = ax.set_xticklabels(dates[12::24], rotation=45)

    """分析数据.将数据点和预测结果显示出来看"""
    def analyseData(self, rides:pd.DataFrame):

        orgData = rides.copy()
        # a. 对于类型变量的处理
        #对于类型变量的特殊处理
        # season=1,2,3,4, weathersi=1,2,3, mnth= 1,2,...,12, hr=0,1, ...,23, weekday=0,1,...,6
        # 经过下面的处理后，将会多出若干特征，例如，对于season变量就会有 season_1, season_2, season_3, season_4
        # 这四种不同的特征。
        for each in self.dummy_fields:
            #利用pandas对象，我们可以很方便地将一个类型变量属性进行one-hot编码，变成多个属性
            dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
            rides = pd.concat([rides, dummies], axis=1)
        #看看数据长什么样子
        print(rides.head())
        fields_to_drop = self.fields_to_drop
        rides = rides.drop(fields_to_drop, axis=1)
        print("热编码后的数据，共59列")
        print(rides)


        data:pd.DataFrame = rides
        # b. 对于数值类型变量进行标准化
        # 调整所有的特征，标准化处理
        quant_features = self.quant_features
        #quant_features = ['temp', 'hum', 'windspeed']

        # 我们将每一个变量的均值和方差都存储到scaled_features变量中。
        for each in quant_features:
            [mean, std] = self.scaled_features[each]
            data.loc[:, each] = (data[each] - mean)/std

        #目标列
        target_fields = self.target_fields
        featuresData, targetsData = data.drop(target_fields, axis=1), data[target_fields]
        
        # 显示图形
        neu = self.neu
        
        # targets是原始数据
        targets = targetsData[self.target_fields] #读取测试集的cnt数值
        # targets = targets.values.reshape([len(targets),1]) #将数据转换成合适的tensor形式
        # 转换成numpy数组
        targets = targets.values
        targets = targets.astype(float) #保证数据为实数

        # 测试样本的x
        x = torch.tensor(featuresData.values, dtype = torch.float, requires_grad = True)

        print(x[:10])
        # 用神经网络进行预测
        # 预测得到的y
        predict = neu(x)
        predict = predict.data.numpy()
        mean, std = self.scaled_features[self.target_fields[0]]
        print((predict * std + mean)[:10])


        # 将后21天的预测数据与真实数据画在一起并比较
        # 横坐标轴是不同的日期，纵坐标轴是预测或者真实数据的值
        fig, ax = plt.subplots(figsize = (10, 7))

        mean, std = self.scaled_features[self.target_fields[0]]
        # plot没有x参数，表示x从0开始，步长为1
        real_predict = predict * std + mean
        real_targets = targets * std + mean
        ax.plot(real_predict, label='Prediction', linestyle = '--')
        ax.plot(real_targets, label='Data', linestyle = '-')
        self.debugData(orgData, real_predict, real_targets)


        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel(self.target_fields[0])
        self.labelX(ax)
        print('aaa')
        plt.show()

    def debugData(self,orgData:pd.DataFrame, real_predict:np.ndarray, real_targets:np.ndarray):
        orgArray:np.ndarray = orgData.values
        print(orgArray.shape)
        print(real_predict.shape)
        print(real_targets.shape)
        i = 159
        print(orgArray[i])
        print(real_predict[i])
        print(real_targets[i])
        print("ok")

