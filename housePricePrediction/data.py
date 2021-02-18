import numpy as np
import json
# 读入训练数据
def load_data():
    datafile = 'housing.data'
    data = np.fromfile(datafile, sep=' ')
    #数据按行处理
    feature_name=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
    feature_num=len(feature_name)
    data=data.reshape(data.shape[0]//feature_num,feature_num)
    #数据分类测试集和训练集
    radio=0.8
    offset=int(data.shape[0]*radio)
    training_data=data[:offset]
    predict_data=data[offset:]
    #数据归一化
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0),training_data.sum(axis=0) / training_data.shape[0]
    for i in range(feature_num):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])
    training_data=data[:offset]
    predict_data=data[offset:]
    return training_data,predict_data
