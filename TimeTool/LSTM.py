import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
# import keras
# from tensorflow import keras
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from numpy import concatenate
from math import sqrt


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # 输入序列(t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 预测序列(t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # 把所有放在一起
    agg = concat(cols, axis=1)
    agg.columns = names
    # 删除空值行
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def normalization(data):
    scaled = np.zeros([len(data), len(data[0])])
    dismin = np.min(data[:, 0])
    disrange = np.max(data[:, 0]) - np.min(data[:, 0])
    for i in range(len(data[0])):
        _range = np.max(data[:, i]) - np.min(data[:, i])
        scaled[:, i] = (data[:, i] - np.min(data[:, i])) / _range
    return scaled, dismin, disrange


def inverse_normalization(data, dismin, disrange):
    inv = np.zeros([len(data), len(data[0])])
    for i in range(len(data[0])):
        inv[:, i] = data[:, i] * disrange + dismin
    return inv


data = pd.read_csv('datatest.csv',parse_dates=True, index_col=0, encoding='gb18030')
values = data.values
weidu = len(values[0])
values = values[:, :weidu]
# values = values.reshape(len(values), 1)
# 对特征标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#scaled, dismin, disrange = normalization(values)
# 构建成监督学习问题
timelag = 15
presteps = 15
reframed = series_to_supervised(scaled, timelag, presteps)
droparr = []
for i in range(presteps-1):
    for j in range(weidu-1):
        droparr.append((timelag+1)*weidu+i*weidu+j+1)  #留下预测列
#print(reframed.columns)
reframed.drop(reframed.columns[droparr],
               axis=1, inplace=True)
# print(reframed.columns)
print(reframed.columns[-presteps+1:])
# print(reframed.columns[-presteps+1:])
# 切分训练集和测试集
values = reframed.values
n_train_hours = 600
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# 切分输入和输出
train_X, train_y = train[:, :-presteps+1], train[:, -presteps+1:]
test_X, test_y = test[:, :-presteps+1], test[:, -presteps+1:]
#print(test_y)
# 将输入转换为三维格式 [samples, timesteps, features]
train_features = np.zeros(shape=(train_X.shape[0], timelag+1, scaled.shape[1]))
test_features = np.zeros(shape=(test_X.shape[0], timelag+1, scaled.shape[1]))
for i in range(scaled.shape[1]):
    train_feature = np.zeros(shape=(train_X.shape[0], timelag+1))
    for j in range(i, train_X.shape[1], scaled.shape[1]):
        train_feature[:, int(j / scaled.shape[1])] = train_X[:, j]
    train_features[:, :, i] = train_feature
    test_feature = np.zeros(shape=(test_X.shape[0], timelag + 1))
    for j in range(i, test_X.shape[1], scaled.shape[1]):
        test_feature[:, int(j / scaled.shape[1])] = test_X[:, j]
    test_features[:, :, i] = test_feature
train_X = train_features
test_X = test_features
# train_X = train_X.reshape((train_X.shape[0], timelag+1, scaled.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], timelag+1, scaled.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# 设计模型
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(presteps-1))
model.compile(loss='mae', optimizer='adam')
# 拟合模型

print("____")
print(train_y[:, weidu])
history = model.fit(train_X, train_y, epochs=50, batch_size=30, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# 绘制损失趋势线
# 绘制损失趋势线
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# # 开始预测
# yhat = model.predict(test_X)
# # 预测值反转缩放
# inv_yhat = inverse_normalization(yhat, dismin, disrange)
# print(inv_yhat)
# # 实际值反转缩放
# inv_y = inverse_normalization(test_y, dismin, disrange)
# print(inv_y)
# output1 = pd.DataFrame(inv_yhat)
# output1.to_csv("predict.csv")
# output2 = pd.DataFrame(inv_y)
# output2.to_csv("real.csv")
# # 计算均方根误差
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)
# pyplot.plot(inv_yhat, label='prediction')
# pyplot.plot(inv_y, label='real')
# pyplot.legend()
# pyplot.show()


# 开始预测
yhat = model.predict(test_X)
print(yhat.shape)
test_X = test_X[:, 0, :]
print(test_X.shape)
# 预测值反转缩放
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]
print(inv_yhat)
# 实际值反转缩放
# test_y = test_y.reshape((len(test_y), 1))
inv_y = test_y[:, 3]
inv_y = inv_y.reshape((len(inv_y), 1))
print(inv_y.shape)
print(test_X[:, 1:].shape)
inv_y = concatenate((inv_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
print(inv_y)
# 计算均方根误差
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
pyplot.plot(inv_yhat, label='prediction')
pyplot.plot(inv_y, label='real')
pyplot.legend()
pyplot.show()

