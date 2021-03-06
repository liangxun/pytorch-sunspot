import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataPreprocess2(object):
    """对每个样本归一化到（均值0，标准差1）"""
    def __init__(self):
        pass

    def lstm_load_data(self, filename, seq_len, ahead=1, dim=1, row=1500):
        """
        :param filename: 数据集
        :param seq_len: 输入序列的时间步
        :param ahead: 预测序列的时间步
        :param dim: 输入特征的维度
        :param row: 划分训练集和测试集的地方
        :return: 数据集，测试集
        """
        self.dim = dim
        df = pd.read_csv(filename, index_col='Date')
        self.col_index = df.columns.get_loc('sunspot_ms')
        print('len(data):'.format(len(df)))
        # 相空间重构和归一化同时进行
        sequence_length = seq_len + ahead
        result = []
        self.info = []
        for index in range(len(df)-sequence_length):
            data = df[index:index+sequence_length]
            mean = np.mean(data[:-ahead])
            std = np.std(data[:-ahead])
            data = (data - mean)/std
            result.append(data.values)
            #self.info.append((mean[self.col_index],std[self.col_index]))
            self.info.append((mean,std))
        result = np.array(result)
        print("train set的长度：", row)
        print(self.dim)
        x_train = result[:row, :-ahead]
        x_test = result[row:,:-ahead]
        if self.dim is 1:
            x_train = x_train[:, :, np.newaxis, self.col_index]
            x_test = x_test[:, :, np.newaxis, self.col_index]
        y_train = result[:row, -ahead:, self.col_index]
        y_test = result[row:, -ahead:, self.col_index]
        self.testinfo = self.info[row:]
        print("test set的长度：", len(y_test))
        #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], dim))
        #y_train = y_train[:,:,np.newaxis]
        print("x_train.shape{}\ty_train.shape{}\nx_test.shape{}\ty_test.shape{}\n".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape))
        return [x_train, y_train, x_test, y_test]


    def recover(self, data):
        info = self.testinfo
        for i in range(len(data)):
            data[i] = data[i] * info[i][1] + info[i][0]
        return data

class DataPreprocess(object):
    """对所有数据归一化到（0,1）"""
    def __init__(self):
        pass

    def lstm_load_data(self, filename, seq_len, ahead=1, dim=1, row=1500):
        """
        :param filename: 数据集
        :param seq_len: 输入序列的时间步
        :param ahead: 预测序列的时间步
        :param dim: 输入特征的维度
        :param row: 划分训练集和测试集的地方
        :return: 数据集，测试集
        """
        self.dim = dim
        df = pd.read_csv(filename, index_col='Date')
        #
        self.col_index = df.columns.get_loc('sunspot_ms')
        data = df.values
        print('len(data):'.format(len(data)))
        self.max = data.max(axis=0)
        self.min = data.min(axis=0)
        data = (data-self.min)/(self.max-self.min)
        # 相空间重构
        sequence_length = seq_len + ahead
        result = []
        for index in range(len(data)-sequence_length):
            result.append(data[index:index+sequence_length])
        result = np.array(result)
        print("train set的长度：", row)
        print(self.dim)
        x_train = result[:row, :-ahead]
        x_test = result[row:,:-ahead]
        if self.dim is 1:
            x_train = x_train[:, :, np.newaxis, self.col_index]
            x_test = x_test[:, :, np.newaxis, self.col_index]
        y_train = result[:row, -ahead:, self.col_index]
        y_test = result[row:, -ahead:, self.col_index]
        print("test set的长度：", len(y_test))
        #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], dim))
        #y_train = y_train[:,:,np.newaxis]
        print("x_train.shape{}\ty_train.shape{}\nx_test.shape{}\ty_test.shape{}\n".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape))
        return [x_train, y_train, x_test, y_test]


    def recover(self, data):
        max = self.max[self.col_index]
        min = self.min[self.col_index]
        recovered_data = [p*(max-min)+min for p in data]
        return recovered_data


def show(data,label='data'):
    plt.figure()
    plt.plot(data, label=label)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    dim = 2
    timesteps = 129
    datapath = './sunspot_ms_dim{}.csv'.format(dim)
    print('> Loading data... ')
    DataLoader = DataPreprocess2()
    #x_train, y_train, x_test, y_test = DataLoader.lstm_load_data(datapath, 50)
    #data, (train,test) = DataLoader.arima_load_data(datapath,50)
    x_train, y_train, x_test, y_test = DataLoader.lstm_load_data(datapath, timesteps,ahead=1,dim=2,row=1686-timesteps)
    data = np.vstack((y_train,y_test))
    show(data)
    show(DataLoader.recover(y_test))