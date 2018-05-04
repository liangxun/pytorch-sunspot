import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataPreprocess(object):
    def __init__(self):
        pass

    def lstm_load_data(self, filename, seq_len, dim=1, row=1686):
        self.dim = dim
        table = pd.read_csv(filename, index_col='Date')
        data = np.array(table['sunspot_ms'])
        print('len(data):'.format(len(data)))
        self.max = max(data)
        self.min = min(data)
        data = (data-self.min)/(self.max-self.min)
        x_train = []
        y_train = []
        for index in range(row - seq_len):
            x_train.append(data[index: index + seq_len])
            y_train.append(data[index+1:index+1+seq_len])
        print("train set的长度：", row)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = data[row-seq_len:]
        y_test = data[row:]
        print("test set的长度：", len(y_test))
        #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], dim))
        #y_train = y_train[:,:,np.newaxis]
        x_test = x_test[:,np.newaxis]
        print("x_train.shape{}\ty_train.shape{}\nx_test.shape{}\ty_test.shape{}\n".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape))
        return [x_train, y_train, x_test, y_test]


    def recover(self, data):
        recovered_data = [p * (self.max - self.min) + self.min for p in data]
        return recovered_data


'''
class DataPreprocess(object):
    def __init__(self):
        pass

    def lstm_load_multidata(self, filename, seq_len, dim=2, row=1686):
        """LSTM多特征输入"""
        self.mode = 'lstm_dim{}'.format(dim)
        self.dim = dim
        table = pd.read_csv(filename, index_col='Date')
        # 定位收盘价所在列
        self.col_sunspot = table.columns.get_loc('sunspot_ms')
        sequence_length = seq_len + 1
        result = []
        info = []
        """求mean和std时只用到输入部分"""
        for index in range(len(table) - sequence_length):
            data = table[index: index + sequence_length]
            mean = np.mean(data[:-1])
            std = np.std(data[:-1])
            data = (data - mean)/std
            info.append((mean[self.col_sunspot], std[self.col_sunspot]))
            result.append(data.values)
        print("相空间重构后数据集的长度：", len(result))
        result = np.array(result)
        train = result[:row, :]
        print("train set的长度：", row)
        # 输入多维特征，但输出依旧是只有sunspot_ms一维
        x_train = train[:, :-1]
        y_train = train[:, -1, self.col_sunspot]
        x_test = result[row:, :-1]
        y_test = result[row:, -1, self.col_sunspot]
        self.testinfo = info[row:]
        print("test set的长度：", len(y_test))
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], dim))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], dim))
        y_train = y_train[:,np.newaxis]
        # y_test = y_test[:,np.newaxis]
        print("x_train.shape{}\ty_train.shape{}\nx_test.shape{}\ty_test.shape{}\n".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape))
        return [x_train, y_train, x_test, y_test]


    def recover(self, data):
        info = self.testinfo
        for i in range(len(data)):
            data[i] = data[i] * info[i][1] + info[i][0]
        return data
'''''

def show(data,label='data'):
    plt.figure()
    plt.plot(data, label=label)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    dim = 2
    datapath = './sunspot_ms_dim{}.csv'.format(dim)
    print('> Loading data... ')
    DataLoader = DataPreprocess()
    #x_train, y_train, x_test, y_test = DataLoader.lstm_load_data(datapath, 50)
    #data, (train,test) = DataLoader.arima_load_data(datapath,50)
    x_train, y_train, x_test, y_test = DataLoader.lstm_load_data(datapath, 200)
    show(y_train)
    show(y_test)
    show(DataLoader.recover(y_test))