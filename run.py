import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import models
import load
import EvaluationIndex


# hyperparams
timesteps = 10
dim = 2
epochs = 10
lr = 0.001
batchsize = 128
ahead = 1
layers = [32,ahead]
datapath = './sunspot_ms_dim{}.csv'.format(2)
modelpath = './models/{}_step{}_dim{}_rmse{:.3f}.pt'

print('>Loading model...')
# model = models.rnn()
model = models.lstm(input_dim=dim, layers=layers,num_layers=2)
optimizer = optim.RMSprop(model.parameters(), lr=lr)
loss_func = nn.MSELoss()
print(model)

# load data
print('> Loading data... ')
DataLoader = load.DataPreprocess()
x_train, y_train, x_test, y_test = DataLoader.lstm_load_data(filename=datapath, seq_len=timesteps, ahead=ahead,dim=dim,
                                                                  row=1686-timesteps)
train_dataset = Data.TensorDataset(torch.Tensor(x_train),torch.Tensor(y_train))
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=batchsize,
    shuffle=True,
)

# train model
log_loss = []
def train():
    loss_epoch = 0
    for step,(input,target) in enumerate(train_loader):
        out = model(input)
        loss = loss_func(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(loss.data)
        loss_epoch += loss.item()
    log_loss.append(loss_epoch)
    print('\tloss={}'.format(loss_epoch))

for epoch in range(epochs):
    print('epoch:{}'.format(epoch+1))
    train()

log_loss = np.array(log_loss)
plt.plot(log_loss,label='loss')
plt.title('the loss during train')
plt.legend()
plt.show()

def predict(x):
    prediction = []
    torch.no_grad()
    inputs = torch.Tensor(x)
    for inp in inputs:
        out = model(inp.view(1,-1,dim))
        prediction.append([a.item() for a in out[0]])
    return prediction

# 对预测结果画图，计算统计指标
def statistic(prediction, target):
    eI = EvaluationIndex.evalueationIndex(prediction, target)
    print("MSE={}\nRMSE={}\nMAPE={}".format(eI.MSE, eI.RMSE, eI.MAPE))
    plt.plot(target, 'o-', label='true_data')
    plt.plot(prediction, 'x-', label='prediction')
    plt.legend()
    plt.title('result of {}\nRMSE{:.3f}  MAPE{:.3f}'.format(model.name, eI.RMSE, eI.MAPE))
    plt.show()
    '''
    eI.plot_e()
    eI.plot_ape()
    eI.correlation()
    '''
    return eI

# 在训练集上预测，看是否欠拟合
prediction = DataLoader.recover(np.squeeze(np.array(predict(x_train))))
target = DataLoader.recover(np.squeeze(y_train))
statistic(prediction, target)

# 在测试集上预测，
prediction = DataLoader.recover(np.squeeze(np.array(predict(x_test))))
target = DataLoader.recover(np.squeeze(y_test))
eI = statistic(prediction, target)

# 存储训练好的参数和一些超参数信息
torch.save({
    'epoch':epochs+1,
    'timesteps':timesteps,
    'inputdim':dim,
    'optimizer':optimizer.state_dict()['param_groups'][0],
    'state_dict':model.state_dict(),
}, modelpath.format(model.name,timesteps,dim,eI.RMSE))
