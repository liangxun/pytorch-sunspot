import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import models
import load
import EvaluationIndex


#hyperparams
timesteps = 42
dim = 1
epochs = 10
lr = 0.01
batchsize = 1
layers = [16,1]
datapath = './sunspot_ms_dim{}.csv'.format(2)
modelpath = './{}_step{}_dim{}_rmse{:.3f}.pt'

print('>Loading model...')
# model = models.rnn()
model = models.lstm(input_dim=dim, layers=layers)
optimizer = optim.RMSprop(model.parameters(), lr=lr)
loss_func = nn.MSELoss()
print(model)

#load data
print('> Loading data... ')
DataLoader = load.DataPreprocess()
x_train, y_train, x_test, y_test = DataLoader.lstm_load_data(filename=datapath, seq_len=timesteps, dim=dim,
                                                                  row=1686)
train_dataset = Data.TensorDataset(torch.Tensor(x_train),torch.Tensor(y_train))
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=batchsize,
    shuffle=True,
)

log_loss = []
def train():
    for step,(input,target) in enumerate(train_loader):
        out = model(input)
        loss = loss_func(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(loss.data)
        log_loss.append(loss.data)

for epoch in range(epochs):
    print('epoch:{}'.format(epoch+1))
    train()

log_loss = np.array(log_loss)
plt.plot(log_loss,label='loss')
plt.title('the loss during train')
plt.legend()
plt.show()


def predict():
    state = model.state_dict()
    pmodel = models.lstmcell(input_dim=dim,layers=layers)
    pmodel.lstm.weight_ih.data = state['lstm.weight_ih_l0']
    pmodel.lstm.weight_hh.data = state['lstm.weight_hh_l0']
    pmodel.lstm.bias_ih.data = state['lstm.bias_ih_l0']
    pmodel.lstm.bias_hh.data = state['lstm.bias_hh_l0']
    pmodel.out.weight.data = state['out.weight']
    pmodel.out.bias.data = state['out.bias']
    inputs = torch.Tensor(x_test[:,np.newaxis,:])
    hx = torch.zeros(1,layers[0])
    cx = torch.zeros(1,layers[0])
    prediction = []

    for i in range(timesteps):
        out,hx,cx = pmodel(inputs[i],hx,cx)
        prediction.append(out.data)
    for i in range(timesteps,len(inputs)):
        out, hx,cx = pmodel(out,hx,cx)
        prediction.append(out.data)
    '''
    for i in range(timesteps,len(inputs)):
        out, hx, cx = pmodel(inputs[i],hx,cx)
        prediction.append(out.data)
    '''
    return prediction
prediction = predict()
print(len(prediction))
plt.plot(prediction)
plt.show()

'''
#在训练集上预测，看是否欠拟合
prediction = predict()
prediction = DataLoader.recover(prediction)
target = DataLoader.recover(np.squeeze(y_test))
eI = EvaluationIndex.evalueationIndex(prediction,target)
print("MSE={}\nRMSE={}\nMAPE={}".format(eI.MSE, eI.RMSE, eI.MAPE))
plt.plot(prediction,label='prediction')
plt.plot(target,label='true_data')
plt.legend()
plt.show()
'''
'''
torch.save({
    'epoch':epochs+1,
    'timesteps':timesteps,
    'inputdim':dim,
    'optimizer':optimizer.state_dict()['param_groups'][0],
    'state_dict':model.state_dict(),
},modelpath.format(model.name,timesteps,dim,eI.RMSE))
'''