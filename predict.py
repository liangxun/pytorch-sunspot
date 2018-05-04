import torch
import matplotlib.pyplot as plt
import numpy as np
import models
import load
import EvaluationIndex


#hyperparams
timesteps = 42
dim = 1
layers = [16,1]
datapath = './sunspot_ms_dim{}.csv'.format(1)
modelpath = 'model_ms_step42_dim1_rmse4.911.pt'

print('>Loading model...')
params = torch.load(modelpath)
state = params['state_dict']
print(state)

# model = models.rnn()
model = models.rnncell(input_dim=dim, layers=layers)

model.rnn.weight_ih.data = state['rnn.weight_ih_l0']
model.rnn.weight_hh.data = state['rnn.weight_hh_l0']
model.rnn.bias_ih.data = state['rnn.bias_ih_l0']
model.rnn.bias_hh.data = state['rnn.bias_hh_l0']
model.out.weight.data = state['out.weight']
model.out.bias.data = state['out.bias']

print(model)
print(model.state_dict())

#load data
print('> Loading data... ')
DataLoader = load.DataPreprocess()
x_train, y_train, x_test, y_test = DataLoader.lstm_load_data(filename=datapath, seq_len=timesteps, dim=1,
                                                                  row=1686 - (timesteps + 1))


inputs = torch.Tensor(x_test[:,np.newaxis,:])
hx = torch.zeros(1,layers[0])
for i in range(timesteps):
    out,hx = model(inputs[i],hx)
prediction = []
for i in range(len(y_test)):
    out,hx = model(inputs[i+timesteps],hx)
    prediction.append(out.data)

print(len(prediction))
prediction = DataLoader.recover(prediction)
y_test = DataLoader.recover(y_test)
plt.plot(prediction,label='prediction')
plt.plot(y_test,label='true_data')
plt.legend()
plt.show()


