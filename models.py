import torch.nn as nn

class rnn(nn.Module):
    def __init__(self,input_dim,layers):
        super(rnn, self).__init__()
        self.name = 'rnn'
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=layers[0], num_layers=1, batch_first=True)
        self.out = nn.Linear(in_features=layers[0], out_features=layers[1])

    def forward(self,input):
        x, hidden = self.rnn(input)
        out = self.out(x[:, -1, :])
        return out


class rnncell(nn.Module):
    def __init__(self,input_dim,layers):
        super(rnncell, self).__init__()
        self.rnn = nn.RNNCell(input_size=input_dim, hidden_size=layers[0])
        self.out = nn.Linear(in_features=layers[0], out_features=layers[1])

    def forward(self, input,hx):
        hidden = self.rnn(input,hx)
        out = self.out(hidden)
        return out,hidden

class lstm(nn.Module):
    def __init__(self,input_dim,layers):
        super(lstm, self).__init__()
        self.name = 'lstm'
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=layers[0],num_layers=1,batch_first=True)
        self.out = nn.Linear(in_features=layers[0],out_features=layers[1])

    def forward(self, input):
        x, hidden = self.lstm(input)
        out = self.out(x)
        return out

class lstmcell(nn.Module):
    def __init__(self, input_dim,layers):
        super(lstmcell, self).__init__()
        self.lstm = nn.LSTMCell(input_size=input_dim,hidden_size=layers[0])
        self.out = nn.Linear(in_features=layers[0],out_features=layers[1])

    def forward(self, input,hx,cx):
        hx, cx = self.lstm(input,(hx,cx))
        out = self.out(hx)
        return out, hx, cx

