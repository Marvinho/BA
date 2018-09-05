import torch
import torch.nn as nn
import torch.nn.functional as F 

class LSTM(nn.Module):
    def __init__(self, num_classes, lstm_size, dropout, bidirectional = True):
        super(LSTM, self).__init__()
        self.dropout = dropout
        #self.hidden_size = hidden_size
        self.lstm_size = lstm_size
        self.lstm = nn.LSTM(1024, self.lstm_size, batch_first=True)
        self.fc = nn.Linear(lstm_size, num_classes)

    def init_hidden(self, batch_size, device=None):
        return (torch.zeros(1, 1, self.lstm_size, device=device),
                torch.zeros(1, 1, self.lstm_size, device=device))

    def forward(self, x, hidden_state):
        x = self.resnet.forward(x)
        x = x.view(1, x.size(0), -1)
        x, hidden_state = self.lstm(x, hidden_state)
        x = x.view(x.size(1), -1) # output.size(2) statt -1
        x = self.fc(x)

        return x, hidden_state
        