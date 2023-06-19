import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
import scipy as sp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn import LSTM, Dropout, Linear
from torch import argmax, softmax

class EMG_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(EMG_LSTM, self).__init__()
        # Modify hidden size to 5 for ActionNet
        self.lstm1 = LSTM(input_size=16, hidden_size=50, batch_first=True)
        # Uncomment the following line to obtain ActionNet
        # self.lstm2 = LSTM(input_size=5, hidden_size=50, batch_first=True)
        self.dropout = Dropout(0.2)
        self.dense = Linear(50, num_classes)

    def forward(self, x):
        x, (h1_T,c1_T) = self.lstm1(x)
        # In order to run ActionNet, please uncomment the following line
        # x, (h2_T,c2_T) = self.lstm2(x)

        # return_sequences = False tensorflow equivalent following
        # For ActionNet, comment the following line and uncomment the second one
        x = torch.squeeze(c1_T)
        # x = torch.squeeze(c2_T)

        x = self.dropout(x)
        x = self.dense(x)
        x = softmax(x, dim=1)
        return x
