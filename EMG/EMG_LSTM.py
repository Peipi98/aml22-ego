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
        self.lstm1 = LSTM(input_size=16, hidden_size=5, batch_first=True)
        # TODO: modify 2nd LSTM in order to output (batch_size,50). 
        # (At the moment the output is (batch_size, 100, 1))
        self.lstm2 = LSTM(input_size=5, hidden_size=50, batch_first=True)
        self.dropout = Dropout(0.2)
        self.dense = Linear(50, num_classes)

    def forward(self, x):
        h1, (h1_T,c1_T) = self.lstm1(x)
        h2, (h2_T,c2_T) = self.lstm2(h1)
        # return_sequences = False equivalent following
        x = h2[:, -1, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = softmax(x, dim=1)
        # x = argmax(x, dim=1)
        return x