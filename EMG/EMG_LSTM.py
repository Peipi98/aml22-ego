import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
import scipy as sp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn import LSTM, Dropout, Linear
from torch import argmax

class EMG_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(EMG_LSTM, self).__init__()
        self.lstm1 = LSTM(input_size=16, hidden_size=5, batch_first=True)
        # TODO: modify 2nd LSTM in order to output (batch_size,50). 
        # (At the moment the output is (batch_size, 100, 1))
        self.lstm2 = LSTM(input_size=5, hidden_size=1, batch_first=True)
        self.dropout = Dropout(0.2)
        self.dense = Linear(50, num_classes)

    def forward(self, x):
        h1, (h1_T,c1_T) = self.lstm1(x)
        h2, (h2_T,c2_T) = self.lstm2(h1)
        x = self.dropout(h2)
        x = self.dense(x)
        x = argmax(x)
        return x