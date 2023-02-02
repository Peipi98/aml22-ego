from torch import nn, relu
import torch.nn.functional as F
import torch
import numpy as np

class LSTMClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(LSTMClassifier ,self).__init__()
        self.lstm = nn.GRU(input_size=1024, hidden_size=256, num_layers=1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        

    def forward(self, x):
        out, _ = self.lstm(x) 
        out = self.fc1(out[:, -1, :])
        x = F.relu(out)
        x = self.fc2(x)
        
        return x, None