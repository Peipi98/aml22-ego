from torch import nn, relu
import torch

class EMGClassifier(nn.Module):
    def __init__(self, input_size=256*1*73, hidden_size=256, num_classes=11):
        super(EMGClassifier ,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = relu

    def forward(self, x):
        f = x
        x = self.dropout(self.relu(self.fc1(x)))
        out = self.fc2(x)
        return out, {'features': f}