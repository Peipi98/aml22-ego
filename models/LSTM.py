from torch import nn, relu
import torch

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, num_layers=2, num_classes=8):
        super(LSTMClassifier ,self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out, None