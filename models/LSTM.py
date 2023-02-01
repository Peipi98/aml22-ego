from torch import nn, relu
import torch

class LSTMClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(LSTMClassifier ,self).__init__()
        self.lstm = nn.LSTM(input_size=1024, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        

    def forward(self, x):
        for t in range(x.size(1)):
            out, hidden = self.lstm(x[:,t,:].unsqueeze(1), hidden) 
        x = self.fc(out[:, -1, :])
        x = relu(x)
        x = self.fc2(x)
        return x, None