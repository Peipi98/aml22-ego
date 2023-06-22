import torch
import torch.nn as nn

class EMG_CNN(nn.Module):
    def __init__(self, num_classes):
        super(EMG_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(256*1*73, 128)  #hop 8 = 123, 16 = 61, 32 = 30 and 25 seconds | 160s -> hop 32 = 148
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(torch.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 256*1*73 )
        feat = x
        #print(x.shape)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x, {"features": feat}