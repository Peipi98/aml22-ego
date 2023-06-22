from torch import nn, relu
import torch

class Classifier(nn.Module):
    '''
    def __init__(self, num_classes):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

    def forward(self, x):
        return self.classifier(x), {}
    '''

    def __init__(self, input_size=1024, hidden_size=512, num_classes=8):
        super(Classifier, self).__init__()
        self.aggregation = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = relu
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.aggregation(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        #x = self.softmax(x)
        return x

