from torch import nn, relu
import torch
from abc import ABC
from tasks import ActionRecognition

class Classifier(ActionRecognition, ABC):
    def __init__(self, input_size, hidden_size, name, task_models, batch_size, total_batch, models_dir, num_classes,
                 num_clips, model_args, args, **kwargs):
        super().__init__(self, name, task_models, batch_size, total_batch, models_dir, num_classes,
                 num_clips, model_args, args, **kwargs)
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

'''
1543 5, 1024
'''
classifier = Classifier(5120, 512, 8)

