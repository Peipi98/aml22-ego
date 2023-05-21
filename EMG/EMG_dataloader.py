import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
import scipy as sp
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class EMG_dataloader(DataLoader):
    def __init__(self):

    def load_data(self, dirname):
        
