import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
import scipy as sp
import pandas as pd
from torch.utils.data import DataLoader
from EMG_dataloader import EMG_dataset

dataset = EMG_dataset()

df = dataset.load_data(split='train')

print(df.shape)