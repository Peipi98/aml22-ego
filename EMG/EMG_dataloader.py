import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
import scipy as sp
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os

activities_to_classify = {
    'Get/replace items from refrigerator/cabinets/drawers': 0,
    'Peel a cucumber': 1,
    'Clear cutting board': 2,
    'Slice a cucumber': 3,
    'Peel a potato': 4,
    'Slice a potato': 5,
    'Slice bread': 6,
    'Spread almond butter on a bread slice': 7,
    'Spread jelly on a bread slice': 8,
    'Open/close a jar of almond butter': 9,
    'Pour water from a pitcher into a glass': 10,
    'Clean a plate with a sponge': 11,
    'Clean a plate with a towel': 12,
    'Clean a pan with a sponge': 13,
    'Clean a pan with a towel': 14,
    'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 15,
    'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 16,
    'Stack on table: 3 each large/small plates, bowls': 17,
    'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 18,
    'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 19,
}
# Some older experiments may have had different labels.
#  Each entry below maps the new name to a list of possible old names.
activities_renamed = {
    'Open a jar of almond butter': 'Open/close a jar of almond butter',
    'Get items from refrigerator/cabinets/drawers': 'Get/replace items from refrigerator/cabinets/drawers',
}

class EMG_dataset(Dataset):
    def __init__(self, directory, filename):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # self.train = self.load_data(os.path.join(script_dir, 'EMG_preprocessed', 'train_EMG_preprocess.pkl'))
        # self.test = self.load_data(os.path.join(script_dir, 'EMG_preprocessed', 'test_EMG_preprocess.pkl'))
        self.df = pd.read_pickle(os.path.join(directory, filename))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        emg, label = self.df.iloc[idx, :]
        label = torch.tensor(activities_to_classify[label]).to(torch.int64)
        emg = torch.from_numpy(emg).to(torch.float32)
        
        return (emg, label)


