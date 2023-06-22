######################################################################################
# IMPORTANT: THIS CODE WAS INTENDED TO RUN ON COLAB, SO SOME DIRECTORIES CAN CHANGE  #
# THE RESAMPLING, EVEN IF IS IMPLEMENTED, DOESN'T WORKS WELL: IF YOU WANT TO USE IT  #
# YOU SHOULD CHANGE THE INPUT OF THE FIRST FC LAYER AND THE VIEW PARAMETERS          #
######################################################################################

import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchvision
import torch.optim as optim
import scipy as sp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

n_fft = 32
win_length = None
hop_length = 16

TIME_CUT = 30 #30

spectrogram = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    normalized=True
)

mel_spectrogram = T.MelSpectrogram(
    n_mels=10,
    sample_rate=160,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    normalized=True,
    f_min=0,
    f_max=80
)

mfccs = T.MFCC(
    sample_rate= 160,
    n_mfcc= 10,
    dct_type= 2,
    norm= 'ortho',
    log_mels = True,
    melkwargs={
        "n_fft": n_fft,
        "n_mels": 15,
        "win_length": win_length,
        "hop_length": hop_length,
        "mel_scale": "htk",
        "pad_mode": "reflect",
        "norm": "slaney",
        "center": True,
        "normalized":True,
        "power":2.0,
        "f_min":0,
        "f_max":80,
    },
)
dict_labels1 = {
    "Get/replace items from refrigerator/cabinets/drawers": 0,
    "Get items from refrigerator/cabinets/drawers": 0,
    "Peel a cucumber" : 1,
    "Clear cutting board": 2,
    "Slice a cucumber": 3,
    "Peel a potato": 4,
    "Slice a potato": 5,
    "Slice bread": 6,
    "Spread almond butter on a bread slice": 7,
    "Spread jelly on a bread slice": 8,
    "Open/close a jar of almond butter": 9,
    "Open a jar of almond butter": 9,
    "Pour water from a pitcher into a glass": 10,
    "Clean a plate with a sponge":  11,
    "Clean a plate with a towel": 12,
    "Clean a pan with a sponge":  13,
    "Clean a pan with a towel": 14,
    "Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 15,
    "Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 16,
    "Stack on table: 3 each large/small plates, bowls": 17,
    "Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 18,
    "Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 19,
}

dict_labels = {
    "Get/replace items from refrigerator/cabinets/drawers": 0,
    "Get items from refrigerator/cabinets/drawers": 0,
    "Peel a cucumber" : 1,
    "Clear cutting board": 2,
    "Slice a cucumber": 3,
    "Peel a potato": 1,
    "Slice a potato": 3,
    "Slice bread": 3,
    "Spread almond butter on a bread slice": 4,
    "Spread jelly on a bread slice": 4,
    "Open/close a jar of almond butter": 5,
    "Open a jar of almond butter": 5,
    "Pour water from a pitcher into a glass": 6,
    "Clean a plate with a sponge":  7,
    "Clean a plate with a towel": 7,
    "Clean a pan with a sponge":  7,
    "Clean a pan with a towel": 7,
    "Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 0,
    "Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 8,
    "Stack on table: 3 each large/small plates, bowls": 9,
    "Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 10,
    "Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 10,
}

class EMGDataset(Dataset):
    def __init__(self, spectrogram_data, labels):
        self.spectrogram_data = spectrogram_data
        self.labels = labels

    def __len__(self):
        return len(self.spectrogram_data)

    def __getitem__(self, idx):
        spectrogram = self.spectrogram_data[idx]
        label = self.labels[idx]
        return spectrogram, label

class CNNold(nn.Module):
    def __init__(self, num_classes):
        super(CNNold, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(256*1*73, 128)  #hop 8 = 123, 16 = 61, 32 = 30 and 25 seconds | 160s -> hop 32 = 148
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 256*1*73)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.
    last_loss = 0.
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 50 == 49:
            last_loss = running_loss / 50 
            print('  batch {} loss: {}'.format(batch_idx + 1, last_loss))

            running_loss = 0.

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Train set accuracy: {accuracy}')

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss

def normalize_tensor(tensor):
    tensor = torch.from_numpy(tensor)
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    normalized_tensor = (tensor - min_value) / (max_value - min_value)
    return normalized_tensor

def get_absolute_tensor(signal):
    return torch.abs(signal)

def cut_and_pad(signal, sampling_rate, seconds):
    padded_signal = torch.zeros(sampling_rate * seconds)
    if signal.shape[0] < sampling_rate * seconds:
        padded_signal = torch.zeros(sampling_rate * seconds)
        padded_signal[:signal.shape[0]] = signal
    else:
        padded_signal = signal[:sampling_rate * seconds]

    return padded_signal

def load_data(filename):
    directory = f"../drive/MyDrive/emg-actionet/{filename}"
    emg_data = pd.read_pickle(directory)
    return emg_data

def n_sec_segmentation(t_L, t_R, sampling_rate=160, seconds=5):
    cut = sampling_rate * seconds

    list_L = []
    list_R = []

    if len(t_L) <= cut:
      list_L.append(t_L)
      list_R.append(t_R)
    else:
      while len(t_L) > cut:
        list_L.append(t_L[:cut])
        list_R.append(t_R[:cut])
        t_L = t_L[cut:]
        t_R = t_R[cut:]
        if len(t_L) <= cut:
          list_L.append(t_L)
          list_R.append(t_R)
          break

    return list_L, list_R

def extract_complete_spectrogram_stack_split_resampled(split):
    L = []
    R = []
    labels_L = []
    labels_R = []
    labels = []
    annotations_spectrograms = []

    emg_ann = pd.read_pickle(f'action-net/ActionNet_{split}.pkl')

    distinct_files = list(map(lambda x: x.split('.')[0].split('_'),emg_ann['file'].unique()))
    data = list()

    print(f"\nExtracting spectrograms ({split})...")
    for idx,file in enumerate(distinct_files):
        if (idx+1) % 7 == 0:
          print(f"{idx+1}/{len(distinct_files)}")
        subject_id, video = file
        file_name = f'{subject_id}_{video}.pkl'

        df_curr_file = emg_ann.query(f"file == '{file_name}'")

        indexes = list(df_curr_file['index'])
        data_byKey = load_data(file_name).loc[indexes]

        first = True
        annotations = load_data(file_name).loc[indexes]

        b, a = sp.signal.iirfilter(4, Wn=5.0, fs=160, btype="low", ftype="butter")
        mel_spectrogram.double()
        mfccs.double()
        spectrogram.double()
        for i in range(1, len(annotations)):
            len_L = len(annotations.iloc[i].myo_left_readings)
            len_R = len(annotations.iloc[i].myo_right_readings)
            min_len = min(len_L, len_R)
            t_L = annotations.iloc[i].myo_left_readings[:min_len]
            t_R = annotations.iloc[i].myo_right_readings[:min_len]
            list_L, list_R = n_sec_segmentation(t_L, t_R, 160, TIME_CUT)

            for left,right in zip(list_L, list_R):
                signal_left = torch.from_numpy(left).float()
                signal_right = torch.from_numpy(right).float()

                temp_L = []
                temp_R = []
                temp_size_L = 0
                temp_size_R = 0
                for j in range(8):

                    filtered_left = sp.signal.lfilter(b, a, get_absolute_tensor(signal_left[:, j]))
                    filtered_right = sp.signal.lfilter(b, a, get_absolute_tensor(signal_right[:, j]))
                    filtered_left = normalize_tensor(filtered_left)
                    filtered_right = normalize_tensor(filtered_right)
                    filtered_left = cut_and_pad(filtered_left, 160, TIME_CUT)
                    filtered_right = cut_and_pad(filtered_right, 160, TIME_CUT)
                    filtered_left = mel_spectrogram(torch.from_numpy(filtered_left.numpy()))
                    filtered_right = mel_spectrogram(torch.from_numpy(filtered_right.numpy()))

                    if first:
                        temp_L = filtered_left[None,:,:]
                        temp_R = filtered_right[None,:,:]
                    else:
                        temp_L = torch.cat((temp_L, filtered_left[None,:,:]), 0)
                        temp_R = torch.cat((temp_R, filtered_right[None,:,:]), 0)

                    first = False

                temp_size_L = filtered_left.shape[1]
                temp_size_R = filtered_right.shape[1]


                labels_L.append(dict_labels[annotations.iloc[i].description])


                labels_R.append(dict_labels[annotations.iloc[i].description])


                annotations_spectrograms.append(torch.cat((temp_L, temp_R),0))
                first = True



    if len(labels_L) > len(labels_R):
        labels = labels_L
    else:
        labels = labels_R

    return annotations_spectrograms, labels

def extract_complete_spectrogram_stack_split(split):
    L = []
    R = []
    labels_L = []
    labels_R = []
    labels = []
    annotations_spectrograms = []

    emg_ann = pd.read_pickle(f'action-net/ActionNet_{split}.pkl')

    distinct_files = list(map(lambda x: x.split('.')[0].split('_'),emg_ann['file'].unique()))
    data = list()

    print(f"\nExtracting spectrograms ({split})...")
    for idx,file in enumerate(distinct_files):
        if (idx+1) % 7 == 0:
          print(f"{idx+1}/{len(distinct_files)}")
        subject_id, video = file
        file_name = f'{subject_id}_{video}.pkl'

        df_curr_file = emg_ann.query(f"file == '{file_name}'")

        indexes = list(df_curr_file['index'])
        data_byKey = load_data(file_name).loc[indexes]

        first = True
        annotations = load_data(file_name).loc[indexes]

        b, a = sp.signal.iirfilter(4, Wn=5.0, fs=160, btype="low", ftype="butter")
        mel_spectrogram.double()
        mfccs.double()
        for i in range(1, len(annotations)):
            signal_left = torch.from_numpy(annotations.iloc[i].myo_left_readings).float()
            signal_right = torch.from_numpy(annotations.iloc[i].myo_right_readings).float()

            temp_L = []
            temp_R = []
            temp_size_L = 0
            temp_size_R = 0
            for j in range(8):

                filtered_left = sp.signal.lfilter(b, a, get_absolute_tensor(signal_left[:, j]))
                filtered_right = sp.signal.lfilter(b, a, get_absolute_tensor(signal_right[:, j]))
                filtered_left = normalize_tensor(filtered_left)
                filtered_right = normalize_tensor(filtered_right)
                filtered_left = cut_and_pad(filtered_left, 160, TIME_CUT)
                filtered_right = cut_and_pad(filtered_right, 160, TIME_CUT)
                filtered_left = mel_spectrogram(torch.from_numpy(filtered_left.numpy()))
                filtered_right = mel_spectrogram(torch.from_numpy(filtered_right.numpy()))

                if first:
                    temp_L = filtered_left[None,:,:]
                    temp_R = filtered_right[None,:,:]
                else:
                    temp_L = torch.cat((temp_L, filtered_left[None,:,:]), 0)
                    temp_R = torch.cat((temp_R, filtered_right[None,:,:]), 0)

                first = False

            temp_size_L = filtered_left.shape[1]
            temp_size_R = filtered_right.shape[1]


            labels_L.append(dict_labels[annotations.iloc[i].description])


            labels_R.append(dict_labels[annotations.iloc[i].description])


            annotations_spectrograms.append(torch.cat((temp_L, temp_R),0))
            first = True



    if len(labels_L) > len(labels_R):
        labels = labels_L
    else:
        labels = labels_R

    return annotations_spectrograms, labels

ST, labels_t = extract_complete_spectrogram_stack_split('train')
print(ST[0].shape)
print(ST[59].shape)
print(len(ST))
print(len(labels_t))

SV, labels_v = extract_complete_spectrogram_stack_split('test')
print(SV[0].shape)
print(SV[1].shape)
print(len(SV))
print(len(labels_v))


# Define data loaders
batch_size = 32
print("Setupping Dataloaders")
train_dataset = EMGDataset(ST, labels_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = EMGDataset(SV, labels_v)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_loader = val_loader
print("End")
print("Model definition")

# Define the model and optimizer
model = CNNold(11)
model = model.double()
optimizer = optim.Adam(model.parameters(), lr=0.0001) #0.0001
scheduler1 = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=20, power=1.0)
# Define the loss function
criterion = nn.CrossEntropyLoss()
model.to(DEVICE)
# Training loop
num_epochs = 20
best_val_acc = 0
print("End")
print("Start training")
for epoch in range(num_epochs):
    print('Epoch: {}'.format(epoch+1))
    train(model, train_loader, optimizer, criterion)
    v_loss = test(model, val_loader, criterion)
    scheduler1.step()

# Test the best model on the test set
# In this case test = val
test(model, test_loader, criterion)
