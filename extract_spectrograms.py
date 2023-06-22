import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchvision
import torch.optim as optim
import scipy as sp
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

SAVE_PATH = './emg_spectrograms/'

n_fft = 32
win_length = None
hop_length = 16

TIME_CUT = 30

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
    directory = f"./Data/ActionNet/ActionNet-EMG/{filename}"
    emg_data = pd.read_pickle(directory)
    zero = emg_data.loc[0].start
    return emg_data, zero

def save_pickle_pd(data, path):
    with open(os.path.join(path,f'Mel_S04.pkl'), 'wb') as f:
        pickle.dump(data, f)

def extract_complete_spectrogram_stack_split(file_name):
    L = []
    R = []
    labels_L = []
    labels_R = []
    labels = []
    annotations_spectrograms = []
    f_n  = file_name.split('.')[0]
    subject_id, video = f_n.split('_')[0], f_n.split('_')[1]

    file_name = f'{subject_id}_{video}.pkl'

    annotations, zero = load_data(file_name)

    fullspect = {'uid': [], 'subject': [], 'data': [], 'start_frame': [], 'stop_frame': []}
    
    first = True

    b, a = sp.signal.iirfilter(4, Wn=5.0, fs=160, btype="low", ftype="butter")
    mel_spectrogram.double()
    for i in range(1, len(annotations)):
        signal_left = torch.from_numpy(annotations.iloc[i].myo_left_readings).double()
        signal_right = torch.from_numpy(annotations.iloc[i].myo_right_readings).double()
        start_frame = int((annotations.iloc[i].start -zero)*30)
        stop_frame = int((annotations.iloc[i].stop -zero)*30)
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
            filtered_left = mel_spectrogram(torch.from_numpy(filtered_left.numpy())).type('torch.FloatTensor')
            filtered_right = mel_spectrogram(torch.from_numpy(filtered_right.numpy())).type('torch.FloatTensor')

            if first:
                temp_L = filtered_left[None,:,:]
                temp_R = filtered_right[None,:,:]
            else:
                temp_L = torch.cat((temp_L, filtered_left[None,:,:]), 0)
                temp_R = torch.cat((temp_R, filtered_right[None,:,:]), 0)

            first = False

        temp_size_L = filtered_left.shape[1]
        temp_size_R = filtered_right.shape[1]

        #dictionary with subject_id, 16channel full annotation, start, stop for every subject
        fullspect['uid'].append(i)
        fullspect['subject'].append(subject_id)
        fullspect['data'].append(torch.cat((temp_L, temp_R),0))
        fullspect['start_frame'].append(start_frame)
        fullspect['stop_frame'].append(stop_frame)
       
        labels_L.append(dict_labels[annotations.iloc[i].description])

        
        labels_R.append(dict_labels[annotations.iloc[i].description])
        
        annotations_spectrograms.append(torch.cat((temp_L, temp_R),0))
        first = True
            
        if i%10 == 0:
            print(i)            

    if len(labels_L) > len(labels_R):
        labels = labels_L
    else:
        labels = labels_R

    return annotations_spectrograms, labels, pd.DataFrame.from_dict(fullspect)


_, _, f = extract_complete_spectrogram_stack_split('S04_1.pkl')

print(f.loc[50].data[0])

save_pickle_pd(f, SAVE_PATH)
#save_pickle_pd(f[f['subject'] == 'S04'], SAVE_PATH, 'test')

