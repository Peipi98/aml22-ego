import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import pickle as pk
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import butter, lfilter, freqz

# Sampling frequency is 160 Hz
# With 32 samples the frequency resolution after FFT is 160 / 32

n_fft = 32
win_length = None
hop_length = 4

spectrogram = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    normalized=True
)

#mel_spectrogram = T.MelScale(
mel_spectrogram = T.MelSpectrogram(
    n_mels=5,
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

dict_labels = {
    "Get/replace items from refrigerator/cabinets/drawers": 0,
    "Peel a cucumber" : 1,
    "Clear cutting board": 2,
    "Slice a cucumber": 3,
    "Peel a potato": 4,
    "Slice a potato": 5,
    "Slice bread": 6,
    "Spread almond butter on a bread slice": 7,
    "Spread jelly on a bread slice": 8,
    "Open/close a jar of almond butter": 9,
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

def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(len(specgram), 1, figsize=(16, 8))

    axs[0].set_title(title or "Spectrogram (db)")

    for i, spec in enumerate(specgram):
        im = axs[i].imshow(librosa.power_to_db(specgram[i]), origin="lower", aspect="auto")
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)

    axs[i].set_xlabel("Frame number")
    axs[i].get_xaxis().set_visible(True)
    plt.show(block=False)

def get_signals_from(filename):
    emg_annotations = pd.read_pickle(f"../Data/ActionNet/ActionNet-EMG/{filename}")

    sample_no = 1
    signal_left = torch.from_numpy(emg_annotations.iloc[sample_no].myo_left_readings).float()
    signal_right = torch.from_numpy(emg_annotations.iloc[sample_no].myo_right_readings).float()

    return signal_left, signal_right

def compute_spectrogram(signal, title):
    freq_signal = [spectrogram(signal[:, i], 5, 32.0, 6) for i in range(8)]
    plot_spectrogram(freq_signal, title=title)

def plot_signals(signal_A, signal_B):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(signal_A)
    plt.subplot(212)
    plt.plot(signal_B)
    plt.show()

def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(len(specgram), 1, figsize=(16, 8))

    axs[0].set_title(title or "Spectrogram (db)")

    for i, spec in enumerate(specgram):
        im = axs[i].imshow(librosa.power_to_db(specgram[i]), origin="lower", aspect="auto")
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)

    axs[i].set_xlabel("Frame number")
    axs[i].get_xaxis().set_visible(True)
    plt.show()

def normalize_tensor(tensor):
    tensor = torch.from_numpy(tensor)
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    normalized_tensor = (tensor - min_value) / (max_value - min_value)
    return normalized_tensor

def get_absolute_tensor(signal):
    return torch.abs(signal)

def extract_complete_spectrogram_sequential(filename):
    L = []
    R = []
    labels_L = []
    labels_R = []
    labels = []

    first = True
    annotations = pd.read_pickle(filename)

    b, a = sp.signal.iirfilter(4, Wn=5.0, fs=160, btype="low", ftype="butter")

    print("\nExtracting spectrograms...")
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

            filtered_left = spectrogram(torch.from_numpy(filtered_left.numpy()))
            filtered_right = spectrogram(torch.from_numpy(filtered_right.numpy()))

            temp_L.append(filtered_left)
            temp_R.append(filtered_right)

        temp_size_L = filtered_left.shape[1]
        temp_size_R = filtered_right.shape[1]
        
        for _ in range(temp_size_L):
            labels_L.append(dict_labels[annotations.iloc[i].description])

        for _ in range(temp_size_R):
            labels_R.append(dict_labels[annotations.iloc[i].description])

        for k in range(8):
            if first:
                L.append(temp_L[k])
                R.append(temp_R[k])                
            else:
                L[k] = torch.cat((L[k], temp_L[k]), 1)
                R[k] = torch.cat((R[k], temp_R[k]), 1)

        first = False

    if len(labels_L) > len(labels_R):
        labels = labels_L
    else:
        labels = labels_R

    return L, R, labels


emg_annotations = pd.read_pickle('../Data/ActionNet/ActionNet-EMG/S04_1.pkl')

sample_no = 1
signal_left = torch.from_numpy(emg_annotations.iloc[sample_no].myo_left_readings).float()
signal_right = torch.from_numpy(emg_annotations.iloc[sample_no].myo_right_readings).float()

b, a = sp.signal.iirfilter(4, Wn=5.0, fs=160, btype="low", ftype="butter")

filtered_l = []
filtered_r = []
mel_spectrogram.double()
for i in range(8):  
  filtered_left = sp.signal.lfilter(b, a, get_absolute_tensor(signal_left[:, i]))
  filtered_right = sp.signal.lfilter(b, a, get_absolute_tensor(signal_right[:,i]))

  #plot_signals(signal_left[:,i], normalize_tensor(filtered_left))
  #plot_signals(signal_right[:,i], normalize_tensor(filtered_right))
  #filtered_l.append(spectrogram(torch.from_numpy(normalize_tensor(filtered_left).numpy())))
  #filtered_r.append(spectrogram(torch.from_numpy(normalize_tensor(filtered_right).numpy())))
  filtered_l.append(mel_spectrogram(torch.from_numpy(normalize_tensor(filtered_left).numpy())))
  filtered_r.append(mel_spectrogram(torch.from_numpy(normalize_tensor(filtered_right).numpy())))

#print(extract_labels(filtered_l, emg_annotations))
plot_spectrogram(filtered_l)
plot_spectrogram(filtered_r)

#L, R, labels = extract_complete_spectrogram('../Data/ActionNet/ActionNet-EMG/S04_1.pkl')
#print(L[0].shape)
#print(R[0].shape)
#print(len(labels))