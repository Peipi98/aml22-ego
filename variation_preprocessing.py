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
    emg_annotations = pd.read_pickle(f"./Data/ActionNet/ActionNet-EMG/{filename}")

    sample_no = 1
    signal_left = torch.from_numpy(emg_annotations.iloc[sample_no].myo_left_readings).float()
    signal_right = torch.from_numpy(emg_annotations.iloc[sample_no].myo_right_readings).float()

    return signal_left, signal_right

def normalize_tensor(tensor):
    tensor = torch.from_numpy(tensor)
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    normalized_tensor = (tensor - min_value) / (max_value - min_value)
    return normalized_tensor

def get_absolute_tensor(signal):
    return torch.abs(signal)

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

emg_annotations = pd.read_pickle('./Data/ActionNet/ActionNet-EMG/S04_1.pkl')

sample_no = 1
signal_left = torch.from_numpy(emg_annotations.iloc[sample_no].myo_left_readings).float()
signal_right = torch.from_numpy(emg_annotations.iloc[sample_no].myo_right_readings).float()

b, a = sp.signal.iirfilter(4, Wn=5.0, fs=160, btype="low", ftype="butter")

filtered_l = []
filtered_r = []
for i in range(8):
  
  filtered_left = sp.signal.lfilter(b, a, get_absolute_tensor(signal_left[:, i]))
  filtered_right = sp.signal.lfilter(b, a, get_absolute_tensor(signal_right[:,i]))

  plot_signals(signal_left[:,i], normalize_tensor(filtered_left))
  plot_signals(signal_right[:,i], normalize_tensor(filtered_right))

  filtered_l.append(spectrogram(torch.from_numpy(normalize_tensor(filtered_left).numpy())))
  filtered_r.append(spectrogram(torch.from_numpy(normalize_tensor(filtered_right).numpy())))


plot_spectrogram(filtered_l)
plot_spectrogram(filtered_r)

