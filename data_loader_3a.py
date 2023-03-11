import numpy as np
from scipy import interpolate # for resampling
from scipy.signal import butter, lfilter # for filtering
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from collections import OrderedDict
import os, glob
import torch
script_dir = os.path.dirname(os.path.realpath(__file__))

##############################
#IMPLEMENTARE CARICAMENTO DATI
##############################
def load_data(filename):
    emg_data = pd.read_pickle(f"./Data/ActionNet/ActionNet-EMG/{filename}")
    return emg_data


#TEST PER UN SOGGETTO
emg = load_data("S04_1.pkl")
#Subject structure:
#myo-right: readings + timestamps | data + time_s 
#myo-left:  readings + timestamps | data + time_s
#start: start label time
#stop: stop label time
#description: label

#dictionary[subject][myo-arm][data]
#dictionary[subject][myo-arm][time_s]

# We need three nested dictionaries
# First level: subject
# Second level: myo-arm
# Third level: myo-arm-content
# INPUT: expirements
# DESIRED OUTPUT: 100x16 matricies for each example.
# To do that we need to preprocess the data and then 
# We have to concatenete the 2 100x8 matricies.
# Each sequence has to be labeled with the start and stop timestamps.
# So we can make a dictionary like this:
# dictionary[subject][(Matrix, label)]


emg_annotations = pd.read_pickle('./action-net/ActionNet_train.pkl')
'''
print(emg_annotations.keys())
print(emg_annotations.query("file == 'S04_1.pkl'")[['index','file']].head(5))
'''

def data_loader(emg_ann):
    #['index', 'file', 'description', 'labels']
    #data_bySubject = map(lambda x: x.split('.')[0].split('_'), emg_ann['file']).distinct()
    data_bySubject = dict()
    print(data_bySubject)
    for (index, row) in emg_ann.iterrows():
        key = row['file']
        #Load for each index at specified file
        data_byKey = load_data(key)[index]

        subject_id, video = key.split('.')[0].split('_')
        if subject_id not in data_bySubject or video not in data_bySubject[subject_id]:
            data_bySubject[subject_id][video] = OrderedDict((index, data_byKey))
        else:
            data_bySubject[subject_id][video].add(index)
    return data_bySubject

emg_data = data_loader(emg_annotations)
#print(emg_data)

# Define segmentation parameters.
resampled_Fs = 10 # define a resampling rate for all sensors to interpolate
num_segments_per_subject = 20
num_baseline_segments_per_subject = 20 # num_segments_per_subject*(max(1, len(activities_to_classify)-1))
segment_duration_s = 10
segment_length = int(round(resampled_Fs*segment_duration_s))
buffer_startActivity_s = 2
buffer_endActivity_s = 2

# Define filtering parameters.
filter_cutoff_emg_Hz = 5
filter_cutoff_tactile_Hz = 2
filter_cutoff_gaze_Hz = 5
num_tactile_rows_aggregated = 4
num_tactile_cols_aggregated = 4

# 

###########################################
#IMPLEMENTARE PREPROCESSING [ABS + LOWPASS]
###########################################
def lowpass_filter(data, cutoff, Fs, order=5):
  nyq = 0.5 * Fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = lfilter(b, a, data.T).T
  return y

data_dict = {"myo_right_readings": [], "myo_left_timestamps": [], "myo_right_timestamps": [], "myo_left_readings": []}

for key in ['myo_right', 'myo_left']:
    for i in range(60):
        if i == 0:
            #calibration
            continue
        data = abs(emg[key + '_readings'][i])
        t =  emg[key + '_timestamps'][i]
        Fs = (t.size -1) / (t[-1] - t[0])
        y =  lowpass_filter(data, 5, Fs)
        # Normalization
        y = y / ((np.amax(y) - np.amin(y))/2)
        # Jointly shift the baseline to -1 instead of 0.
        y = y - np.amin(y) - 1
    data_dict[key + '_readings'].append(y)
    data_dict[key + '_timestamps'].append(t)