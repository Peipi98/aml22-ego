import numpy as np
from scipy import interpolate # for resampling
from scipy.signal import butter, lfilter # for filtering
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from collections import OrderedDict
import os, glob
import torch

##############################
#IMPLEMENTARE CARICAMENTO DATI
##############################
def load_data(filename):
    emg_annotations = pd.read_pickle(f"./Data/ActionNet/ActionNet-EMG/{filename}")
    return emg_annotations


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
# Third level: myo-arm-contentù
# INPUT: expirements
# DESIRED OUTPUT: 100x16 matricies for each example.
# To do that we need to preprocess the data and then 
# We have to concatenete the 2 100x8 matricies.
# Each sequence has to be labeled with the start and stop timestamps.
# So we can make a dictionary like this:
# dictionary[subject][(Matrix, label)]



print(emg.keys())

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



print(data_dict)

########################################
#IMPLEMENTARE SEGMENTAZIONE E DATALOADER
#######################################

#We take one stream of data and we segment it creating a matrix of 100x8 for each arm.
#To do so we have to re-sampling it at 10Hz and taking 10s of 10Hz = 100 samples.
#

def get_feature_matrices(data, start, end, count=20):
    # Determine start/end times for each example segment.
    start_time_s = start + 0.5
    end_time_s = end - 0.5
    segment_start_times_s = np.linspace(start_time_s, end_time_s - 10,
                                        num=count,
                                        endpoint=True)
    # Create a feature matrix by concatenating each desired sensor stream.
    feature_matrices = []
    for segment_start_time_s in segment_start_times_s:
        # print('Processing segment starting at %f' % segment_start_time_s)
        segment_end_time_s = segment_start_time_s + 10
        feature_matrix = np.empty(shape=(100, 0))
        for (device_name, stream_name, extraction_fn) in data:
            # print(' Adding data from [%s][%s]' % (device_name, stream_name))
            data = np.squeeze(np.array(data[device_name][stream_name]['data']))
            time_s = np.squeeze(np.array(data[device_name][stream_name]['time_s']))
            time_indexes = np.where((time_s >= segment_start_time_s) & (time_s <= segment_end_time_s))[0]
            # Expand if needed until the desired segment length is reached.
            time_indexes = list(time_indexes)
            while len(time_indexes) < 100:
                print(' Increasing segment length from %d to %d for %s %s for segment starting at %f' % (len(time_indexes), 100, device_name, stream_name, segment_start_time_s))
                if time_indexes[0] > 0:
                    time_indexes = [time_indexes[0]-1] + time_indexes
                elif time_indexes[-1] < len(time_s)-1:
                    time_indexes.append(time_indexes[-1]+1)
                else:
                    raise AssertionError
            while len(time_indexes) > 100:
                print(' Decreasing segment length from %d to %d for %s %s for segment starting at %f' % (len(time_indexes), 100, device_name, stream_name, segment_start_time_s))
                time_indexes.pop()
            time_indexes = np.array(time_indexes)
            
            # Extract the data.
            time_s = time_s[time_indexes]
            data = data[time_indexes,:]
            data = extraction_fn(data)
            # print('  Got data of shape', data.shape)
            feature_matrix = np.concatenate((feature_matrix, data), axis=1)
        feature_matrices.append(feature_matrix)
    return feature_matrices

