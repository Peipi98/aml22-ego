'''
INSTRUCTIONS

In order to retrieve the desirable preprocessing split,
please modify the variable `split` and write one of the following:
    ['train', 'test']
'''

import numpy as np
from scipy import interpolate # for resampling
from scipy.signal import butter, lfilter # for filtering
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from collections import OrderedDict
import os, glob
import sys 

stdoutOrigin=sys.stdout 
sys.stdout = open("log.txt", "w")

script_dir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(script_dir, 'EMG_preprocessed')
split = 'test'

activities_to_classify = [
  'Get/replace items from refrigerator/cabinets/drawers',
  'Peel a cucumber',
  'Clear cutting board',
  'Slice a cucumber',
  'Peel a potato',
  'Slice a potato',
  'Slice bread',
  'Spread almond butter on a bread slice',
  'Spread jelly on a bread slice',
  'Open/close a jar of almond butter',
  'Pour water from a pitcher into a glass',
  'Clean a plate with a sponge',
  'Clean a plate with a towel',
  'Clean a pan with a sponge',
  'Clean a pan with a towel',
  'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  'Stack on table: 3 each large/small plates, bowls',
  'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  ]
# Some older experiments may have had different labels.
#  Each entry below maps the new name to a list of possible old names.
activities_renamed = {
  'Open/close a jar of almond butter': ['Open a jar of almond butter'],
  'Get/replace items from refrigerator/cabinets/drawers': ['Get items from refrigerator/cabinets/drawers'],
}

##############################
#IMPLEMENTARE CARICAMENTO DATI
##############################
def load_data(filename):
    emg_data = pd.read_pickle(f"Data/ActionNet/ActionNet-EMG/{filename}")
    return emg_data

def data_loader(emg_ann):
    '''
    Retrieve data from emg_annotations file (train or test).
    Then it loads each experiment in the dictionary 'data_bySubject'
    starting from their header:

    `['index', 'file', 'description', 'labels']`

    data_bySubject[subject_id][video] contains the entire dataframe for
    that specific experiment, whom have the following header:

    `['description', 'start', 'stop', 'myo_left_timestamps', 'myo_left_readings',
    'myo_right_timestamps', 'myo_right_readings']`

    '''
    #['index', 'file', 'description', 'labels']
    #print(data_bySubject)

    distinct_files = list(map(lambda x: x.split('.')[0].split('_'), emg_ann['file'].unique()))
    data_bySubject = dict()

    for file in distinct_files:
        subject_id, video = file
        file_name = f'{subject_id}_{video}.pkl'
    
        df_curr_file = emg_ann.query(f"file == '{file_name}'")
        
        indexes = list(df_curr_file['index'])
        data_byKey = load_data(file_name).loc[indexes]

        if subject_id not in data_bySubject:
            data_bySubject[subject_id] = dict()
        data_bySubject[subject_id][video] = data_byKey

    return data_bySubject

#TEST PER UN SOGGETTO
# emg = load_data("S04_1.pkl")

emg_annotations = pd.read_pickle(f'action-net/ActionNet_{split}.pkl')
emg = data_loader(emg_annotations)

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

#emg_annotations = pd.read_pickle('./action-net/ActionNet_train.pkl')

print(emg.keys())

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

for (subject_id, content) in emg.items():
    for (video, df) in content.items():
        for key in ['myo_right', 'myo_left']:
            for i, _ in df.iterrows():
                if i == 0:
                    #calibration
                    continue
                data = abs(emg[subject_id][video].loc[i, key + '_readings'])
                t =  emg[subject_id][video].loc[i, key + '_timestamps']
                Fs = (t.size -1) / (t[-1] - t[0])
                y =  lowpass_filter(data, 5, Fs)
                # Normalization
                y = y / ((np.amax(y) - np.amin(y))/2)
                # Jointly shift the baseline to -1 instead of 0.
                y = y - np.amin(y) - 1
                emg[subject_id][video].at[i, key + '_readings'] = y 
                emg[subject_id][video].at[i, key + '_timestamps'] = t
                #data_dict[key + '_readings'].append(y)
                #data_dict[key + '_timestamps'].append(t)
#########
#RESAMPLE
#########

#This is an example of resample of one row. Now we have to do it for every row.
#It resamples from Fs (around 160Hz) to 10Hz
#(Number sample / Fs )* 10 = New number of sample
#We do that through interpolation to preserve informations and timestamps
#It can happen that some timestamps doens't have any reading
#In this case we convert "NaN" into 0.

for (subject_id, content) in emg.items():
    for (video, df) in content.items():
        for key in ['myo_right', 'myo_left']:
            for i, _ in df.iterrows():
                if i == 0:
                    #calibration
                    continue
                #data1 = np.squeeze(np.array(data_dict['myo_right_readings'][1]))
                data1 = np.squeeze(np.array(emg[subject_id][video].loc[i, f'{key}_readings']))
                # print(f'data: ', data1.shape)
                #time_s = np.squeeze(np.array(data_dict['myo_right_timestamps'][1]))
                time_s = np.squeeze(np.array(emg[subject_id][video].loc[i, f'{key}_timestamps']))

                target_time_s = np.linspace(time_s[0], time_s[-1],
                                                num=int(round(1+resampled_Fs*(time_s[-1] - time_s[0]))),
                                                endpoint=True)
                fn_interpolate = interpolate.interp1d(
                        time_s, # x values
                        data1,   # y values
                        axis=0,              # axis of the data along which to interpolate
                        kind='linear',       # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
                        fill_value='extrapolate' # how to handle x values outside the original range
                    )

                data_resampled = fn_interpolate(target_time_s)
                if np.any(np.isnan(data_resampled)):
                        print('\n'*5)
                        print('='*50)
                        print('='*50)
                        print('FOUND NAN')
                        timesteps_have_nan = np.any(np.isnan(data_resampled), axis=tuple(np.arange(1,np.ndim(data_resampled))))
                        print('Timestep indexes with NaN:', np.where(timesteps_have_nan)[0])
                        print('\n'*5)
                        data_resampled[np.isnan(data_resampled)] = 0
                #print((time_s.size -1) / (time_s[-1] - time_s[0]))
                # print(f'{subject_id=} | {video=}')
                # print(len(data1))
                # print(len(data_resampled))
                # print(len(emg[subject_id][video].loc[i, f'{key}_readings']))
                emg[subject_id][video].at[i, f'{key}_readings'] = data_resampled
                emg[subject_id][video].at[i, f'{key}_timestamps'] = target_time_s
#print(emg[['myo_right_readings','myo_right_timestamps']])

'''
      data = np.squeeze(np.array(file_data[device_name][stream_name]['data']))
      time_s = np.squeeze(np.array(file_data[device_name][stream_name]['time_s']))
      target_time_s = np.linspace(time_s[0], time_s[-1],
                                  num=int(round(1+resampled_Fs*(time_s[-1] - time_s[0]))),
                                  endpoint=True)
      fn_interpolate = interpolate.interp1d(
          time_s, # x values
          data,   # y values
          axis=0,              # axis of the data along which to interpolate
          kind='linear',       # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
          fill_value='extrapolate' # how to handle x values outside the original range
      )
      data_resampled = fn_interpolate(target_time_s)
      if np.any(np.isnan(data_resampled)):
        print('\n'*5)
        print('='*50)
        print('='*50)
        print('FOUND NAN')
        print(subject_id, device_name, stream_name)
        timesteps_have_nan = np.any(np.isnan(data_resampled), axis=tuple(np.arange(1,np.ndim(data_resampled))))
        print('Timestep indexes with NaN:', np.where(timesteps_have_nan)[0])
        print_var(data_resampled)
        # input('Press enter to continue ')
        print('\n'*5)
        time.sleep(10)
        data_resampled[np.isnan(data_resampled)] = 0
      file_data[device_name][stream_name]['time_s'] = target_time_s
      file_data[device_name][stream_name]['data'] = data_resampled
    data_bySubject[subject_id][data_file_index] = file_data

print(data_dict)
'''

########################################
#IMPLEMENTARE SEGMENTAZIONE E DATALOADER
#######################################

#We take one stream of data and we segment it creating a matrix of 100x8 for each arm.
#We take the resampled data and we create features matricies.

'''
    TEMPORARY:
    -   Added `correct_shape` boolean to handle the case in which
        left and right readings don't match the shape 100x16.
        At the moment there isn't 'raise AssertionError'.

    MODIFICATIONS:
    -   The buffers for `start_time_s` and `end_time_s` are
        commented since the dataset is already cleaned.
'''
example_matrices_byLabel = {}

for (subject_id, content) in emg.items():
    for (video, dataframe) in content.items():
        for (label_index, activity_label) in enumerate(activities_to_classify):
            # Extract num_segments_per_subject examples from each instance of the activity.
            # Then later, will select num_segments_per_subject in total from all instances.
            activities_labels = dataframe['description'].copy()
            file_label_indexes = [i for (i, label) in activities_labels.items() if label==activity_label]
            if len(file_label_indexes) == 0 and activity_label in activities_renamed:
                for alternate_label in activities_renamed[activity_label]:
                    file_label_indexes = [i for (i, label) in activities_labels.items() if label==activity_label]
                    if len(file_label_indexes) > 0:
                        print('  Found renamed activity from "%s"' % alternate_label)
                        break
            print('  Found %d instances of %s' % (len(file_label_indexes), activity_label))

            count_iter = 0
            for i in file_label_indexes:
                
                start = emg[subject_id][video].loc[i, 'start']
                end = emg[subject_id][video].loc[i, 'stop']

                #we try to extract 20 segments of 10s at 10Hz (100)
                start_time_s = start #+ 0.5
                end_time_s = end #- 0.5
                duration_s = end_time_s - start_time_s

                # Extract example segments and generate a feature matrix for each one.
                num_examples = num_segments_per_subject
                print('  Extracting %d examples from activity "%s" with duration %0.2fs' % (num_examples, activity_label, duration_s))
                
                segment_start_times_s = np.linspace(start_time_s, end_time_s - 10.0,
                                                        num = num_examples,
                                                        endpoint=True)
                
                feature_matrices = []
                correct_shape = True

                for segment_start_time_s in segment_start_times_s:
                    # print('Processing segment starting at %f' % segment_start_time_s)
                    segment_end_time_s = segment_start_time_s + 10.0
                    feature_matrix = np.empty(shape=(100, 0))
                    for key in ['myo_right', 'myo_left']:
                        # print(' Adding data from [%s][%s]' % (device_name, stream_name))
                        data = np.squeeze(np.array(emg[subject_id][video].loc[i, f'{key}_readings']))
                        time_s = np.squeeze(np.array(emg[subject_id][video].loc[i, f'{key}_timestamps']))
                        time_indexes = np.where((time_s >= segment_start_time_s) & (time_s <= segment_end_time_s))[0]
                        # Expand if needed until the desired segment length is reached.
                        time_indexes = list(time_indexes)
                        while len(time_indexes) < 100:
                            # print(' Increasing segment length from %d to %d for segment starting at %f' % (len(time_indexes), 100, segment_start_time_s))
                            if time_indexes[0] > 0:
                                time_indexes = [time_indexes[0]-1] + time_indexes
                            elif time_indexes[-1] < len(time_s)-1:
                                time_indexes.append(time_indexes[-1]+1)
                            else:
                                # print('time_indexes: ', len(time_indexes))
                                # print(f'{correct_shape=}')
                                correct_shape = False
                                break
                        while len(time_indexes) > 100:
                            # print(' Decreasing segment length from %d to %d for segment starting at %f' % (len(time_indexes), 100, segment_start_time_s))
                            time_indexes.pop()
                        time_indexes = np.array(time_indexes)
                            
                        # Extract the data.
                        time_s = time_s[time_indexes]
                        data = data[time_indexes,:]

                        # # Zero padding over the feature matrices with shape[0] < 100 (ex. (54, 8)->(100, 8))
                        # if not correct_shape:
                        #     #[time_indexes.append(len(time_indexes) + _) for _ in range(100-len(time_indexes))]
                        #     data = np.pad(data, [(0, 100-data.shape[0]), (0, 0)], mode='constant', constant_values=0)
                        #     print(data.shape)
                        
                        # data = np.reshape(data, (segment_length, -1))
                        # print(data.shape)
                        # print('  Got data of shape', data.shape)
                        #try:
                        if correct_shape:
                            feature_matrix = np.concatenate((feature_matrix, data), axis=1)
                        #except:
                            #correct_shape = False
                    if correct_shape:
                        feature_matrices.append(feature_matrix)
                    #reset boolean
                correct_shape = True
                print()
                example_matrices_byLabel.setdefault(activity_label, [])
                example_matrices_byLabel[activity_label].extend(feature_matrices)
                # print(len(feature_matrices))
                # print(len(feature_matrices[count_iter]))
                # print(len(feature_matrices[count_iter][0]))
                # if correct_shape:
                #     count_iter += 1

df_list = []
for activity_label in example_matrices_byLabel:
    len_dict = len(example_matrices_byLabel[activity_label])
    features = example_matrices_byLabel[activity_label]
    labels = [activity_label] * len_dict

    df_activity = pd.DataFrame({"features": features, 'activity_label': labels})
    df_list.append(df_activity)
df_save = pd.concat(df_list)
df_save.to_pickle(os.path.join(save_dir, f'{split}_EMG_preprocess.pkl'))
# with open(os.path.join(save_dir, f'{split}_EMG_preprocess.pkl'), 'wb') as f:
    # pickle.dump(example_matrices_byLabel, f)


sys.stdout.close()
sys.stdout=stdoutOrigin
