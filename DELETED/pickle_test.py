import pandas as pd
import pickle
import os

EK_PATH = '/home/h4r/Desktop/advanced-machine-learning/project/fork/aml22-ego/train_val/D1_test.pkl'
AN_PATH = '/home/h4r/Desktop/advanced-machine-learning/project/fork/aml22-ego/action-net/S04_1.pkl'

#Load pickle file
def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# Load pickle from pandas
def load_pickle_pd(path):
    with open(path, 'rb') as f:
        data = pd.read_pickle(f)
    return data

#print(load_pickle(EK_PATH).keys())
#print(load_pickle_pd(AN_PATH))
print(load_pickle(EK_PATH))
a = load_pickle_pd(AN_PATH)
#print(a['stop'][0]-a['start'][0])
'''
EK = 'uid', 'participant_id', 'video_id', 'narration', 'start_timestamp', 'stop_timestamp', 'start_frame', 'stop_frame', 'verb', 'verb_class'
AN = 'description', 'start', 'stop', 'myo_left_timestamps', 'myo_left_readings', 'myo_right_timestamps', 'myo_right_readings'
'''
verbs_class = {
        "Get": 0,
        "Peel" : 1,
        "Clear": 2,
        "Slice": 3,
        "Spread": 4,
        "Open/Close": 5,
        "Pour": 6,
        "Clean":  7,
        "Set": 8,
        "Stack": 9,
        "Load": 10,
    }
verbs = {
        "Get/replace items from refrigerator/cabinets/drawers": "Get",
        "Get items from refrigerator/cabinets/drawers": "Get",
        "Peel a cucumber" : "Peel",
        "Clear cutting board": "Clear",
        "Slice a cucumber": "Slice",
        "Peel a potato": "Peel",
        "Slice a potato": "Slice",
        "Slice bread": "Slice",
        "Spread almond butter on a bread slice": "Spread",
        "Spread jelly on a bread slice": "Spread",
        "Open/close a jar of almond butter": "Open/Close",
        "Open a jar of almond butter": "Open/Close",
        "Pour water from a pitcher into a glass": "Pour",
        "Clean a plate with a sponge":  "Clean",
        "Clean a plate with a towel": "Clean",
        "Clean a pan with a sponge":  "Clean",
        "Clean a pan with a towel": "Clean",
        "Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": "Get",
        "Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": "Set",
        "Stack on table: 3 each large/small plates, bowls": "Stack",
        "Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": "Load",
        "Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": "Load",
    }
# Get an AN pickle file and return a EK pickle file
def get_EK_from_AN(AN_pickle, filename):
    video_id = filename
    p_id = filename.split('_')[0]
    zero_time = 0
    EK_pickle = {'uid' : [], 'participant_id' : [], 'video_id' : [], 'narration' : [], 'start_timestamp' : [], 'stop_timestamp' : [], 'start_frame' : [], 'stop_frame' : [], 'verb' : [], 'verb_class' : []}
    for idx, row in AN_pickle.iterrows():
        if idx == 0:
            zero_time = row['start']
            continue
        EK_pickle['uid'].append(idx)
        EK_pickle['participant_id'].append(p_id)
        EK_pickle['video_id'].append(video_id)
        EK_pickle['narration'].append(row['description'])
        EK_pickle['start_timestamp'].append(row['start']-zero_time)
        EK_pickle['stop_timestamp'].append(row['stop']-zero_time)
        EK_pickle['start_frame'].append(int((row['start']-zero_time)*30))
        EK_pickle['stop_frame'].append(int((row['stop']-zero_time)*30))
        EK_pickle['verb'].append(verbs[row['description']])
        EK_pickle['verb_class'].append(verbs_class[verbs[row['description']]])
    
    return EK_pickle

b= pd.DataFrame.from_dict(get_EK_from_AN(a,'S04_1'))
print(a)
print(b)
# EX tuple AN
# AN | 1, S04, S04_1, Open a jar of peanut butter, 00:00:01.00, 00:01:00.00, ?, ? , Open, 1 |    

