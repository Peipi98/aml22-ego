# Project Code for AML 2022 @ Politecnico di Torino

## Team members
- [Giuseppe Atanasio](s300733@studenti.polito.it)
- [Federico Mustich](federico.mustich@studenti.polito.it)
- [Francesco Sorrentino](s301655@studenti.polito.it)

To run the instructions, be careful to set the correct configs for the specific configuration you want to run.

## Feature Extraction

### 1. Extract EK-RGB features
`python save_feat.py config=configs/I3D_save_feat.yaml dataset.shift=D1-D1 name=save_feat_I3D_EK`

### 2. Resampling EMG for LSTM 
`python EMG/EMG_preprocessing.py`

### 3. Extract ActionSense RGB+EMG features
`python save_feat_action-net.py config=configs/I3D_save_feat.yaml dataset.shift=D1-S04 name=save_feat_I3D_AS`

## Training

### 1. Fully Connected Classifier 
`python train_classifier.py name=classifierD1 dataset.shift=D1-D1`

using Classifier2 on model inside configs/default.yaml

### 2. TRN Classifier
`python train_TRN.py name=classifierD1 dataset.shift=D1-D1`

using TRNClassifier on model inside configs/default.yaml

### 3. EMG-LSTM Classifier
`python EMG/EMG_train.py`

### 4. EMG-CNN Classifier
`python EMG_CNN.py`

### 5. Multimodal Classifier
`python train_multimodal.py name=classifierS04 dataset.shift=S04-S04`

using configs/multi_modalities.yaml
