import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
import scipy as sp
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 39 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 39 * 5)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train(model, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

def normalize_tensor(tensor):
    tensor = torch.from_numpy(tensor)
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    normalized_tensor = (tensor - min_value) / (max_value - min_value)
    return normalized_tensor

def get_absolute_tensor(signal):
    return torch.abs(signal)

def extract_complete_spectrogram(filename):
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

L, R, labels = extract_complete_spectrogram('./Data/ActionNet/ActionNet-EMG/S04_1.pkl')

print("LESSS GOOOOOO")

# Data preprocessing
# load and preprocess your spectrogram data
#spectrogram_data = # DA CAPIRE COME GESTIRE IL FATTO CHE I NOSTRI SPETTROGRAMMI SONO DUE LISTE DI OTTO MATRICI 2D 
#labels = # Labels is a list of integers representing the target labels for each time step

# Split data into training, validation, and test sets
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

num_samples = len(spectrogram_data)
train_samples = int(train_ratio * num_samples)
val_samples = int(val_ratio * num_samples)
test_samples = num_samples - train_samples - val_samples

train_data = spectrogram_data[:train_samples]
train_labels = labels[:train_samples]
val_data = spectrogram_data[train_samples:train_samples+val_samples]
val_labels = labels[train_samples:train_samples+val_samples]
test_data = spectrogram_data[-test_samples:]
test_labels = labels[-test_samples:]

# Define data loaders
batch_size = 64

train_dataset = EMGDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = EMGDataset(val_data, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = EMGDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model and optimizer
model = CNN(num_classes= 20)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
best_val_acc = 0

for epoch in range(num_epochs):
    train(model, train_loader, optimizer, criterion)
    print('Epoch: {}'.format(epoch+1))
    test(model, val_loader, criterion)
    
    with torch.no_grad():
        model.eval()
        output = model(val_data.unsqueeze(1).float())
        pred = output.argmax(dim=1)
        val_acc = (pred == val_labels).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')

# Test the best model on the test set
model.load_state_dict(torch.load('best_model.pt'))
test(model, test_loader, criterion)