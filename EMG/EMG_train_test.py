import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from torch.optim.lr_scheduler import LRScheduler

import numpy as np
import os
import pickle as pk

from EMG_dataset import EMG_dataset
from EMG_LSTM import EMG_LSTM
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(batch_size=32):
    directory = os.path.join(os.path.dirname(__file__), "EMG_preprocessed")
    train_set = EMG_dataset(directory, 'train_EMG_preprocess.pkl')
    test_set = EMG_dataset(directory, 'test_EMG_preprocess.pkl')

    train_indexes, val_indexes = train_test_split(
        range(len(train_set)),
        test_size=0.2,
        shuffle=True,
        stratify=[y for (x, y) in train_set]
    )

    train_set, val_set = Subset(train_set, train_indexes), Subset(train_set, val_indexes)

    train_dataset = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_dataset = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    test_dataset = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_dataset, val_dataset, test_dataset


def train(model, train_dataloader, val_dataloader, num_epochs=20, save_model=False):
    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_acc = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_running_loss = 0.0
        train_correct_predictions = 0
        train_total_samples = 0
        
        for _, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total_samples += labels.size(0)
            train_correct_predictions += (predicted == labels).sum().item()
            
            train_running_loss += loss.item()
        
        train_epoch_loss = train_running_loss / len(train_dataloader)
        train_epoch_accuracy = 100 * train_correct_predictions / train_total_samples
        train_acc[epoch] = train_epoch_accuracy
        train_loss[epoch] = train_epoch_loss

        # Validation
        model.eval()
        val_correct_predictions = 0
        val_total_samples = 0
        running_vloss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
                
                # Forward pass
                outputs = model(inputs)
                vloss = criterion(outputs, labels)
                running_vloss += vloss
                # Track accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()
        
        val_epoch_accuracy = 100 * val_correct_predictions / val_total_samples
        val_acc[epoch] = val_epoch_accuracy
        avg_vloss = running_vloss / len(val_dataloader)
        val_loss[epoch] = avg_vloss

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accuracy:.4f}, ', f'Val Loss: {avg_vloss:.4f}, '
              f'Val Accuracy: {val_epoch_accuracy:.4f}')
        
    if save_model:
        torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "models", "emg_model.pth"))
        with open(os.path.join(os.path.dirname(__file__), "models", "acc_loss.pkl"), "wb") as f:
            acc_loss = {
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss
            }
            pk.dump(acc_loss, f)

    return (train_acc, train_loss), (val_acc, val_loss)

def test(model, dataloader):    
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            
            # Forward pass
            outputs = model(inputs)
            
            # Track accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    accuracy = correct_predictions / total_samples
    print(f"Test Accuracy:\n{accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    train_data, val_data, test_data = load_data(batch_size=32)

    model = EMG_LSTM(20)
    model.to(device)
    train(model, train_data, val_data, num_epochs=200, save_model=True)
    test(model, test_data)