import pickle as pk
import os
import matplotlib.pyplot as plt
import numpy as np

def load_acc_loss() -> dict:
    acc_loss=None
    with open(os.path.join(os.path.dirname(__file__), "models", "acc_loss.pkl"), "rb") as f:
        acc_loss = pk.load(f)
    return acc_loss


def plot_images(train, valid, type="Accuracy") -> None:
    epochs = np.array(range(1, len(train) + 1))
    plt.plot(epochs, train, 'b', label=f'Training {type}')
    plt.plot(epochs, valid, 'r', label=f'Validation {type}')
    plt.title(f'Training and Validation {type}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{type}')
    if type == "Accuracy":
        plt.ylim(0, 100)

    plt.legend()
    plt.savefig(f'{type}_plot.png')
    plt.show()

if __name__ == "__main__":
    acc_loss = load_acc_loss()

    plot_images(acc_loss["train_acc"], acc_loss["val_acc"], type="Accuracy")
    plot_images(acc_loss["train_loss"], acc_loss["val_loss"], type="Loss")