import numpy as np

def normalize_tensor(tensor):
    min_value = np.min(tensor)
    max_value = np.max(tensor)
    normalized_tensor = (tensor - min_value) / (max_value - min_value) * 2 - 1
    return normalized_tensor