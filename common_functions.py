import matplotlib.pyplot as plt 
from sklearn.preprocessing import normalize
from scipy.signal import argrelextrema
import numpy as np 
import pandas as pd 


def load_ecg(csv_path, normalize=True):
    record = pd.read_csv(csv_path).values
    if normalize:
        record = normalize(record, axis=0)
    return record


def split_ecg(ecg_record, length=900, r_peak_search_threshold=0.35):
    # Find indeces of R peaks
    max_peak = np.amax(ecg_record)
    thresh = max_peak - (r_peak_search_threshold * max_peak)
    local_maxima = argrelextrema(ecg_record, np.greater)[0]
    r_peak_indeces = local_maxima[ecg_record[local_maxima] > thresh]
    # Validate that the split was reasonable and meets criteria
    try:
        max_distance = np.amax(np.diff(r_peak_indeces))
    except:
        print(r_peak_indeces)
        return None
    if max_distance < 500 or max_distance > length:
        return np.array([])
    # Find points on which to split
    split_points = []
    for i in range(len(r_peak_indeces)-1):
        split_points.append((r_peak_indeces[i] + r_peak_indeces[i+1]) // 2)
    raw_splits = np.split(ecg_record, split_indeces)[1:-1]
    # Make splits
    all_lines = []
    for split in raw_splits:
        all_lines.append(np.pad(split, (0, 900-len(split)), 'constant'))
    return np.array(all_lines)

def plot_ecg(ecg_record, label):
    pass

def plot_ecg_splits(ecg_record):
    pass
