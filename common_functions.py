import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
from sklearn.preprocessing import normalize
from scipy.signal import argrelextrema
import numpy as np 
import pandas as pd 


def load_ecg(csv_path, normalize_data=True):
    '''
    Loads an ecg from a csv and optionally normalizes the data contained within.

    Parameters:
        csv_path (string): The file path to the ecg csv. The csv must contain one lead per column
    Returns:
        ndarray containing the ecg data
        
    '''
    record = pd.read_csv(csv_path).values
    if normalize_data:
        record = normalize(record, axis=0)
    return record


def split_ecg(ecg_lead, length=900, r_peak_search_threshold=0.35):
    '''
    Splits an ecg lead into individual wave forms. Cuts are made between T and P waves.

    Parameters:
        ecg_lead (ndarray): 1D array representing the lead to split. This must resemble a traditional
                            ecg in order for the spliting to work
        length (int): The length of each output split/slice. Waves will be padded with trailing zeros
                        in order to achieve this length if they do not meet it.
        r_peak_search_threshold (float): The margin below the highest R wave to still count as
                                         an R wave. This is useful in tuning the method to split
                                        unusual ecg's
    Returns:
        ndarray containing the ecg data
        
    '''
    # Find indeces of R peaks
    max_peak = np.amax(ecg_lead)
    thresh = max_peak - (r_peak_search_threshold * max_peak)
    local_maxima = argrelextrema(ecg_lead, np.greater)[0]
    r_peak_indeces = local_maxima[ecg_lead[local_maxima] > thresh]
    # Validate that the split was reasonable and meets criteria
    try:
        max_distance = np.amax(np.diff(r_peak_indeces))
    except:
        return np.array([])
    if max_distance < 350 or max_distance > length:
        return np.array([])
    # Find points on which to split
    split_points = []
    for i in range(len(r_peak_indeces)-1):
        split_points.append((r_peak_indeces[i] + r_peak_indeces[i+1]) // 2)
    raw_splits = np.split(ecg_lead, split_points)[1:-1]
    # Make splits
    all_lines = []
    for split in raw_splits:
        all_lines.append(np.pad(split, (0, 900-len(split)), 'constant'))
    return np.array(all_lines)


def plot_ecg(ecg_record, output_file, label='ECG'):
    '''
    Makes plots sized appropriately for ecg files containing 8000
    samples. This method can will plot any number of leads.

    Parameters:
        ecg_record (ndarray): Numpy array containing the ecg data. This may
                                  be a 1D array representing a single lead
                                  or a 2D array with each column representing a 
                                  lead
        output_file (string): The path (including file name) for the graph output png
        label (string): A unique label for the ecg. This will be used to label the plots
    Returns:
        None
    '''
    plt.clf()
    x_axis = np.arange(len(ecg_record))
    if ecg_record.ndim == 2:
        fig, axs = plt.subplots(ecg_record.shape[1], figsize=(80, 50))
        for i in range(len(axs)):
            axs[i].plot(x_axis, ecg_record[:, i])
            axs[i].set_title('Lead {0}'.format(i))
        fig.title(label)
        fig.savefig(output_file)
   
    else:
        figure(figsize=(80, 5))
        plt.plot(x_axis, ecg_record)
        plt.title(label)
        plt.savefig(output_file)


def plot_ecg_splits(ecg_lead_splits, output_file, label='ECG Splits'):
    '''
    Makes a plot of all of the splits of an ecg lead. Each split is represented
    by a different color

    Parameters:
        ecg_lead_splits (list or list like): A 2d array with each row representing a
                                             split of an ecg
        output_file (string): The path (including file name) for the graph output png
        label (string): A unique label for the ecg splits. This will be used to label the plots
    Returns:
        None
    '''
    plt.clf()
    offset = 0
    for split in ecg_lead_splits:
        plt.plot(range(offset, offset + len(split)), split)
        plt.title(label)
        offset += len(split)
    plt.savefig(output_file)
