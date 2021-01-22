import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
from sklearn.preprocessing import normalize
from scipy.signal import argrelextrema
import numpy as np 
import pandas as pd 
from scipy.signal import butter, sosfiltfilt
import h5py
from os import walk


###########################################################################################################################################################
##########################################################################IO###############################################################################

def load_annotations(annotations_file, csv_dir):
    df_annotations = pd.read_csv(annotations_file) if annotations_file.endswith('.csv') else pd.read_excel(annotations_file) 

    # Ensure all annotations file names are linked to an ECG
    csv_files = []
    for (dirpath, dirnames, filenames) in walk(csv_dir):
        csv_files.extend(filenames)
    df_existing_files = pd.DataFrame({'ExistingFileName': csv_files})
    df_annotations = df_annotations.merge(df_existing_files, left_on='FileName', right_on='ExistingFileName', how='inner')
    df_annotations = df_annotations.drop(['ExistingFileName'], axis=1)
    return df_annotations


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


def write_hdf5(ecgs, filename):
    f = h5py.File(filename, "w")
    f.create_dataset('tracings', ecgs.shape, dtype='float32', data=ecgs)
    f.close()


##################################################################################################################################################################
##########################################################################Splitting###############################################################################

def split_ecg(ecg_lead, length=700, r_peak_search_threshold=0.35, min_dist=400):
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
    local_maxima = argrelextrema(ecg_lead, np.greater, order=50)[0]
    filtered_local_maxima = []
    for i in range(len(local_maxima)):
        point_height =  ecg_lead[local_maxima[i]]
        if np.amax(ecg_lead[local_maxima][max(0, i-1):i+2]) == point_height:
            #print(ecg_lead[local_maxima][max(0, i-1):i+2])
            filtered_local_maxima.append(local_maxima[i])

    local_maxima = np.array(filtered_local_maxima)

    r_peak_indeces = local_maxima[ecg_lead[local_maxima] > thresh]
    # Validate that the split was reasonable and meets criteria
    try:
        max_distance = np.amax(np.diff(r_peak_indeces))
    except:
        return np.array([])
    if max_distance < min_dist or max_distance > length:
        return np.array([])
    # Find points on which to split
    split_points = []
    for i in range(len(r_peak_indeces)-1):
        split_points.append((r_peak_indeces[i] + r_peak_indeces[i+1]) // 2) #+ int(abs(r_peak_indeces[i] - r_peak_indeces[i+1]) * .125))
    raw_splits = np.split(ecg_lead, split_points)[1:-1]
    # Make splits
    all_lines = []
    for split in raw_splits:
        all_lines.append(np.pad(split, (0, length-len(split)), 'constant'))
    return np.array(all_lines)

###########################################################################################################################################################
##########################################################################FILTERS##########################################################################

def hi_lo_filter(sig):
        # high pass
        hp = butter(2, 2, 'hp', fs=1000, output='sos')
        sig = sosfiltfilt(hp, sig, axis=0)
        # low pass
        lp = butter(2, 25, 'lp', fs=1000, output='sos')
        sig = sosfiltfilt(lp, sig, axis=0)
        return sig

############################################################################################################################################################
##########################################################################Plotting##########################################################################

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
        fig, axs = plt.subplots(ecg_record.shape[1], figsize=(100, 100))
        for i in range(len(axs)):
            axs[i].plot(x_axis, ecg_record[:, i])
            axs[i].set_title('Lead {0}'.format(i))
        fig.suptitle(label, fontsize=16)
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
