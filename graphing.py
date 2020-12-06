import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import numpy as np 
from scipy.signal import argrelextrema
from sklearn.preprocessing import normalize
import pandas as pd
import os

abnormal_categories = ['RAA', 'LAA', 'CorSinus', 'Erratic', 'PAC', 'Pause', 'AFlut', 'AFib', 'LongQRS', 'N-Ectopy', 'Ectopy', 'PolyEct', 'PRAnom', 'LAD', 'GConD', 'RAD', 'LBBB', 'RBBB', 'ICRBBB', 'SpatialQRST', 'WPW', 'DiagQX', 'DiagQY', 'DiagQZ', 'ARVD', 'RVH', 'LowV', 'Brugada', 'STDep', 'STE', 'LQTS', 'SQTS', 'LTInv', 'ITInv', 'ATInv', 'TaVR', 'LimbRev', 'Rx<Sx', 'MI', 'RecCon']

def get_csvs():
    csv_files = []
    for (dirpath, dirnames, filenames) in os.walk('./records/'):
        csv_files.extend(filenames)
    return csv_files

def get_normal_records():
    csv_files = get_csvs()
    df_annotations = pd.read_excel('attributes.xlsx')
    df_existing_files = pd.DataFrame({'ExistingFileName': csv_files})
    df_annotations = df_annotations.merge(df_existing_files, left_on='FileName', right_on='ExistingFileName', how='inner')
    df_annotations = df_annotations.drop(['ExistingFileName'], axis=1)
    for cat in abnormal_categories:
        df_annotations[cat] = (df_annotations[cat] - 1).abs()
    return df_annotations[df_annotations[abnormal_categories].all(axis=1)]

def get_splits(record):
    max_peak = np.amax(record)
    thresh = max_peak - (.25 * max_peak)
    local_maxima = argrelextrema(record, np.greater)[0]
    r_peak_indeces = local_maxima[record[local_maxima] > thresh]
    r_peak_values = record[r_peak_indeces]
    return (r_peak_indeces, r_peak_values)


def get_reasonable_splits():
    filenames = get_normal_records()['FileName']
    reasonable_files = []
    for filename in filenames:
        record = normalize(pd.read_csv('./records/' + filename).values, axis=0)[:, 1]
        r_peak_indeces, r_peak_values = get_splits(record)
        try:
            max_distance = np.amax(np.diff(r_peak_indeces))
            if max_distance >500 and max_distance < 900:
                reasonable_files.append(filename)
        except:
            print('fail')
    return reasonable_files






files = get_reasonable_splits()
for f in files:
    record = normalize(pd.read_csv('./records/' + f).values, axis=0)[:, 1]
    r_peak_indeces, r_peak_values = get_splits(record)
    figure(figsize=(80, 10))
    plt.plot(np.arange(len(record)), record)
    plt.scatter(r_peak_indeces, r_peak_values, c='red')
    plt.savefig('images/{0}_graph.png'.format(f))


# Make graph

