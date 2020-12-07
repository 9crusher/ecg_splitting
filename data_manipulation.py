import pandas as pd
import numpy as np
import os

def get_peaks(record):
    max_peak = np.amax(record)
    thresh = max_peak - (.35 * max_peak)
    local_maxima = argrelextrema(record, np.greater)[0]
    r_peak_indeces = local_maxima[record[local_maxima] > thresh]
    r_peak_values = record[r_peak_indeces]
    return (r_peak_indeces, r_peak_values)

def make_splits(record, split_indeces):
    raw_splits = np.split(record, split_indeces)[1:-1]
    padding = np.zeros((len(raw_splits), 900))

# Filter out annotations that are not in our csv's directory
csv_files = []
for (dirpath, dirnames, filenames) in os.walk('./records/'):
    csv_files.extend(filenames)
df_annotations = pd.read_excel('attributes.xlsx')
df_existing_files = pd.DataFrame({'ExistingFileName': csv_files})
df_annotations = df_annotations.merge(df_existing_files, left_on='FileName', right_on='ExistingFileName', how='inner')
df_annotations = df_annotations.drop(['ExistingFileName'], axis=1)

# Filter out irregular csv's
abnormal_categories = ['RAA', 'LAA', 'CorSinus', 'Erratic', 'PAC', 'Pause', 'AFlut', 'AFib', 'LongQRS', 'N-Ectopy', 'Ectopy', 'PolyEct', 'PRAnom', 'LAD', 'GConD', 'RAD', 'LBBB', 'RBBB', 'ICRBBB', 'SpatialQRST', 'WPW', 'DiagQX', 'DiagQY', 'DiagQZ', 'ARVD', 'RVH', 'LowV', 'Brugada', 'STDep', 'STE', 'LQTS', 'SQTS', 'LTInv', 'ITInv', 'ATInv', 'TaVR', 'LimbRev', 'Rx<Sx', 'MI', 'RecCon']
for cat in abnormal_categories:
    df_annotations[cat] = (df_annotations[cat] - 1).abs()
df_annotations = df_annotations[df_annotations[abnormal_categories].all(axis=1)]
for cat in abnormal_categories:
    df_annotations[cat] = (df_annotations[cat] - 1).abs()

for f in df_annotations['FileNames']:
    record = normalize(pd.read_csv('./records/' + f).values, axis=0)[:, 1]
    r_peak_indeces, r_peak_values = get_peaks(record)
    split_points = []
    for i in range(len(r_peak_indeces)-1):
        split_points.append((r_peak_indeces[i] + r_peak_indeces[i+1]) // 2)
    splits = make_splits(record, split_points)


