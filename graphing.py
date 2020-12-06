import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import numpy as np 
from scipy.signal import argrelextrema
from sklearn.preprocessing import normalize
import pandas as pd
import os

abnormal_categories = ['RAA', 'LAA', 'CorSinus', 'Erratic', 'PAC', 'Pause', 'AFlut', 'AFib', 'LongQRS', 'N-Ectopy', 'Ectopy', 'PolyEct', 'PRAnom', 'LAD', 'GConD', 'RAD', 'LBBB', 'RBBB', 'ICRBBB', 'SpatialQRST', 'WPW', 'DiagQX', 'DiagQY', 'DiagQZ', 'ARVD', 'RVH', 'LowV', 'Brugada', 'STDep', 'STE', 'LQTS', 'SQTS', 'LTInv', 'ITInv', 'ATInv', 'TaVR', 'LimbRev', 'Rx<Sx', 'MI', 'RecCon']

def analyze_average_diffs():
    csv_files = []
    for (dirpath, dirnames, filenames) in os.walk('./records/'):
        csv_files.extend(filenames)

    df_annotations = pd.read_excel('attributes.xlsx')
    df_annotations = df_annotations[~df_annotations[abnormal_categories].all()]
    print(len(df_annotations))

    max_diffs = []
    """
    for filename in csv_files:
        try:
            record = normalize(pd.read_csv('./records/' + filename).values, axis=0)[:, 1]
            max_peak = np.amax(record)
            thresh = max_peak - (.25 * max_peak)
            local_maxima = argrelextrema(record, np.greater)[0]
            r_peak_indeces = local_maxima[record[local_maxima] > thresh]
            r_peak_values = record[r_peak_indeces]
            max_diff = np.amax(np.diff(r_peak_indeces))
            if max_diff > 1000:
                print('fail: greater than 1000 {0}'.format(max_diff))
                print(filename)
            else: 
                max_diffs.append(max_diff)
        except:
            print('fail')
    print(max(max_diffs))
    """



record = normalize(pd.read_csv('./records/BCS41.190.07.20.2004.2.5.2020.csv').values, axis=0)[:, 1]
max_peak = np.amax(record)
thresh = max_peak - (.25 * max_peak)
local_maxima = argrelextrema(record, np.greater)[0]
r_peak_indeces = local_maxima[record[local_maxima] > thresh]
r_peak_values = record[r_peak_indeces]

# Create slice boundaries
#print(np.amax(np.diff(r_peak_indeces)))
analyze_average_diffs()
# Make graph
figure(figsize=(80, 10))
plt.plot(np.arange(len(record)), record)
plt.scatter(r_peak_indeces, r_peak_values, c='red')
plt.savefig('graph.png')
