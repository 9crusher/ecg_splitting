import pandas as pd 
import numpy as np 
import ecg_helpers as ecg
from sklearn.model_selection import train_test_split
import os

abnormal_categories = ['RAA', 'LAA', 'CorSinus', 'Erratic', 'PAC', 'Pause', 'AFlut', 'AFib', 'LongQRS', 'N-Ectopy', 'Ectopy', 'PolyEct', 'PRAnom', 'LAD', 'GConD', 'RAD', 'LBBB', 'RBBB', 'ICRBBB', 'SpatialQRST', 'WPW', 'DiagQX', 'DiagQY', 'DiagQZ', 'ARVD', 'RVH', 'LowV', 'Brugada', 'STDep', 'STE', 'LQTS', 'SQTS', 'LTInv', 'ITInv', 'ATInv', 'TaVR', 'LimbRev', 'Rx<Sx', 'MI', 'RecCon']

# Grab csv names
csv_files = []
for (dirpath, dirnames, filenames) in os.walk('./records/'):
    csv_files.extend(filenames)

# Exclude attribute files not in our files
df_annotations = pd.read_csv('./new_attributes.csv')
df_existing_files = pd.DataFrame({'ExistingFileName': csv_files})
df_annotations = df_annotations.merge(df_existing_files, left_on='FileName', right_on='ExistingFileName', how='inner')
df_annotations = df_annotations.drop(['ExistingFileName'], axis=1)


# Remove all abnormal records 
for cat in abnormal_categories:
    df_annotations[cat] = (df_annotations[cat] - 1).abs()
df_annotations = df_annotations[df_annotations[abnormal_categories].all(axis=1)]
for cat in abnormal_categories:
    df_annotations[cat] = (df_annotations[cat] - 1).abs()

# Make segments
segments = []
for filename in df_annotations['FileName']:
    data = ecg.load_ecg('./records/' + filename)[:, 1][250:-250]
    data = ecg.hi_lo_filter(data)
    segments.append(data)

# Split and write
train, test = train_test_split(segments, test_size=0.2)
np.savetxt('smooth_train_7500.csv', train, delimiter=',')
np.savetxt('smooth_test_7500.csv', test, delimiter=',')
print('train length: ' + str(len(train)))