import pandas as pd 
import numpy as np 
import ecg_helpers as ecg
from sklearn.model_selection import train_test_split
import os


abnormal_categories = ['RAA', 'LAA', 'CorSinus', 'Erratic', 'PAC', 'Pause', 'AFlut', 'AFib', 'LongQRS', 'N-Ectopy', 'Ectopy', 'PolyEct', 'PRAnom', 'LAD', 'GConD', 'RAD', 'LBBB', 'RBBB', 'ICRBBB', 'SpatialQRST', 'WPW', 'DiagQX', 'DiagQY', 'DiagQZ', 'ARVD', 'RVH', 'LowV', 'Brugada', 'STDep', 'STE', 'LQTS', 'SQTS', 'LTInv', 'ITInv', 'ATInv', 'TaVR', 'LimbRev', 'Rx<Sx', 'MI', 'RecCon']
def plot_and_split(files, dest_dir='./images/', csv_dir='./records/'):
    all_splits = []
    for f in files:
        record_name = ''.join(f.split('.')[:-1])
        lead_2 = ecg.load_ecg(csv_dir + f)[:, 1]
        #ecg.plot_ecg(lead_2, dest_dir + 'original_{0}.png'.format(record_name))
        splits = ecg.split_ecg(lead_2, r_peak_search_threshold=0.3, length=700)
        
        if len(splits) > 0:
            #ecg.plot_ecg_splits(splits, dest_dir + 'split_{0}'.format(record_name))
            all_splits.extend(splits)
    return np.array(all_splits)


csv_files = []
for (dirpath, dirnames, filenames) in os.walk('./records/'):
    csv_files.extend(filenames)
df_annotations = pd.read_excel('not_new_attributes.xlsx')
df_existing_files = pd.DataFrame({'ExistingFileName': csv_files})
df_annotations = df_annotations.merge(df_existing_files, left_on='FileName', right_on='ExistingFileName', how='inner')
df_annotations = df_annotations.drop(['ExistingFileName'], axis=1)

print(len(df_annotations))
for cat in abnormal_categories:
    df_annotations[cat] = (df_annotations[cat] - 1).abs()
df_annotations = df_annotations[df_annotations[abnormal_categories].all(axis=1)]
for cat in abnormal_categories:
    df_annotations[cat] = (df_annotations[cat] - 1).abs()
print(len(df_annotations))
splits = plot_and_split(df_annotations['FileName'].values)
print(len(splits))
train, test = train_test_split(splits, test_size=0.2)
np.savetxt('train_clean_split.csv', train, delimiter=',')
np.savetxt('test_clean_split.csv', test, delimiter=',')
