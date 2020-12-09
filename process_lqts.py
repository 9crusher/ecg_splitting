import pandas as pd 
import numpy as np 
import ecg_helpers as ecg
import os



def plot_and_split(files, dest_dir='./images/', csv_dir='./records/'):
    all_splits = []
    for f in files:
        print(f)
        record_name = ''.join(f.split('.')[:-1])
        lead_2 = ecg.load_ecg(csv_dir + f)[:, 1]
        ecg.plot_ecg(lead_2, dest_dir + 'original_{0}.png'.format(record_name))
        splits = ecg.split_ecg(lead_2, r_peak_search_threshold=0.5, length=750)
        ecg.plot_ecg_splits(splits, dest_dir + 'split_{0}'.format(record_name))
        all_splits.extend(splits)
    return np.array(all_splits)

csv_files = []
for (dirpath, dirnames, filenames) in os.walk('./records/'):
    csv_files.extend(filenames)
df_annotations = pd.read_csv('./new_attributes.csv')
df_existing_files = pd.DataFrame({'ExistingFileName': csv_files})
df_annotations = df_annotations.merge(df_existing_files, left_on='FileName', right_on='ExistingFileName', how='inner')
df_annotations = df_annotations.drop(['ExistingFileName'], axis=1)
print('Accurate LQTS')
lqts_files = df_annotations[df_annotations['followup_lethal___6'] == 1]['FileName']
splits = plot_and_split(lqts_files)
print(splits.shape)
np.savetxt('lqts_splits.csv', splits, delimiter=',')


