import pandas as pd
import numpy as np
import os

label_column = 'LQTS'
csv_directory = './records/'

df_annotations = pd.read_excel('attributes.xlsx')
required_annotations = ['FileName', label_column]
df_annotations = df_annotations[required_annotations]
df_annotations = df_annotations.dropna()
df_annotations[label_column] = df_annotations[label_column].astype(int)

# Ensure all annotations file names are linked to an ECG
csv_files = []
for (dirpath, dirnames, filenames) in os.walk(csv_directory):
    csv_files.extend(filenames)
df_existing_files = pd.DataFrame({'ExistingFileName': csv_files})
df_annotations = df_annotations.merge(df_existing_files, left_on='FileName', right_on='ExistingFileName', how='inner')


data = np.array([pd.read_csv(csv_directory + filename).values[:, 1].flatten() / 8000.0 for filename in df_annotations['FileName'].values])