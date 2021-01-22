# Builds hdf5 files and annotations csv's from test data
# Outputs:
# train.hdf5 -- A 3d hdf5 file with a training "tracings" dataset in shape (<individual_ecg_test_results>, <ecg_samples>, <leads>)
# test.hdf5 -- A 3d hdf5 file with a test "tracings" dataset in shape (<individual_ecg_test_results>, <ecg_samples>, <leads>)
# train_annotations.csv -- A csv file with annotations for the train.hdf5 file
# test_annotations.csv -- A csv file with annotations for the test.hdf5 file
# pylint: disable=import-error

import argparse
import random
import h5py
import os, sys, inspect
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))) 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import ecg_helpers as ecg

CSV_DIRECTORY = '/data/mcw_ecg/Nick of Time Records/Nick of Time Records/'
ANNOTATIONS_FILE = '../attributes.csv'
OUTPUT_DIRECTORY = '../../mcw_ecg/data/'

def validation(attributes, hdf5_path):
    # Print top 20 rows of the attributes df
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = None
    pd.options.display.max_colwidth = -1
    print(attributes.head(20))

    # Print top 2 rows of each of the first 20 scans
    print('\nHDF5 contents:')
    f = h5py.File(hdf5_path, 'r')
    print(f['tracings'][0:20, 0:2])

    # Graph three samples from the hdf5
    for i in range(3):
        ecg.plot_ecg(f['tracings'][random.randint(0,len(f['tracings']) - 1)], OUTPUT_DIRECTORY + 'random' + str(i), hdf5_path)
    f.close()


def afib_mapper(df_annotations):
    required_annotations = ['FileName', 'AFib']
    df_annotations = df_annotations[required_annotations]
    df_annotations = df_annotations.dropna()
    return df_annotations

def blood_pressure_mapper(df_annotations):
    required_annotations = ['FileName', 'Systolic', 'Diastolic']
    # Drop bad data and unused cols
    df_annotations = df_annotations[required_annotations].dropna()
    df_annotations = df_annotations[np.logical_and(df_annotations['Systolic'] != 0, df_annotations['Diastolic'] != 0)]
    # Five category mapping
    df_annotations['Normal'] = np.logical_and(df_annotations['Systolic'].between(0, 120), df_annotations['Diastolic'].between(0, 80)).astype(int)
    df_annotations['Elevated'] = np.logical_and(df_annotations['Systolic'].between(121, 129), df_annotations['Diastolic'].between(0, 80)).astype(int)
    df_annotations['High1'] = np.logical_or(df_annotations['Systolic'].between(130, 139), df_annotations['Diastolic'].between(81, 89)).astype(int)
    df_annotations['High2'] = np.logical_or(df_annotations['Systolic'].between(140, 179), df_annotations['Diastolic'].between(90, 119)).astype(int)
    df_annotations['High3'] = np.logical_or(df_annotations['Systolic'].between(180, 500), df_annotations['Diastolic'].between(120, 500)).astype(int)
    # Take the higher category of the categories that can overlap
    df_annotations['High2'] = (df_annotations['High2'] - df_annotations['High3']).clip(lower=0)
    df_annotations['High1'] = (df_annotations['High1'] - (df_annotations['High2'] + df_annotations['High3'])).clip(lower=0)
    return df_annotations[['FileName', 'Normal', 'Elevated', 'High1', 'High2', 'High3']]


# Load annotations file and use selected mapper
df_annotations = ecg.load_annotations(ANNOTATIONS_FILE, CSV_DIRECTORY)
df_annotations = afib_mapper(df_annotations)


# Train test split
df_train, df_test = train_test_split(df_annotations, test_size=0.2)
print('Test length: ' + str(len(df_test)))
print('Train length: ' + str(len(df_train)))

# Write to files
used_cols = df_annotations.columns[df_annotations.columns != 'FileName']
print('Making test files')
df_test.to_csv(OUTPUT_DIRECTORY + 'test_annotations.csv', columns=used_cols, index=False)
ecg.write_hdf5(np.array([ecg.hi_lo_filter(ecg.load_ecg(CSV_DIRECTORY + fi)[:, 1]) for fi in df_test['FileName']]), OUTPUT_DIRECTORY + "test.hdf5")
print('Making train files')
df_train.to_csv(OUTPUT_DIRECTORY + 'train_annotations.csv', columns=used_cols, index=False)
ecg.write_hdf5(np.array([ecg.hi_lo_filter(ecg.load_ecg(CSV_DIRECTORY + fi)[:, 1]) for fi in df_train['FileName']]), OUTPUT_DIRECTORY + "train.hdf5")
print('Done')

# Print section of outputs to validate data integrity
print("Validate test data:")
validation(df_test, OUTPUT_DIRECTORY + "test.hdf5")
