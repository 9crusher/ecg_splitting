#!/bin/bash

. /usr/local/anaconda3/bin/activate ./datagen_env
if ([ $? -ne 0 ] && [ ! -d "./datagen_env" ]); then # If conda activate failed and the ./mcw_ecg_env directory does not exist
    echo Virtual env does not exist. Creating datagen_env
    conda create --prefix ./datagen_env python=3.7 > /dev/null 2>&1  || exit 1 # Create virtual env silently
    echo mcw_ecg_env created
    . /usr/local/anaconda3/bin/activate ./datagen_env
fi

pip install -r requirements.txt
python3 ./classifier.py

. /usr/local/anaconda3/bin/deactivate ./datagen_env