# TUH EEG Seizure Detection Preprocessing

This directory contains code to reproduce the preprocessing steps for the paper:  
*"Real-Time Seizure Detection using EEG: A Comprehensive Comparison of Recent Approaches under a Realistic Setting"*

## Prerequisites

Install the required dependencies:

```bash
pip install numpy scipy pandas torch mne pyedflib tqdm
```


## Test Mock Data

```
python test_cases.py --generate_mock_data
```

Generates mock data and annotations into train and dev dirs. 

```
python test_cases.py
```

Tests processor on generated mock data.

## Processing TUH Dataset:
```
python process_TUH_dataset.py \
    --data_folder /path/to/tuh/train \
    --data_type train \
    --save_directory ./output \
    --cpu_num 12 \
    --label_type tse_bi \
    --samplerate 200 \
    --feature_sample_rate 50
```

```
python process_TUH_dataset.py \
    --data_folder /path/to/tuh/dev \
    --data_type dev \
    --save_directory ./output \
    --cpu_num 12 \
    --label_type tse_bi \
    --samplerate 200 \
    --feature_sample_rate 50 \
    --use_dev_function
```