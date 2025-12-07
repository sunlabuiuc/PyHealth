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

---

## Required Dependencies

The preprocessing pipeline requires the following Python packages:

- **numpy**: Core numerical computing library for array operations and signal processing
- **scipy**: Scientific computing library, specifically used for signal resampling (`scipy.signal.resample`)
- **pandas**: Data manipulation library (used for data handling and organization)
- **torch**: PyTorch deep learning framework for tensor operations and data type conversions
- **mne**: MNE-Python library for EEG/MEG data analysis (used for EEG signal processing utilities)
- **pyedflib**: Python library for reading and writing EDF (European Data Format) files, which is the standard format for storing EEG recordings
- **tqdm**: Progress bar library for displaying processing status during long-running operations

These dependencies are essential for reading EDF files, processing EEG signals, resampling data, and saving processed outputs in the required format.

---

## File Descriptions

### `process_TUH_dataset.py` - TUH EEG Dataset Preprocessor

This is the main preprocessing script that processes raw TUH (Temple University Hospital) EEG seizure detection dataset files. It performs comprehensive data preprocessing to convert raw EDF files and their annotations into a format suitable for machine learning model training and evaluation.

#### Overview

The script processes EEG recordings stored in EDF (European Data Format) files along with their corresponding annotation files (`.tse`, `.tse_bi`, or `.csv_bi`). It performs signal preprocessing, label sampling, data slicing, and saves the processed data as pickle files ready for model training.

#### Input Data Format

**EDF Files:**
- The script expects EDF (`.edf` or `.EDF`) files containing multi-channel EEG recordings
- Each EDF file should contain 19 standard EEG channels: `EEG FP1`, `EEG FP2`, `EEG F3`, `EEG F4`, `EEG F7`, `EEG F8`, `EEG C3`, `EEG C4`, `EEG CZ`, `EEG T3`, `EEG T4`, `EEG P3`, `EEG P4`, `EEG O1`, `EEG O2`, `EEG T5`, `EEG T6`, `EEG PZ`, `EEG FZ`
- Channel labels in the EDF file may include reference suffixes (e.g., `EEG FP1-REF`), which are automatically stripped during processing
- The original sample rate of EDF files can vary, but must be at least as high as the specified `--samplerate` parameter

**Annotation Files:**
- Each EDF file must have a corresponding annotation file with the same base filename
- Supported annotation formats:
  - **`.tse_bi`**: Binary seizure annotation format (Time-Stamped Events, binary)
  - **`.csv_bi`**: CSV-based binary annotation format (fallback if `.tse_bi` not found)
  - **`.tse`**: Multi-class annotation format (for non-binary tasks)
- Annotation file structure:
  - **TSE format**: Each line contains `start_time end_time label confidence` (e.g., `0.0000 10.0000 bckg 1.0000`)
  - **CSV format**: Contains columns `channel, start_time, end_time, label` with header row
  - The first two lines are typically headers/metadata and are skipped
  - Labels include:
    - `bckg`: Background/normal EEG activity
    - `seiz`: Seizure activity (binary mode)
    - Specific seizure types: `cpsz`, `mysz`, `gnsz`, `fnsz`, `tnsz`, `tcsz`, `spsz`, `absz` (multi-class mode)

#### Processing Pipeline

1. **File Discovery**: Recursively searches the specified data folder for all EDF files (`.edf` and `.EDF`)

2. **EDF File Reading**: 
   - Uses `pyedflib.highlevel.read_edf()` to read EDF files
   - Extracts signal data, signal headers (containing channel information and sample rates), and file headers
   - Validates that all required channels are present in the file

3. **Signal Preprocessing**:
   - **Channel Selection**: Filters and reorders signals to match the standard 19-channel layout
   - **Resampling**: If the original sample rate exceeds the target `--samplerate`, signals are downsampled using `scipy.signal.resample()` to the specified rate (default: 200 Hz)
   - **Data Type Conversion**: Converts signals to PyTorch tensors and then to `float16` format for memory efficiency

4. **Label Processing**:
   - Reads annotation files and parses time-stamped events
   - **Label Sampling**: Converts continuous time-based annotations to discrete samples based on `--feature_sample_rate` (default: 50 Hz)
   - For binary mode (`tse_bi`), all non-background seizure types are mapped to `seiz`
   - Labels are sampled at the feature sample rate, creating a string of label indices for each time point

5. **Patient History Detection** (Training mode only):
   - Checks other EDF files in the same patient directory for seizure history
   - Marks files with `_patT` (patient with history) or `_patF` (patient without history) suffix
   - This information is used for creating more nuanced training samples

6. **Data Slicing**:
   - Segments the continuous EEG recordings into fixed-length windows
   - **Minimum slice length**: Controlled by `--min_binary_slicelength` (default: 30 seconds)
   - **Slice types** (for training data):
     - Background slices: `0_patT` or `0_patF`
     - Seizure slices: `{label}_beg` (beginning), `{label}_middle` (middle), `{label}_end` (end), `{label}_whole` (complete seizure)
   - **Dev/Test mode**: Uses different slicing logic that preserves seizure boundaries more naturally, with variable-length slices that respect seizure transitions

7. **Label Alignment**:
   - Ensures label sequences match the length of resampled signal data
   - Handles mismatches by truncating or padding labels appropriately

8. **Output Generation**:
   - Each slice is saved as a separate pickle file containing:
     - `RAW_DATA`: NumPy array of shape `[num_channels, num_samples]` (float16)
     - `LABEL1`: NumPy array of label indices (uint8) sampled at feature rate
     - `LABEL2`: Optional binary target mapping (if `--binary_target1` specified)
     - `LABEL3`: Optional secondary binary target mapping (if `--binary_target2` specified)
   - Filenames include: original filename, slice index, and label type
   - Saves preprocessing metadata to `preprocess_info.infopkl` containing all configuration parameters

#### Key Parameters

- `--data_folder`: Path to directory containing EDF files (searches recursively)
- `--data_type`: `train` or `dev` (affects processing logic)
- `--samplerate`: Target sample rate for EEG signals (Hz, default: 200)
- `--feature_sample_rate`: Sample rate for label sequences (Hz, default: 50)
- `--label_type`: `tse_bi` (binary) or `tse` (multi-class)
- `--cpu_num`: Number of parallel processes for multi-processing
- `--min_binary_slicelength`: Minimum slice duration in seconds (default: 30)
- `--use_dev_function`: Use dev-specific slicing logic (recommended for dev/test sets)

#### Output Structure

Processed data is saved in a directory structure:
```
{save_directory}/dataset-{dataset}_task-{task_type}_datatype-{data_type}_v6/
├── preprocess_info.infopkl          # Preprocessing configuration metadata
├── {filename}_c0_label_{label}.pkl  # Processed slice files
├── {filename}_c1_label_{label}.pkl
└── ...
```

#### Multi-Processing

The script uses Python's `multiprocessing.Pool` to process multiple EDF files in parallel, significantly speeding up preprocessing for large datasets. Each EDF file is processed independently, making this approach highly scalable.

---

### `testcases.py` - Test Suite for Preprocessing Pipeline

This is a test file designed to verify that the preprocessing code works correctly with all required dependencies. It uses Python's `unittest` framework to create a comprehensive test suite.

The test file generates mock data that mimics the structure and format of the original TUH dataset, including:
- Mock EDF files with 19 standard EEG channels containing synthetic signal data
- Mock annotation files (`.tse_bi` format) with seizure and background segments
- Proper directory structure matching the expected TUH dataset layout

**Test Modes:**
- **Generate Mode** (`--generate_mock_data`): Creates mock EDF and annotation files in `test_tuh_env/mock_data/` directories without running tests
- **Test Mode** (default): Runs the full test suite that:
  1. Verifies mock data generation
  2. Executes the preprocessing script on mock data
  3. Validates output directory structure
  4. Checks data integrity of processed pickle files

The test suite ensures that all dependencies are properly installed and that the preprocessing pipeline correctly handles EDF file reading, annotation parsing, signal processing, and output generation. This is particularly useful for validating the setup before processing large-scale TUH datasets.