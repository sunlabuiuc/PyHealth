"""
Author: Max Piazza
NetID: mpiazza2
Author: Franco Blacutt
NetID: blacutt2
Paper Title: Contrastive Learning of Electrodermal Activity Representations for Stress Detection
Paper Link: https://proceedings.mlr.press/v209/matton23a.html

Example: Contrastive Learning for EDA Stress Detection with WESAD Dataset

This example demonstrates how to preprocess the WESAD dataset and apply
EDA-specific augmentations, inspired by the framework of Matton et al. (2023).

Implemented steps:
1. Load and preprocess the WESAD dataset.
2. Apply EDA-specific augmentations (jittering, scaling, cropping, permutation).

Outlined for future work:
3. Define a 1D CNN encoder and contrastive learning model.
4. Train the model using contrastive loss (NT-Xent).
5. Fine-tune the pretrained model for stress classification with limited labeled data.

This example serves as a PyHealth use case for physiological signal data integration.
The current version focuses on data processing and augmentation, providing a scaffold
for further development of the modeling and training pipeline.
"""

# ============================
# Imports
# ============================

# Import necessary libraries
import os
import pickle
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import interp1d

print("Starting WESAD PyHealth example...")

# ============================
# Step 1: Load WESAD Dataset
# ============================

def load_wesad_subjects(data_dir):
    """
    Loads EDA signal and labels from the WESAD dataset.

    Args:
        data_dir (str): Path to the WESAD dataset directory.

    Returns:
        tuple: 
            - eda_data (List[np.ndarray]): List of EDA signals per subject.
            - label_data (List[np.ndarray]): List of corresponding label arrays.

    Example:
        eda_data, label_data = load_wesad_subjects("./WESAD")
    """
    subject_ids = [f"S{i}" for i in range(1, 18) if i not in [1, 12]]
    eda_data = []
    label_data = []
    for sid in subject_ids:
        pkl_path = os.path.join(data_dir, sid, f"{sid}.pkl")
        if not os.path.exists(pkl_path):
            print(f"Missing file: {pkl_path}")
            continue
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f, encoding='latin1')
            eda_signal = d['signal']['wrist']['EDA']  # 4 Hz EDA signal
            labels = d['label']  # label array (same length as EDA)
            eda_data.append(eda_signal)
            label_data.append(labels)
    print(f"Loaded {len(eda_data)} subjects from WESAD.")
    return eda_data, label_data

# ============================
# Step 2: Preprocess Data
# ============================

# Define preprocessing pipeline
# (e.g., segment EDA into 60s windows, normalize, handle missing data)

def preprocess_eda_data(eda_data, label_data, window_size_sec=60, sample_rate=4):
    """
    Preprocesses EDA data: windowing, normalization, and label assignment.

    Args:
        eda_data (list): List of EDA signal arrays (np.ndarray) per subject.
        label_data (list): List of label arrays (np.ndarray) per subject.
        window_size_sec (int): Size of each window in seconds. Defaults to 60.
        sample_rate (int): Sampling rate (Hz) of EDA signals. Defaults to 4.

    Returns:
        tuple: 
            - windowed_data (List[np.ndarray]): List of windowed EDA signals.
            - windowed_labels (List[int]): List of majority labels per window.

    Example:
        windowed_data, windowed_labels = preprocess_eda_data(eda_data, label_data)
    """
    window_size = window_size_sec * sample_rate  # e.g., 60s * 4Hz = 240 samples
    windowed_data = []
    windowed_labels = []

    for eda, lbl in zip(eda_data, label_data):
        n_windows = len(eda) // window_size
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            window = eda[start:end]
            window_label = lbl[start:end]
            if len(window) < window_size:
                continue  # Skip incomplete windows
            # Normalize window (z-score)
            window = (window - np.mean(window)) / np.std(window)
            # Assign majority label in the window
            counts = np.bincount(window_label)
            main_label = np.argmax(counts)
            windowed_data.append(window)
            windowed_labels.append(main_label)

    print(f"Processed {len(windowed_data)} windows in total.")
    return windowed_data, windowed_labels

# ============================
# Step 3: Apply EDA-Specific Augmentations
# ============================

def jitter(signal, sigma=0.05) -> np.ndarray:
    """
    Adds random Gaussian noise (jitter) to the input signal.

    Args:
        signal (np.ndarray): The input EDA signal array of shape (N,).
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: The jittered signal with the same shape as input.

    Example:
        jittered_signal = jitter(signal, sigma=0.05)
    """
    return signal + np.random.normal(loc=0., scale=sigma, size=signal.shape)

def scaling(signal, sigma=0.1) -> np.ndarray:
    """
    Scales the amplitude of the signal by a random factor.

    Args:
        signal (np.ndarray): The input EDA signal array of shape (N,).
        sigma (float): Standard deviation of the scaling factor around 1.0.

    Returns:
        np.ndarray: The scaled signal.

    Example:
        scaled_signal = scaling(signal, sigma=0.1)
    """
    factor = np.random.normal(loc=1.0, scale=sigma)
    return signal * factor


def cropping(signal, crop_size=0.9) -> np.ndarray:
    """
    Randomly crops a segment of the signal and pads zeros to maintain length.

    Args:
        signal (np.ndarray): The input EDA signal array of shape (N,).
        crop_size (float): Fraction (0 < crop_size <= 1) of the original length to keep.

    Returns:
        np.ndarray: Cropped and zero-padded signal with same shape as input.

    Example:
        cropped_signal = cropping(signal, crop_size=0.9)
    """
    L = signal.shape[0]
    cropped_length = int(L * crop_size)
    start = np.random.randint(0, L - cropped_length)
    cropped = signal[start:start + cropped_length]
    # Pad to original length
    pad_width = L - cropped_length
    return np.pad(cropped, (0, pad_width), 'constant')

def permutation(signal, n_perm=4) -> np.ndarray:
    """
    Divides the signal into equal chunks and randomly permutes them.

    Args:
        signal (np.ndarray): The input EDA signal array of shape (N,).
        n_perm (int): Number of chunks to divide the signal into.

    Returns:
        np.ndarray: Signal with permuted chunks.

    Example:
        permuted_signal = permutation(signal, n_perm=4)
    """
    chunk_size = int(len(signal) / n_perm)
    chunks = [signal[i * chunk_size:(i + 1) * chunk_size] for i in range(n_perm)]
    np.random.shuffle(chunks)
    return np.concatenate(chunks)

# ============================
# Step 4: Define 1D CNN Encoder + Contrastive Framework
# ============================

# TODO: Define your 1D CNN encoder model
# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Define Conv1D layers here

# TODO: Define projection head + contrastive loss

print("# TODO: Define encoder, projection head, and NT-Xent loss")

# ============================
# Step 5: Pretrain Model (Contrastive Learning)
# ============================

# TODO: Write training loop for contrastive learning
# - Use unlabeled data
# - Apply augmentations on-the-fly
# - Optimize contrastive loss

print("# TODO: Pretrain model using contrastive learning")

# ============================
# Step 6: Fine-tune on Labeled Data
# ============================

# TODO: Replace projection head with classification head
# TODO: Fine-tune on small labeled set (e.g., 1% data)

print("# TODO: Fine-tune model for stress classification")

# ============================
# Step 7: Evaluate Model
# ============================

# TODO: Evaluate using accuracy, AUC, etc.
# - Implement Leave-N-Subjects-Out CV

print("# TODO: Evaluate model performance")


if __name__ == "__main__":
    data_dir = "/path/to/WESAD"  # <-- user should customize this
    eda_data, label_data = load_wesad_subjects(data_dir)
    windowed_data, windowed_labels = preprocess_eda_data(eda_data, label_data)

    # Example: Apply all augmentations to the first window
    sample = windowed_data[0]

    aug_jitter = jitter(sample)
    aug_scale = scaling(sample)
    aug_crop = cropping(sample)
    aug_perm = permutation(sample)

    print("Applied augmentations to one sample window:")
    print(f"Original shape: {sample.shape}")
    print(f"Jittered shape: {aug_jitter.shape}")
    print(f"Scaled shape: {aug_scale.shape}")
    print(f"Cropped shape: {aug_crop.shape}")
    print(f"Permuted shape: {aug_perm.shape}")

    print(f"Example run completed. {len(windowed_data)} windows processed.")