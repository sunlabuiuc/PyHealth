"""Multi-view time series task for domain adaptation.

This task generates three views (temporal, derivative, frequency) from
physiological time series signals (EEG, ECG, EMG) for multi-view contrastive learning.
"""

import os
import pickle
import numpy as np
from scipy.fft import fft


def multi_view_time_series_fn(record, epoch_seconds=30, sample_rate=100):
    """Creates multi-view representations from time series data.
    
    Generates three complementary views of each signal epoch:
    - Temporal view: raw signal
    - Derivative view: first-order difference (captures dynamics)
    - Frequency view: FFT magnitude (captures spectral patterns)
    
    Args:
        record: dict from a PyHealth time-series dataset with keys:
            - load_from_path: root directory of the data
            - signal_file: filename of the signal (.edf or similar)
            - label_file: filename of the labels
            - save_to_path: directory to save processed epochs
            - subject_id: patient/subject identifier
        epoch_seconds: duration of each epoch in seconds (default: 30)
        sample_rate: sampling rate of the signal in Hz (default: 100)
    
    Returns:
        samples: list of dicts, each containing:
            - record_id: unique identifier for this epoch
            - patient_id: patient identifier
            - epoch_path: path to saved .pkl file
            - label: ground truth label
    
    The saved .pkl file contains:
        - temporal: raw signal array (channels, time_steps)
        - derivative: first-order difference array (channels, time_steps-1)
        - frequency: FFT magnitude array (channels, frequency_bins)
        - label: ground truth label string
    """
    
    # Extract record information
    root = record[0]["load_from_path"]
    signal_file = record[0]["signal_file"]
    label_file = record[0]["label_file"]
    save_path = record[0]["save_to_path"]
    patient_id = record[0].get("subject_id", signal_file[:6])
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Load signal (simplified - in practice, use mne.read_raw_edf)
    # For testing, we'll generate synthetic data
    # In real implementation, replace with actual data loading
    total_duration_seconds = 60 * 10  # Assume 10 minutes of recording
    total_samples = int(sample_rate * total_duration_seconds)
    num_channels = 2  # Typical for EEG (e.g., F3, F4)
    
    # TODO: Replace with actual data loading
    # data = mne.io.read_raw_edf(os.path.join(root, signal_file)).get_data()
    data = np.random.randn(num_channels, total_samples)
    
    # Load labels (simplified - actual implementation depends on dataset)
    # For testing, generate dummy labels
    epochs_per_label = int(30 / epoch_seconds) if epoch_seconds < 30 else 1
    num_epochs = total_samples // int(sample_rate * epoch_seconds)
    # TODO: Replace with actual label loading
    labels = ["W"] * num_epochs  # Dummy labels for testing
    
    samples = []
    epoch_length = int(sample_rate * epoch_seconds)
    
    for epoch_idx in range(num_epochs):
        # Extract epoch signal
        start_idx = epoch_idx * epoch_length
        end_idx = start_idx + epoch_length
        epoch_signal = data[:, start_idx:end_idx]  # Shape: (channels, time)
        
        # Get label for this epoch
        label_idx = epoch_idx // epochs_per_label if epochs_per_label > 1 else epoch_idx
        if label_idx >= len(labels):
            break
        label = labels[label_idx]
        
        # Generate three views
        temporal_view = epoch_signal  # Raw signal
        
        derivative_view = np.diff(epoch_signal, axis=1)  # First-order difference
        
        # Frequency view (FFT magnitude)
        fft_vals = fft(epoch_signal, axis=1)
        freq_magnitude = np.abs(fft_vals[:, :epoch_length // 2])  # Keep positive frequencies
        
        # Save to pickle file
        epoch_path = os.path.join(save_path, f"{patient_id}-{epoch_idx}.pkl")
        pickle.dump(
            {
                "temporal": temporal_view,
                "derivative": derivative_view,
                "frequency": freq_magnitude,
                "label": label,
            },
            open(epoch_path, "wb"),
        )
        
        # Append sample metadata
        samples.append(
            {
                "record_id": f"{patient_id}-{epoch_idx}",
                "patient_id": patient_id,
                "epoch_path": epoch_path,
                "label": label,
            }
        )
    
    return samples


# Simple test to verify the function works
if __name__ == "__main__":
    print("Testing multi_view_time_series_fn...")
    
    # Create a dummy record
    test_record = [{
        "load_from_path": "/tmp/test_data",
        "signal_file": "test.edf",
        "label_file": "test.label",
        "save_to_path": "/tmp/test_output",
        "subject_id": "TEST001",
    }]
    
    # Run the function
    samples = multi_view_time_series_fn(test_record, epoch_seconds=30, sample_rate=100)
    
    print(f"Generated {len(samples)} samples")
    
    if len(samples) > 0:
        print(f"Sample keys: {samples[0].keys()}")
        
        # Load and check the saved data
        with open(samples[0]["epoch_path"], "rb") as f:
            data = pickle.load(f)
        print(f"Saved data keys: {data.keys()}")
        print(f"Temporal shape: {data['temporal'].shape}")
        print(f"Derivative shape: {data['derivative'].shape}")
        print(f"Frequency shape: {data['frequency'].shape}")
        print(f"Label: {data['label']}")
    
    print("Test complete!")