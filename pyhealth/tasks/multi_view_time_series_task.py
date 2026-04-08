"""Multi-view time series task for domain adaptation.

This task generates three complementary views (temporal, derivative, frequency)
from physiological time series signals (EEG, ECG, EMG) for multi-view contrastive
learning. The three views capture different aspects of the signal:
- Temporal view: raw signal preserving original patterns
- Derivative view: rate of change capturing signal dynamics
- Frequency view: spectral content capturing periodic patterns
"""

import os
import pickle
import numpy as np
from scipy.fft import fft
from typing import List, Dict, Any, Optional, Tuple


def multi_view_time_series_fn(
    record: List[Dict[str, Any]],
    epoch_seconds: int = 30,
    sample_rate: int = 100,
    num_channels: int = 2,
) -> List[Dict[str, Any]]:
    """Creates multi-view representations from time series data.
    
    This function processes a single patient's recording by:
    1. Loading the raw time series signal
    2. Slicing it into non-overlapping epochs (windows)
    3. For each epoch, generating three views:
       - Temporal: raw signal
       - Derivative: first-order difference (signal[i+1] - signal[i])
       - Frequency: FFT magnitude spectrum
    4. Saving each epoch as a pickle file
    5. Returning metadata for each epoch
    
    Args:
        record: A list containing one dictionary with the following keys:
            - load_from_path (str): Root directory containing the data files
            - signal_file (str): Filename of the signal (.edf or similar)
            - label_file (str): Filename containing labels/annotations
            - save_to_path (str): Directory where processed epochs will be saved
            - subject_id (str, optional): Patient identifier. If not provided,
              will be extracted from signal_file.
        epoch_seconds: Duration of each epoch in seconds. Default 30.
        sample_rate: Sampling rate of the signal in Hz. Default 100.
        num_channels: Number of channels in the signal. Default 2 (e.g., F3, F4 for EEG).
    
    Returns:
        A list of sample dictionaries, each containing:
            - record_id (str): Unique identifier for this epoch
            - patient_id (str): Patient identifier
            - epoch_path (str): Absolute path to saved .pkl file
            - label (str): Ground truth label for this epoch
    
    The saved .pkl file contains a dictionary with:
        - temporal (np.ndarray): Raw signal, shape (num_channels, time_steps)
        - derivative (np.ndarray): First-order difference, shape (num_channels, time_steps-1)
        - frequency (np.ndarray): FFT magnitude, shape (num_channels, frequency_bins)
        - label (str): Ground truth label
    
    Example:
        >>> from pyhealth.datasets import SleepEDFDataset
        >>> dataset = SleepEDFDataset(root="/path/to/data")
        >>> dataset.set_task(multi_view_time_series_fn)
        >>> sample = dataset.samples[0]
        >>> print(sample.keys())
        dict_keys(['record_id', 'patient_id', 'epoch_path', 'label'])
        
        >>> # Load the saved views
        >>> import pickle
        >>> with open(sample['epoch_path'], 'rb') as f:
        ...     views = pickle.load(f)
        >>> print(views['temporal'].shape)
        (2, 3000)  # 2 channels, 3000 time points (100 Hz * 30 seconds)
    """
    
    # ==================== STEP 1: Extract record information ====================
    # Record is a list with one element per patient/recording
    # For sleep staging datasets, it's a singleton list
    record_data = record[0]
    
    root = record_data["load_from_path"]
    signal_file = record_data["signal_file"]
    label_file = record_data["label_file"]
    save_path = record_data["save_to_path"]
    
    # Get patient ID - use subject_id if provided, otherwise extract from filename
    patient_id = record_data.get("subject_id", signal_file[:6])
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # ==================== STEP 2: Load the raw signal ====================
    # TODO: Replace with actual data loading for your specific dataset
    # For SleepEDF, use: import mne; data = mne.io.read_raw_edf(filepath).get_data()
    # For now, we generate synthetic data for demonstration
    
    # Calculate total duration based on typical recording length
    # Real implementation would read the actual file duration
    total_duration_seconds = 60 * 10  # Assume 10 minutes for demo
    total_samples = int(sample_rate * total_duration_seconds)
    
    # Generate synthetic signal with some structure
    # In reality, this would be loaded from the EDF file
    np.random.seed(42)  # For reproducibility
    time = np.linspace(0, total_duration_seconds, total_samples)
    # Create a signal with: sine wave + noise + some drift
    synthetic_signal = np.zeros((num_channels, total_samples))
    for ch in range(num_channels):
        # Add a sine wave (simulating alpha rhythm for EEG)
        synthetic_signal[ch] = (
            np.sin(2 * np.pi * 10 * time) +  # 10 Hz alpha wave
            0.5 * np.sin(2 * np.pi * 0.5 * time) +  # 0.5 Hz drift
            0.3 * np.random.randn(total_samples)  # random noise
        )
    
    data = synthetic_signal
    
    # ==================== STEP 3: Load labels ====================
    # TODO: Replace with actual label loading for your specific dataset
    # For SleepEDF, labels are in .hyp or annotation files
    # For now, we generate dummy labels
    
    # Calculate number of epochs
    epoch_length_samples = int(sample_rate * epoch_seconds)
    num_epochs = total_samples // epoch_length_samples
    
    # Generate dummy labels (sleep stages: W, N1, N2, N3, REM)
    possible_labels = ["W", "N1", "N2", "N3", "REM"]
    labels = [possible_labels[i % len(possible_labels)] for i in range(num_epochs)]
    
    # ==================== STEP 4: Process each epoch ====================
    samples = []
    
    for epoch_idx in range(num_epochs):
        # ----- 4a: Extract the signal segment for this epoch -----
        start_idx = epoch_idx * epoch_length_samples
        end_idx = start_idx + epoch_length_samples
        epoch_signal = data[:, start_idx:end_idx]  # Shape: (num_channels, time_steps)
        
        # Get label for this epoch
        label = labels[epoch_idx]
        
        # ----- 4b: Generate the three views -----
        
        # View 1: TEMPORAL - Raw signal
        # Preserves original amplitude, phase, and temporal relationships
        temporal_view = epoch_signal  # Shape: (channels, time)
        
        # View 2: DERIVATIVE - First-order difference
        # Captures rate of change, emphasizes transitions and dynamics
        # Formula: derivative(t) = signal(t+1) - signal(t)
        # This removes baseline drift and highlights rapid changes
        derivative_view = np.diff(epoch_signal, axis=1)  # Shape: (channels, time-1)
        
        # View 3: FREQUENCY - FFT magnitude spectrum
        # Captures periodic patterns and frequency band power
        # Useful for identifying rhythms (alpha, beta, theta, delta in EEG)
        fft_vals = fft(epoch_signal, axis=1)
        # Keep only positive frequencies (Nyquist limit)
        # Shape: (channels, time//2) - half the time points
        freq_magnitude = np.abs(fft_vals[:, :epoch_length_samples // 2])
        
        # ----- 4c: Save to pickle file -----
        epoch_path = os.path.join(save_path, f"{patient_id}-epoch-{epoch_idx}.pkl")
        
        # Create dictionary with all three views + label
        epoch_data = {
            "temporal": temporal_view,
            "derivative": derivative_view,
            "frequency": freq_magnitude,
            "label": label,
        }
        
        # Save to disk using pickle (PyHealth's standard format)
        with open(epoch_path, "wb") as f:
            pickle.dump(epoch_data, f)
        
        # ----- 4d: Create sample metadata -----
        # This is what PyHealth's dataset uses to track each epoch
        samples.append(
            {
                "record_id": f"{patient_id}-epoch-{epoch_idx}",
                "patient_id": patient_id,
                "epoch_path": epoch_path,
                "label": label,  # Stored here for easy access without loading pickle
            }
        )
    
    return samples


# ==================== HELPER FUNCTIONS ====================

def load_epoch_views(epoch_path: str) -> Dict[str, np.ndarray]:
    """Helper function to load the three views from a saved epoch file.
    
    Args:
        epoch_path: Path to the .pkl file saved by multi_view_time_series_fn
    
    Returns:
        Dictionary with keys: 'temporal', 'derivative', 'frequency', 'label'
    
    Example:
        >>> views = load_epoch_views('/path/to/patient-epoch-0.pkl')
        >>> temporal = views['temporal']  # Use for training
    """
    with open(epoch_path, "rb") as f:
        return pickle.load(f)


def get_view_shapes(
    sample_rate: int = 100, 
    epoch_seconds: int = 30, 
    num_channels: int = 2
) -> Dict[str, Tuple[int, int]]:
    """Returns expected shapes for each view given parameters.
    
    Useful for setting up model input dimensions.
    
    Args:
        sample_rate: Sampling rate in Hz
        epoch_seconds: Duration of each epoch in seconds
        num_channels: Number of signal channels
    
    Returns:
        Dictionary with expected shapes for temporal, derivative, and frequency views
    """
    time_steps = sample_rate * epoch_seconds
    
    return {
        "temporal": (num_channels, time_steps),
        "derivative": (num_channels, time_steps - 1),
        "frequency": (num_channels, time_steps // 2),
    }


# ==================== SELF-TEST (only runs when executed directly) ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing multi_view_time_series_fn")
    print("=" * 60)
    
    # Create a dummy record
    test_record = [{
        "load_from_path": "/tmp/test_data",
        "signal_file": "test_signal.edf",
        "label_file": "test_labels.txt",
        "save_to_path": "/tmp/test_output",
        "subject_id": "TEST001",
    }]
    
    # Run the function
    samples = multi_view_time_series_fn(
        test_record, 
        epoch_seconds=30, 
        sample_rate=100,
        num_channels=2
    )
    
    print(f"\n✓ Generated {len(samples)} samples")
    
    if len(samples) > 0:
        sample = samples[0]
        print(f"\nSample metadata keys: {list(sample.keys())}")
        print(f"  - record_id: {sample['record_id']}")
        print(f"  - patient_id: {sample['patient_id']}")
        print(f"  - label: {sample['label']}")
        print(f"  - epoch_path: {sample['epoch_path']}")
        
        # Load and check the saved data
        with open(sample["epoch_path"], "rb") as f:
            views = pickle.load(f)
        
        print(f"\nSaved views keys: {list(views.keys())}")
        print(f"\nView shapes:")
        print(f"  - temporal: {views['temporal'].shape}")
        print(f"  - derivative: {views['derivative'].shape}")
        print(f"  - frequency: {views['frequency'].shape}")
        
        # Verify shapes are correct
        expected = get_view_shapes(sample_rate=100, epoch_seconds=30, num_channels=2)
        assert views['temporal'].shape == expected['temporal'], "Temporal shape mismatch"
        assert views['derivative'].shape == expected['derivative'], "Derivative shape mismatch"
        assert views['frequency'].shape == expected['frequency'], "Frequency shape mismatch"
        print("\n✓ All shape checks passed!")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)