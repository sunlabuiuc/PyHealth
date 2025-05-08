#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Bio-Impedance Data Generation for Cuff-less Blood Pressure Estimation Study

This script generates synthetic multi-channel bio-impedance (BioZ) beat data 
and corresponding synthetic Diastolic (DBP) and Systolic (SBP) Blood Pressure 
labels for multiple subjects. 

CONTEXT:
This script was developed as part of a reproducibility study for the paper:
    Zhang, L., Hurley, N. C., Ibrahim, B., Spatz, E., Krumholz, H. M., 
    Jafari, R., & Mortazavi, B. J. (2020). Domain-Adversarial Neural Networks 
    for Cuff-less Blood-Pressure Estimation. arXiv preprint arXiv:2007.12802.
    (Referenced as Zhang et al., 2020)

The original dataset used in the paper was not publicly available. Therefore, 
this script aims to generate synthetic data that structurally mimics the 
pre-processed beat-level input tensors described in the paper (Section 3.1), 
which include raw BioZ channels and their derivatives.

LIMITATIONS:
- **Synthetic Data:** The generated BioZ signals (based on Gaussian pulses) and 
  BP values (based on constrained random walks) are synthetic and do not 
  represent real physiological measurements. They lack the complexity, noise 
  characteristics, and true physiological correlations found in real-world data.
- **No Guarantee of Physiological Accuracy:** While efforts were made to keep BP 
  values within plausible ranges and generate beat-like shapes, there is no 
  guarantee that the relationship between the synthetic signals and synthetic 
  BP labels reflects true physiological principles.
- **Simplified Variability:** Inter-subject variability is modeled simplistically 
  through parameters in the BP generation and minor variations in beat synthesis. 
  It likely does not capture the full spectrum of real-world differences.

PURPOSE IN REPOSITORY:
This script is provided to allow regeneration of the synthetic dataset used in 
the associated reproducibility study. It demonstrates the data format and 
preprocessing steps applied. Users should be aware of the limitations of this 
synthetic data when interpreting results based on it.

OUTPUT:
For each subject ID defined in SUBJECTS, this script creates a subdirectory 
(e.g., 'bioz/Subject01/') containing:
- beats.csv: Flattened feature vectors for each beat (N_beats x 900). 
             Each row represents one beat (100 time steps * 9 features).
             The 9 features per time step are:
                 [raw_ch1, raw_ch2, raw_ch3, raw_ch4, time, 
                  deriv_ch1, deriv_ch2, deriv_ch3, deriv_ch4]
             where 'raw' channels are centered, 'deriv' channels are 
             z-scaled derivatives of the raw channels. Data is right-padded to 100 steps.
             Rows are saved in a random order defined by order.csv.
- labels.csv: Scaled DBP and SBP labels corresponding to each beat.
              Each row is [scaled_DBP, scaled_SBP]. Labels are MinMax scaled 
              globally across all subjects to [0, 1].
              Rows are saved in the same random order as beats.csv.
- order.csv: A single column listing the original indices (0 to N_beats-1) 
             representing the random permutation applied to beats.csv and 
             labels.csv. This allows reconstruction of the original sequence order.
- label_min.npy, label_max.npy: Saved in the script's execution directory. 
                                Contain the global minimum and maximum DBP/SBP 
                                values (in mmHg) used for scaling labels.csv. 
                                Needed to inverse-transform labels back to mmHg.
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- Configuration Constants ---

# Root directory for saving the generated subject data folders
ROOT_DIR = "bioz_synthetic_dataset" 

# Dictionary defining subject IDs and the number of beats to generate for each
# Based on subject IDs mentioned in Zhang et al., 2020. Beat counts are arbitrary examples.
SUBJECTS = {
    1: 800, 3: 800, 6: 800, 7: 800, 9: 800,
    10: 800, 11: 1200, 12: 800, 13: 1200, 14: 800, 15: 800
}

# Sampling frequency of the synthetic signal (matches paper's downsampled rate)
SAMPLE_RATE = 100  # Hz 

# Minimum and maximum length (in samples) for initially generated synthetic beats
BEAT_LENGTH_MIN, BEAT_LENGTH_MAX = 60, 100  

# Target sequence length after padding (matches LSTM input size in paper/code)
PAD_LENGTH = 100   

# Subject ID used to generate sample beats for fitting the derivative scaler
# Choosing a subject with more beats might provide a more stable scaler.
DERIVATIVE_SCALER_FIT_SUBJECT = 11

# Padding value used when extending sequences to PAD_LENGTH
# Using 0 as it's common for normalized/scaled data, though paper mentioned -3.
PADDING_VALUE = 0.0

# --- Synthetic Data Generation Functions ---

def generate_bp_sequence(num_beats, subject_id):
    """
    Generates a sequence of somewhat plausible synthetic DBP/SBP values (mmHg).

    Uses a constrained random walk model with subject-specific parameters 
    (start points, drift, volatility) to simulate BP fluctuations over time.
    Includes occasional larger jumps to mimic activity changes.
    Values are clipped to physiologically plausible ranges.

    Args:
        num_beats (int): The number of consecutive beats for which to generate BP values.
        subject_id (int): The ID of the subject (used here mainly for potential 
                          future extension, currently only influences random seed implicitly).

    Returns:
        np.ndarray: A NumPy array of shape (num_beats, 2) containing [DBP, SBP] 
                    values in mmHg, with dtype float32.
    """
    # Initialize BP values with some randomness based loosely on typical ranges
    dbp_start = np.random.uniform(60, 80)
    sbp_start = dbp_start + np.random.uniform(30, 50) # Ensure initial SBP > DBP
    
    # Parameters for the random walk (can be made subject-dependent if needed)
    drift_factor = np.random.uniform(-0.05, 0.05) # Small average trend per beat
    volatility = np.random.uniform(0.7, 1.3)      # Controls magnitude of random fluctuations

    dbps = [dbp_start]
    sbps = [sbp_start]
    
    for _ in range(num_beats - 1):
        # Calculate random step for DBP based on drift and volatility
        dbp_step = np.random.normal(drift_factor, volatility * 1.0)
        
        # Simulate occasional larger BP changes (e.g., due to posture/activity)
        if np.random.rand() < 0.01:  # 1% chance of a larger jump
            dbp_step += np.random.normal(0, 4) # Add extra random jump

        # Calculate next DBP, clipping to a reasonable physiological range
        next_dbp = np.clip(dbps[-1] + dbp_step, 50, 110) # Clip DBP between 50 and 110 mmHg

        # Calculate SBP step: related to DBP change plus its own independent variation
        sbp_step = (dbp_step * np.random.uniform(0.7, 1.1) +  # SBP change correlated with DBP change
                    np.random.normal(0, volatility * 1.5))   # Add extra SBP variability

        # Calculate next SBP, clipping to range and ensuring minimum pulse pressure
        next_sbp = np.clip(sbps[-1] + sbp_step, next_dbp + 25, 180) # Clip SBP between DBP+25 and 180 mmHg
        # Ensure minimum pulse pressure constraint again after clipping upper bound
        next_sbp = np.maximum(next_dbp + 25, next_sbp) 

        dbps.append(next_dbp)
        sbps.append(next_sbp)

    # Stack DBP and SBP columns and return as float32 array
    return np.stack([dbps, sbps], axis=1).astype("f4")


def generate_synthetic_beat(length):
    """
    Generates a single synthetic beat signal with 4 raw channels and a time channel.

    Models the beat shape as a simple Gaussian pulse with added noise. 
    This is a significant simplification of real BioZ signals.

    Args:
        length (int): The number of time samples for this beat.

    Returns:
        np.ndarray: A NumPy array of shape (length, 5) containing 
                    [ch1, ch2, ch3, ch4, time], with dtype float32.
    """
    # Create time vector for the beat duration
    time = np.arange(length) / SAMPLE_RATE
    
    # Define parameters for the Gaussian pulse shape (randomized center and width)
    center = np.random.uniform(0.3, 0.7) * length # Pulse center relative to beat length
    width = np.random.uniform(0.05, 0.1) * length # Pulse width relative to beat length
    pulse = np.exp(-0.5 * ((np.arange(length) - center) / width) ** 2) # Gaussian function
    
    # Generate 4 'raw' signal channels
    channels = []
    for _ in range(4):
        # Apply slight random amplitude scaling and add Gaussian noise to the base pulse
        channel = (np.random.uniform(0.9, 1.1) * pulse +  # Amplitude variation
                   np.random.normal(0, 0.06, length))    # Additive noise
        channels.append(channel)
    
    # Stack the 4 channels and the time vector horizontally
    return np.stack(channels + [time], axis=1).astype("f4")


def main():
    """
    Main function to generate the synthetic dataset for all subjects.
    Includes preprocessing steps like centering, derivative calculation, 
    scaling, padding, and saving the data in the specified format.
    """
    # Ensure the root output directory exists
    os.makedirs(ROOT_DIR, exist_ok=True)
    
    # --- Step 1: Fit Derivative Scaler ---
    # A StandardScaler is fitted to the derivatives of synthetic beats generated 
    # from one subject (e.g., Subject 11). This scaler is then used to standardize 
    # the derivative features for all subjects. This mimics standard practice 
    # where scaling factors are derived from a representative training set partition.
    print("Fitting derivative feature scaler (StandardScaler)...")
    deriv_samples = [] # List to store derivative samples for fitting
    n_scaler_samples = SUBJECTS[DERIVATIVE_SCALER_FIT_SUBJECT] # Number of beats to generate for fitting
    
    for _ in range(n_scaler_samples):
        # Generate a temporary beat for fitting purposes
        beat_len = np.random.randint(BEAT_LENGTH_MIN, BEAT_LENGTH_MAX + 1)
        beat_raw = generate_synthetic_beat(beat_len) # Shape: (beat_len, 5)
        
        # Calculate derivatives (gradient) only on the 4 signal channels (first 4 columns)
        if beat_len > 1: 
            # np.gradient computes the gradient along the time axis (axis=0)
            deriv = np.gradient(beat_raw[:, :4], axis=0) # Shape: (beat_len, 4)
            deriv_samples.append(deriv)
    
    if not deriv_samples:
        raise ValueError("No derivative samples generated for scaler fitting. Check BEAT_LENGTH_MIN/MAX.")
    
    # Fit the StandardScaler (computes mean and std dev) on all collected derivative samples
    deriv_scaler = StandardScaler().fit(np.vstack(deriv_samples)) 
    print("Derivative scaler fitted.")
    
    # --- Step 2: Generate Beats and Labels for All Subjects ---
    beats_dict = {} # Dictionary to store processed beat features for each subject
    labels_dict = {} # Dictionary to store raw DBP/SBP labels (in mmHg) for each subject
    print("Generating synthetic beats and BP labels for all subjects...")
    
    for subject_id, num_beats in SUBJECTS.items():
        print(f"  Generating data for Subject {subject_id:02d}...")
        
        # 1. Generate the sequence of raw BP labels (mmHg) for this subject
        raw_labels = generate_bp_sequence(num_beats, subject_id) # Shape: (num_beats, 2)
        
        # 2. Generate and process the corresponding beats
        processed_beats = [] # List to store processed feature vectors for this subject
        skipped_beats_count = 0
        
        for i in range(num_beats):
            # Generate a raw beat signal (4 channels + time)
            beat_len = np.random.randint(BEAT_LENGTH_MIN, BEAT_LENGTH_MAX + 1)
            beat_raw = generate_synthetic_beat(beat_len) # Shape: (beat_len, 5)
            
            # Preprocessing Step: Center the raw signal channels (remove DC offset)
            beat_raw[:, :4] -= beat_raw[:, :4].mean(axis=0, keepdims=True)
            
            # Preprocessing Step: Calculate derivatives
            if beat_len > 1:
                derivs = np.gradient(beat_raw[:, :4], axis=0) # Shape: (beat_len, 4)
            else:
                derivs = np.zeros_like(beat_raw[:, :4]) # Handle very short beats (unlikely)
            
            # Preprocessing Step: Concatenate features (raw + time + derivatives)
            # Resulting shape: (beat_len, 9)
            beat_features = np.concatenate([beat_raw, derivs], axis=1) 
            
            # Preprocessing Step: Scale derivative features (columns 5 to 8) using the fitted scaler
            try:
                if derivs.shape[1] == 4: # Ensure correct shape before transform
                    scaled_derivs = deriv_scaler.transform(derivs) # Apply z-scaling
                    beat_features[:, 5:9] = scaled_derivs # Replace original derivs with scaled ones
                else:
                    # This warning indicates an unexpected issue in derivative calculation
                    print(f"Warning: Derivative shape mismatch for S{subject_id} beat {i}. Skipping scaling.")
            except Exception as e:
                # Catch potential errors during transform (e.g., if scaler wasn't fitted properly)
                print(f"Error scaling derivatives for S{subject_id} beat {i}: {e}. Using unscaled derivatives.")
                skipped_beats_count += 1
                continue # Skip this beat if scaling fails critically
            
            # Preprocessing Step: Pad the beat sequence to the target length (PAD_LENGTH)
            # Padding is added to the beginning (before the actual beat data)
            padding_config = ((PAD_LENGTH - beat_len, 0), (0, 0)) # ((before, after), (left, right))
            padded_beat = np.pad(beat_features, padding_config, 'constant', constant_values=PADDING_VALUE)
            
            # Reshape the padded beat (PAD_LENGTH, 9) into a flat 1D array (PAD_LENGTH * 9)
            # This matches the format expected by some models or for saving in a simple CSV.
            flattened_beat = padded_beat.reshape(-1) # Shape: (PAD_LENGTH * 9,)
            processed_beats.append(flattened_beat)
        
        # Store results for the subject if beats were successfully generated
        if processed_beats:
            beats_dict[subject_id] = np.array(processed_beats, dtype="f4")
            # Store the corresponding raw labels, ensuring alignment if beats were skipped
            labels_dict[subject_id] = raw_labels[:len(processed_beats)] 
            print(f"  Subject {subject_id:02d}: Generated and processed {len(processed_beats)} beats ({skipped_beats_count} skipped).")
        else:
            # This indicates a potential issue if no beats were processed for a subject
            print(f"  Warning: No beats were successfully processed for Subject {subject_id:02d}.")
    
    # Filter out subjects for whom no beats were processed
    valid_subject_ids = list(beats_dict.keys())
    labels_dict = {sid: labels_dict[sid] for sid in valid_subject_ids if sid in labels_dict}
    
    if not labels_dict:
        raise ValueError("No labels generated or retained. Cannot proceed.")
    
    # --- Step 3: Fit Global Label Scaler ---
    # A MinMaxScaler is fitted to *all* raw BP labels (mmHg) collected from *all* subjects.
    # This ensures a consistent scaling range [0, 1] across the entire dataset.
    print("Fitting label scaler (MinMaxScaler) globally across all subjects...")
    all_raw_labels = np.vstack(list(labels_dict.values())) # Combine labels from all subjects
    
    if all_raw_labels.size == 0:
        raise ValueError("Combined label array is empty. Cannot fit label scaler.")
    
    # Fit the MinMaxScaler to find the global min/max for DBP and SBP
    y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(all_raw_labels) 
    print("Label scaler fitted.")
    
    # --- Step 4: Save Scaled Data and Metadata ---
    # Save the parameters of the fitted label scaler (min and max values per feature)
    # These are required to inverse-transform model predictions back to mmHg.
    np.save("label_min.npy", y_scaler.data_min_) # Saves [global_min_DBP, global_min_SBP]
    np.save("label_max.npy", y_scaler.data_max_) # Saves [global_max_DBP, global_max_SBP]
    print(f"Saved global label scaling parameters (min/max) to label_min.npy and label_max.npy")
    
    print("Saving processed data for each subject...")
    for subject_id in valid_subject_ids:
        subject_beats_processed = beats_dict[subject_id] # Processed, flattened feature vectors
        subject_labels_raw = labels_dict[subject_id]     # Raw labels in mmHg
        
        if subject_beats_processed.size == 0 or subject_labels_raw.size == 0:
            print(f"  Skipping Subject {subject_id:02d} due to empty data after processing.")
            continue
        
        # Apply the fitted global scaler to this subject's raw labels
        subject_labels_scaled = y_scaler.transform(subject_labels_raw) # Shape: (num_beats, 2)
        
        # Generate a random permutation order for saving data shuffled
        # Saving the order allows reconstruction of the original time sequence if needed.
        num_beats_subj = len(subject_beats_processed)
        random_order = np.random.permutation(num_beats_subj).astype("i4")
        
        # Define the output directory for this subject
        subject_dir = os.path.join(ROOT_DIR, f"Subject{subject_id:02d}")
        os.makedirs(subject_dir, exist_ok=True) # Create directory if it doesn't exist
        
        # Save the processed data files
        # 1. Beats: Flattened feature vectors, saved in random order
        np.savetxt(os.path.join(subject_dir, "beats.csv"), 
                   subject_beats_processed[random_order], 
                   delimiter=",", fmt='%.6f') # Use sufficient precision for float data
                   
        # 2. Labels: Scaled DBP/SBP labels, saved in the same random order
        np.savetxt(os.path.join(subject_dir, "labels.csv"), 
                   subject_labels_scaled[random_order], 
                   delimiter=",", fmt='%.6f')
                   
        # 3. Order: The random permutation indices used
        np.savetxt(os.path.join(subject_dir, "order.csv"), 
                   random_order, 
                   delimiter=",", fmt="%d") # Save indices as integers
        
        print(f"  Subject {subject_id:02d}: Saved {num_beats_subj} processed beats/labels to {subject_dir}")
    
    print(f"\nSynthetic data generation complete. Output saved in directory: '{ROOT_DIR}'")


if __name__ == "__main__":
    # Set a random seed for reproducibility of the synthetic data generation
    np.random.seed(42) 
    main()
