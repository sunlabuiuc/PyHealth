#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Bio-Impedance Data Generation Module for Cuff-less Blood Pressure Estimation Study

This module provides the `SyntheticBioZDataset` class. Instantiating this class 
generates a synthetic multi-channel bio-impedance (BioZ) beat dataset and 
corresponding synthetic Diastolic (DBP) and Systolic (SBP) Blood Pressure 
labels for multiple subjects. The generated data is stored as attributes of 
the class instance.

CONTEXT:
This script was developed as part of a reproducibility study for the paper:
    Zhang, L., Hurley, N. C., Ibrahim, B., Spatz, E., Krumholz, H. M., 
    Jafari, R., & Mortazavi, B. J. (2020). Domain-Adversarial Neural Networks 
    for Cuff-less Blood-Pressure Estimation. arXiv preprint arXiv:2007.12802.
    (Referenced as Zhang et al., 2020)

The original dataset used in the paper was not publicly available. Therefore, 
this module aims to generate synthetic data that structurally mimics the 
pre-processed beat-level input tensors described in the paper (Section 3.1).

LIMITATIONS (Applies to the generated synthetic data):
- **Synthetic Nature:** BioZ signals (Gaussian pulses) and BP values (constrained 
  random walks) are artificial and lack real physiological complexity and noise.
- **No Physiological Guarantee:** Relationships between synthetic signals and 
  labels may not reflect true physiological principles.
- **Simplified Variability:** Inter-subject variability is modeled simplistically.

PRIMARY CLASS:
- SyntheticBioZDataset: Upon instantiation, generates the dataset.

DATA ACCESSIBLE VIA INSTANCE ATTRIBUTES:
After `dataset = SyntheticBioZDataset()`:
- `dataset.subject_data`: Dict; keys are subject IDs. Each maps to a dict with:
    - 'beats': np.ndarray (N_beats x (PAD_LENGTH * 9)), flattened feature vectors.
               Features: [raw_ch1-4, time, deriv_ch1-4] per timestep.
               Padded, raw centered, derivatives z-scaled, shuffled.
    - 'labels_scaled': np.ndarray (N_beats x 2), [scaled_DBP, scaled_SBP].
                       MinMax scaled globally to [0, 1], shuffled.
    - 'labels_raw': np.ndarray (N_beats x 2), [DBP, SBP] in mmHg, shuffled.
    - 'order': np.ndarray (N_beats,), original indices for permutation.
- `dataset.derivative_scaler`: Fitted sklearn.preprocessing.StandardScaler for derivatives.
- `dataset.label_scaler_min`: np.ndarray (2,), global [min_DBP, min_SBP] in mmHg.
- `dataset.label_scaler_max`: np.ndarray (2,), global [max_DBP, max_SBP] in mmHg.
- `dataset.config`: Dict of configuration parameters used for generation.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- Configuration Constants (Defaults for class __init__) ---

DEFAULT_SUBJECTS_CONFIG = {
    1: 800, 3: 800, 6: 800, 7: 800, 9: 800,
    10: 800, 11: 1200, 12: 800, 13: 1200, 14: 800, 15: 800
}
DEFAULT_SAMPLE_RATE = 100  # Hz
DEFAULT_BEAT_LENGTH_MIN, DEFAULT_BEAT_LENGTH_MAX = 60, 100
DEFAULT_PAD_LENGTH = 100
DEFAULT_DERIVATIVE_SCALER_FIT_SUBJECT_ID = 11
DEFAULT_PADDING_VALUE = 0.0
DEFAULT_RANDOM_SEED = 42


class SyntheticBioZDataset:
    """
    Generates and holds a synthetic bio-impedance and blood pressure dataset.

    Upon initialization, the entire dataset is generated based on the provided
    configuration and stored in instance attributes.
    """

    def __init__(self,
                 subjects_config=None,
                 sample_rate=DEFAULT_SAMPLE_RATE,
                 beat_length_min=DEFAULT_BEAT_LENGTH_MIN,
                 beat_length_max=DEFAULT_BEAT_LENGTH_MAX,
                 pad_length=DEFAULT_PAD_LENGTH,
                 derivative_scaler_fit_subject_id=DEFAULT_DERIVATIVE_SCALER_FIT_SUBJECT_ID,
                 padding_value=DEFAULT_PADDING_VALUE,
                 random_seed=DEFAULT_RANDOM_SEED):
        """
        Initializes the dataset generation process.

        Args:
            subjects_config (dict, optional): Dict of subject_id: num_beats.
            sample_rate (int, optional): Sampling rate in Hz.
            beat_length_min (int, optional): Min samples per beat.
            beat_length_max (int, optional): Max samples per beat.
            pad_length (int, optional): Target sequence length after padding.
            derivative_scaler_fit_subject_id (int, optional): Subject ID for fitting derivative scaler.
            padding_value (float, optional): Value for padding.
            random_seed (int, optional): Seed for NumPy's random number generator.
        """
        if subjects_config is None:
            subjects_config = DEFAULT_SUBJECTS_CONFIG.copy()

        self.config = {
            "subjects_config": subjects_config,
            "sample_rate": sample_rate,
            "beat_length_min": beat_length_min,
            "beat_length_max": beat_length_max,
            "pad_length": pad_length,
            "derivative_scaler_fit_subject_id": derivative_scaler_fit_subject_id,
            "padding_value": padding_value,
            "random_seed": random_seed
        }
        self.rng = np.random.RandomState(self.config["random_seed"])

        print("Initializing synthetic dataset generation...")
        self._generate_full_dataset()
        print("Synthetic dataset generation complete. Data available in instance attributes.")

    def _generate_bp_sequence(self, num_beats, subject_id=None):
        """Generates a sequence of synthetic DBP/SBP values (mmHg)."""
        dbp_start = self.rng.uniform(60, 80)
        sbp_start = dbp_start + self.rng.uniform(30, 50)

        drift_factor = self.rng.uniform(-0.05, 0.05)
        volatility = self.rng.uniform(0.7, 1.3)

        dbps = [dbp_start]
        sbps = [sbp_start]

        for _ in range(num_beats - 1):
            dbp_step = self.rng.normal(drift_factor, volatility * 1.0)
            if self.rng.rand() < 0.01:
                dbp_step += self.rng.normal(0, 4)
            next_dbp = np.clip(dbps[-1] + dbp_step, 50, 110)

            sbp_step = (dbp_step * self.rng.uniform(0.7, 1.1) +
                        self.rng.normal(0, volatility * 1.5))
            next_sbp = np.clip(sbps[-1] + sbp_step, next_dbp + 25, 180)
            next_sbp = np.maximum(next_dbp + 25, next_sbp)

            dbps.append(next_dbp)
            sbps.append(next_sbp)

        return np.stack([dbps, sbps], axis=1).astype("f4")

    def _generate_synthetic_beat(self, length):
        """Generates a single synthetic beat signal [ch1-4, time]."""
        time_vec = np.arange(length) / self.config["sample_rate"]

        center = self.rng.uniform(0.3, 0.7) * length
        width = self.rng.uniform(0.05, 0.1) * length
        pulse = np.exp(-0.5 * ((np.arange(length) - center) / width) ** 2)

        channels = []
        for _ in range(4):
            channel = (self.rng.uniform(0.9, 1.1) * pulse +
                       self.rng.normal(0, 0.06, length))
            channels.append(channel)

        return np.stack(channels + [time_vec], axis=1).astype("f4")

    def _generate_full_dataset(self):
        """Orchestrates the full dataset generation."""
        # --- Step 1: Fit Derivative Scaler ---
        print("Fitting derivative feature scaler (StandardScaler)...")
        deriv_samples = []
        fit_subject_id = self.config["derivative_scaler_fit_subject_id"]
        if fit_subject_id not in self.config["subjects_config"]:
            raise ValueError(f"derivative_scaler_fit_subject_id {fit_subject_id} "
                             f"not in subjects_config keys.")
        n_scaler_samples = self.config["subjects_config"][fit_subject_id]

        for _ in range(n_scaler_samples):
            beat_len = self.rng.randint(self.config["beat_length_min"], self.config["beat_length_max"] + 1)
            beat_raw_for_scaler = self._generate_synthetic_beat(beat_len)
            if beat_len > 1:
                deriv = np.gradient(beat_raw_for_scaler[:, :4], axis=0)
                deriv_samples.append(deriv)

        if not deriv_samples:
            raise ValueError("No derivative samples for scaler. Check beat_length_min/max.")
        self.derivative_scaler = StandardScaler().fit(np.vstack(deriv_samples))
        print("Derivative scaler fitted.")

        # --- Step 2: Generate Beats and Raw Labels for All Subjects ---
        subject_data_intermediate = {}
        all_raw_labels_list = []
        print("Generating synthetic beats and BP labels for all subjects...")

        for subject_id, num_beats_total in self.config["subjects_config"].items():
            print(f"  Generating data for Subject {subject_id:02d}...")
            raw_labels_subject = self._generate_bp_sequence(num_beats_total, subject_id)
            processed_beats_list = []
            valid_raw_labels_list = []

            for i in range(num_beats_total):
                beat_len = self.rng.randint(self.config["beat_length_min"], self.config["beat_length_max"] + 1)
                beat_raw = self._generate_synthetic_beat(beat_len)
                beat_raw[:, :4] -= beat_raw[:, :4].mean(axis=0, keepdims=True) # Centering

                derivs = np.gradient(beat_raw[:, :4], axis=0) if beat_len > 1 else np.zeros_like(beat_raw[:, :4])
                beat_features = np.concatenate([beat_raw, derivs], axis=1)

                try:
                    if derivs.shape[1] == 4: # Ensure correct shape
                        scaled_derivs = self.derivative_scaler.transform(derivs)
                        beat_features[:, 5:9] = scaled_derivs
                    else:
                        print(f"Warning: Derivative shape mismatch for S{subject_id} beat {i}. Using unscaled.")
                except Exception as e:
                    print(f"Error scaling derivatives for S{subject_id} beat {i}: {e}. Using unscaled.")

                padding_config = ((self.config["pad_length"] - beat_len, 0), (0, 0))
                padded_beat = np.pad(beat_features, padding_config, 'constant',
                                     constant_values=self.config["padding_value"])
                processed_beats_list.append(padded_beat.reshape(-1))
                valid_raw_labels_list.append(raw_labels_subject[i])

            if processed_beats_list:
                subject_data_intermediate[subject_id] = {
                    'beats_list': processed_beats_list,
                    'raw_labels_list': valid_raw_labels_list
                }
                all_raw_labels_list.extend(valid_raw_labels_list)
                print(f"  Subject {subject_id:02d}: Generated {len(processed_beats_list)} beats.")
            else:
                print(f"  Warning: No beats processed for Subject {subject_id:02d}.")

        if not all_raw_labels_list:
            raise ValueError("No labels generated. Cannot proceed.")

        # --- Step 3: Fit Global Label Scaler ---
        print("Fitting label scaler (MinMaxScaler) globally...")
        all_raw_labels_array = np.array(all_raw_labels_list, dtype="f4")
        y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(all_raw_labels_array)
        self.label_scaler_min = y_scaler.data_min_.astype("f4")
        self.label_scaler_max = y_scaler.data_max_.astype("f4")
        # self.label_scaler = y_scaler # Optionally store the full scaler
        print("Label scaler fitted.")

        # --- Step 4: Finalize Data Structures ---
        self.subject_data = {}
        print("Finalizing data for each subject (scaling labels, shuffling)...")

        for subject_id, data_lists in subject_data_intermediate.items():
            subject_beats_array = np.array(data_lists['beats_list'], dtype="f4")
            subject_raw_labels_array = np.array(data_lists['raw_labels_list'], dtype="f4")

            if subject_beats_array.size == 0 or subject_raw_labels_array.size == 0:
                print(f"  Skipping Subject {subject_id:02d} due to empty data.")
                continue

            subject_labels_scaled_array = y_scaler.transform(subject_raw_labels_array).astype("f4")
            num_beats_subj = len(subject_beats_array)
            random_order = self.rng.permutation(num_beats_subj).astype("i4")

            self.subject_data[subject_id] = {
                'beats': subject_beats_array[random_order],
                'labels_scaled': subject_labels_scaled_array[random_order],
                'labels_raw': subject_raw_labels_array[random_order],
                'order': random_order
            }
            print(f"  Subject {subject_id:02d}: Finalized {num_beats_subj} beats/labels.")


if __name__ == "__main__":
    print("Demonstrating SyntheticBioZDataset class usage...")

    # Instantiate the class to generate data with default settings
    # This will print progress messages from the __init__ and _generate_full_dataset methods.
    synthetic_dataset = SyntheticBioZDataset(random_seed=42)

    print(f"\n--- Dataset Details ---")
    print(f"Data generated for {len(synthetic_dataset.subject_data)} subjects.")

    # Access data for a specific subject (e.g., the first one available)
    if synthetic_dataset.subject_data:
        example_subject_id = list(synthetic_dataset.subject_data.keys())[0]
        print(f"\nExample data for Subject {example_subject_id:02d}:")

        subject_beats = synthetic_dataset.subject_data[example_subject_id]['beats']
        subject_labels_scaled = synthetic_dataset.subject_data[example_subject_id]['labels_scaled']
        subject_labels_raw = synthetic_dataset.subject_data[example_subject_id]['labels_raw']

        print(f"  Number of beats: {len(subject_beats)}")
        print(f"  Shape of beats array: {subject_beats.shape}")
        print(f"  Shape of scaled labels array: {subject_labels_scaled.shape}")
        print(f"  First 3 scaled labels:\n{subject_labels_scaled[:3]}")
        print(f"  First 3 raw labels (mmHg):\n{subject_labels_raw[:3]}")

    # Access global scalers and config
    deriv_scaler = synthetic_dataset.derivative_scaler
    label_min = synthetic_dataset.label_scaler_min
    label_max = synthetic_dataset.label_scaler_max
    config_used = synthetic_dataset.config

    print("\nDerivative Scaler Info:")
    print(f"  Mean per feature: {deriv_scaler.mean_}")

    print("\nLabel Scaler Info (Global Min/Max for DBP, SBP in mmHg):")
    print(f"  Min DBP, SBP: {label_min}")
    print(f"  Max DBP, SBP: {label_max}")

    print("\nConfiguration used for generation snapshot:")
    print(f"  Subjects config sample (first item): {list(config_used['subjects_config'].items())[0] if config_used['subjects_config'] else 'N/A'}")
    print(f"  Sample rate: {config_used['sample_rate']}")
    print(f"  Random seed: {config_used['random_seed']}")

    print("\nTo use this class in another script:")
    print("1. Ensure this file ('your_module_name.py') is in your Python path.")
    print("2. From your_module_name import SyntheticBioZDataset")
    print("3. dataset_instance = SyntheticBioZDataset(subjects_config={1:100, 2:150}, random_seed=123) # Customize params")
    print("4. Access data: dataset_instance.subject_data[1]['beats']")