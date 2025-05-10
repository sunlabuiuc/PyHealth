import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def generate_synthetic_hirid_from_mimic(
    chartevents_path: str,
    output_dir: str,
    n_patients: int = 50,
    input_channels: int = 36,
    window_size: int = 48,
    seed: int = 42
):
    """
    Generates synthetic HiRID-style time series data from MIMIC-IV chartevents.

    Args:
        chartevents_path: Path to `chartevents.csv`.
        output_dir: Directory to store `.npy` files.
        n_patients: Number of synthetic patients to generate.
        input_channels: Number of vital sign channels.
        window_size: Number of time steps per sequence.
        seed: Random seed.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    # Synthetic data shape: (n_patients, input_channels, window_size)
    print(f"Generating synthetic HiRID-style data for {n_patients} patients...")
    data = np.random.normal(
        loc=0, scale=1, size=(n_patients, input_channels, window_size)
    ).astype(np.float32)

    labels = np.random.randint(0, 2, size=(n_patients,))  # Binary task placeholder

    np.save(os.path.join(output_dir, "synthetic_data.npy"), data)
    np.save(os.path.join(output_dir, "synthetic_labels.npy"), labels)
    print(f"Synthetic data saved to {output_dir}")


if __name__ == "__main__":
    generate_synthetic_hirid_from_mimic(
        chartevents_path="path/to/chartevents.csv",  # optional for now
        output_dir="data/synthetic_hirid",
        n_patients=100,
        input_channels=36,
        window_size=48
    )
