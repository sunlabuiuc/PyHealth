#!/usr/bin/env python3
"""
Generate synthetic MIMIC-III patients using a trained MedGAN checkpoint.
Uses simple 0.5 threshold - MedGAN doesn't require post-processing.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from pyhealth.models.generators.medgan import MedGAN


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic patients using trained MedGAN")
    parser.add_argument("--checkpoint", required=True, help="Path to trained MedGAN checkpoint (.pth)")
    parser.add_argument("--vocab", required=True, help="Path to ICD-9 vocabulary file (.txt)")
    parser.add_argument("--data_matrix", required=True, help="Path to training data matrix (.npy)")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of synthetic patients to generate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarization (binary mode only)")

    # Mode parameters
    parser.add_argument("--data_mode", type=str, default="binary", choices=["binary", "count"],
                       help="Data mode: 'binary' (default) or 'count'")
    parser.add_argument("--count_activation", type=str, default="relu", choices=["relu", "softplus"],
                       help="Activation for count mode: 'relu' (default) or 'softplus'")
    parser.add_argument("--count_loss", type=str, default="mse", choices=["mse", "poisson"],
                       help="Loss function for count mode: 'mse' (default) or 'poisson'")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Data mode: {args.data_mode}")

    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab}")
    with open(args.vocab, 'r') as f:
        code_vocab = [line.strip() for line in f]
    print(f"Loaded {len(code_vocab)} ICD-9 codes")

    # Load data matrix to get architecture dimensions
    print(f"Loading data matrix from {args.data_matrix}")
    data_matrix = np.load(args.data_matrix)
    n_codes = data_matrix.shape[1]
    print(f"Data matrix shape: {data_matrix.shape}")
    if args.data_mode == "binary":
        print(f"Real data avg codes/patient: {data_matrix.sum(axis=1).mean():.2f}")
    else:
        print(f"Real data avg code occurrences/patient: {data_matrix.sum(axis=1).mean():.2f}")
        print(f"Real data max count: {data_matrix.max():.0f}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Initialize MedGAN with same architecture
    print("Initializing MedGAN model...")
    if args.data_mode == "binary":
        model = MedGAN.from_binary_matrix(
            binary_matrix=data_matrix,
            latent_dim=128,
            autoencoder_hidden_dim=128,
            discriminator_hidden_dim=256,
            minibatch_averaging=True,
            data_mode=args.data_mode
        ).to(device)
    else:  # count mode
        model = MedGAN.from_count_matrix(
            count_matrix=data_matrix,
            latent_dim=128,
            autoencoder_hidden_dim=128,
            discriminator_hidden_dim=256,
            minibatch_averaging=True,
            count_activation=args.count_activation,
            count_loss=args.count_loss
        ).to(device)

    # Load trained weights
    model.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    model.eval()
    print("Model loaded successfully")

    # Generate synthetic patients
    print(f"\nGenerating {args.n_samples} synthetic patients...")

    with torch.no_grad():
        # Generate data
        synthetic_data = model.generate(args.n_samples, device)

        # Apply transform (binary threshold or count round+clip)
        if args.data_mode == "binary":
            discrete_data = model.sample_transform(synthetic_data, threshold=args.threshold)
        else:
            discrete_data = model.sample_transform(synthetic_data)

    data_matrix_synthetic = discrete_data.cpu().numpy()

    # Calculate statistics
    avg_codes = data_matrix_synthetic.sum(axis=1).mean()
    std_codes = data_matrix_synthetic.sum(axis=1).std()
    min_codes = data_matrix_synthetic.sum(axis=1).min()
    max_codes = data_matrix_synthetic.sum(axis=1).max()
    sparsity = (data_matrix_synthetic == 0).mean()

    print(f"\nSynthetic data statistics:")
    if args.data_mode == "binary":
        print(f"  Avg codes per patient: {avg_codes:.2f} ± {std_codes:.2f}")
    else:
        print(f"  Avg code occurrences per patient: {avg_codes:.2f} ± {std_codes:.2f}")
        print(f"  Max count: {data_matrix_synthetic.max():.0f}")
    print(f"  Range: [{min_codes:.0f}, {max_codes:.0f}]")
    print(f"  Sparsity: {sparsity:.4f}")

    # Check heterogeneity
    unique_profiles = len(set(tuple(row) for row in data_matrix_synthetic))
    print(f"  Unique patient profiles: {unique_profiles}/{args.n_samples} ({unique_profiles/args.n_samples*100:.1f}%)")

    # Convert to CSV format (SUBJECT_ID, ICD9_CODE)
    print(f"\nConverting to CSV format...")
    records = []
    for patient_idx in range(args.n_samples):
        patient_id = f"SYNTHETIC_{patient_idx+1:06d}"

        if args.data_mode == "binary":
            # Binary mode: include codes where value == 1
            code_indices = np.where(data_matrix_synthetic[patient_idx] == 1)[0]
            for code_idx in code_indices:
                records.append({
                    'SUBJECT_ID': patient_id,
                    'ICD9_CODE': code_vocab[code_idx]
                })
        else:  # count mode
            # Count mode: repeat codes based on their counts
            for code_idx in range(n_codes):
                count = int(data_matrix_synthetic[patient_idx, code_idx])
                for _ in range(count):
                    records.append({
                        'SUBJECT_ID': patient_id,
                        'ICD9_CODE': code_vocab[code_idx]
                    })

    df = pd.DataFrame(records)
    print(f"Created {len(df)} diagnosis records for {args.n_samples} patients")

    # Save to CSV
    print(f"\nSaving to {args.output}")
    df.to_csv(args.output, index=False)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved {file_size_mb:.1f} MB")

    print("\n✓ Generation complete!")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
