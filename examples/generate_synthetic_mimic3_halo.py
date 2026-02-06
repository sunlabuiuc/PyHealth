#!/usr/bin/env python3
"""
Generate synthetic MIMIC-III patients using trained HALO checkpoint.
Outputs sequential visit data (temporal format) as pickle file.
"""

import os
import sys
sys.path.insert(0, '/u/jalenj4/PyHealth')
import argparse
import torch
import pickle
import pandas as pd
from pyhealth.datasets.halo_mimic3 import HALO_MIMIC3Dataset
from pyhealth.models.generators.halo import HALO
from pyhealth.models.generators.halo_resources.halo_config import HALOConfig

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic patients using trained HALO")
    parser.add_argument("--checkpoint", required=True, help="Path to trained HALO checkpoint directory")
    parser.add_argument("--output", required=True, help="Path to output pickle file")
    parser.add_argument("--csv_output", help="Optional: Path to output CSV file (converts from pickle)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vocabulary and configuration from checkpoint directory
    pkl_data_dir = args.checkpoint + "pkl_data/"
    print(f"\nLoading vocabulary from {pkl_data_dir}")

    code_to_index = pickle.load(open(f"{pkl_data_dir}codeToIndex.pkl", "rb"))
    index_to_code = pickle.load(open(f"{pkl_data_dir}indexToCode.pkl", "rb"))
    id_to_label = pickle.load(open(f"{pkl_data_dir}idToLabel.pkl", "rb"))
    train_dataset = pickle.load(open(f"{pkl_data_dir}trainDataset.pkl", "rb"))

    code_vocab_size = len(code_to_index)
    label_vocab_size = len(id_to_label)
    special_vocab_size = 3
    total_vocab_size = code_vocab_size + label_vocab_size + special_vocab_size

    print(f"Vocabulary sizes:")
    print(f"  Code vocabulary: {code_vocab_size}")
    print(f"  Label vocabulary: {label_vocab_size}")
    print(f"  Total vocabulary: {total_vocab_size}")

    # Create config with same parameters as training
    config = HALOConfig(
        total_vocab_size=total_vocab_size,
        code_vocab_size=code_vocab_size,
        label_vocab_size=label_vocab_size,
        special_vocab_size=special_vocab_size,
        n_positions=56,
        n_ctx=48,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        batch_size=48,
        sample_batch_size=256,  # Generation batch size
        epoch=50,
        pos_loss_weight=None,
        lr=1e-4
    )

    # Create a minimal dataset object (just for interface compatibility)
    class MinimalDataset:
        def __init__(self, pkl_data_dir):
            self.pkl_data_dir = pkl_data_dir

    dataset = MinimalDataset(pkl_data_dir)

    # Load trained model
    print(f"\nLoading checkpoint from {args.checkpoint}halo_model")
    from pyhealth.models.generators.halo_resources.halo_model import HALOModel

    model = HALOModel(config).to(device)
    checkpoint = torch.load(f'{args.checkpoint}halo_model', map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print("Model loaded successfully")

    # Generate synthetic patients
    n_samples = 10000  # Generate 10k synthetic patients
    print(f"\nGenerating {n_samples} synthetic patients...")
    print("This will take 1-2 hours...")

    # Create HALO instance for generation
    halo = HALO(dataset=dataset, config=config, save_dir=args.checkpoint, train_on_init=False)
    halo.model = model
    halo.train_ehr_dataset = train_dataset[:n_samples]  # Limit to 10k
    halo.index_to_code = index_to_code

    # Generate synthetic data
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    halo.synthesize_dataset(pkl_save_dir=output_dir + "/")

    # Move the generated file to the requested output path
    generated_file = os.path.join(output_dir, "haloDataset.pkl")
    if generated_file != args.output:
        os.rename(generated_file, args.output)

    print(f"\nGeneration complete!")
    print(f"Output saved to: {args.output}")

    # Load and print statistics
    synthetic_data = pickle.load(open(args.output, "rb"))
    print(f"\nSynthetic data statistics:")
    print(f"  Total patients: {len(synthetic_data)}")
    print(f"  Avg visits per patient: {sum(len(p['visits']) for p in synthetic_data) / len(synthetic_data):.2f}")
    print(f"  Total visits: {sum(len(p['visits']) for p in synthetic_data)}")
    print(f"  Avg codes per visit: {sum(len(c) for p in synthetic_data for v in p['visits'] for c in v) / sum(len(p['visits']) for p in synthetic_data):.2f}")

    # Optionally convert to CSV format
    if args.csv_output:
        print(f"\nConverting to CSV format: {args.csv_output}")
        convert_to_csv(synthetic_data, index_to_code, args.csv_output)

def convert_to_csv(synthetic_data, index_to_code, csv_path):
    """Convert pickle format to CSV with temporal information."""
    records = []
    for patient_idx, patient in enumerate(synthetic_data):
        patient_id = f"SYNTHETIC_{patient_idx+1:06d}"
        for visit_num, visit in enumerate(patient['visits'], 1):
            for code_idx in visit:
                icd9_code = index_to_code[code_idx]
                records.append({
                    'SUBJECT_ID': patient_id,
                    'VISIT_NUM': visit_num,
                    'ICD9_CODE': icd9_code
                })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"CSV saved with {len(df)} records")

if __name__ == '__main__':
    main()
