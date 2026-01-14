"""Split MIMIC-III data into training and holdout sets.

Randomly selects 1,000 patients as holdout and creates separate CSV files
for training (45,520 patients) and holdout (1,000 patients).

Usage:
    python examples/split_mimic_train_holdout.py \
        --mimic3_root /u/jalenj4/pehr_scratch/data_files \
        --train_output /u/jalenj4/pehr_scratch/data_files_train \
        --holdout_output /u/jalenj4/pehr_scratch/data_files_holdout \
        --n_holdout 1000 \
        --seed 42
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split MIMIC-III into train/holdout sets")
    parser.add_argument("--mimic3_root", type=str, required=True,
                        help="Path to original MIMIC-III data directory")
    parser.add_argument("--train_output", type=str, required=True,
                        help="Output directory for training data")
    parser.add_argument("--holdout_output", type=str, required=True,
                        help="Output directory for holdout data")
    parser.add_argument("--n_holdout", type=int, default=1000,
                        help="Number of patients to hold out")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directories
    train_dir = Path(args.train_output)
    holdout_dir = Path(args.holdout_output)
    train_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MIMIC-III Data Split: Training vs Holdout")
    print("=" * 80)

    # Load PATIENTS.csv
    print("\n[1/3] Loading PATIENTS.csv...")
    patients_path = Path(args.mimic3_root) / "PATIENTS.csv"
    patients_df = pd.read_csv(patients_path)
    all_patient_ids = patients_df['SUBJECT_ID'].unique()
    print(f"  Total patients: {len(all_patient_ids)}")

    # Randomly sample holdout patient IDs
    print(f"\n[2/3] Randomly selecting {args.n_holdout} holdout patients (seed={args.seed})...")
    holdout_ids = np.random.choice(all_patient_ids, size=args.n_holdout, replace=False)
    holdout_ids_set = set(holdout_ids)
    train_ids_set = set(all_patient_ids) - holdout_ids_set

    print(f"  Holdout patients: {len(holdout_ids_set)}")
    print(f"  Training patients: {len(train_ids_set)}")

    # Split PATIENTS.csv
    print("\n[3/3] Splitting CSV files...")
    print("  - PATIENTS.csv")
    patients_train = patients_df[patients_df['SUBJECT_ID'].isin(train_ids_set)]
    patients_holdout = patients_df[patients_df['SUBJECT_ID'].isin(holdout_ids_set)]

    patients_train.to_csv(train_dir / "PATIENTS.csv", index=False)
    patients_holdout.to_csv(holdout_dir / "PATIENTS.csv", index=False)
    print(f"    Train: {len(patients_train)} rows -> {train_dir / 'PATIENTS.csv'}")
    print(f"    Holdout: {len(patients_holdout)} rows -> {holdout_dir / 'PATIENTS.csv'}")

    # Load and split ADMISSIONS.csv
    print("  - ADMISSIONS.csv")
    admissions_path = Path(args.mimic3_root) / "ADMISSIONS.csv"
    admissions_df = pd.read_csv(admissions_path)

    admissions_train = admissions_df[admissions_df['SUBJECT_ID'].isin(train_ids_set)]
    admissions_holdout = admissions_df[admissions_df['SUBJECT_ID'].isin(holdout_ids_set)]

    admissions_train.to_csv(train_dir / "ADMISSIONS.csv", index=False)
    admissions_holdout.to_csv(holdout_dir / "ADMISSIONS.csv", index=False)
    print(f"    Train: {len(admissions_train)} rows -> {train_dir / 'ADMISSIONS.csv'}")
    print(f"    Holdout: {len(admissions_holdout)} rows -> {holdout_dir / 'ADMISSIONS.csv'}")

    # Load and split DIAGNOSES_ICD.csv
    print("  - DIAGNOSES_ICD.csv")
    diagnoses_path = Path(args.mimic3_root) / "DIAGNOSES_ICD.csv"
    diagnoses_df = pd.read_csv(diagnoses_path)

    diagnoses_train = diagnoses_df[diagnoses_df['SUBJECT_ID'].isin(train_ids_set)]
    diagnoses_holdout = diagnoses_df[diagnoses_df['SUBJECT_ID'].isin(holdout_ids_set)]

    diagnoses_train.to_csv(train_dir / "DIAGNOSES_ICD.csv", index=False)
    diagnoses_holdout.to_csv(holdout_dir / "DIAGNOSES_ICD.csv", index=False)
    print(f"    Train: {len(diagnoses_train)} rows -> {train_dir / 'DIAGNOSES_ICD.csv'}")
    print(f"    Holdout: {len(diagnoses_holdout)} rows -> {holdout_dir / 'DIAGNOSES_ICD.csv'}")

    # Save patient ID lists for reference
    print("\n[4/4] Saving patient ID lists...")
    with open(train_dir / "patient_ids.txt", 'w') as f:
        for pid in sorted(train_ids_set):
            f.write(f"{pid}\n")
    print(f"  Train IDs: {train_dir / 'patient_ids.txt'}")

    with open(holdout_dir / "patient_ids.txt", 'w') as f:
        for pid in sorted(holdout_ids_set):
            f.write(f"{pid}\n")
    print(f"  Holdout IDs: {holdout_dir / 'patient_ids.txt'}")

    print("\n" + "=" * 80)
    print("Split Complete!")
    print("=" * 80)
    print(f"Training data: {train_dir}")
    print(f"  - {len(patients_train)} patients")
    print(f"  - {len(admissions_train)} admissions")
    print(f"  - {len(diagnoses_train)} diagnoses")
    print(f"\nHoldout data: {holdout_dir}")
    print(f"  - {len(patients_holdout)} patients")
    print(f"  - {len(admissions_holdout)} admissions")
    print(f"  - {len(diagnoses_holdout)} diagnoses")
    print("=" * 80)


if __name__ == "__main__":
    main()
