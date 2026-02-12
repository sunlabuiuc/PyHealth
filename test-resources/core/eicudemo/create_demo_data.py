"""
Script to create eICU demo data subset from the PhysioNet eICU-CRD demo dataset.

This script loads the eICU demo data and creates a smaller subset with ~100 patients
for use in unit tests. It ensures patients with multiple visits and complete clinical
data are included for task testing.
"""

import gzip
import os
import random
from pathlib import Path

import pandas as pd

# Set random seed for reproducibility
random.seed(42)

# Source and destination paths
EICU_DEMO_ROOT = Path("/home/johnwu3/projects/PyHealth_Branch_Testing/datasets/physionet.org/files/eicu-crd-demo/2.0.1")
OUTPUT_DIR = Path(__file__).parent

# Tables to include in the demo subset
TABLES_TO_INCLUDE = [
    "patient",
    "hospital", 
    "diagnosis",
    "medication",
    "treatment",
    "lab",
    "physicalExam",
    "admissionDx",
]


def load_csv_gz(filepath: Path) -> pd.DataFrame:
    """Load a gzipped CSV file."""
    print(f"Loading {filepath.name}...")
    return pd.read_csv(filepath, compression="gzip", low_memory=False)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Save a DataFrame to CSV."""
    df.to_csv(filepath, index=False)
    print(f"Saved {filepath.name} with {len(df)} rows")


def main():
    print("=" * 60)
    print("Creating eICU Demo Data Subset")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load patient table first to select patients
    patient_df = load_csv_gz(EICU_DEMO_ROOT / "patient.csv.gz")
    print(f"Total patients (rows) in source: {len(patient_df)}")
    print(f"Unique patients: {patient_df['uniquepid'].nunique()}")
    
    # Find patients with multiple ICU stays (for task testing)
    patient_visit_counts = patient_df.groupby('uniquepid').size()
    multi_visit_patients = patient_visit_counts[patient_visit_counts >= 2].index.tolist()
    single_visit_patients = patient_visit_counts[patient_visit_counts == 1].index.tolist()
    
    print(f"Patients with multiple visits: {len(multi_visit_patients)}")
    print(f"Patients with single visit: {len(single_visit_patients)}")
    
    # Select patients: prioritize multi-visit patients for task testing
    # Take all multi-visit patients + some single-visit to get ~100 total
    num_multi = min(len(multi_visit_patients), 50)
    num_single = min(100 - num_multi, len(single_visit_patients))
    
    selected_multi = random.sample(multi_visit_patients, num_multi)
    selected_single = random.sample(single_visit_patients, num_single)
    selected_patients = selected_multi + selected_single
    
    print(f"\nSelected {len(selected_patients)} patients:")
    print(f"  - {len(selected_multi)} multi-visit patients")
    print(f"  - {len(selected_single)} single-visit patients")
    
    # Filter patient table
    patient_subset = patient_df[patient_df['uniquepid'].isin(selected_patients)]
    selected_stay_ids = set(patient_subset['patientunitstayid'].astype(str))
    selected_hospital_ids = set(patient_subset['hospitalid'].astype(str))
    
    print(f"Total ICU stays in subset: {len(patient_subset)}")
    
    # Save patient table
    save_csv(patient_subset, OUTPUT_DIR / "patient.csv")
    
    # Load and filter hospital table
    hospital_df = load_csv_gz(EICU_DEMO_ROOT / "hospital.csv.gz")
    hospital_subset = hospital_df[hospital_df['hospitalid'].astype(str).isin(selected_hospital_ids)]
    save_csv(hospital_subset, OUTPUT_DIR / "hospital.csv")
    
    # Process clinical tables (filter by patientunitstayid)
    clinical_tables = ["diagnosis", "medication", "treatment", "lab", "physicalExam", "admissionDx"]
    
    for table_name in clinical_tables:
        source_file = EICU_DEMO_ROOT / f"{table_name}.csv.gz"
        if not source_file.exists():
            print(f"Warning: {source_file.name} not found, skipping")
            continue
            
        df = load_csv_gz(source_file)
        
        # Filter by patientunitstayid
        if 'patientunitstayid' in df.columns:
            df_subset = df[df['patientunitstayid'].astype(str).isin(selected_stay_ids)]
            save_csv(df_subset, OUTPUT_DIR / f"{table_name}.csv")
        else:
            print(f"Warning: {table_name} has no patientunitstayid column, saving full table")
            save_csv(df, OUTPUT_DIR / f"{table_name}.csv")
    
    print("\n" + "=" * 60)
    print("Demo data creation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Print summary statistics
    print("\nSummary of created files:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df)} rows")


if __name__ == "__main__":
    main()





