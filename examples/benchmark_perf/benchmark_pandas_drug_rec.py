"""
Benchmark script for MIMIC-IV drug recommendation using pandas
(analogous to PyHealth DrugRecommendationMIMIC4 task).

This benchmark mimics the DrugRecommendationMIMIC4 task:
1. Creates visit-level samples with cumulative history
2. For each visit, extracts conditions, procedures, and drugs
3. Builds cumulative history of conditions, procedures, and drug history
4. Target is the drugs for the current visit
5. Excludes target drugs from history

Drug Recommendation Task:
- Input: conditions (nested), procedures (nested), drugs_hist (nested)
- Output: drugs (multilabel) for the current visit
"""

import argparse
import time
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import psutil


PEAK_MEM_USAGE = 0
SELF_PROC = psutil.Process(os.getpid())
STOP_TRACKING = False


def track_mem():
    """Background thread to track peak memory usage."""
    global PEAK_MEM_USAGE
    while not STOP_TRACKING:
        m = SELF_PROC.memory_info().rss
        if m > PEAK_MEM_USAGE:
            PEAK_MEM_USAGE = m
        time.sleep(0.1)


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def process_patient_drug_recommendation(
    subject_id: int,
    admissions_df: pd.DataFrame,
    diagnoses_df: pd.DataFrame,
    procedures_df: pd.DataFrame,
    prescriptions_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Process a single patient for drug recommendation task.

    Creates visit-level samples with cumulative history.
    Each sample includes all previous visits' conditions, procedures, and drugs.

    Args:
        subject_id: Patient ID
        admissions_df: Admission records (pre-filtered for this patient)
        diagnoses_df: Diagnosis ICD codes (pre-filtered for this patient)
        procedures_df: Procedure ICD codes (pre-filtered for this patient)
        prescriptions_df: Prescription records (pre-filtered for this patient)

    Returns:
        List of sample dictionaries, or empty list if patient doesn't qualify
    """
    samples = []

    # Get all admissions for this patient, sorted by time
    patient_admissions = admissions_df[
        admissions_df["subject_id"] == subject_id
    ].sort_values("admittime")

    if len(patient_admissions) < 2:
        # Need at least 2 visits for history-based prediction
        return []

    # Process each admission to collect visit data
    visit_data = []
    for _, admission in patient_admissions.iterrows():
        hadm_id = admission["hadm_id"]

        # Get diagnosis codes for this admission
        visit_diagnoses = diagnoses_df[diagnoses_df["hadm_id"] == hadm_id]
        # Combine ICD version with code (e.g., "10_A123" or "9_456")
        conditions = []
        for _, row in visit_diagnoses.iterrows():
            if pd.notna(row.get("icd_code")) and pd.notna(row.get("icd_version")):
                conditions.append(f"{int(row['icd_version'])}_{row['icd_code']}")

        # Get procedure codes for this admission
        visit_procedures = procedures_df[procedures_df["hadm_id"] == hadm_id]
        procedures = []
        for _, row in visit_procedures.iterrows():
            if pd.notna(row.get("icd_code")) and pd.notna(row.get("icd_version")):
                procedures.append(f"{int(row['icd_version'])}_{row['icd_code']}")

        # Get prescriptions for this admission
        visit_prescriptions = prescriptions_df[prescriptions_df["hadm_id"] == hadm_id]
        drugs = []
        for _, row in visit_prescriptions.iterrows():
            ndc = row.get("ndc")
            if pd.notna(ndc) and ndc:
                ndc_str = str(ndc)
                if len(ndc_str) >= 4:
                    # ATC 3 level (first 4 characters)
                    drugs.append(ndc_str[:4])

        # Exclude visits without condition, procedure, or drug code
        if len(conditions) == 0 or len(procedures) == 0 or len(drugs) == 0:
            continue

        visit_data.append({
            "hadm_id": hadm_id,
            "conditions": conditions,
            "procedures": procedures,
            "drugs": drugs,
        })

    # Exclude patients with less than 2 valid visits
    if len(visit_data) < 2:
        return []

    # Build samples with cumulative history
    for i, visit in enumerate(visit_data):
        sample = {
            "visit_id": visit["hadm_id"],
            "patient_id": subject_id,
            "conditions": [],
            "procedures": [],
            "drugs_hist": [],
            "drugs": visit["drugs"],  # Target drugs for this visit
        }

        # Add cumulative history from all visits up to and including current
        for j in range(i + 1):
            sample["conditions"].append(visit_data[j]["conditions"])
            sample["procedures"].append(visit_data[j]["procedures"])
            # drugs_hist includes history up to current, but current visit's
            # drugs will be emptied
            sample["drugs_hist"].append(visit_data[j]["drugs"])

        # Remove target drug from history (set current visit drugs_hist to empty)
        sample["drugs_hist"][i] = []

        samples.append(sample)

    return samples


def benchmark_drug_recommendation(
    admissions_df: pd.DataFrame,
    diagnoses_df: pd.DataFrame,
    procedures_df: pd.DataFrame,
    prescriptions_df: pd.DataFrame,
    n_patients: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Benchmark MIMIC-IV drug recommendation processing.

    Args:
        admissions_df: Admissions dataframe
        diagnoses_df: Diagnoses dataframe
        procedures_df: Procedures dataframe
        prescriptions_df: Prescriptions dataframe
        n_patients: Number of patients to process (None = all patients)

    Returns:
        Tuple of (list of samples, processing time in seconds)
    """
    print("=" * 80)
    print("BENCHMARK: Pandas Drug Recommendation (MIMIC-IV format)")
    print("=" * 80)

    # Get unique patients
    all_patients = admissions_df["subject_id"].unique().tolist()

    if n_patients is None:
        patients_to_process = all_patients
        print(f"Processing all {len(patients_to_process)} patients...")
    else:
        patients_to_process = all_patients[:n_patients]
        print(f"Processing first {len(patients_to_process)} patients...")

    # Parse datetime columns
    admissions_df = admissions_df.copy()
    admissions_df["admittime"] = pd.to_datetime(admissions_df["admittime"])

    # Start processing timer
    start_time = time.perf_counter()

    samples = []
    processed_patients = 0
    patients_with_samples = 0

    for subject_id in patients_to_process:
        patient_samples = process_patient_drug_recommendation(
            subject_id,
            admissions_df,
            diagnoses_df,
            procedures_df,
            prescriptions_df,
        )

        if patient_samples:
            samples.extend(patient_samples)
            patients_with_samples += 1

        processed_patients += 1
        if processed_patients % 1000 == 0:
            print(f"Processed {processed_patients} patients, "
                  f"{len(samples)} samples so far...")

    # End processing timer
    processing_time = time.perf_counter() - start_time

    print("\nCompleted processing:")
    print(f"  - Total patients processed: {processed_patients}")
    print(f"  - Patients with valid samples: {patients_with_samples}")
    print(f"  - Total samples created: {len(samples)}")
    print(f"  - Processing time: {processing_time:.2f}s")
    print("=" * 80)

    return samples, processing_time


def load_mimic_data(
    data_root: str = "/srv/local/data/physionet.org/files/mimiciv/2.2/hosp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load MIMIC-IV tables needed for drug recommendation.

    Args:
        data_root: Root directory for MIMIC-IV hosp data

    Returns:
        Tuple of dataframes: (admissions, diagnoses, procedures, prescriptions)
    """
    print("Loading MIMIC-IV data tables...")
    load_start = time.perf_counter()

    admissions_df = pd.read_csv(f"{data_root}/admissions.csv")
    diagnoses_df = pd.read_csv(f"{data_root}/diagnoses_icd.csv.gz")
    procedures_df = pd.read_csv(f"{data_root}/procedures_icd.csv.gz")
    prescriptions_df = pd.read_csv(f"{data_root}/prescriptions.csv.gz", low_memory=False)

    load_time = time.perf_counter() - load_start
    print(f"Data loaded in {load_time:.2f}s")
    print(f"  - Admissions: {len(admissions_df):,}")
    print(f"  - Diagnoses: {len(diagnoses_df):,}")
    print(f"  - Procedures: {len(procedures_df):,}")
    print(f"  - Prescriptions: {len(prescriptions_df):,}")
    print()

    return (
        admissions_df,
        diagnoses_df,
        procedures_df,
        prescriptions_df,
    )


def main():
    """Main function to run the benchmark."""
    global STOP_TRACKING

    parser = argparse.ArgumentParser(
        description="Benchmark MIMIC-IV drug recommendation with pandas"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/srv/local/data/physionet.org/files/mimiciv/2.2/hosp",
        help="Path to MIMIC-IV hosp directory",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=None,
        help="Number of patients to process (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results_pandas_drug_rec.txt",
        help="Output file for results",
    )
    args = parser.parse_args()

    # Start memory tracking thread
    mem_thread = threading.Thread(target=track_mem, daemon=True)
    mem_thread.start()

    # Load data
    total_start = time.perf_counter()
    (
        admissions_df,
        diagnoses_df,
        procedures_df,
        prescriptions_df,
    ) = load_mimic_data(args.data_root)
    load_time = time.perf_counter() - total_start

    # Run benchmark
    samples, processing_time = benchmark_drug_recommendation(
        admissions_df,
        diagnoses_df,
        procedures_df,
        prescriptions_df,
        n_patients=args.n_patients,
    )

    total_time = time.perf_counter() - total_start

    # Stop memory tracking
    STOP_TRACKING = True
    time.sleep(0.2)  # Allow final memory sample

    # Get peak memory
    peak_mem = PEAK_MEM_USAGE

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Data loading time: {load_time:.2f}s")
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total samples: {len(samples)}")
    print(f"Peak memory usage: {format_size(peak_mem)}")
    print("=" * 80)

    # Save results
    with open(args.output, "w") as f:
        f.write("BENCHMARK RESULTS: Pandas Drug Recommendation (MIMIC-IV)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Data root: {args.data_root}\n")
        f.write(f"N patients: {args.n_patients or 'all'}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Data loading time: {load_time:.2f}s\n")
        f.write(f"Processing time: {processing_time:.2f}s\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"Total samples: {len(samples)}\n")
        f.write(f"Peak memory usage: {format_size(peak_mem)}\n")
        f.write("=" * 80 + "\n")

    print(f"\n✓ Results saved to {args.output}")

    # Show example sample
    if samples:
        print("\nExample sample (first sample):")
        first_sample = samples[0]
        print(f"  Patient ID: {first_sample['patient_id']}")
        print(f"  Visit ID: {first_sample['visit_id']}")
        print(f"  Num visits in history: {len(first_sample['conditions'])}")
        print(f"  Conditions (last visit): {first_sample['conditions'][-1][:5]}...")
        print(f"  Procedures (last visit): {first_sample['procedures'][-1][:3]}...")
        print(f"  Target drugs: {first_sample['drugs'][:5]}...")


if __name__ == "__main__":
    main()

