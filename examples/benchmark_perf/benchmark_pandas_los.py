"""
Benchmark script for MIMIC-IV length of stay prediction using pandas
(analogous to PyHealth LengthOfStayPredictionMIMIC4 task).

This benchmark mimics the LengthOfStayPredictionMIMIC4 task:
1. Creates visit-level samples for each admission
2. For each visit, extracts conditions, procedures, and drugs
3. Calculates length of stay from admission to discharge
4. Categorizes LOS into 10 categories (0-9)

Length of Stay Categories:
- 0: < 1 day
- 1-7: 1-7 days (each day is its own category)
- 8: 8-14 days (over one week, less than two)
- 9: > 14 days (over two weeks)
"""

import argparse
import time
import os
import threading
from datetime import datetime
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


def categorize_los(days: int) -> int:
    """Categorizes length of stay into 10 categories.

    One for ICU stays shorter than a day, seven day-long categories for each day of
    the first week, one for stays of over one week but less than two,
    and one for stays of over two weeks.

    Args:
        days: int, length of stay in days

    Returns:
        category: int, category of length of stay (0-9)
    """
    # ICU stays shorter than a day
    if days < 1:
        return 0
    # each day of the first week
    elif 1 <= days <= 7:
        return days
    # stays of over one week but less than two
    elif 7 < days <= 14:
        return 8
    # stays of over two weeks
    else:
        return 9


def process_patient_length_of_stay(
    subject_id: int,
    admissions_df: pd.DataFrame,
    diagnoses_df: pd.DataFrame,
    procedures_df: pd.DataFrame,
    prescriptions_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Process a single patient for length of stay prediction task.

    Creates visit-level samples with conditions, procedures, drugs, and LOS label.

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

    # Get all admissions for this patient
    patient_admissions = admissions_df[admissions_df["subject_id"] == subject_id]

    if len(patient_admissions) == 0:
        return []

    # Process each admission
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
                drugs.append(str(ndc))

        # Exclude visits without condition, procedure, or drug code
        if len(conditions) == 0 or len(procedures) == 0 or len(drugs) == 0:
            continue

        # Calculate length of stay
        admittime = admission["admittime"]
        dischtime = admission["dischtime"]

        if pd.isna(admittime) or pd.isna(dischtime):
            continue

        # Calculate LOS in days
        los_days = (dischtime - admittime).days
        los_category = categorize_los(los_days)

        samples.append({
            "visit_id": hadm_id,
            "patient_id": subject_id,
            "conditions": conditions,
            "procedures": procedures,
            "drugs": drugs,
            "los": los_category,
            "los_days": los_days,  # Also store raw days for debugging
        })

    return samples


def benchmark_length_of_stay(
    admissions_df: pd.DataFrame,
    diagnoses_df: pd.DataFrame,
    procedures_df: pd.DataFrame,
    prescriptions_df: pd.DataFrame,
    n_patients: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Benchmark MIMIC-IV length of stay processing.

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
    print("BENCHMARK: Pandas Length of Stay Prediction (MIMIC-IV format)")
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
    admissions_df["dischtime"] = pd.to_datetime(admissions_df["dischtime"])

    # Start processing timer
    start_time = time.perf_counter()

    samples = []
    processed_patients = 0
    patients_with_samples = 0

    # Track LOS distribution
    los_distribution = {i: 0 for i in range(10)}

    for subject_id in patients_to_process:
        patient_samples = process_patient_length_of_stay(
            subject_id,
            admissions_df,
            diagnoses_df,
            procedures_df,
            prescriptions_df,
        )

        if patient_samples:
            samples.extend(patient_samples)
            patients_with_samples += 1
            # Update LOS distribution
            for sample in patient_samples:
                los_distribution[sample["los"]] += 1

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
    print("\nLOS Category Distribution:")
    for cat, count in los_distribution.items():
        pct = (count / len(samples) * 100) if samples else 0
        label = {
            0: "<1 day",
            1: "1 day", 2: "2 days", 3: "3 days", 4: "4 days",
            5: "5 days", 6: "6 days", 7: "7 days",
            8: "8-14 days",
            9: ">14 days",
        }.get(cat, str(cat))
        print(f"    Category {cat} ({label}): {count} ({pct:.1f}%)")
    print("=" * 80)

    return samples, processing_time


def load_mimic_data(
    data_root: str = "/srv/local/data/physionet.org/files/mimiciv/2.2/hosp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load MIMIC-IV tables needed for length of stay prediction.

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
        description="Benchmark MIMIC-IV length of stay prediction with pandas"
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
        default="benchmark_results_pandas_los.txt",
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
    samples, processing_time = benchmark_length_of_stay(
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
        f.write("BENCHMARK RESULTS: Pandas Length of Stay Prediction (MIMIC-IV)\n")
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
        print(f"  Conditions: {first_sample['conditions'][:5]}..." if len(first_sample['conditions']) > 5 else f"  Conditions: {first_sample['conditions']}")
        print(f"  Procedures: {first_sample['procedures'][:3]}..." if len(first_sample['procedures']) > 3 else f"  Procedures: {first_sample['procedures']}")
        print(f"  Drugs: {first_sample['drugs'][:5]}..." if len(first_sample['drugs']) > 5 else f"  Drugs: {first_sample['drugs']}")
        print(f"  LOS (days): {first_sample['los_days']}")
        print(f"  LOS (category): {first_sample['los']}")


if __name__ == "__main__":
    main()

