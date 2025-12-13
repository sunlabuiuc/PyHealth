"""
Benchmark script for MIMIC-IV mortality prediction using pandas
(analogous to PyHealth task).

This benchmark mimics the MortalityPredictionStageNetMIMIC4 task:
1. Creates PATIENT-LEVEL samples (not visit-level)
2. Aggregates all admissions per patient
3. Combines ICD codes (diagnoses + procedures) across all visits
4. Extracts lab events in 10-dimensional vectors per timestamp
5. Calculates time intervals between consecutive admissions

Lab Categories (10 dimensions):
- Sodium, Potassium, Chloride, Bicarbonate, Glucose
- Calcium, Magnesium, Anion Gap, Osmolality, Phosphate
"""

import time
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import psutil


PEAK_MEM_USAGE = 0
SELF_PROC = psutil.Process(os.getpid())


def track_mem():
    """Background thread to track peak memory usage."""
    global PEAK_MEM_USAGE
    while True:
        m = SELF_PROC.memory_info().rss
        if m > PEAK_MEM_USAGE:
            PEAK_MEM_USAGE = m
        time.sleep(0.1)


# Lab item organization by category (matches MortalityPredictionStageNetMIMIC4)
LAB_CATEGORIES = {
    "Sodium": ["50824", "52455", "50983", "52623"],
    "Potassium": ["50822", "52452", "50971", "52610"],
    "Chloride": ["50806", "52434", "50902", "52535"],
    "Bicarbonate": ["50803", "50804"],
    "Glucose": ["50809", "52027", "50931", "52569"],
    "Calcium": ["50808", "51624"],
    "Magnesium": ["50960"],
    "Anion Gap": ["50868", "52500"],
    "Osmolality": ["52031", "50964", "51701"],
    "Phosphate": ["50970"],
}

# Ordered list of category names (defines vector dimension order)
LAB_CATEGORY_NAMES = [
    "Sodium",
    "Potassium",
    "Chloride",
    "Bicarbonate",
    "Glucose",
    "Calcium",
    "Magnesium",
    "Anion Gap",
    "Osmolality",
    "Phosphate",
]

# Flat list of all lab item IDs for filtering
LABITEMS = [item for itemids in LAB_CATEGORIES.values() for item in itemids]


def process_patient_mortality(
    subject_id: int,
    patients_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    diagnoses_df: pd.DataFrame,
    procedures_df: pd.DataFrame,
    labevents_df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Process a single patient for mortality prediction task.

    Creates ONE patient-level sample by aggregating all admissions,
    ICD codes, and lab events.

    Args:
        subject_id: Patient ID
        patients_df: Patient demographics
        admissions_df: Admission records
        diagnoses_df: Diagnosis ICD codes
        procedures_df: Procedure ICD codes
        labevents_df: Lab event measurements

    Returns:
        Dictionary with patient sample or None if patient doesn't qualify
    """
    # Get patient demographics
    patient_demo = patients_df[patients_df["subject_id"] == subject_id]
    if len(patient_demo) == 0:
        return None

    # Skip if under 18
    anchor_age = patient_demo.iloc[0]["anchor_age"]
    if anchor_age < 18:
        return None

    # Get all admissions for this patient, sorted by time
    patient_admissions = admissions_df[
        admissions_df["subject_id"] == subject_id
    ].sort_values("admittime")

    if len(patient_admissions) < 1:
        return None

    # Initialize aggregated data structures
    all_icd_codes = []  # Nested list: [[visit1_codes], [visit2_codes], ...]
    all_icd_times = []  # Time from previous admission per visit
    all_lab_values = []  # List of 10D vectors
    all_lab_times = []  # Time from admission start per measurement

    previous_admission_time = None
    final_mortality = 0

    # Process each admission
    for _, admission in patient_admissions.iterrows():
        hadm_id = admission["hadm_id"]
        admit_time = admission["admittime"]
        discharge_time = admission["dischtime"]

        # Skip if invalid timestamps
        if pd.isna(discharge_time) or discharge_time < admit_time:
            continue

        # Calculate time from previous admission (hours)
        if previous_admission_time is None:
            time_from_previous = 0.0
        else:
            time_from_previous = (
                admit_time - previous_admission_time
            ).total_seconds() / 3600.0

        previous_admission_time = admit_time

        # Update mortality label if this admission had mortality
        if int(admission.get("hospital_expire_flag", 0)) == 1:
            final_mortality = 1

        # Get diagnosis codes for this admission
        visit_diagnoses = diagnoses_df[diagnoses_df["hadm_id"] == hadm_id]
        diagnoses_codes = visit_diagnoses["icd_code"].dropna().tolist()

        # Get procedure codes for this admission
        visit_procedures = procedures_df[procedures_df["hadm_id"] == hadm_id]
        procedures_codes = visit_procedures["icd_code"].dropna().tolist()

        # Combine diagnoses and procedures
        visit_icd_codes = diagnoses_codes + procedures_codes

        if visit_icd_codes:
            all_icd_codes.append(visit_icd_codes)
            all_icd_times.append(time_from_previous)

        # Get lab events for this admission
        admission_labs = labevents_df[
            (labevents_df["subject_id"] == subject_id)
            & (labevents_df["hadm_id"] == hadm_id)
        ]

        # Filter to relevant lab items
        admission_labs = admission_labs[
            admission_labs["itemid"].astype(str).isin(LABITEMS)
        ]

        if len(admission_labs) > 0:
            # Parse storetime
            admission_labs = admission_labs.copy()
            admission_labs["storetime"] = pd.to_datetime(admission_labs["storetime"])

            # Filter to valid times (before discharge)
            admission_labs = admission_labs[
                admission_labs["storetime"] <= discharge_time
            ]

            if len(admission_labs) > 0:
                # Group by timestamp and create 10D vectors
                unique_timestamps = sorted(admission_labs["storetime"].unique())

                for lab_ts in unique_timestamps:
                    # Get all labs at this timestamp
                    ts_labs = admission_labs[admission_labs["storetime"] == lab_ts]

                    # Create 10-dimensional vector
                    lab_vector = []
                    for category_name in LAB_CATEGORY_NAMES:
                        category_itemids = LAB_CATEGORIES[category_name]

                        # Find first matching value for this category
                        category_value = None
                        for itemid in category_itemids:
                            matching = ts_labs[ts_labs["itemid"].astype(str) == itemid]
                            if len(matching) > 0:
                                category_value = matching.iloc[0]["valuenum"]
                                break

                        lab_vector.append(category_value)

                    # Calculate time from admission start (hours)
                    time_from_admission = (lab_ts - admit_time).total_seconds() / 3600.0

                    all_lab_values.append(lab_vector)
                    all_lab_times.append(time_from_admission)

    # Skip if no lab events (required for this task)
    if len(all_lab_values) == 0:
        return None

    # Skip if no ICD codes
    if len(all_icd_codes) == 0:
        return None

    # Create patient-level sample
    sample = {
        "patient_id": subject_id,
        "icd_codes": (all_icd_times, all_icd_codes),
        "labs": (all_lab_times, all_lab_values),
        "mortality": final_mortality,
        "num_visits": len(all_icd_codes),
        "num_lab_measurements": len(all_lab_values),
    }

    return sample


def benchmark_mortality_prediction(
    patients_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    diagnoses_df: pd.DataFrame,
    procedures_df: pd.DataFrame,
    labevents_df: pd.DataFrame,
    n_patients: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Benchmark MIMIC-IV mortality prediction processing.

    Args:
        patients_df: Patient demographics
        admissions_df: Admissions dataframe
        diagnoses_df: Diagnoses dataframe
        procedures_df: Procedures dataframe
        labevents_df: Lab events dataframe
        n_patients: Number of patients to process (None = all patients)

    Returns:
        Tuple of (list of samples, processing time in seconds)
    """
    print("=" * 80)
    print("BENCHMARK: Pandas Mortality Prediction (StageNet format)")
    print("=" * 80)

    # Get patients to process
    if n_patients is None:
        patients_to_process = patients_df["subject_id"].tolist()
        print(f"Processing all {len(patients_to_process)} patients...")
    else:
        patients_to_process = patients_df["subject_id"].head(n_patients).tolist()
        print(f"Processing first {len(patients_to_process)} patients...")

    # Parse datetime columns
    admissions_df = admissions_df.copy()
    admissions_df["admittime"] = pd.to_datetime(admissions_df["admittime"])
    admissions_df["dischtime"] = pd.to_datetime(admissions_df["dischtime"])

    # Start processing timer
    start_time = time.perf_counter()

    samples = []
    processed_patients = 0

    for subject_id in patients_to_process:
        sample = process_patient_mortality(
            subject_id,
            patients_df,
            admissions_df,
            diagnoses_df,
            procedures_df,
            labevents_df,
        )

        if sample is not None:
            samples.append(sample)

        processed_patients += 1
        if processed_patients % 100 == 0:
            print(f"Processed {processed_patients} patients...")

    # End processing timer
    processing_time = time.perf_counter() - start_time

    print("\nCompleted processing:")
    print(f"  - Total patients processed: {processed_patients}")
    print(f"  - Valid samples created: {len(samples)}")
    print(f"  - Processing time: {processing_time:.2f}s")
    print("=" * 80)

    return samples, processing_time


def load_mimic_data(data_root: str = "/srv/local/data/MIMIC-IV/2.0/hosp"):
    """Load MIMIC-IV tables needed for mortality prediction.

    Args:
        data_root: Root directory for MIMIC-IV data

    Returns:
        Tuple of dataframes: (patients, admissions, diagnoses,
                              procedures, labevents)
    """
    print("Loading MIMIC-IV data tables...")
    load_start = time.perf_counter()

    patients_df = pd.read_csv(f"{data_root}/patients.csv")
    admissions_df = pd.read_csv(f"{data_root}/admissions.csv")
    diagnoses_df = pd.read_csv(f"{data_root}/diagnoses_icd.csv")
    procedures_df = pd.read_csv(f"{data_root}/procedures_icd.csv")
    labevents_df = pd.read_csv(f"{data_root}/labevents.csv")

    load_time = time.perf_counter() - load_start
    print(f"Data loaded in {load_time:.2f}s")
    print(f"  - Patients: {len(patients_df):,}")
    print(f"  - Admissions: {len(admissions_df):,}")
    print(f"  - Diagnoses: {len(diagnoses_df):,}")
    print(f"  - Procedures: {len(procedures_df):,}")
    print(f"  - Lab events: {len(labevents_df):,}")
    print()

    return (
        patients_df,
        admissions_df,
        diagnoses_df,
        procedures_df,
        labevents_df,
    )


def main():
    """Main function to run the benchmark."""
    # Start memory tracking thread
    mem_thread = threading.Thread(target=track_mem, daemon=True)
    mem_thread.start()

    # Load data
    data_root = "/srv/local/data/MIMIC-IV/2.0/hosp"
    (
        patients_df,
        admissions_df,
        diagnoses_df,
        procedures_df,
        labevents_df,
    ) = load_mimic_data(data_root)

    # Run benchmark (process all patients)
    samples, processing_time = benchmark_mortality_prediction(
        patients_df,
        admissions_df,
        diagnoses_df,
        procedures_df,
        labevents_df,
        n_patients=None,  # Change to a number to limit patients
    )

    # Get peak memory
    peak_mem = PEAK_MEM_USAGE

    # Helper function for formatting size
    def format_size(size_bytes):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    # Save results
    results_file = "benchmark_results_pandas.txt"
    with open(results_file, "w") as f:
        f.write("BENCHMARK RESULTS: Pandas Mortality Prediction\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total samples: {len(samples)}\n")
        f.write(f"Processing time: {processing_time:.2f}s\n")
        f.write(f"Peak memory usage: {format_size(peak_mem)}\n")
        f.write("=" * 80 + "\n")

    print(f"\n✓ Results saved to {results_file}")
    print(f"Peak memory usage: {format_size(peak_mem)}")

    # Optional: Save samples for inspection
    if samples:
        print("\nExample sample (first patient):")
        first_sample = samples[0]
        print(f"  Patient ID: {first_sample['patient_id']}")
        print(f"  Mortality: {first_sample['mortality']}")
        print(f"  Number of visits: {first_sample['num_visits']}")
        print(
            f"  Number of lab measurements: " f"{first_sample['num_lab_measurements']}"
        )


if __name__ == "__main__":
    main()
