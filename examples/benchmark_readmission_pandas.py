import pandas as pd
import time
from datetime import datetime
import pandas as pd
import time

path_to_massive_lab_events_table = "/srv/local/data/MIMIC-IV/2.0/hosp/labevents.csv"
path_to_massive_admissions_table = "/srv/local/data/MIMIC-IV/2.0/hosp/admissions.csv"
path_to_massive_diagnoses_table = "/srv/local/data/MIMIC-IV/2.0/hosp/diagnoses_icd.csv"
path_to_massive_procedures_table = (
    "/srv/local/data/MIMIC-IV/2.0/hosp/procedures_icd.csv"
)
path_to_massive_prescriptions_table = (
    "/srv/local/data/MIMIC-IV/2.0/hosp/prescriptions.csv"
)
path_to_massive_patients_table = "/srv/local/data/MIMIC-IV/2.0/hosp/patients.csv"

lab_df = pd.read_csv(path_to_massive_lab_events_table)
adm_df = pd.read_csv(path_to_massive_admissions_table)
diag_df = pd.read_csv(path_to_massive_diagnoses_table)
proc_df = pd.read_csv(path_to_massive_procedures_table)
presc_df = pd.read_csv(path_to_massive_prescriptions_table)
pat_df = pd.read_csv(path_to_massive_patients_table)


def benchmark_mimic_processing(adm_df, diag_df, proc_df, presc_df, patients_df, n=1000):
    """
    Benchmark MIMIC-IV data processing for readmission prediction task.

    Args:
        adm_df: Admissions dataframe
        diag_df: Diagnoses dataframe
        proc_df: Procedures dataframe
        presc_df: Prescriptions dataframe
        patients_df: Patients dataframe
        n: Number of patients to process (None = all patients)

    Returns:
        DataFrame with processed samples
    """
    print("Starting MIMIC-IV processing benchmark...")
    start_total = time.perf_counter()

    # Get patients to process
    if n is None:
        patients_to_process = patients_df["subject_id"].tolist()
        print(f"Processing all {len(patients_to_process)} patients...")
    else:
        patients_to_process = patients_df["subject_id"].head(n).tolist()
        print(f"Processing first {len(patients_to_process)} patients...")

    adm_filtered = adm_df[adm_df["subject_id"].isin(patients_to_process)].copy()
    diag_filtered = diag_df[diag_df["subject_id"].isin(patients_to_process)].copy()
    proc_filtered = proc_df[proc_df["subject_id"].isin(patients_to_process)].copy()
    presc_filtered = presc_df[presc_df["subject_id"].isin(patients_to_process)].copy()
    patients_filtered = patients_df[
        patients_df["subject_id"].isin(patients_to_process)
    ].copy()

    adm_filtered["admittime"] = pd.to_datetime(adm_filtered["admittime"])
    adm_filtered["dischtime"] = pd.to_datetime(adm_filtered["dischtime"])

    # Process each patient (mimicking the pyhealth task logic)
    start_processing = time.perf_counter()

    samples = []
    processed_patients = 0

    for subject_id in patients_to_process:
        # Get patient demographics
        patient_demo = patients_filtered[patients_filtered["subject_id"] == subject_id]
        if len(patient_demo) == 0:
            continue

        # Skip if under 18 (mimicking the age filter from pyhealth code)
        anchor_age = patient_demo.iloc[0]["anchor_age"]
        if anchor_age < 18:
            continue

        # Get admissions for this patient, sorted by admit time
        patient_admissions = adm_filtered[
            adm_filtered["subject_id"] == subject_id
        ].sort_values("admittime")

        for i, (_, admission) in enumerate(patient_admissions.iterrows()):
            hadm_id = admission["hadm_id"]
            admit_time = admission["admittime"]
            discharge_time = admission["dischtime"]

            if pd.isna(discharge_time):
                continue

            # Calculate admission duration (skip if <= 12 hours)
            duration_hours = (discharge_time - admit_time).total_seconds() / 3600
            if duration_hours <= 12:
                continue

            # Check for readmission within 30 days
            readmission = 0
            if i < len(patient_admissions) - 1:
                next_admission = patient_admissions.iloc[i + 1]
                next_admit_time = next_admission["admittime"]
                time_diff_hours = (
                    next_admit_time - discharge_time
                ).total_seconds() / 3600

                if time_diff_hours <= 3:  # Skip if too close
                    continue

                readmission = 1 if time_diff_hours < 30 * 24 else 0

            # Get events for this admission
            admission_diagnoses = diag_filtered[diag_filtered["hadm_id"] == hadm_id]
            admission_procedures = proc_filtered[proc_filtered["hadm_id"] == hadm_id]
            admission_prescriptions = presc_filtered[
                presc_filtered["hadm_id"] == hadm_id
            ]

            # Create condition codes (mimicking the pyhealth format)
            conditions = []
            if len(admission_diagnoses) > 0:
                conditions = [
                    f"{row['icd_version']}_{row['icd_code']}"
                    for _, row in admission_diagnoses.iterrows()
                    if pd.notna(row["icd_code"])
                ]

            # Create procedure codes
            procedures = []
            if len(admission_procedures) > 0:
                procedures = [
                    f"{row['icd_version']}_{row['icd_code']}"
                    for _, row in admission_procedures.iterrows()
                    if pd.notna(row["icd_code"])
                ]

            # Create drug codes
            drugs = []
            if len(admission_prescriptions) > 0:
                drugs = [
                    f"{row['drug']}"
                    for _, row in admission_prescriptions.iterrows()
                    if pd.notna(row["drug"])
                ]

            # Skip if any category is empty (mimicking pyhealth logic)
            if len(conditions) == 0 or len(procedures) == 0 or len(drugs) == 0:
                continue

            samples.append(
                {
                    "patient_id": subject_id,
                    "admission_id": hadm_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "readmission": readmission,
                    "num_conditions": len(conditions),
                    "num_procedures": len(procedures),
                    "num_drugs": len(drugs),
                }
            )

        processed_patients += 1
        if processed_patients % 100 == 0:
            print(f"Processed {processed_patients} patients...")

    # Create results dataframe
    results_df = pd.DataFrame(samples)

    return results_df

start_processing = time.perf_counter()
results = benchmark_mimic_processing(adm_df, diag_df, proc_df, presc_df, pat_df, n=None)
end_processing = time.perf_counter()
processing_time = end_processing - start_processing
print(f"Processing time for {len(results)} samples: {processing_time:.2f} seconds")