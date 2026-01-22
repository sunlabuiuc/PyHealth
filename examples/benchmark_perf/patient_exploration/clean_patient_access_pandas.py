import pandas as pd
MIMIC_ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.2/hosp"
PATIENT_ID = 10014729
if __name__ == "__main__":
    patients = pd.read_csv(f"{MIMIC_ROOT}/patients.csv.gz")
    admissions = pd.read_csv(f"{MIMIC_ROOT}/admissions.csv.gz")
    diagnoses = pd.read_csv(f"{MIMIC_ROOT}/diagnoses_icd.csv.gz")
    procedures = pd.read_csv(f"{MIMIC_ROOT}/procedures_icd.csv.gz")
    labevents = pd.read_csv(f"{MIMIC_ROOT}/labevents.csv.gz", low_memory=False)
    patient_info = patients[patients["subject_id"] == PATIENT_ID]
    patient_hadm_ids = admissions[admissions["subject_id"] == PATIENT_ID]["hadm_id"]
    events = pd.concat([
        diagnoses[diagnoses["hadm_id"].isin(patient_hadm_ids)],
        procedures[procedures["hadm_id"].isin(patient_hadm_ids)],
        labevents[labevents["hadm_id"].isin(patient_hadm_ids)],
    ])