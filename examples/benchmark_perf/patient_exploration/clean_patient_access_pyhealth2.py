from pyhealth.datasets import MIMIC4Dataset
MIMIC_ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.2/"
PATIENT_ID = "10014729"
if __name__ == "__main__":
    dataset = MIMIC4Dataset(
        ehr_root=MIMIC_ROOT,
        ehr_tables=["patients", "admissions", "diagnoses_icd", "procedures_icd", "labevents"],
    )
    patient = dataset.get_patient(PATIENT_ID)
    events = patient.get_events()