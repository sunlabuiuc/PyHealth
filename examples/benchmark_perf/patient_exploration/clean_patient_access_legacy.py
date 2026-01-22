from pyhealth.datasets import MIMIC4Dataset
MIMIC_ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.0/hosp"
PATIENT_ID = "10014729"
if __name__ == "__main__":
    dataset = MIMIC4Dataset(
        root=MIMIC_ROOT,
        tables=["diagnoses_icd", "procedures_icd", "labevents"],
        refresh_cache=True,
    )
    patient = dataset.patients[PATIENT_ID]
    events = []
    for visit in patient.visits.values():
        for table in visit.available_tables:
            events.extend(visit.get_event_list(table))
