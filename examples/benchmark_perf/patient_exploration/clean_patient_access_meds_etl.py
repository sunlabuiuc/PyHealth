import subprocess
import meds_reader
MIMIC_ROOT = "/srv/local/data/physionet.org/files/mimiciv"
MEDS_DIR = "/tmp/mimic4_meds"
MEDS_READER_DIR = "/tmp/mimic4_meds_reader"
PATIENT_ID = 10014729
if __name__ == "__main__":
    subprocess.run(["meds_etl_mimic", MIMIC_ROOT, MEDS_DIR], check=True)
    subprocess.run(["meds_reader_convert", MEDS_DIR, MEDS_READER_DIR], check=True)
    with meds_reader.SubjectDatabase(MEDS_READER_DIR) as db:
        patient = db[PATIENT_ID]
        events = list(patient.events)