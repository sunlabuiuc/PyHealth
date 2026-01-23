import os, shutil, subprocess
from typing import Iterator
import meds_reader
def run_meds_etl_mimic(src_mimic: str, output_dir: str, num_shards: int = 100, num_proc: int = 1):
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    subprocess.run(["meds_etl_mimic", src_mimic, output_dir, "--num_shards", str(num_shards), "--num_proc", str(num_proc), "--backend", "polars"], check=True)
def run_meds_reader_convert(input_dir: str, output_dir: str, num_threads: int = 10):
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    subprocess.run(["meds_reader_convert", input_dir, output_dir, "--num_threads", str(num_threads)], check=True)
def get_los_samples(subjects: Iterator[meds_reader.Subject]):
    samples = []
    for subject in subjects:
        admission_data = {}
        for event in subject.events:
            if event.code.startswith("MIMIC_IV_Admission/"):
                visit_id, end_time = getattr(event, 'visit_id', None), getattr(event, 'end', None)
                if visit_id and event.time and end_time:
                    los_days = (end_time - event.time).days
                    admission_data[visit_id] = {'label': 0 if los_days < 1 else (los_days if los_days <= 7 else (8 if los_days <= 14 else 9)),
                                                'conditions': set(), 'procedures': set(), 'drugs': set()}
        for event in subject.events:
            visit_id = getattr(event, 'visit_id', None)
            if visit_id and visit_id in admission_data:
                if event.code.startswith("ICD"): (admission_data[visit_id]['conditions'] if "CM" in event.code else admission_data[visit_id]['procedures']).add(event.code)
                elif event.code.startswith("NDC/") or event.code.startswith("MIMIC_IV_Drug/"): admission_data[visit_id]['drugs'].add(event.code)
        for visit_id, data in admission_data.items():
            if data['conditions'] and data['procedures'] and data['drugs']:
                samples.append({"visit_id": visit_id, "patient_id": subject.subject_id, "conditions": list(data['conditions']),
                                "procedures": list(data['procedures']), "drugs": list(data['drugs']), "label": data['label']})
    return samples
if __name__ == "__main__":
    MIMIC_ROOT, CACHE_DIR = "/srv/local/data/physionet.org/files/mimiciv", "/srv/local/data/REDACTED_USER/meds_reader"
    MEDS_DIR, MEDS_READER_DIR = f"{CACHE_DIR}/mimic4_meds_los", f"{CACHE_DIR}/mimic4_meds_reader_los"
    run_meds_etl_mimic(MIMIC_ROOT, MEDS_DIR); run_meds_reader_convert(MEDS_DIR, MEDS_READER_DIR)
    samples = []
    with meds_reader.SubjectDatabase(MEDS_READER_DIR, num_threads=4) as db:
        for s in db.map(get_los_samples): samples.extend(s)
    print(f"Samples: {len(samples)}")
