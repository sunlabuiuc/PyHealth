import os, shutil, subprocess
from typing import Iterator
import meds_reader
def run_meds_etl_mimic(src_mimic: str, output_dir: str, num_shards: int = 100, num_proc: int = 1):
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    subprocess.run(["meds_etl_mimic", src_mimic, output_dir, "--num_shards", str(num_shards), "--num_proc", str(num_proc), "--backend", "polars"], check=True)
def run_meds_reader_convert(input_dir: str, output_dir: str, num_threads: int = 10):
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    subprocess.run(["meds_reader_convert", input_dir, output_dir, "--num_threads", str(num_threads)], check=True)
def get_drug_rec_samples(subjects: Iterator[meds_reader.Subject]):
    samples = []
    for subject in subjects:
        admissions = {}
        for event in subject.events:
            if event.code.startswith("MIMIC_IV_Admission/"):
                visit_id = getattr(event, 'visit_id', None)
                if visit_id and event.time: admissions[visit_id] = {'time': event.time, 'conditions': set(), 'procedures': set(), 'drugs': set()}
        for event in subject.events:
            visit_id = getattr(event, 'visit_id', None)
            if visit_id and visit_id in admissions:
                if event.code.startswith("ICD"): (admissions[visit_id]['conditions'] if "CM" in event.code else admissions[visit_id]['procedures']).add(event.code)
                elif event.code.startswith("NDC/") or event.code.startswith("MIMIC_IV_Drug/"): admissions[visit_id]['drugs'].add(event.code)
        sorted_visits = sorted(admissions.items(), key=lambda x: x[1]['time'])
        valid = [(vid, d) for vid, d in sorted_visits if d['conditions'] and d['procedures'] and d['drugs']]
        if len(valid) < 2: continue
        for i, (vid, data) in enumerate(valid):
            samples.append({"visit_id": vid, "patient_id": subject.subject_id,
                            "conditions": [list(valid[j][1]['conditions']) for j in range(i+1)],
                            "procedures": [list(valid[j][1]['procedures']) for j in range(i+1)],
                            "drugs_hist": [list(valid[j][1]['drugs']) if j < i else [] for j in range(i+1)], "drugs": list(data['drugs'])})
    return samples
if __name__ == "__main__":
    MIMIC_ROOT, CACHE_DIR = "/srv/local/data/physionet.org/files/mimiciv", "/srv/local/data/johnwu3/meds_reader"
    MEDS_DIR, MEDS_READER_DIR = f"{CACHE_DIR}/mimic4_meds_drug_rec", f"{CACHE_DIR}/mimic4_meds_reader_drug_rec"
    run_meds_etl_mimic(MIMIC_ROOT, MEDS_DIR); run_meds_reader_convert(MEDS_DIR, MEDS_READER_DIR)
    samples = []
    with meds_reader.SubjectDatabase(MEDS_READER_DIR, num_threads=4) as db:
        for s in db.map(get_drug_rec_samples): samples.extend(s)
    print(f"Samples: {len(samples)}")
