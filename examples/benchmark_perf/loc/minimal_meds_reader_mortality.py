import os, shutil, subprocess
from typing import Iterator
import meds_reader
LAB_ITEM_IDS = {"50824", "52455", "50983", "52623", "50822", "52452", "50971", "52610", "50806", "52434", "50902", "52535",
                "50803", "50804", "50809", "52027", "50931", "52569", "50808", "51624", "50960", "50868", "52500", "52031", "50964", "51701", "50970"}
def run_meds_etl_mimic(src_mimic: str, output_dir: str, num_shards: int = 100, num_proc: int = 1):
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    subprocess.run(["meds_etl_mimic", src_mimic, output_dir, "--num_shards", str(num_shards), "--num_proc", str(num_proc), "--backend", "polars"], check=True)
def run_meds_reader_convert(input_dir: str, output_dir: str, num_threads: int = 10):
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    subprocess.run(["meds_reader_convert", input_dir, output_dir, "--num_threads", str(num_threads)], check=True)
def get_mortality_samples(subjects: Iterator[meds_reader.Subject]):
    samples = []
    for subject in subjects:
        admissions, death_time = {}, None
        for event in subject.events:
            if event.code == "meds/death": death_time = event.time; break
        for event in subject.events:
            if event.code.startswith("MIMIC_IV_Admission/"):
                visit_id, end_time = getattr(event, 'visit_id', None), getattr(event, 'end', None)
                if visit_id is not None and event.time is not None:
                    admissions[visit_id] = {'time': event.time, 'conditions': set(), 'procedures': set(), 'labs': set(),
                                            'discharge_status': 1 if death_time and end_time and death_time <= end_time else 0}
        for event in subject.events:
            visit_id = getattr(event, 'visit_id', None)
            if visit_id and visit_id in admissions:
                if event.code.startswith("ICD"): (admissions[visit_id]['conditions'] if "CM" in event.code else admissions[visit_id]['procedures']).add(event.code)
                elif event.code.startswith("MIMIC_IV_LABITEM/") and event.code.split("/")[-1] in LAB_ITEM_IDS: admissions[visit_id]['labs'].add(event.code)
        sorted_visits = sorted(admissions.items(), key=lambda x: x[1]['time'])
        for i in range(len(sorted_visits) - 1):
            visit_id, cur = sorted_visits[i]
            if cur['conditions'] and cur['labs']:
                samples.append({"visit_id": visit_id, "patient_id": subject.subject_id, "conditions": list(cur['conditions']),
                                "procedures": list(cur['procedures']), "labs": list(cur['labs']), "label": sorted_visits[i+1][1]['discharge_status']})
    return samples
if __name__ == "__main__":
    MIMIC_ROOT, CACHE_DIR = "/srv/local/data/physionet.org/files/mimiciv", "/srv/local/data/REDACTED_USER/meds_reader"
    MEDS_DIR, MEDS_READER_DIR = f"{CACHE_DIR}/mimic4_meds_mortality", f"{CACHE_DIR}/mimic4_meds_reader_mortality"
    run_meds_etl_mimic(MIMIC_ROOT, MEDS_DIR); run_meds_reader_convert(MEDS_DIR, MEDS_READER_DIR)
    samples = []
    with meds_reader.SubjectDatabase(MEDS_READER_DIR, num_threads=4) as db:
        for s in db.map(get_mortality_samples): samples.extend(s)
    print(f"Samples: {len(samples)}")
