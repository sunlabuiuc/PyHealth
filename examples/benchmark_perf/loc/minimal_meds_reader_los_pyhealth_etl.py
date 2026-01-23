import collections, datetime, os, shutil, subprocess
from typing import Iterator, List
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
import meds_reader
from pyhealth.datasets import MIMIC4Dataset
def pyhealth_to_meds(pyhealth_root: str, output_dir: str, tables: List[str], num_shards: int = 100):
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    dataset = MIMIC4Dataset(root=pyhealth_root, tables=tables, dev=False, refresh_cache=True)
    results = collections.defaultdict(list)
    for patient_id, patient in dataset.patients.items():
        sid = int(patient_id)
        if patient.birth_datetime: results[sid].append({'subject_id': sid, 'code': 'meds/birth', 'time': patient.birth_datetime})
        if patient.death_datetime: results[sid].append({'subject_id': sid, 'code': 'meds/death', 'time': patient.death_datetime})
        for visit_id, visit in patient.visits.items():
            vid = int(visit_id)
            ve = {'subject_id': sid, 'code': 'MIMIC_IV_Admission/unknown', 'time': visit.encounter_time, 'visit_id': vid}
            if visit.discharge_time: ve['end'] = visit.discharge_time
            results[sid].append(ve)
            for table in visit.available_tables:
                for event in visit.get_event_list(table):
                    results[sid].append({'subject_id': sid, 'visit_id': vid, 'code': f'{event.vocabulary}/{event.code}', 'time': event.timestamp or visit.discharge_time})
        results[sid].sort(key=lambda a: a['time'] if a['time'] else datetime.datetime.min)
    os.makedirs(f"{output_dir}/metadata", exist_ok=True); os.makedirs(f"{output_dir}/data", exist_ok=True)
    attr_map = {str: pa.string(), int: pa.int64(), np.int64: pa.int64(), float: pa.float64(), datetime.datetime: pa.timestamp('us')}
    attr_schema = {k: attr_map.get(type(v), pa.string()) for subj in results.values() for row in subj for k, v in row.items() if k not in {'subject_id', 'time'} and v is not None}
    schema = pa.schema([('subject_id', pa.int64()), ('time', pa.timestamp('us'))] + [(k, v) for k, v in sorted(attr_schema.items())])
    for i, sids in enumerate(np.array_split(list(results.keys()), num_shards)):
        rows = [v for sid in sids for v in results[sid]]
        if rows: pq.write_table(pa.Table.from_pylist(rows, schema=schema), f"{output_dir}/data/{i}.parquet")
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
    PYHEALTH_ROOT, CACHE_DIR = "/srv/local/data/physionet.org/files/mimiciv/2.0/hosp", "/srv/local/data/REDACTED_USER/meds_reader"
    MEDS_DIR, MEDS_READER_DIR = f"{CACHE_DIR}/mimic4_meds_los_pyhealth", f"{CACHE_DIR}/mimic4_meds_reader_los_pyhealth"
    pyhealth_to_meds(PYHEALTH_ROOT, MEDS_DIR, ["diagnoses_icd", "procedures_icd", "prescriptions"]); run_meds_reader_convert(MEDS_DIR, MEDS_READER_DIR)
    samples = []
    with meds_reader.SubjectDatabase(MEDS_READER_DIR, num_threads=4) as db:
        for s in db.map(get_los_samples): samples.extend(s)
    print(f"Samples: {len(samples)}")
