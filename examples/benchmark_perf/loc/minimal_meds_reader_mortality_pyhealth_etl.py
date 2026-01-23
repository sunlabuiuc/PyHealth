import collections, datetime, os, shutil, subprocess
from typing import Iterator, List
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
import meds_reader
from pyhealth.datasets import MIMIC4Dataset
LAB_ITEM_IDS = {"50824", "52455", "50983", "52623", "50822", "52452", "50971", "52610", "50806", "52434", "50902", "52535",
                "50803", "50804", "50809", "52027", "50931", "52569", "50808", "51624", "50960", "50868", "52500", "52031", "50964", "51701", "50970"}
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
def get_mortality_samples(subjects: Iterator[meds_reader.Subject]):
    samples = []
    for subject in subjects:
        admissions, death_time = {}, None
        for event in subject.events:
            if event.code == "meds/death": death_time = event.time; break
        for event in subject.events:
            if event.code.startswith("MIMIC_IV_Admission/"):
                visit_id, end_time = getattr(event, 'visit_id', None), getattr(event, 'end', None)
                if visit_id and event.time:
                    admissions[visit_id] = {'time': event.time, 'conditions': set(), 'procedures': set(), 'labs': set(),
                                            'discharge_status': 1 if death_time and end_time and death_time <= end_time else 0}
        for event in subject.events:
            visit_id = getattr(event, 'visit_id', None)
            if visit_id and visit_id in admissions:
                if event.code.startswith("ICD"): (admissions[visit_id]['conditions'] if "CM" in event.code else admissions[visit_id]['procedures']).add(event.code)
                elif "LABITEM" in event.code and event.code.split("/")[-1] in LAB_ITEM_IDS: admissions[visit_id]['labs'].add(event.code)
        sorted_visits = sorted(admissions.items(), key=lambda x: x[1]['time'])
        for i in range(len(sorted_visits) - 1):
            vid, cur = sorted_visits[i]
            if cur['conditions'] and cur['labs']:
                samples.append({"visit_id": vid, "patient_id": subject.subject_id, "conditions": list(cur['conditions']),
                                "procedures": list(cur['procedures']), "labs": list(cur['labs']), "label": sorted_visits[i+1][1]['discharge_status']})
    return samples
if __name__ == "__main__":
    PYHEALTH_ROOT, CACHE_DIR = "/srv/local/data/physionet.org/files/mimiciv/2.0/hosp", "/srv/local/data/REDACTED_USER/meds_reader"
    MEDS_DIR, MEDS_READER_DIR = f"{CACHE_DIR}/mimic4_meds_mortality_pyhealth", f"{CACHE_DIR}/mimic4_meds_reader_mortality_pyhealth"
    pyhealth_to_meds(PYHEALTH_ROOT, MEDS_DIR, ["diagnoses_icd", "procedures_icd", "labevents"]); run_meds_reader_convert(MEDS_DIR, MEDS_READER_DIR)
    samples = []
    with meds_reader.SubjectDatabase(MEDS_READER_DIR, num_threads=4) as db:
        for s in db.map(get_mortality_samples): samples.extend(s)
    print(f"Samples: {len(samples)}")
