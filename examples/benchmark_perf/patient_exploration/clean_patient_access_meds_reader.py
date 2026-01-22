import os
import shutil
import subprocess
import datetime
import collections
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyhealth.datasets import MIMIC4Dataset
import meds_reader
MIMIC_ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.0/hosp"
MEDS_DIR = "/tmp/pyhealth_meds"
MEDS_READER_DIR = "/tmp/pyhealth_meds_reader"
PATIENT_ID = 10014729
if __name__ == "__main__":
    dataset = MIMIC4Dataset(
        root=MIMIC_ROOT,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
        refresh_cache=True,
    )
    results = collections.defaultdict(list)
    for patient_id, patient in dataset.patients.items():
        subject_id = int(patient_id)
        birth_obj = {"subject_id": subject_id, "code": "Birth", "time": patient.birth_datetime}
        birth_obj["gender"] = patient.gender
        birth_obj["ethnicity"] = patient.ethnicity
        for k, v in patient.attr_dict.items():
            if v != v:  # Skip NaN
                continue
            birth_obj[k] = v
        results[subject_id].append(birth_obj)
        if patient.death_datetime is not None:
            results[subject_id].append({"subject_id": subject_id, "code": "Death", "time": patient.death_datetime})
        for visit_id, visit in patient.visits.items():
            vid = int(visit_id)
            visit_event = {
                "subject_id": subject_id,
                "code": "Visit",
                "time": visit.encounter_time,
                "visit_id": vid,
                "discharge_time": visit.discharge_time,
                "discharge_status": visit.discharge_status,
            }
            for k, v in visit.attr_dict.items():
                if v != v:
                    continue
                visit_event[k] = v
            results[subject_id].append(visit_event)
            for table in visit.available_tables:
                for event in visit.get_event_list(table):
                    event_obj = {
                        "subject_id": subject_id,
                        "visit_id": vid,
                        "code": f"{event.vocabulary}/{event.code}",
                        "time": event.timestamp or visit.discharge_time,
                    }
                    for k, v in event.attr_dict.items():
                        if v != v:
                            continue
                        event_obj[k] = v
                    results[subject_id].append(event_obj)
        results[subject_id].sort(key=lambda a: a["time"])
    attr_map = {str: pa.string(), int: pa.int64(), np.int64: pa.int64(), float: pa.float64(), np.float64: pa.float64(), datetime.datetime: pa.timestamp("us")}
    attr_schema = set()
    for subject_values in results.values():
        for row in subject_values:
            for k, v in row.items():
                if k not in {"subject_id", "time", "numeric_value"} and type(v) in attr_map:
                    attr_schema.add((k, attr_map[type(v)]))
    schema = pa.schema([("subject_id", pa.int64()), ("time", pa.timestamp("us"))] + sorted(list(attr_schema)))
    shutil.rmtree(MEDS_DIR, ignore_errors=True)
    os.makedirs(f"{MEDS_DIR}/data")
    os.makedirs(f"{MEDS_DIR}/metadata")
    all_subjects = list(results.keys())
    for i, subject_ids in enumerate(np.array_split(all_subjects, 100)):
        rows = [v for sid in subject_ids for v in results[sid]]
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), f"{MEDS_DIR}/data/{i}.parquet")
    shutil.rmtree(MEDS_READER_DIR, ignore_errors=True)
    subprocess.run(["meds_reader_convert", MEDS_DIR, MEDS_READER_DIR], check=True)
    with meds_reader.SubjectDatabase(MEDS_READER_DIR) as db:
        patient = db[PATIENT_ID]
        events = list(patient.events)
