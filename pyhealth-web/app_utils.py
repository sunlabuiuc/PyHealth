from datetime import datetime, timedelta, timezone
import os
import time
import sys
sys.path.append("/home/chaoqiy2/github/PyHealth-OMOP")
from pyhealth.api import run_healthcare_ml_job

dataset_mapping = {
    'mimic_iii': 'MIMIC-III',
    'eicu': 'eICU',
    'mimic_iv': 'MIMIC-IV',
    'iqvia': 'IQVIA',
}

task_mapping = {
    'drug_rec': 'Drug Recommendation',
    'hf_predict': 'Heart Failure Prediction',
    'mortality': 'Mortality Prediction',
    'readmission': 'Readmission Prediction',
}

model_mapping = {
    "safedrug": "(SafeDrug) Molecule Structure Info. Enhanced Model",
    "gamenet": "(GAMENet) Memory based Sequential Model",
    "micron": "(MICRON) Drug Replacement Prediction Model",
    "retain": "(RETAIN) General Healthcare Predictive Model",
}

def create_new_record(Job, db, config):
    # trigger_time = datetime.now(timezone(timedelta(hours=0), 'EST'))
    run_id = Job.query.count()
    trigger_time = datetime.now()
    dataset = dataset_mapping[config['dataset']]
    task_name = task_mapping[config['task']]
    model = model_mapping[config['model']]
    # TODO: update this
    run_stats = 'Pending'
    example_job = Job(
        run_id = run_id,
        trigger_time = trigger_time,
        dataset = dataset,
        task_name = task_name,
        model = model,
        run_stats = run_stats,
        downloads = "Wait",
    )
    db.session.add(example_job)
    db.session.commit()
    trigger_ML_task(Job, db, run_id, trigger_time, config)

def trigger_ML_task(Job, db, run_id, trigger_time, config):
    output_file = run_healthcare_ml_job(run_id, trigger_time, config)
    Job.query.filter_by(run_id=run_id).update(dict(run_stats='SUCCESS', downloads=output_file))
    db.session.commit()