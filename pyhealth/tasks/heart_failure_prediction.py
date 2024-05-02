# Create custom visit time difference calculation and heart failure prediction task
from pyhealth.data import Patient, Visit
import numpy as np


def hf_prediction_mimic3_fn(patient: Patient):
    """Processes a single patient for the heart failure detection task.

    Heart failure prediction aims at predicting whether the patient will be 
    diagnosed with heart failure in the next hospital visit based on the 
    clinical information from current visit (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id,
            visit_id, and other task-specific attributes as key

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> mimic3_base = MIMIC3Dataset(
        ...    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        ...    code_mapping={"ICD9CM": "CCSCM"},
        ... )
        >>> from pyhealth.tasks import hf_prediction_mimic3_fn
        >>> mimic3_sample = mimic3_base.set_task(visit_time_diff_mimic3_fn)
        >>> mimic3_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'visit_diff': [[0.0, 0.0, 0.0, 0.0]] 'label': 0}]
    """
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]

        next_visit: Visit = patient[i + 1]
        hf_label = 0

        if '108' in next_visit.get_code_list(table="DIAGNOSES_ICD"):
            hf_label = 1

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        
        # exclude: visits without condition, procedure, and drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": hf_label,
            }
        )
    # no cohort selection
    return samples
