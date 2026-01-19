from typing import List
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.data import Patient, Visit
LAB_ITEM_IDS = {"50824", "52455", "50983", "52623", "50822", "52452", "50971", "52610",
                "50806", "52434", "50902", "52535", "50803", "50804", "50809", "52027",
                "50931", "52569", "50808", "51624", "50960", "50868", "52500", "52031",
                "50964", "51701", "50970"}
def mortality_task_fn(patient: Patient) -> List[dict]:
    samples = []
    for i in range(len(patient) - 1):
        visit, next_visit = patient[i], patient[i + 1]
        mortality_label = int(next_visit.discharge_status) if next_visit.discharge_status in [0, 1] else 0
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        labs = list(dict.fromkeys([e.code for e in visit.get_event_list(table="labevents") if e.code in LAB_ITEM_IDS]))
        if conditions and labs:
            samples.append({"visit_id": visit.visit_id, "patient_id": patient.patient_id,
                            "conditions": [conditions], "procedures": [procedures] if procedures else [[]],
                            "labs": [labs], "label": mortality_label})
    return samples
base_dataset = MIMIC4Dataset(root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
                             tables=["diagnoses_icd", "procedures_icd", "labevents"], dev=False, refresh_cache=True)
sample_dataset = base_dataset.set_task(task_fn=mortality_task_fn)
print(f"Samples: {len(sample_dataset.samples)}")
