from pyhealth.datasets import MIMIC4Dataset

def categorize_los(d): return 0 if d<1 else (d if d<=7 else (8 if d<=14 else 9))

def length_of_stay_prediction_mimic4_fn(patient):
    S=[]
    for v in patient:
        c=v.get_code_list(table="diagnoses_icd"); r=v.get_code_list(table="procedures_icd"); d=v.get_code_list(table="prescriptions")
        if not(c and r and d): continue
        S.append({"visit_id":v.visit_id,"patient_id":patient.patient_id,"conditions":[c],"procedures":[r],"drugs":[d],"label":categorize_los((v.discharge_time-v.encounter_time).days)})
    return S

base_dataset=MIMIC4Dataset(root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",tables=["diagnoses_icd","procedures_icd","prescriptions"],dev=False,code_mapping={"ICD10PROC":"CCSPROC","NDC":"ATC"},refresh_cache=True)
sample_dataset=base_dataset.set_task(task_fn=length_of_stay_prediction_mimic4_fn)
