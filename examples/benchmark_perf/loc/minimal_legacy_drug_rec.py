from pyhealth.data import Patient,Visit; from pyhealth.datasets import MIMIC4Dataset

def drug_recommendation_mimic4_fn(patient):
    S=[]
    for v in patient:
        c=v.get_code_list(table="diagnoses_icd"); r=v.get_code_list(table="procedures_icd"); d=[x[:4] for x in v.get_code_list(table="prescriptions")]
        if not(c and r and d): continue
        S.append({"visit_id":v.visit_id,"patient_id":patient.patient_id,"conditions":c,"procedures":r,"drugs":d,"drugs_hist":d})
    if len(S)<2: return []
    S[0].update({"conditions":[S[0]["conditions"]],"procedures":[S[0]["procedures"]],"drugs_hist":[S[0]["drugs_hist"]]})
    for i in range(1,len(S)): S[i]["conditions"]=S[i-1]["conditions"]+[S[i]["conditions"]]; S[i]["procedures"]=S[i-1]["procedures"]+[S[i]["procedures"]]; S[i]["drugs_hist"]=S[i-1]["drugs_hist"]+[S[i]["drugs_hist"]]
    for i in range(len(S)): S[i]["drugs_hist"][i]=[]
    return S

base_dataset=MIMIC4Dataset(root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",tables=["diagnoses_icd","procedures_icd","prescriptions"],dev=False,code_mapping={"NDC":"ATC"},refresh_cache=True)
sample_dataset=base_dataset.set_task(task_fn=drug_recommendation_mimic4_fn)
