from datetime import datetime; from pyhealth.datasets import MIMIC4Dataset; from pyhealth.tasks.base_task import BaseTask
def categorize_los(d): return 0 if d<1 else (d if d<=7 else (8 if d<=14 else 9))class LengthOfStayPredictionMIMIC4(BaseTask):
    task_name="LengthOfStayPredictionMIMIC4"
    input_schema={"conditions":"sequence","procedures":"sequence","drugs":"sequence"}
    output_schema={"los":"multiclass"}
    def __call__(self,p):
        S=[]
        for adm in p.get_events(event_type="admissions"):
            f=[("hadm_id","==",adm.hadm_id)]
            c=[e.icd_code for e in p.get_events("diagnoses_icd",filters=f)]
            r=[e.icd_code for e in p.get_events("procedures_icd",filters=f)]
            d=[e.ndc for e in p.get_events("prescriptions",filters=f)]
            if not(c and r and d): continue
            los=categorize_los((datetime.strptime(adm.dischtime,"%Y-%m-%d %H:%M:%S")-adm.timestamp).days)
            S.append({"visit_id":adm.hadm_id,"patient_id":p.patient_id,"conditions":c,"procedures":r,"drugs":d,"los":los})
        return S
base_dataset=MIMIC4Dataset(ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",ehr_tables=["patients","admissions","diagnoses_icd","procedures_icd","prescriptions"])
sample_dataset=base_dataset.set_task(LengthOfStayPredictionMIMIC4())
