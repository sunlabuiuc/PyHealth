import polars as pl; from pyhealth.datasets import MIMIC4Dataset; from pyhealth.tasks.base_task import BaseTask
class DrugRecommendationMIMIC4(BaseTask):
    task_name="DrugRecommendationMIMIC4"
    input_schema={"conditions":"nested_sequence","procedures":"nested_sequence","drugs_hist":"nested_sequence"}
    output_schema={"drugs":"multilabel"}
    def __call__(self,p):
        adms=p.get_events(event_type="admissions")
        if len(adms)<2: return []
        S=[]
        for adm in adms:
            f=[("hadm_id","==",adm.hadm_id)]
            c=p.get_events("diagnoses_icd",filters=f,return_df=True).select(pl.concat_str(["diagnoses_icd/icd_version","diagnoses_icd/icd_code"],separator="_")).to_series().to_list()
            r=p.get_events("procedures_icd",filters=f,return_df=True).select(pl.concat_str(["procedures_icd/icd_version","procedures_icd/icd_code"],separator="_")).to_series().to_list()
            d=[x[:4] for x in p.get_events("prescriptions",filters=f,return_df=True).select(pl.col("prescriptions/ndc")).to_series().to_list() if x]
            if not(c and r and d): continue
            S.append({"visit_id":adm.hadm_id,"patient_id":p.patient_id,"conditions":c,"procedures":r,"drugs":d,"drugs_hist":d})
        if len(S)<2: return []
        S[0].update({"conditions":[S[0]["conditions"]],"procedures":[S[0]["procedures"]],"drugs_hist":[S[0]["drugs_hist"]]})
        for i in range(1,len(S)): S[i]["conditions"]=S[i-1]["conditions"]+[S[i]["conditions"]]; S[i]["procedures"]=S[i-1]["procedures"]+[S[i]["procedures"]]; S[i]["drugs_hist"]=S[i-1]["drugs_hist"]+[S[i]["drugs_hist"]]
        for i in range(len(S)): S[i]["drugs_hist"][i]=[]
        return S
base_dataset=MIMIC4Dataset(ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",ehr_tables=["patients","admissions","diagnoses_icd","procedures_icd","prescriptions"])
sample_dataset=base_dataset.set_task(DrugRecommendationMIMIC4())
