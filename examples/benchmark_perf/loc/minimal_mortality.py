from datetime import datetime; from typing import ClassVar,Dict,List; import polars as pl
from pyhealth.datasets import MIMIC4Dataset; from pyhealth.tasks.base_task import BaseTask
class MortalityPredictionStageNetMIMIC4(BaseTask):
    task_name="MortalityPredictionStageNetMIMIC4"
    LAB_CATS:ClassVar[Dict[str,List[str]]]={"Sodium":["50824","52455","50983","52623"],"Potassium":["50822","52452","50971","52610"],"Chloride":["50806","52434","50902","52535"],"Bicarbonate":["50803","50804"],"Glucose":["50809","52027","50931","52569"],"Calcium":["50808","51624"],"Magnesium":["50960"],"Anion Gap":["50868","52500"],"Osmolality":["52031","50964","51701"],"Phosphate":["50970"]}
    LAB_NAMES:ClassVar[List[str]]=list(LAB_CATS); LABITEMS:ClassVar[List[str]]=[x for ids in LAB_CATS.values() for x in ids]
    def __init__(self,padding=0):
        self.padding=padding; self.input_schema={"icd_codes":("stagenet",{"padding":padding}),"labs":("stagenet_tensor",{})}; self.output_schema={"mortality":"binary"}
    def __call__(self,p):
        demo=p.get_events(event_type="patients")
        if not demo or int(getattr(demo[0],"anchor_age",0)or 0)<18: return []
        adms=p.get_events(event_type="admissions")
        if not adms: return []
        icd_d,icd_t,lab_v,lab_t,mort,prev=[],[],[],[],0,None
        for adm in adms:
            try: at=adm.timestamp; dt=datetime.strptime(adm.dischtime,"%Y-%m-%d %H:%M:%S")
            except: continue
            if dt<at: continue
            tp=0.0 if prev is None else (at-prev).total_seconds()/3600.0; prev=at
            if int(getattr(adm,"hospital_expire_flag",0)or 0)==1: mort=1
            f=[("hadm_id","==",adm.hadm_id)]
            codes=[e.icd_code for e in p.get_events("diagnoses_icd",filters=f) if getattr(e,"icd_code",None)]+[e.icd_code for e in p.get_events("procedures_icd",filters=f) if getattr(e,"icd_code",None)]
            if codes: icd_d.append(codes); icd_t.append(tp)
            ldf=p.get_events("labevents",start=at,end=dt,return_df=True).filter(pl.col("labevents/itemid").is_in(self.LABITEMS))
            if ldf.height>0:
                ldf=ldf.with_columns(pl.col("labevents/storetime").str.strptime(pl.Datetime,"%Y-%m-%d %H:%M:%S")).filter(pl.col("labevents/storetime")<=dt).select(pl.col("timestamp"),pl.col("labevents/itemid"),pl.col("labevents/valuenum").cast(pl.Float64))
                for ts in sorted(ldf["timestamp"].unique().to_list()):
                    r=ldf.filter(pl.col("timestamp")==ts)
                    vec=[next((r.filter(pl.col("labevents/itemid")==iid)["labevents/valuenum"][0] for iid in self.LAB_CATS[cat] if r.filter(pl.col("labevents/itemid")==iid).height>0),None) for cat in self.LAB_NAMES]
                    lab_v.append(vec); lab_t.append((ts-at).total_seconds()/3600.0)
        if not lab_v or not icd_d: return []
        return [{"patient_id":p.patient_id,"icd_codes":(icd_t,icd_d),"labs":(lab_t,lab_v),"mortality":mort}]
base_dataset=MIMIC4Dataset(ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",ehr_tables=["patients","admissions","diagnoses_icd","procedures_icd","labevents"])
sample_dataset=base_dataset.set_task(MortalityPredictionStageNetMIMIC4())
