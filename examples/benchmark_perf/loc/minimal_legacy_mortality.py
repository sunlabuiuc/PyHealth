from collections import defaultdict; from pyhealth.data import Patient; from pyhealth.datasets import MIMIC4Dataset
from typing import Dict,List

LAB_CATS:Dict[str,List[str]]={"Sodium":["50824","52455","50983","52623"],"Potassium":["50822","52452","50971","52610"],"Chloride":["50806","52434","50902","52535"],"Bicarbonate":["50803","50804"],"Glucose":["50809","52027","50931","52569"],"Calcium":["50808","51624"],"Magnesium":["50960"],"Anion Gap":["50868","52500"],"Osmolality":["52031","50964","51701"],"Phosphate":["50970"]}
LAB_NAMES=list(LAB_CATS); LABITEMS=[x for ids in LAB_CATS.values() for x in ids]

def mortality_task_fn(patient):
    icd_d,icd_t,lab_v,lab_t,mort,prev=[],[],[],[],0,None
    for v in patient:
        at=v.encounter_time; dt=v.discharge_time
        if at is None or dt is None or dt<at: continue
        tp=0.0 if prev is None else (at-prev).total_seconds()/3600.0; prev=at
        if getattr(v,"discharge_status",None)==1: mort=1
        codes=v.get_code_list(table="diagnoses_icd")+v.get_code_list(table="procedures_icd")
        if codes: icd_d.append(codes); icd_t.append(tp)
        ts_groups=defaultdict(dict)
        for e in v.get_event_list(table="labevents"):
            if e.code in LABITEMS and e.timestamp and e.value is not None:
                ts_groups[e.timestamp][e.code]=e.value
        for ts in sorted(ts_groups.keys()):
            vec=[next((float(ts_groups[ts][iid]) for iid in LAB_CATS[cat] if iid in ts_groups[ts]),None) for cat in LAB_NAMES]
            lab_v.append(vec); lab_t.append((ts-at).total_seconds()/3600.0)
    if not lab_v or not icd_d: return []
    return [{"patient_id":patient.patient_id,"icd_codes":(icd_t,icd_d),"labs":(lab_t,lab_v),"mortality":mort}]

base_dataset=MIMIC4Dataset(root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",tables=["diagnoses_icd","procedures_icd","labevents"],dev=False,refresh_cache=True)
sample_dataset=base_dataset.set_task(task_fn=mortality_task_fn)
