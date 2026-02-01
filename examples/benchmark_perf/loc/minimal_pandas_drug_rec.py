from typing import Any, Dict, List
import pandas as pd
def process_patient(subject_id, admissions_df, diagnoses_df, procedures_df, prescriptions_df) -> List[Dict[str, Any]]:
    patient_admissions = admissions_df[admissions_df["subject_id"] == subject_id].sort_values("admittime")
    if len(patient_admissions) < 2: return []
    visit_data = []
    for _, admission in patient_admissions.iterrows():
        hadm_id = admission["hadm_id"]
        conditions = [f"{int(r['icd_version'])}_{r['icd_code']}" for _, r in diagnoses_df[diagnoses_df["hadm_id"] == hadm_id].iterrows() if pd.notna(r.get("icd_code")) and pd.notna(r.get("icd_version"))]
        procedures = [f"{int(r['icd_version'])}_{r['icd_code']}" for _, r in procedures_df[procedures_df["hadm_id"] == hadm_id].iterrows() if pd.notna(r.get("icd_code")) and pd.notna(r.get("icd_version"))]
        drugs = [str(r["ndc"])[:4] for _, r in prescriptions_df[prescriptions_df["hadm_id"] == hadm_id].iterrows() if pd.notna(r.get("ndc")) and len(str(r["ndc"])) >= 4]
        if conditions and procedures and drugs: visit_data.append({"hadm_id": hadm_id, "conditions": conditions, "procedures": procedures, "drugs": drugs})
    if len(visit_data) < 2: return []
    return [{"visit_id": visit_data[i]["hadm_id"], "patient_id": subject_id,
             "conditions": [v["conditions"] for v in visit_data[:i+1]], "procedures": [v["procedures"] for v in visit_data[:i+1]],
             "drugs_hist": [v["drugs"] if j < i else [] for j, v in enumerate(visit_data[:i+1])], "drugs": visit_data[i]["drugs"]} for i in range(len(visit_data))]
if __name__ == "__main__":
    DATA_ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.2/hosp"
    admissions_df = pd.read_csv(f"{DATA_ROOT}/admissions.csv")
    admissions_df["admittime"] = pd.to_datetime(admissions_df["admittime"])
    diagnoses_df, procedures_df = pd.read_csv(f"{DATA_ROOT}/diagnoses_icd.csv.gz"), pd.read_csv(f"{DATA_ROOT}/procedures_icd.csv.gz")
    prescriptions_df = pd.read_csv(f"{DATA_ROOT}/prescriptions.csv.gz", low_memory=False)
    samples = [s for sid in admissions_df["subject_id"].unique() for s in process_patient(sid, admissions_df, diagnoses_df, procedures_df, prescriptions_df)]
    print(f"Samples: {len(samples)}")
