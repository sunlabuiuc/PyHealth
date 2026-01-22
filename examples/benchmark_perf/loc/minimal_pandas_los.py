from typing import Any, Dict, List
import pandas as pd
def categorize_los(days: int) -> int:
    return 0 if days < 1 else (days if days <= 7 else (8 if days <= 14 else 9))
def process_patient(subject_id, admissions_df, diagnoses_df, procedures_df, prescriptions_df) -> List[Dict[str, Any]]:
    samples = []
    for _, admission in admissions_df[admissions_df["subject_id"] == subject_id].iterrows():
        hadm_id = admission["hadm_id"]
        conditions = [f"{int(r['icd_version'])}_{r['icd_code']}" for _, r in diagnoses_df[diagnoses_df["hadm_id"] == hadm_id].iterrows() if pd.notna(r.get("icd_code")) and pd.notna(r.get("icd_version"))]
        procedures = [f"{int(r['icd_version'])}_{r['icd_code']}" for _, r in procedures_df[procedures_df["hadm_id"] == hadm_id].iterrows() if pd.notna(r.get("icd_code")) and pd.notna(r.get("icd_version"))]
        drugs = [str(r["ndc"]) for _, r in prescriptions_df[prescriptions_df["hadm_id"] == hadm_id].iterrows() if pd.notna(r.get("ndc"))]
        if not conditions or not procedures or not drugs or pd.isna(admission["admittime"]) or pd.isna(admission["dischtime"]): continue
        samples.append({"visit_id": hadm_id, "patient_id": subject_id, "conditions": conditions, "procedures": procedures, "drugs": drugs, "los": categorize_los((admission["dischtime"] - admission["admittime"]).days)})
    return samples
if __name__ == "__main__":
    DATA_ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.2/hosp"
    admissions_df = pd.read_csv(f"{DATA_ROOT}/admissions.csv")
    admissions_df["admittime"], admissions_df["dischtime"] = pd.to_datetime(admissions_df["admittime"]), pd.to_datetime(admissions_df["dischtime"])
    diagnoses_df, procedures_df = pd.read_csv(f"{DATA_ROOT}/diagnoses_icd.csv.gz"), pd.read_csv(f"{DATA_ROOT}/procedures_icd.csv.gz")
    prescriptions_df = pd.read_csv(f"{DATA_ROOT}/prescriptions.csv.gz", low_memory=False)
    samples = [s for sid in admissions_df["subject_id"].unique() for s in process_patient(sid, admissions_df, diagnoses_df, procedures_df, prescriptions_df)]
    print(f"Samples: {len(samples)}")
