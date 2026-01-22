from typing import Any, Dict, Optional
import pandas as pd
LAB_CATEGORIES = {"Sodium": ["50824", "52455", "50983", "52623"], "Potassium": ["50822", "52452", "50971", "52610"],
                  "Chloride": ["50806", "52434", "50902", "52535"], "Bicarbonate": ["50803", "50804"],
                  "Glucose": ["50809", "52027", "50931", "52569"], "Calcium": ["50808", "51624"], "Magnesium": ["50960"],
                  "Anion Gap": ["50868", "52500"], "Osmolality": ["52031", "50964", "51701"], "Phosphate": ["50970"]}
LAB_CATEGORY_NAMES = list(LAB_CATEGORIES.keys())
LABITEMS = [item for itemids in LAB_CATEGORIES.values() for item in itemids]
def process_patient(subject_id, patients_df, admissions_df, diagnoses_df, procedures_df, labevents_df) -> Optional[Dict[str, Any]]:
    patient_demo = patients_df[patients_df["subject_id"] == subject_id]
    if len(patient_demo) == 0 or patient_demo.iloc[0]["anchor_age"] < 18: return None
    patient_admissions = admissions_df[admissions_df["subject_id"] == subject_id].sort_values("admittime")
    if len(patient_admissions) < 1: return None
    all_icd_codes, all_icd_times, all_lab_values, all_lab_times = [], [], [], []
    previous_admission_time, final_mortality = None, 0
    for _, admission in patient_admissions.iterrows():
        hadm_id, admit_time, discharge_time = admission["hadm_id"], admission["admittime"], admission["dischtime"]
        if pd.isna(discharge_time) or discharge_time < admit_time: continue
        time_from_previous = 0.0 if previous_admission_time is None else (admit_time - previous_admission_time).total_seconds() / 3600.0
        previous_admission_time = admit_time
        if int(admission.get("hospital_expire_flag", 0)) == 1: final_mortality = 1
        visit_icd = diagnoses_df[diagnoses_df["hadm_id"] == hadm_id]["icd_code"].dropna().tolist()
        visit_icd += procedures_df[procedures_df["hadm_id"] == hadm_id]["icd_code"].dropna().tolist()
        if visit_icd: all_icd_codes.append(visit_icd); all_icd_times.append(time_from_previous)
        admission_labs = labevents_df[(labevents_df["subject_id"] == subject_id) & (labevents_df["hadm_id"] == hadm_id)]
        admission_labs = admission_labs[admission_labs["itemid"].astype(str).isin(LABITEMS)].copy()
        if len(admission_labs) > 0:
            admission_labs["storetime"] = pd.to_datetime(admission_labs["storetime"])
            admission_labs = admission_labs[admission_labs["storetime"] <= discharge_time]
            for lab_ts in sorted(admission_labs["storetime"].unique()):
                ts_labs = admission_labs[admission_labs["storetime"] == lab_ts]
                lab_vector = []
                for cat in LAB_CATEGORY_NAMES:
                    val = None
                    for itemid in LAB_CATEGORIES[cat]:
                        m = ts_labs[ts_labs["itemid"].astype(str) == itemid]
                        if len(m) > 0: val = m.iloc[0]["valuenum"]; break
                    lab_vector.append(val)
                all_lab_values.append(lab_vector)
                all_lab_times.append((lab_ts - admit_time).total_seconds() / 3600.0)
    if not all_lab_values or not all_icd_codes: return None
    return {"patient_id": subject_id, "icd_codes": (all_icd_times, all_icd_codes), "labs": (all_lab_times, all_lab_values), "mortality": final_mortality}
if __name__ == "__main__":
    DATA_ROOT = "/srv/local/data/MIMIC-IV/2.0/hosp"
    patients_df = pd.read_csv(f"{DATA_ROOT}/patients.csv")
    admissions_df = pd.read_csv(f"{DATA_ROOT}/admissions.csv")
    admissions_df["admittime"], admissions_df["dischtime"] = pd.to_datetime(admissions_df["admittime"]), pd.to_datetime(admissions_df["dischtime"])
    diagnoses_df, procedures_df = pd.read_csv(f"{DATA_ROOT}/diagnoses_icd.csv"), pd.read_csv(f"{DATA_ROOT}/procedures_icd.csv")
    labevents_df = pd.read_csv(f"{DATA_ROOT}/labevents.csv")
    samples = [s for sid in patients_df["subject_id"] if (s := process_patient(sid, patients_df, admissions_df, diagnoses_df, procedures_df, labevents_df))]
    print(f"Samples: {len(samples)}")
