import polars as pl
from pyhealth.datasets import BaseDataset
import os
import gzip
import shutil
import json
import pandas as pd
import re
import requests
import time
from typing import Dict, Optional
from tqdm import tqdm


class MedAllergyConflictDataset(BaseDataset):
    STOPWORDS = {
        'tab', 'tablet', 'po', 'daily', 'chewable', 'cap', 'capsule', 'unit',
        'sc', 'si', 'bid', 'prn', 'ml', 'mg', 'drop', 'puff', 'tid', 'qid',
        'qh', 'qhs', 'ih', 'inh', 'soln', 'suspension', 'ointment', 'ophth', 'sol'
    }

    PLACEHOLDERS = {'___', 'nka', 'none', 'n/a'}

    def __init__(self, root: str, **kwargs):
        config_path = os.path.join(root, "pyhealth.yaml")

        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                f.write("""
dataset_name: MedAllergyConflictDataset
version: 1.0.0
description: Dummy config for medication-allergy conflict detection.
tables:
  dummy_table:
    path: dummy.csv
    file_path: dummy.csv
    type: custom
    attributes: ["dummy_attr"]
""")

        super().__init__(
            dataset_name="MedAllergyConflictDataset",
            root=root,
            tables=[],
            dev=kwargs.get("dev", False),
            config_path=config_path
        )

        self.patients = {}
        self.extract_dir = os.path.join(root, "file")
        os.makedirs(self.extract_dir, exist_ok=True)

        self.gz_path = os.path.join(root, "note", "discharge.csv.gz")
        self.csv_path = os.path.join(self.extract_dir, "discharge.csv")
        self.med_cache_path = os.path.join(self.extract_dir, "med_cache.json")
        self.allergy_cache_path = os.path.join(self.extract_dir, "allergy_cache.json")

        self._load_data()

    def load_data(self):
        return pl.DataFrame([])

    def _load_data(self):
        if not os.path.exists(self.csv_path):
            with gzip.open(self.gz_path, 'rb') as f_in:
                with open(self.csv_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        self.df = pd.read_csv(self.csv_path)
        self.rxcui_atc_cache = self._load_json(self.med_cache_path)
        self.allergy_cui_cache = self._load_json(self.allergy_cache_path)
        self._preprocess()

    def _load_json(self, path: str) -> Dict:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def _save_json(self, obj: Dict, path: str):
        with open(path, 'w') as f:
            json.dump(obj, f)

    def _normalize_med_name(self, name: str) -> str:
        name = name.lower()
        name = re.sub(r'[^a-z0-9\s/-]', '', name)
        tokens = name.split()
        filtered = [t for t in tokens if t not in self.STOPWORDS and not t.isdigit()]
        return ' '.join(filtered).strip().title()

    def _extract_allergies(self, text):
        match = re.search(r"Allergies:\s*(.*?)(?:\n|$)", text)
        return match.group(1).strip() if match else None

    def _extract_medications(self, text):
        start = text.find("Discharge Medications:")
        if start == -1:
            return []
        end = text.find("\n\n", start)
        section = text[start:end].replace('Discharge Medications:', '').strip() if end != -1 else text[start:]
        entries = re.split(r'\n?\d+\.\s+', section)
        meds = []
        for entry in entries:
            name = re.split(r'\d+ *mg|mEq|mcg|g|units', entry, flags=re.IGNORECASE)[0]
            name = re.sub(r'\(.*?\)', '', name)
            name = re.sub(r'[^a-zA-Z0-9\s/]+', '', name)
            cleaned = self._normalize_med_name(name)
            if cleaned:
                meds.append(cleaned)
        return meds

    def _preprocess(self):
        print("Starting preprocessing loop...")
        count = 0
        for _, row in self.df.iterrows():
            if count >= 200:
                break
            hadm_id = str(row['hadm_id'])
            text = row['text']
            allergies = self._extract_allergies(text)
            meds = self._extract_medications(text)
            if not allergies or not meds:
                continue

            parsed_allergies = [a.strip().title() for a in re.split(r'[,/]', allergies) if a.strip()]
            if all(a.lower() in self.PLACEHOLDERS for a in parsed_allergies):
                continue

            allergy_cuis = [self._resolve_allergy_cui(a) for a in parsed_allergies if a]
            med_cuis = [self._resolve_med_ingredient_cui(m) for m in meds if m]

            self.patients[hadm_id] = {
                "hadm_id": hadm_id,
                "allergies": parsed_allergies,
                "medications": meds,
                "allergy_cuis": allergy_cuis,
                "medication_cuis": med_cuis,
            }
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} patients...")

        print(f"Finished preprocessing. Total patients processed: {count}")
        self._save_json(self.rxcui_atc_cache, self.med_cache_path)
        self._save_json(self.allergy_cui_cache, self.allergy_cache_path)

    def _resolve_allergy_cui(self, term: str, max_retries=3) -> Optional[str]:
        base_url = "https://rxnav.nlm.nih.gov/REST"
        term = term.strip().title()
        if term in self.allergy_cui_cache:
            return self.allergy_cui_cache[term]

        retries = 0
        ingredient_cui = None
        while retries < max_retries:
            try:
                r = requests.get(f"{base_url}/rxcui.json", params={"name": term, "search": 1}, timeout=5)
                r.raise_for_status()
                rxcui = r.json().get("idGroup", {}).get("rxnormId", [None])[0]
                if not rxcui:
                    a = requests.get(f"{base_url}/approximateTerm.json", params={"term": term}, timeout=5)
                    a.raise_for_status()
                    candidates = a.json().get("approximateGroup", {}).get("candidate", [])
                    rxcui = candidates[0].get("rxcui") if candidates else None
                if not rxcui:
                    break
                rel = requests.get(f"{base_url}/rxcui/{rxcui}/related.json", params={"tty": "IN"}, timeout=5)
                rel.raise_for_status()
                concept_group = rel.json().get("relatedGroup", {}).get("conceptGroup", [])
                if concept_group:
                    props = concept_group[0].get("conceptProperties", [])
                    if props:
                        ingredient_cui = props[0]['rxcui']
                if not ingredient_cui:
                    ingredient_cui = rxcui
                break
            except requests.exceptions.RequestException:
                pass
            retries += 1
            time.sleep(1)

        self.allergy_cui_cache[term] = ingredient_cui
        return ingredient_cui

    def _resolve_med_ingredient_cui(self, drug_name: str, max_retries=3) -> Optional[str]:
        base_url = "https://rxnav.nlm.nih.gov/REST"
        drug_name = drug_name.strip().title()
        if drug_name in self.rxcui_atc_cache:
            return self.rxcui_atc_cache[drug_name][2]

        retries = 0
        rxcui = atc_code = ingredient_cui = None
        while retries < max_retries:
            try:
                r = requests.get(f"{base_url}/rxcui.json", params={"name": drug_name, "search": 1}, timeout=5)
                r.raise_for_status()
                rxcui = r.json().get("idGroup", {}).get("rxnormId", [None])[0]
                if not rxcui:
                    approx = requests.get(f"{base_url}/approximateTerm.json", params={"term": drug_name}, timeout=5)
                    approx.raise_for_status()
                    candidates = approx.json().get("approximateGroup", {}).get("candidate", [])
                    rxcui = candidates[0]['rxcui'] if candidates else None
                if not rxcui:
                    break
                atc_resp = requests.get(f"{base_url}/rxcui/{rxcui}/class.json", timeout=5)
                if atc_resp.status_code == 200:
                    atc_json = atc_resp.json()
                    atc_codes = [
                        c["rxclassMinConceptItem"]["classId"]
                        for c in atc_json.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
                        if "ATC" in c["rxclassMinConceptItem"]["className"]
                    ]
                    atc_code = atc_codes[0] if atc_codes else None
                rel_resp = requests.get(f"{base_url}/rxcui/{rxcui}/related.json", params={"tty": "IN"}, timeout=5)
                rel_resp.raise_for_status()
                related = rel_resp.json().get("relatedGroup", {}).get("conceptGroup", [])
                if related:
                    props = related[0].get("conceptProperties", [])
                    if props:
                        ingredient_cui = props[0]['rxcui']
                if not ingredient_cui:
                    ingredient_cui = rxcui
                break
            except requests.exceptions.RequestException:
                pass
            retries += 1
            time.sleep(1)

        self.rxcui_atc_cache[drug_name] = [rxcui, atc_code, ingredient_cui]
        return ingredient_cui

    def get_all_patient_ids(self):
        return list(self.patients.keys())

    def get_patient_by_id(self, patient_id: str) -> Optional[Dict]:
        return self.patients.get(patient_id)

    def export_medications(self, output_path: str):
        rows = []
        for p in self.patients.values():
            for med, cui in zip(p["medications"], p["medication_cuis"]):
                rows.append({"hadm_id": p["hadm_id"], "medication": med, "ingredient_cui": cui})
        pd.DataFrame(rows).to_csv(output_path, index=False)

    def export_allergies(self, output_path: str):
        rows = []
        for p in self.patients.values():
            for allergen, cui in zip(p["allergies"], p["allergy_cuis"]):
                rows.append({"hadm_id": p["hadm_id"], "allergy": allergen, "allergy_cui": cui})
        pd.DataFrame(rows).to_csv(output_path, index=False)

if __name__ == "__main__":
    dataset = MedAllergyConflictDataset(root="/your/data/path")
    print(f"Total patients: {len(dataset.get_all_patient_ids())}")
