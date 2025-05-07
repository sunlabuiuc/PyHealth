# ------------------------------------------------------------------------------
# Author: Chidiebere Anichebe , Naveen Baskaran
# NetID: cpa4@illinois.edu, nc42#illinois.edu
# Contribution: Medication-Allergy Conflict Detection Task 
# Paper Title: A Data-Centric Approach to Generate Faithful and High-Quality Patient Summaries with Large Language Models
# Paper Link: https://physionet.org/content/ann-pt-summ/1.0.1/
# Description:
#   Labels whether any medication ingredient CUI in a patient admission
#   matches a known allergy CUI.
# ------------------------------------------------------------------------------


from pyhealth.tasks.base_task import BaseTask
from typing import Dict, Any
import pandas as pd
from pyhealth.datasets.med_allergy_conflict_dataset import MedAllergyConflictDataset


class AllergyConflictDetectionTask(BaseTask):
    """
    A PyHealth Task that detects whether a medication-allergy conflict exists for a given patient.
    Returns 1 if any ingredient CUI from medications appears in the allergy CUIs, else 0.
    """

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.feature_keys = ["medication_cuis", "allergy_cuis"]
        self.label_key = "conflict"

    def get_label(self, sample: Dict[str, Any]) -> int:
        med_cuis = set(str(cui) for cui in sample.get("medication_cuis", []) if cui)
        allergy_cuis = set(str(cui) for cui in sample.get("allergy_cuis", []) if cui)
        return int(not med_cuis.isdisjoint(allergy_cuis))

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        label = self.get_label(sample)
        return {"label": label}

    def export_conflicts(self, output_path: str = "conflict_patients.csv"):
        """
        Iterates through the dataset and exports all patients with medication-allergy conflicts.

        Args:
            output_path (str): Path to save the CSV file with conflict details.
        """
        conflict_rows = []
        for pid in self.dataset.get_all_patient_ids():
            patient = self.dataset.get_patient_by_id(pid)
            if not patient:
                continue

            label = self.get_label(patient)
            if label == 1:
                conflict_rows.append({
                    "hadm_id": pid,
                    "medications": patient.get("medications"),
                    "medication_cuis": patient.get("medication_cuis"),
                    "allergies": patient.get("allergies"),
                    "allergy_cuis": patient.get("allergy_cuis")
                })

        df = pd.DataFrame(conflict_rows)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} conflict cases to {output_path}")

if __name__ == "__main__":
    from pyhealth.datasets.med_allergy_conflict_dataset import MedAllergyConflictDataset

    dataset = MedAllergyConflictDataset(root="/Users/royal/Documents/Pyhealth_testing/mimic-iv-note-deidentified-free-text-clinical-notes-2")
    task = AllergyConflictDetectionTask(dataset)
    task.export_conflicts("conflict_patients.csv")
