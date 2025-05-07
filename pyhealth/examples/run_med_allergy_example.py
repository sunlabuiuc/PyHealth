print("🚀 STARTING run_med_allergy_example.py")

import pandas as pd
from pyhealth.tasks.allergy_conflict_task import AllergyConflictDetectionTask
from pyhealth.datasets.med_allergy_conflict_dataset import MedAllergyConflictDataset  # Adjust if in a different file

def main():
    # Step 1: Set path to the data folder containing 'note/discharge.csv.gz'
    root_path = "/Users/royal/Documents/Pyhealth_testing/mimic-iv-note-deidentified-free-text-clinical-notes-2"  

    # Step 2: Initialize dataset
    dataset = MedAllergyConflictDataset(root=root_path)

    # Step 3: Run the conflict detection task
    task = AllergyConflictDetectionTask(dataset)

    # Step 4: Export conflicts to CSV
    task.export_conflicts(output_path="conflict_patients.csv")

    # Step 5: Preview the output
    df = pd.read_csv("conflict_patients.csv")
    print("\n🧾 Preview of exported conflicts:")
    print(df.head())

if __name__ == "__main__":
    main()
