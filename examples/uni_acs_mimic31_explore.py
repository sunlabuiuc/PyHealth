from pathlib import Path
import pandas as pd


def main():
    data_path = Path.home() / "Desktop" / "mimiciv-3.1"

    files = {
        "patients": data_path / "hosp" / "patients.csv.gz",
        "admissions": data_path / "hosp" / "admissions.csv.gz",
        "icustays": data_path / "icu" / "icustays.csv.gz",
        "diagnoses": data_path / "hosp" / "diagnoses_icd.csv.gz",
        "labitems": data_path / "hosp" / "d_labitems.csv.gz",
        "icuitems": data_path / "icu" / "d_items.csv.gz",
    }

    print("D3.1: MIMIC-IV 3.1 Data Inspection")

    for name, path in files.items():
        print(f"\n{name}")
        print("Exists:", path.exists())

        if not path.exists():
            continue

        print("Size (MB):", round(path.stat().st_size / 1024**2, 1))

        # Only read five rows to avoid loading large datasets
        sample = pd.read_csv(path, nrows=5)

        print("Columns:", sample.columns.tolist())
        print(sample.head(2).to_string(index=False))

    # Check temporal metadata
    patients = pd.read_csv(
        files["patients"],
        usecols=[
            "subject_id",
            "anchor_age",
            "anchor_year",
            "anchor_year_group",
        ],
    )

    print("\nPatient count:", patients["subject_id"].nunique())
    print("\nAnchor year groups:")
    print(patients["anchor_year_group"].value_counts().sort_index())


if __name__ == "__main__":
    main()