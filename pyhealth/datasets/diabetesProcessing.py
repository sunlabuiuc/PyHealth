"""
preprocessing.py
----------------
Loads and processes the Pima Indians Diabetes CSV
"""

import pandas as pd
from pathlib import Path

RAW_CSV = Path("data/diabetes.csv")
OUT_CSV = Path("data/diabetes_preprocessed.csv")

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

def main() -> None:
    # 1. Load
    df = pd.read_csv(RAW_CSV)

    # 2. Drop missing
    df = df.dropna(subset=FEATURES + ["Outcome"]).copy()

    # 3. Binary target
    df["outcome"] = df["Outcome"]

    # 4. Synthetic year split
    df = df.reset_index(drop=True)
    df["year"] = [2016 if i < len(df) // 2 else 2017 for i in range(len(df))]

    # 5. Save
    OUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[preprocessing] Saved cleaned file → {OUT_CSV}")

if __name__ == "__main__":
    main()
