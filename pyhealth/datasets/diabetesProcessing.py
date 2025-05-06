# diabetesProcessing.py
# ----------------
# Defines a DiabetesPreprocessor class that downloads,
# cleans, and saves the Pima Indians Diabetes dataset.

import pandas as pd
from pathlib import Path
from typing import Union, List

DEFAULT_RAW_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
DEFAULT_OUT_CSV = Path("data/diabetes_preprocessed.csv")
DEFAULT_FEATURES: List[str] = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

class DiabetesPreprocessor:
    def __init__(
        self,
        raw_source: Union[str, Path] = DEFAULT_RAW_URL,
        out_csv: Union[str, Path] = DEFAULT_OUT_CSV,
        features: List[str] = DEFAULT_FEATURES,
    ):
        """
        :param raw_source: URL or local path to the raw CSV
        :param out_csv:  path where the cleaned CSV will be saved
        :param features:  list of feature column names to keep
        """
        self.raw_source = str(raw_source)
        self.out_csv = Path(out_csv)
        self.features = features

    def load(self) -> pd.DataFrame:
        """Load CSV from a URL or local path."""
        return pd.read_csv(self.raw_source)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing feature or outcome values."""
        return df.dropna(subset=self.features + ["Outcome"]).copy()

    def add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a lowercase 'outcome' column from 'Outcome'."""
        df["outcome"] = df["Outcome"]
        return df

    def add_year_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a synthetic 'year' column:
        first half → 2016, second half → 2017
        """
        df = df.reset_index(drop=True)
        midpoint = len(df) // 2
        df["year"] = [2016 if i < midpoint else 2017 for i in range(len(df))]
        return df

    def save(self, df: pd.DataFrame) -> None:
        """Ensure output directory exists and write CSV."""
        self.out_csv.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(self.out_csv, index=False)
        print(f"[preprocessing] Saved cleaned file → {self.out_csv}")

    def run(self) -> None:
        """Full pipeline: load, clean, augment, and save."""
        df = self.load()
        df = self.clean(df)
        df = self.add_target(df)
        df = self.add_year_split(df)
        self.save(df)

if __name__ == "__main__":
    DiabetesPreprocessor().run()
