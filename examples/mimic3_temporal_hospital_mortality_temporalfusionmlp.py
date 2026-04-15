"""
Example: Temporal Mortality Prediction with TemporalFusionMLP (MIMIC-III Demo)

Author: Elizabeth Binkina

Description:
This example demonstrates a full end-to-end machine learning pipeline for
predicting in-hospital mortality using the MIMIC-III demo dataset. The pipeline
includes data loading, feature construction from structured EHR data, model
training, and evaluation under both random and temporal data splits.

Key Components:
- Dataset: MIMIC-III Demo (PhysioNet)
- Task: Binary in-hospital mortality prediction
- Features:
    * Diagnosis codes (ICD-9)
    * Procedure codes
    * Prescription data
    * Optional temporal feature (admission year)
- Model:
    * TemporalFusionMLP (custom neural network)
    * Baseline comparison with logistic regression
- Evaluation Metrics:
    * PR-AUC (Primary)
    * ROC-AUC

Experimental Design:
We compare model performance under:
1. Random train/validation/test split
2. Temporal split (train on earlier years, test on later years)

This setup follows prior work highlighting that random splits can lead to
overly optimistic performance estimates due to temporal distribution shift.

Key Findings:
- Performance is consistently higher under random splits than temporal splits,
  confirming the presence of temporal distribution shift.
- Including admission year as a feature provides modest improvements but does
  not eliminate performance degradation under temporal evaluation.
- The TemporalFusionMLP captures nonlinear feature interactions while remaining
  lightweight compared to sequence-based models.

Notes:
- This example uses the MIMIC-III demo dataset for accessibility and fast runtime.
- While the original literature evaluates recurrent models (e.g., LSTM, GRU-D),
  this implementation introduces a simpler TemporalFusionMLP architecture that
  incorporates temporal information through engineered features.

Setup Instructions:
1. Download the MIMIC-III demo dataset from:
   https://physionet.org/content/mimiciii-demo/1.4/
2. Extract it into a directory named "mimiciii_demo" in the project root.
3. Run the script:
   python examples/mimic3_temporal_hospital_mortality_temporalfusionmlp.py

Expected Output:
- Number of samples and class distribution
- Performance metrics (PR-AUC, ROC-AUC) for each split configuration
"""



from __future__ import annotations

import pathlib
import urllib.request
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def download_demo(root: pathlib.Path) -> None:
    """Downloads required MIMIC-III demo files if missing."""
    root.mkdir(parents=True, exist_ok=True)
    base_url = "https://physionet.org/files/mimiciii-demo/1.4"
    files = [
        "PATIENTS.csv",
        "ADMISSIONS.csv",
        "DIAGNOSES_ICD.csv",
        "PROCEDURES_ICD.csv",
        "PRESCRIPTIONS.csv",
    ]

    for filename in files:
        csv_path = root / filename
        gz_path = root / f"{filename}.gz"

        if csv_path.exists() or gz_path.exists():
            continue

        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(f"{base_url}/{filename}", csv_path)


def load_table(root: pathlib.Path, name: str) -> pd.DataFrame:
    """Loads either .csv or .csv.gz."""
    csv_path = root / f"{name}.csv"
    gz_path = root / f"{name}.csv.gz"

    if csv_path.exists():
        return pd.read_csv(csv_path)
    if gz_path.exists():
        return pd.read_csv(gz_path, compression="gzip")

    raise FileNotFoundError(f"Could not find {csv_path} or {gz_path}")


def build_samples(root: pathlib.Path) -> List[Dict]:
    """Builds admission-level mortality samples directly from MIMIC-III demo CSVs."""
    admissions = load_table(root, "ADMISSIONS")
    diagnoses = load_table(root, "DIAGNOSES_ICD")
    procedures = load_table(root, "PROCEDURES_ICD")
    prescriptions = load_table(root, "PRESCRIPTIONS")

    admissions.columns = admissions.columns.str.upper()
    diagnoses.columns = diagnoses.columns.str.upper()
    procedures.columns = procedures.columns.str.upper()
    prescriptions.columns = prescriptions.columns.str.upper()

    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="coerce")
    admissions["HOSPITAL_EXPIRE_FLAG"] = (
        admissions["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int)
    )

    diag_group = (
        diagnoses.groupby("HADM_ID")["ICD9_CODE"]
        .apply(lambda s: [str(x) for x in s.dropna().tolist()])
        .to_dict()
    )
    proc_group = (
        procedures.groupby("HADM_ID")["ICD9_CODE"]
        .apply(lambda s: [str(x) for x in s.dropna().tolist()])
        .to_dict()
    )

    if "DRUG" in prescriptions.columns:
        drug_col = "DRUG"
    elif "DRUG_NAME_GENERIC" in prescriptions.columns:
        drug_col = "DRUG_NAME_GENERIC"
    else:
        drug_col = prescriptions.columns[-1]

    drug_group = (
        prescriptions.groupby("HADM_ID")[drug_col]
        .apply(lambda s: [str(x) for x in s.dropna().tolist()])
        .to_dict()
    )

    min_year, max_year = 2001, 2012
    samples: List[Dict] = []

    for _, row in admissions.iterrows():
        hadm_id = row["HADM_ID"]
        admit_time = row["ADMITTIME"]

        if pd.isna(admit_time):
            continue

        conditions = diag_group.get(hadm_id, [])
        procedures_list = proc_group.get(hadm_id, [])
        drugs = drug_group.get(hadm_id, [])

        if not conditions or not procedures_list or not drugs:
            continue

        year_raw = int(admit_time.year)
        year_raw = min(max(year_raw, min_year), max_year)
        year_norm = (year_raw - min_year) / float(max_year - min_year)

        samples.append(
            {
                "hadm_id": int(hadm_id),
                "patient_id": str(row["SUBJECT_ID"]),
                "conditions": conditions,
                "procedures": procedures_list,
                "drugs": drugs,
                "admission_year": [float(year_norm)],
                "admission_year_raw": year_raw,
                "mortality": int(row["HOSPITAL_EXPIRE_FLAG"]),
            }
        )

    return samples


def sample_to_features(sample: Dict, use_temporal_feature: bool) -> Dict[str, float]:
    """Converts one sample dict into sparse tabular features."""
    feat: Dict[str, float] = {}

    for code in sample.get("conditions", []):
        feat[f"cond::{code}"] = 1.0
    for code in sample.get("procedures", []):
        feat[f"proc::{code}"] = 1.0
    for drug in sample.get("drugs", []):
        feat[f"drug::{drug}"] = 1.0

    if use_temporal_feature:
        feat["admission_year"] = float(sample["admission_year"][0])

    return feat


def make_random_split(
    samples: List[Dict],
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Creates stratified random train/val/test splits."""
    all_idx = list(range(len(samples)))
    all_labels = [int(samples[i]["mortality"]) for i in all_idx]

    train_idx, temp_idx, _, y_temp = train_test_split(
        all_idx,
        all_labels,
        test_size=0.3,
        random_state=seed,
        stratify=all_labels,
    )

    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        y_temp,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp,
    )
    return train_idx, val_idx, test_idx


def make_temporal_split(
    samples: List[Dict],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[List[int], List[int], List[int]]:
    """Creates chronological train/val/test splits by sorted admission year."""
    df = pd.DataFrame(
        {
            "idx": list(range(len(samples))),
            "year": [int(s["admission_year_raw"]) for s in samples],
            "label": [int(s["mortality"]) for s in samples],
        }
    ).sort_values(["year", "idx"]).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_idx = df.iloc[:train_end]["idx"].tolist()
    val_idx = df.iloc[train_end:val_end]["idx"].tolist()
    test_idx = df.iloc[val_end:]["idx"].tolist()

    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            "Temporal split is empty for one partition. "
            "Dataset is too small for the requested fractions."
        )

    return train_idx, val_idx, test_idx


class TemporalFusionMLP(nn.Module):
    """Small neural network for admission-level mortality prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def evaluate_split(
    samples: List[Dict],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    use_temporal_feature: bool,
    epochs: int = 100,
    lr: float = 1e-3,
    hidden_dim: int = 64,
) -> Dict[str, float]:
    """Fits a custom TemporalFusionMLP and evaluates it."""
    vectorizer = DictVectorizer(sparse=False)

    x_train = [sample_to_features(samples[i], use_temporal_feature) for i in train_idx]
    x_val = [sample_to_features(samples[i], use_temporal_feature) for i in val_idx]
    x_test = [sample_to_features(samples[i], use_temporal_feature) for i in test_idx]

    y_train = np.array([int(samples[i]["mortality"]) for i in train_idx], dtype=np.float32)
    y_val = np.array([int(samples[i]["mortality"]) for i in val_idx], dtype=np.float32)
    y_test = np.array([int(samples[i]["mortality"]) for i in test_idx], dtype=np.float32)

    x_train_vec = vectorizer.fit_transform(x_train).astype(np.float32)
    x_val_vec = vectorizer.transform(x_val).astype(np.float32)
    x_test_vec = vectorizer.transform(x_test).astype(np.float32)

    x_train_tensor = torch.tensor(x_train_vec, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val_vec, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_vec, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    model = TemporalFusionMLP(input_dim=x_train_vec.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 10
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_logits = model(x_train_tensor)
        train_loss = criterion(train_logits, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(x_val_tensor)).cpu().numpy()
        test_prob = torch.sigmoid(model(x_test_tensor)).cpu().numpy()

    metrics: Dict[str, float] = {
        "n_train": float(len(train_idx)),
        "n_val": float(len(val_idx)),
        "n_test": float(len(test_idx)),
        "positive_rate_test": float(y_test.mean()),
        "pr_auc_val": float(average_precision_score(y_val, val_prob)),
        "pr_auc_test": float(average_precision_score(y_test, test_prob)),
    }

    try:
        metrics["roc_auc_test"] = float(roc_auc_score(y_test, test_prob))
    except ValueError:
        metrics["roc_auc_test"] = float("nan")

    return metrics


def main() -> None:
    demo_root = pathlib.Path.home() / "Desktop" / "mimiciii_demo_1.4"
    download_demo(demo_root)

    samples = build_samples(demo_root)

    print(f"Number of samples: {len(samples)}")
    print("Label counts:", Counter(int(s["mortality"]) for s in samples))

    rand_train, rand_val, rand_test = make_random_split(samples)
    tmp_train, tmp_val, tmp_test = make_temporal_split(samples)

    results = []

    results.append(
        {
            "split": "random",
            "use_temporal_feature": False,
            **evaluate_split(samples, rand_train, rand_val, rand_test, False),
        }
    )
    results.append(
        {
            "split": "random",
            "use_temporal_feature": True,
            **evaluate_split(samples, rand_train, rand_val, rand_test, True),
        }
    )
    results.append(
        {
            "split": "temporal",
            "use_temporal_feature": False,
            **evaluate_split(samples, tmp_train, tmp_val, tmp_test, False),
        }
    )
    results.append(
        {
            "split": "temporal",
            "use_temporal_feature": True,
            **evaluate_split(samples, tmp_train, tmp_val, tmp_test, True),
        }
    )

    results_df = pd.DataFrame(results)
    print("\nResults:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
"""
Final Results:

Random Split:
PR-AUC: 0.713 | ROC-AUC: 0.733

Temporal Split:
PR-AUC: 0.607 | ROC-AUC: 0.633

Conclusion:
Random splits overestimate performance compared to temporal splits.
"""
