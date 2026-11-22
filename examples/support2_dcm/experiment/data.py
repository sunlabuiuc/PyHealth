# Contributor: Xingyao Wang (xingyao6)
# Paper: Deep Cox Mixtures for Survival Regression
# https://proceedings.mlr.press/v149/nagpal21a.html
"""SUPPORT2 CSV loading, train-only preprocessing and laboratory masking."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.util import Surv

from pyhealth.tasks.support2_survival import (
    SUPPORT2_DCM_CATEGORICAL,
    SUPPORT2_DCM_NUMERIC,
)

# The covariate list is defined once, in the PyHealth task.
CATEGORICAL = list(SUPPORT2_DCM_CATEGORICAL)
NUMERIC = list(SUPPORT2_DCM_NUMERIC)
LABS = ["wblc", "alb", "bili", "crea", "sod", "ph", "glucose", "bun"]


def load_data(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Read the SUPPORT2 CSV, check labels and return covariates and outcomes."""
    frame = pd.read_csv(path)
    required = set(CATEGORICAL + NUMERIC + ["death", "d.time"])
    if not required.issubset(frame):
        raise ValueError(f"Missing SUPPORT2 fields: {required - set(frame)}")
    if not frame.death.isin([0, 1]).all() or not (frame["d.time"] > 0).all():
        raise ValueError(
            "Expected positive follow-up and death=1/event, death=0/censored"
        )
    # Row ordinal in the verified archive order; not a clinical identifier.
    frame["patient_id"] = [f"support2-row-{i:05d}" for i in range(len(frame))]
    y = Surv.from_arrays(frame.death.astype(bool), frame["d.time"].astype(float))
    return frame, y


def preprocessing(indicators: bool = False) -> ColumnTransformer:
    """Return an unfitted median/scale + one-hot transformer for the covariates."""
    transformers = [
        (
            "numeric",
            make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
            ),
            NUMERIC,
        ),
        (
            "categorical",
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
            CATEGORICAL,
        ),
    ]
    if indicators:
        # features() adds the indicators, covering original missingness too.
        transformers.append(
            ("lab_missing", "passthrough", [x + "__missing" for x in LABS])
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def features(frame: pd.DataFrame, indicators: bool = False) -> pd.DataFrame:
    """Select covariates as strings/numbers, optionally adding lab indicators."""
    out = frame[CATEGORICAL + NUMERIC].copy()
    for key in CATEGORICAL:
        out[key] = out[key].map(lambda x: str(x) if pd.notna(x) else np.nan)
    if indicators:
        for key in LABS:
            out[key + "__missing"] = out[key].isna().astype(float)
    return out


def mask_labs(
    frame: pd.DataFrame, rate: float, seed: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Hide a random share of observed lab values; return the copy and counts."""
    if not 0 <= rate <= 1:
        raise ValueError("Mask rate must be in [0,1]")
    out = frame.copy()
    observed = out[LABS].notna().to_numpy()
    induced = (np.random.default_rng(seed).random(observed.shape) < rate) & observed
    out.loc[:, LABS] = out[LABS].mask(induced)
    return out, {
        "observed_lab_values": int(observed.sum()),
        "induced_missing": int(induced.sum()),
        "original_missing": int((~observed).sum()),
        "realized_rate": float(induced.sum() / observed.sum()),
    }


def sha256(path: str) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
