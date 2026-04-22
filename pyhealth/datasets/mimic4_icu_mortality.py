"""MIMIC-IV ICU In-Hospital Mortality Dataset for PyHealth

Loads the MIMIC-IV dataset and extracts the 34 features (mean and standard
deviation over the first 48 hours of ICU stay for 17 physiological variables)
used in Hegselmann et al. (MLHC 2020) for in-hospital mortality prediction.

Reference:
    Hegselmann et al., "An Evaluation of the Doctor-Interpretability of
    Generalized Additive Models with Interactions", MLHC 2020.
    https://proceedings.mlr.press/v126/hegselmann20a.html

Data Access:
    Full MIMIC-IV requires credentialed PhysioNet access:
        https://physionet.org/content/mimiciv/
    A demo subset (100 patients) is freely available. We are using this.
        https://physionet.org/content/mimic-iv-demo/2.2/

Usage:
    - from pyhealth.datasets import MIMIC4ICUMortalityDataset
    - dataset = MIMIC4ICUMortalityDataset(root="path/to/mimic-iv-demo/")
    - print(len(dataset))
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pyhealth.datasets import SampleDataset, create_sample_dataset


# mapping of variable names to lists of mimic-iv itemids to use
# order of preference
VARIABLE_ITEMIDS: Dict[str, List[int]] = {
    "heart_rate":           [220045],
    "respiratory_rate":     [220210],
    "temperature":          [223762, 223761], # celsius preferred, fahrenheit fallback
    "systolic_bp":          [220179, 220050], # non-invasive preferred, arterial fallback
    "diastolic_bp":         [220180, 220051],
    "mean_bp":              [220181, 220052],
    "spo2":                 [220277],
    "gcs_eye":              [220739],
    "gcs_motor":            [223901],
    "gcs_verbal":           [223900],
    "ph":                   [223830],
    "fio2":                 [223835],
    "glucose":              [220621],
    "potassium":            [227442],
    "sodium":               [220645],
    "hematocrit":           [220545],
    "wbc":                  [220546],
}

# unknown/missing sentinel value
UNKNOWN_SENTINEL = -1.0

# temp conversion threshold: values above this are assumed Fahrenheit.
_FAHRENHEIT_THRESHOLD = 50.0


def _to_celsius(series: pd.Series) -> pd.Series:
    # convert F° to C° for values
    mask = series > _FAHRENHEIT_THRESHOLD
    series = series.copy()
    series[mask] = (series[mask] - 32.0) * 5.0 / 9.0
    return series


def build_mortality_samples(
    root: str,
    max_icu_hours: float = 48.0,
    unknown_sentinel: float = UNKNOWN_SENTINEL,
) -> List[Dict]:
    """Extract 34-feature mortality prediction samples from MIMIC-IV CSVs.

    Replicates the preprocessing pipeline from Hegselmann et al. (MLHC 2020),
        - first 48 hours of each ICU stay
        - mean and std dev for each of 17 variables, so 34 features per stay.
        - missing vals: "unknown_sentinel" (-1).
        - temp vals: to celsius.

    Args:
        root: Path to the MIMIC-IV (or demo) root directory, which should
            contain ``hosp/`` and ``icu/`` subdirectories.
        max_icu_hours: Number of hours from ICU admission to use (default 48).
        unknown_sentinel: Value to impute for missing features (default -1.0).

    Returns:
        List of sample dicts, each containing:
            - patient_id (str)
            - visit_id (str)
            - features (list of 34 floats: mean+std for 17 variables)
            - label (int): 1 = in-hospital death, 0 = survived
    """
    hosp_dir = os.path.join(root, "hosp")
    icu_dir = os.path.join(root, "icu")

    # table load
    admissions = pd.read_csv(
        os.path.join(hosp_dir, "admissions.csv.gz"),
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    icustays = pd.read_csv(
        os.path.join(icu_dir, "icustays.csv.gz"),
        parse_dates=["intime", "outtime"],
    )
    chartevents = pd.read_csv(
        os.path.join(icu_dir, "chartevents.csv.gz"),
        parse_dates=["charttime"],
        usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"],
    )

    # var name look up by itemid
    itemid_to_var: Dict[int, str] = {}
    for var_name, item_ids in VARIABLE_ITEMIDS.items():
        for iid in item_ids:
            # register first occurrence
            if iid not in itemid_to_var:
                itemid_to_var[iid] = var_name

    # chartevents - only keep rows w/ itemids we want. map to var names
    all_itemids = list(itemid_to_var.keys())
    chart = chartevents[chartevents["itemid"].isin(all_itemids)].copy()
    chart["variable"] = chart["itemid"].map(itemid_to_var)

    # merge in ICU stay data for admission times
    chart = chart.merge(
        icustays[["stay_id", "hadm_id", "intime"]],
        on="stay_id",
        how="inner",
    )

    # within first max icu ahours
    chart["hours_from_intime"] = (
        chart["charttime"] - chart["intime"]
    ).dt.total_seconds() / 3600.0
    chart = chart[
        (chart["hours_from_intime"] >= 0)
        & (chart["hours_from_intime"] <= max_icu_hours)
    ]

    # temp conversion
    temp_mask = chart["variable"] == "temperature"
    chart.loc[temp_mask, "valuenum"] = _to_celsius(
        chart.loc[temp_mask, "valuenum"]
    )

    # merge mortaility labels from admissions
    mortality = admissions[["hadm_id", "hospital_expire_flag"]].copy()
    chart = chart.merge(mortality, on="hadm_id", how="inner")

    # compute per-stay mean and std for each variable
    var_names = list(VARIABLE_ITEMIDS.keys())
    samples = []

    for stay_id, stay_df in chart.groupby("stay_id"):
        hadm_id = stay_df["hadm_id"].iloc[0]
        subject_id = stay_df["subject_id"].iloc[0]
        label = int(stay_df["hospital_expire_flag"].iloc[0])

        features = []
        for var in var_names:
            var_vals = stay_df.loc[
                stay_df["variable"] == var, "valuenum"
            ].dropna()

            if len(var_vals) == 0:
                mean_val = unknown_sentinel
                std_val = unknown_sentinel
            elif len(var_vals) == 1:
                mean_val = float(var_vals.iloc[0])
                std_val = unknown_sentinel # std undefined for single obs
            else:
                mean_val = float(var_vals.mean())
                std_val = float(var_vals.std())

            features.append(mean_val)
            features.append(std_val)

        samples.append({
            "patient_id": str(subject_id),
            "visit_id": str(stay_id),
            "features": features,
            "label": label,
        })

    return samples


def MIMIC4ICUMortalityDataset(
    root: str,
    max_icu_hours: float = 48.0,
) -> SampleDataset:
    """Extract mortality prediction samples from MIMIC-IV:

    - Extracts 34 features
        - (mean + std of 17 physiological variables over the
    first 48 ICU hours)
        - missing values filled with unknown_sentinel (-1)
        - temperature values converted to Celsius
        
    - following the preprocessing pipeline from Hegselmann et al. (MLHC 2020).

    Args:
        root: Path to MIMIC-IV root directory (containing hosp/ and icu/).
        max_icu_hours: Hours of ICU data to use per stay (default: 48).
        unknown_sentinel: Value used to impute missing features (default: -1.0).

    Returns:
        List of sample dicts w/:
            - patient_id (str)
            - visit_id (str)
            - features (list[float]) — length 34
            - label (int) — 1 = in-hospital death, 0 = survived
    """
    samples = build_mortality_samples(root=root, max_icu_hours=max_icu_hours)

    # pyhealth needs 2 or more unique labels for bin class tasks
    labels = [s["label"] for s in samples]
    if len(set(labels)) < 2:
        raise ValueError(
            f"Dataset must contain both positive and negative mortality "
            f"labels. Found only: {set(labels)}. Try using a larger subset."
        )

    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"label": "binary"},
        dataset_name="mimic4_icu_mortality",
        task_name="in_hospital_mortality",
        in_memory=True,
    )