"""
PyHealth task for ICU Re-entry classification using the MIMIC-III dataset.

Reproducing and extending: Feature Robustness in Non-Stationary Health Records
(Nestor et al. 2019). This task predicts whether a patient will have an
unplanned return to the ICU within 7 days of their current ICU episode end,
using the first 24 hours of hourly vitals and labs as input features.

A direct transfer (ICU stay beginning within 24 hours of a prior stay) is not
considered a re-entry. Re-entry is defined at the episode level: a patient must
return to the ICU after a gap of > 24 hours and <= 168 hours (7 days) from the
end of their current episode.

The clinical aggregation mapping collapses MIMIC_Extract LEVEL2 columns into
65 expert-defined clinical categories following Nestor et al. 2019. Three
categories from the paper's original 68 (bicarbonate, capillary refill rate,
lactate dehydrogenase) were absent from the available MIMIC_Extract output and
are excluded, resulting in a 65-category feature set.

Reference:
    Nestor et al. (2019). Feature Robustness in Non-Stationary Health Records:
    Caveats to Deployable Model Performance in Common Clinical Machine Learning
    Tasks. Proceedings of the 4th Machine Learning for Healthcare Conference,
    PMLR 106, 381-405.

Author:
    Jenna Reno (jlreno2@illinois.edu)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clinical Aggregation Map — 65 categories
#
# Maps expert-defined clinical category names to one or more LEVEL2 column
# names from MIMIC_Extract output (all_hourly_data.h5). Where multiple LEVEL2
# names map to one category, their hourly values are averaged. This is the
# mechanism that provides robustness across the CareVue (pre-2008) and
# MetaVision (post-2008) EHR systems in MIMIC-III.
#
# All LEVEL2 strings are lowercase and match the HDF5 column MultiIndex
# produced by MIMIC_Extract exactly.
#
# Deviations from Nestor et al. 2019 (original 68 categories):
#   - bicarbonate: absent from available MIMIC_Extract output — excluded
#   - capillary refill rate: absent from available MIMIC_Extract output — excluded
#   - lactate dehydrogenase: absent from available MIMIC_Extract output — excluded
# ---------------------------------------------------------------------------

CLINICAL_AGGREGATION_MAP: Dict[str, List[str]] = {

    # --- Vitals (9) ---
    "heart rate":                               ["heart rate"],
    "systolic blood pressure":                  ["systolic blood pressure"],
    "diastolic blood pressure":                 ["diastolic blood pressure"],
    "mean blood pressure":                      ["mean blood pressure"],
    "respiratory rate":                         ["respiratory rate"],
    "temperature":                              ["temperature"],
    "oxygen saturation":                        ["oxygen saturation"],
    "weight":                                   ["weight"],
    "height":                                   ["height"],

    # --- Neurological (1) ---
    "glascow coma scale total":                 ["glascow coma scale total"],

    # --- Respiratory / Ventilation (13) ---
    # Measured and set values kept separate: they represent distinct clinical
    # quantities (what the patient is doing vs. what the ventilator is told).
    # PaCO2: 'co2 (etco2, pco2, etc.)' and 'co2' merged as proxies.
    "respiratory rate set":                     ["respiratory rate set"],
    "fraction inspired oxygen":                 ["fraction inspired oxygen"],
    "fraction inspired oxygen set":             ["fraction inspired oxygen set"],
    "positive end-expiratory pressure":         ["positive end-expiratory pressure"],
    "positive end-expiratory pressure set":     ["positive end-expiratory pressure set"],
    "tidal volume observed":                    ["tidal volume observed"],
    "tidal volume set":                         ["tidal volume set"],
    "tidal volume spontaneous":                 ["tidal volume spontaneous"],
    "peak inspiratory pressure":                ["peak inspiratory pressure"],
    "plateau pressure":                         ["plateau pressure"],
    "partial pressure of oxygen":               ["partial pressure of oxygen"],
    "partial pressure of carbon dioxide":       ["partial pressure of carbon dioxide",
                                                 "co2 (etco2, pco2, etc.)",
                                                 "co2"],
    "ph":                                       ["ph"],

    # --- Chemistry / Metabolic (13) ---
    # Potassium and potassium serum kept separate: specimen context differs
    # and the two columns come from different EHR systems.
    "sodium":                                   ["sodium"],
    "potassium":                                ["potassium"],
    "potassium serum":                          ["potassium serum"],
    "chloride":                                 ["chloride"],
    "anion gap":                                ["anion gap"],
    "blood urea nitrogen":                      ["blood urea nitrogen"],
    "creatinine":                               ["creatinine"],
    "glucose":                                  ["glucose"],
    "calcium":                                  ["calcium"],
    "calcium ionized":                          ["calcium ionized"],
    "magnesium":                                ["magnesium"],
    "phosphorous":                              ["phosphorous"],
    "lactic acid":                              ["lactic acid"],

    # --- Other Chemistry (2) ---
    "cholesterol":                              ["cholesterol"],
    "total protein":                            ["total protein"],

    # --- Liver (4) ---
    "bilirubin":                                ["bilirubin"],
    "alanine aminotransferase":                 ["alanine aminotransferase"],
    "asparate aminotransferase":                ["asparate aminotransferase"],
    "alkaline phosphate":                       ["alkaline phosphate"],

    # --- Hematology (9) ---
    "white blood cell count":                   ["white blood cell count"],
    "hemoglobin":                               ["hemoglobin"],
    "hematocrit":                               ["hematocrit"],
    "platelets":                                ["platelets"],
    "red blood cell count":                     ["red blood cell count"],
    "partial thromboplastin time":              ["partial thromboplastin time"],
    "prothrombin time inr":                     ["prothrombin time inr"],
    "prothrombin time pt":                      ["prothrombin time pt"],
    "fibrinogen":                               ["fibrinogen"],

    # --- Cardiac / Hemodynamic (12) ---
    # Cardiac output fick and thermodilution kept separate: distinct
    # measurement methods, not EHR-system aliases of the same event.
    "cardiac output fick":                      ["cardiac output fick"],
    "cardiac output thermodilution":            ["cardiac output thermodilution"],
    "cardiac index":                            ["cardiac index"],
    "central venous pressure":                  ["central venous pressure"],
    "pulmonary artery pressure systolic":       ["pulmonary artery pressure systolic"],
    "pulmonary artery pressure mean":           ["pulmonary artery pressure mean"],
    "pulmonary capillary wedge pressure":       ["pulmonary capillary wedge pressure"],
    "troponin-i":                               ["troponin-i"],
    "troponin-t":                               ["troponin-t"],
    "albumin":                                  ["albumin"],
    "systemic vascular resistance":             ["systemic vascular resistance"],
    "venous pvo2":                              ["venous pvo2"],

    # --- Other / Renal (2) ---
    "urine output":                             ["urine output"],
    "post void residual":                       ["post void residual"],
}

# Ordered list of the 65 category names — defines feature ordering in tensors.
CLINICAL_CATEGORIES: List[str] = list(CLINICAL_AGGREGATION_MAP.keys())

assert len(CLINICAL_CATEGORIES) == 65, (
    f"Expected 65 clinical categories, got {len(CLINICAL_CATEGORIES)}. "
    "Check CLINICAL_AGGREGATION_MAP for accidental additions or removals."
)


def apply_clinical_aggregation(
    hourly_data: "pd.DataFrame",
    agg_func: str = "mean",
    n_hours: int = 24,
) -> "tuple[np.ndarray, List[str], List[int]]":
    """
    Applies clinical aggregation to MIMIC_Extract hourly output.

    Collapses LEVEL2 feature columns into 65 expert-defined clinical categories
    following Nestor et al. 2019. Where multiple LEVEL2 columns map to one
    category, their values are averaged per hour.

    Args:
        hourly_data: DataFrame loaded from all_hourly_data.h5 'vitals_labs'.
                     MultiIndex columns: (LEVEL2, Aggregation Function).
                     MultiIndex rows: (subject_id, hadm_id, icustay_id,
                     hours_in).
        agg_func: Aggregation function to select from column MultiIndex.
                  Default: 'mean'.
        n_hours: Number of hourly windows per stay. Default: 24.

    Returns:
        Tuple of:
        - clinical_array: np.ndarray of shape (n_stays, n_hours, 65).
        - category_names: List of 65 category name strings.
        - stay_ids: List of icustay_id integers.

    Examples:
        >>> import pandas as pd
        >>> data = pd.read_hdf("all_hourly_data.h5", "vitals_labs")
        >>> arr, cats, stays = apply_clinical_aggregation(data)
        >>> arr.shape
        (n_stays, 24, 65)
    """
    import pandas as pd

    if isinstance(hourly_data.columns, pd.MultiIndex):
        df = hourly_data.xs(agg_func, axis=1, level="Aggregation Function")
    else:
        df = hourly_data.copy()

    available = set(df.columns.tolist())
    category_names = list(CLINICAL_AGGREGATION_MAP.keys())
    n_categories = len(category_names)

    # Build one column per clinical category
    clinical_cols = {}
    for category, source_cols in CLINICAL_AGGREGATION_MAP.items():
        present = [c for c in source_cols if c in available]
        if present:
            clinical_cols[category] = df[present].mean(axis=1)
        else:
            clinical_cols[category] = pd.Series(
                np.nan, index=df.index, dtype=np.float32
            )
    clinical_df = pd.DataFrame(clinical_cols, index=df.index)

    # Reshape into (n_stays, n_hours, n_categories)
    stay_ids = clinical_df.index.get_level_values("icustay_id").unique().tolist()
    n_stays = len(stay_ids)
    clinical_array = np.full(
        (n_stays, n_hours, n_categories), fill_value=np.nan, dtype=np.float32
    )

    for i, stay_id in enumerate(stay_ids):
        stay_data = clinical_df.xs(stay_id, level="icustay_id").reset_index()
        hours_col = "hours_in" if "hours_in" in stay_data.columns \
            else stay_data.columns[0]
        for _, row in stay_data.iterrows():
            try:
                hour = int(row[hours_col])
            except (TypeError, ValueError):
                continue
            if 0 <= hour < n_hours:
                clinical_array[i, hour, :] = row[category_names].values.astype(
                    np.float32
                )

    return clinical_array, category_names, stay_ids


