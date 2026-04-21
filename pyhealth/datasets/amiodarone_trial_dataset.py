"""Amiodarone Clinical Trial Dataset for PyHealth.

Dataset for the amiodarone case study from Kaul & Gordon (2024).
Amiodarone is assessed for its effectiveness in converting atrial
fibrillation (AF) to normal sinus rhythm.

The embedded AMIODARONE_RAW_TRIALS data below was extracted from the
21 trials reviewed in Letelier et al. (2003). Feature values were
extracted using an LLM following the prompt in Figure 5 of
Kaul & Gordon (2024). Numeric conversion follows the rules in
Figure 6 of the same paper.

Data split:
    - 11 non-placebo trials ("untrusted"): training data for the
      prior encoder
    - 10 placebo-controlled trials ("trusted"): used by the CMA model
      for conformal meta-analysis

Mapping to PyHealth's Patient-Visit-Event structure:
    - Patient = one clinical trial
    - Visit   = single aggregated observation from that trial
    - Event   = the trial's features, computed effect, and variance

References:
    Kaul, S.; and Gordon, G. J. 2024. Meta-Analysis with Untrusted Data.
    Proceedings of Machine Learning Research, 259:563-593.

    Letelier, L. M., Udol, K., Ena, J., Weaver, B., and Guyatt, G. H.
    2003. Effectiveness of amiodarone for conversion of atrial
    fibrillation to sinus rhythm: a meta-analysis. Archives of Internal
    Medicine, 163(7):777-785.

Examples:
    >>> from pyhealth.datasets import AmiodaroneTrialDataset
    >>> dataset = AmiodaroneTrialDataset(root="./data/amiodarone")
    >>> samples = dataset.set_task()
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Raw trial data (extracted via LLM per Figures 5-7 of the paper)
# ---------------------------------------------------------------------
AMIODARONE_RAW_TRIALS = [
    {
        "Name": "Cowan et al.16 (England) 1986",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 7 mg/kg in 30 min + 1000 mg in 23 h",
            "Comparison Treatment": "IV digoxin, 0.5 mg in 30 min twice, 30 min apart",
            "Time to Outcome Measure": "24 h",
            "Number of Amiodarone Patients": 18,
            "Number of Control Patients": 16,
            "Fraction with CV Disease": 76,
            "Mean Left Atrium Size, mm": "NA",
            "Mean AF Duration": "NA",
            "Mean Age": 68,
            "Fraction Male": "NA",
            "Adequate Concealment of Treatment": "Yes",
            "Follow-up Fraction": 100,
            "Masked Patients": "No",
            "Masked Caregiver": "no",
            "Masked Assessor": "no",
        },
        "Results": [5, 18, 4, 16],
    },
    {
        "Name": "Noc et al.17 (Slovenia) 1990",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 5 mg/kg in 3 min",
            "Comparison Treatment": "IV verapamil hydrochloride, 0.075 mg/kg in 1 min, repeated after 10 min",
            "Time to Outcome Measure": "3 h",
            "Number of Amiodarone Patients": 13,
            "Number of Control Patients": 11,
            "Fraction with CV Disease": "NA",
            "Mean Left Atrium Size, mm": "NA",
            "Mean AF Duration": "20 min to 48 h",
            "Mean Age": 71,
            "Fraction Male": 63,
            "Adequate Concealment of Treatment": "No",
            "Follow-up Fraction": 100,
            "Masked Patients": "Yes",
            "Masked Caregiver": "no",
            "Masked Assessor": "no",
        },
        "Results": [10, 13, 0, 11],
    },
    {
        "Name": "Capucci et al.18 (Italy) 1992",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 5 mg/kg in 5 min + 1.8 g in 24 h",
            "Comparison Treatment": "Placebo",
            "Time to Outcome Measure": "3 and 8 h",
            "Number of Amiodarone Patients": 19,
            "Number of Control Patients": 21,
            "Fraction with CV Disease": 31,
            "Mean Left Atrium Size, mm": 46,
            "Mean AF Duration": "28 h",
            "Mean Age": 58,
            "Fraction Male": 56,
            "Adequate Concealment of Treatment": "NA",
            "Follow-up Fraction": 100,
            "Masked Patients": "Yes",
            "Masked Caregiver": "no",
            "Masked Assessor": "no",
        },
        "Results": [7, 19, 10, 21],
    },
    {
        "Name": "Cochrane et al.19 (Australia) 1994",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 5 mg/kg in 30 min + 25-40 mg/h in 24 h",
            "Comparison Treatment": "IV digoxin, 0.5 mg in 30 min + 0.25 mg at 2 h + 0.125 mg at 5 h + 0.125 mg at 9 h",
            "Time to Outcome Measure": "24 h",
            "Number of Amiodarone Patients": 15,
            "Number of Control Patients": 15,
            "Fraction with CV Disease": 100,
            "Mean Left Atrium Size, mm": "NA",
            "Mean AF Duration": "1 h",
            "Mean Age": 63,
            "Fraction Male": 70,
            "Adequate Concealment of Treatment": "No",
            "Follow-up Fraction": 100,
            "Masked Patients": "Yes",
            "Masked Caregiver": "no",
            "Masked Assessor": "no",
        },
        "Results": [15, 15, 13, 15],
    },
    {
        "Name": "Donovan et al.20 (Australia) 1995",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 7 mg/kg in 30 min",
            "Comparison Treatment": "Placebo",
            "Time to Outcome Measure": "2 and 8 h",
            "Number of Amiodarone Patients": 32,
            "Number of Control Patients": 32,
            "Fraction with CV Disease": 65,
            "Mean Left Atrium Size, mm": "NA",
            "Mean AF Duration": "10 h",
            "Mean Age": 58,
            "Fraction Male": "NA",
            "Adequate Concealment of Treatment": "Yes",
            "Follow-up Fraction": 100,
            "Masked Patients": "Yes",
            "Masked Caregiver": "yes",
            "Masked Assessor": "yes",
        },
        "Results": [21, 32, 20, 32],
    },
    {
        "Name": "Hou et al.21 (Taiwan) 1995",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 5 mg/min for 1 h + 3 mg/min for 3 h + 1 mg/min for 6 h + 0.5 mg/min"
                                           " for 14 h",
            "Comparison Treatment": "IV digoxin, 0.0043 mg/kg in 30 min every 2 h for 3 dosages",
            "Time to Outcome Measure": "24 h",
            "Number of Amiodarone Patients": 20,
            "Number of Control Patients": 19,
            "Fraction with CV Disease": 62,
            "Mean Left Atrium Size, mm": 48,
            "Mean AF Duration": "9 h",
            "Mean Age": 70,
            "Fraction Male": 86,
            "Adequate Concealment of Treatment": "NA",
            "Follow-up Fraction": 100,
            "Masked Patients": "No",
            "Masked Caregiver": "no",
            "Masked Assessor": "no",
        },
        "Results": [19, 20, 14, 19],
    },
    {
        "Name": "Kondili et al.22 (Albania) 1995",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 300 mg + 900 mg in 24 h",
            "Comparison Treatment": "IV verapamil, 2 bolus of 5 mg each in 30 min",
            "Time to Outcome Measure": "3, 6, and 12 h",
            "Number of Amiodarone Patients": 21,
            "Number of Control Patients": 21,
            "Fraction with CV Disease": 72,
            "Mean Left Atrium Size, mm": 31,
            "Mean AF Duration": "30 h",
            "Mean Age": "NA",
            "Fraction Male": "NA",
            "Adequate Concealment of Treatment": "No",
            "Follow-up Fraction": 100,
            "Masked Patients": "Yes",
            "Masked Caregiver": "no",
            "Masked Assessor": "no",
        },
        "Results": [4, 21, 3, 21],
    },
    {
        "Name": "Galve et al.23 (Spain) 1996",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 5 mg/kg in 30 min + 1.2 g in 24 h",
            "Comparison Treatment": "Placebo",
            "Time to Outcome Measure": "2, 6, and 12 h",
            "Number of Amiodarone Patients": 50,
            "Number of Control Patients": 50,
            "Fraction with CV Disease": 52,
            "Mean Left Atrium Size, mm": 42,
            "Mean AF Duration": "21 h",
            "Mean Age": 61,
            "Fraction Male": 55,
            "Adequate Concealment of Treatment": "Yes",
            "Follow-up Fraction": 100,
            "Masked Patients": "NA",
            "Masked Caregiver": "yes",
            "Masked Assessor": "no",
        },
        "Results": [26, 50, 23, 50],
    },
    {
        "Name": "Kontoyannis et al.24 (Greece) 2001",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 300 mg in 2 h + 44 mg/h in 22 h",
            "Comparison Treatment": "IV digoxin, 0.5 mg bolus + 0.25 mg 1 h later + PRN",
            "Time to Outcome Measure": "2, 8, and 96 h",
            "Number of Amiodarone Patients": 16,
            "Number of Control Patients": 26,
            "Fraction with CV Disease": 100,
            "Mean Left Atrium Size, mm": 43,
            "Mean AF Duration": "30 min",
            "Mean Age": "NA",
            "Fraction Male": "NA",
            "Adequate Concealment of Treatment": "NA",
            "Follow-up Fraction": 100,
            "Masked Patients": "Yes",
            "Masked Caregiver": "no",
            "Masked Assessor": "no",
        },
        "Results": [7, 16, 8, 26],
    },
    {
        "Name": "Bellandi et al.26 (Italy) 1999",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 120 mg/h for 24 h",
            "Comparison Treatment": "Placebo",
            "Time to Outcome Measure": "24 h",
            "Number of Amiodarone Patients": 60,
            "Number of Control Patients": 60,
            "Fraction with CV Disease": "NA",
            "Mean Left Atrium Size, mm": "NA",
            "Mean AF Duration": "48 h",
            "Mean Age": "NA",
            "Fraction Male": "NA",
            "Adequate Concealment of Treatment": "NA",
            "Follow-up Fraction": 100,
            "Masked Patients": "NA",
            "Masked Caregiver": "NA",
            "Masked Assessor": "NA",
        },
        "Results": [55, 60, 39, 60],
    },
    {
        "Name": "Cotter et al.27 (Israel) 1999",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 125 mg/h for 24 h",
            "Comparison Treatment": "Placebo",
            "Time to Outcome Measure": "24 h",
            "Number of Amiodarone Patients": 50,
            "Number of Control Patients": 50,
            "Fraction with CV Disease": 67,
            "Mean Left Atrium Size, mm": 45,
            "Mean AF Duration": "10 h",
            "Mean Age": 66,
            "Fraction Male": 43,
            "Adequate Concealment of Treatment": "NA",
            "Follow-up Fraction": 100,
            "Masked Patients": "Yes",
            "Masked Caregiver": "NA",
            "Masked Assessor": "NA",
        },
        "Results": [10, 50, 7, 50],
    },
    {
        "Name": "Kochiadakis et al.12 (Greece) 1999",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 300 mg in 1 h + 20 mg/kg in 1 d + 15 mg/kg in 1 d or oral 500 mg "
                                           "4 times for 1 d + 200 mg 4 times for 1 d",
            "Comparison Treatment": "Placebo",
            "Time to Outcome Measure": "48 h",
            "Number of Amiodarone Patients": 135,
            "Number of Control Patients": 69,
            "Fraction with CV Disease": 60,
            "Mean Left Atrium Size, mm": 42,
            "Mean AF Duration": "13 h",
            "Mean Age": 65,
            "Fraction Male": 47,
            "Adequate Concealment of Treatment": "Yes",
            "Follow-up Fraction": 100,
            "Masked Patients": "Yes",
            "Masked Caregiver": "no",
            "Masked Assessor": "no",
        },
        "Results": [20, 135, 7, 69],
    },
    {
        "Name": "Peuhkurinen et al.30 (Finland) 2000",
        "Features": {
            "Amiodarone Therapy Protocol": "Oral, 30 mg/kg single dose",
            "Comparison Treatment": "Placebo",
            "Time to Outcome Measure": "24 h",
            "Number of Amiodarone Patients": 31,
            "Number of Control Patients": 31,
            "Fraction with CV Disease": 74,
            "Mean Left Atrium Size, mm": 39,
            "Mean AF Duration": "3 to 48 h",
            "Mean Age": 59,
            "Fraction Male": 73,
            "Adequate Concealment of Treatment": "Yes",
            "Follow-up Fraction": 86,
            "Masked Patients": "Yes",
            "Masked Caregiver": "yes",
            "Masked Assessor": "yes",
        },
        "Results": [27, 31, 11, 31],
    },
    {
        "Name": "Vardas et al.31 (Greece) 2000",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 300 mg in 1 h + 20 mg/kg in 24 h + oral 600 mg/d for 1 wk + 400 mg/d "
                                           "for 3 wk",
            "Comparison Treatment": "Placebo",
            "Time to Outcome Measure": "1 and 24 h, 28 d",
            "Number of Amiodarone Patients": 108,
            "Number of Control Patients": 100,
            "Fraction with CV Disease": 43,
            "Mean Left Atrium Size, mm": 43,
            "Mean AF Duration": "26 h",
            "Mean Age": 65,
            "Fraction Male": 49,
            "Adequate Concealment of Treatment": "Yes",
            "Follow-up Fraction": 100,
            "Masked Patients": "Yes",
            "Masked Caregiver": "no",
            "Masked Assessor": "no",
        },
        "Results": [89, 108, 41, 100],
    },
    {
        "Name": "Joseph and Ward32 (Australia) 2000",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 5 mg/kg in 30 min + oral, 400 mg 3 times per day for 2 d",
            "Comparison Treatment": "IV digoxin, 0.5 mg in 30 min + oral 0.25 mg every 6 h for 24 h + 0.25 mg/d",
            "Time to Outcome Measure": "4, 24, and 48 h",
            "Number of Amiodarone Patients": 39,
            "Number of Control Patients": 36,
            "Fraction with CV Disease": 46,
            "Mean Left Atrium Size, mm": 39,
            "Mean AF Duration": "24 h",
            "Mean Age": 63,
            "Fraction Male": 56,
            "Adequate Concealment of Treatment": "No",
            "Follow-up Fraction": 96,
            "Masked Patients": "No",
            "Masked Caregiver": "no",
            "Masked Assessor": "no",
        },
        "Results": [30, 39, 21, 36],
    },
    {
        "Name": "Cybulski et al.33 (Poland) 2001",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 5 mg/kg in 30 min + 10 mg/kg in 20 h",
            "Comparison Treatment": "Control group",
            "Time to Outcome Measure": "20 h",
            "Number of Amiodarone Patients": 106,
            "Number of Control Patients": 54,
            "Fraction with CV Disease": 92,
            "Mean Left Atrium Size, mm": 41,
            "Mean AF Duration": "18 h",
            "Mean Age": "NA",
            "Fraction Male": "NA",
            "Adequate Concealment of Treatment": "Yes",
            "Follow-up Fraction": 100,
            "Masked Patients": "Yes",
            "Masked Caregiver": "no",
            "Masked Assessor": "no",
        },
        "Results": [11, 106, 3, 54],
    },
    {
        "Name": "Natale et al.25 (United States) 1998",
        "Features": {
            "Amiodarone Therapy Protocol": "IV, 600 mg in 20 min + 66 mg/h",
            "Comparison Treatment": "IV diltiazem hydrochloride",
            "Time to Outcome Measure": "12 h",
            "Number of Amiodarone Patients": 42,
            "Number of Control Patients": 43,
            "Fraction with CV Disease": "NA",
            "Mean Left Atrium Size, mm": "NA",
            "Mean AF Duration": "48 h",
            "Mean Age": "NA",
            "Fraction Male": "NA",
            "Adequate Concealment of Treatment": "NA",
            "Follow-up Fraction": 100,
            "Masked Patients": "NA",
            "Masked Caregiver": "NA",
            "Masked Assessor": "NA",
        },
        "Results": [15, 42, 3, 43],
    },
    {
        "Name": "Bianconi et al.28 (Italy) 2000",
        "Features": {
            "Amiodarone Therapy Protocol": "IV 5 mg/kg in 15 min",
            "Comparison Treatment": "Placebo",
            "Time to Outcome Measure": "3 h",
            "Number of Amiodarone Patients": 41,
            "Number of Control Patients": 42,
            "Fraction with CV Disease": 73,
            "Mean Left Atrium Size, mm": 44,
            "Mean AF Duration": "7 d",
            "Mean Age": 64,
            "Fraction Male": 56,
            "Adequate Concealment of Treatment": "Yes",
            "Follow-up Fraction": 94,
            "Masked Patients": "Yes",
            "Masked Caregiver": "no",
            "Masked Assessor": "no",
        },
        "Results": [2, 41, 1, 42],
    },
    {
        "Name": "Galperin et al.29 (Argentina) 2000",
        "Features": {
            "Amiodarone Therapy Protocol": "Oral, 600 mg/d for 4 wk",
            "Comparison Treatment": "Placebo",
            "Time to Outcome Measure": "28 d",
            "Number of Amiodarone Patients": 47,
            "Number of Control Patients": 48,
            "Fraction with CV Disease": 94,
            "Mean Left Atrium Size, mm": 48,
            "Mean AF Duration": "35 mo",
            "Mean Age": 63,
            "Fraction Male": 73,
            "Adequate Concealment of Treatment": "Yes",
            "Follow-up Fraction": 100,
            "Masked Patients": "Yes",
            "Masked Caregiver": "yes",
            "Masked Assessor": "yes",
        },
        "Results": [33, 47, 1, 48],
    },
    {
        "Name": "Hohnloser et al.3 (Germany) 2000",
        "Features": {
            "Amiodarone Therapy Protocol": "Oral, 600 mg/d for 3 wk",
            "Comparison Treatment": "Oral diltiazem hydrochloride 180-270 mg/d for 3 wk",
            "Time to Outcome Measure": "3 wk",
            "Number of Amiodarone Patients": 95,
            "Number of Control Patients": 108,
            "Fraction with CV Disease": 85,
            "Mean Left Atrium Size, mm": 46,
            "Mean AF Duration": "16 wk",
            "Mean Age": 61,
            "Fraction Male": 73,
            "Adequate Concealment of Treatment": "NA",
            "Follow-up Fraction": 81,
            "Masked Patients": "No",
            "Masked Caregiver": "no",
            "Masked Assessor": "no",
        },
        "Results": [33, 95, 12, 108],
    },
    {
        "Name": "Villani et al.11 (Italy) 2000",
        "Features": {
            "Amiodarone Therapy Protocol": "Oral, 400 mg/d for 1 mo",
            "Comparison Treatment": "Oral digoxin, 0.25 mg/d or oral diltiazem hydrochloride 180-360 mg/d for 1 mo",
            "Time to Outcome Measure": "1 mo",
            "Number of Amiodarone Patients": 44,
            "Number of Control Patients": 76,
            "Fraction with CV Disease": 47,
            "Mean Left Atrium Size, mm": 50,
            "Mean AF Duration": "17 wk",
            "Mean Age": 58,
            "Fraction Male": 67,
            "Adequate Concealment of Treatment": "No",
            "Follow-up Fraction": 100,
            "Masked Patients": "Yes",
            "Masked Caregiver": "no",
            "Masked Assessor": "no",
        },
        "Results": [11, 44, 4, 76],
    },
]


# ---------------------------------------------------------------------
# Manually computed total amiodarone dose (mg) in first 24 hours.
# The paper's LLM-generated parser (Figure 7) had bugs, so these were
# hand-computed from each trial's protocol. Using AVG_WEIGHT_KG = 70.
# ---------------------------------------------------------------------
MANUAL_AMIODARONE_DOSE_MG = {
    "Cowan et al.16 (England) 1986": 1490.0,
    "Noc et al.17 (Slovenia) 1990": 350.0,
    "Capucci et al.18 (Italy) 1992": 2150.0,
    "Cochrane et al.19 (Australia) 1994": 1130.0,
    "Donovan et al.20 (Australia) 1995": 490.0,
    "Hou et al.21 (Taiwan) 1995": 1620.0,
    "Kondili et al.22 (Albania) 1995": 1200.0,
    "Galve et al.23 (Spain) 1996": 1550.0,
    "Kontoyannis et al.24 (Greece) 2001": 1268.0,
    "Bellandi et al.26 (Italy) 1999": 2880.0,
    "Cotter et al.27 (Israel) 1999": 3000.0,
    "Kochiadakis et al.12 (Greece) 1999": 1700.0,
    "Peuhkurinen et al.30 (Finland) 2000": 2100.0,
    "Vardas et al.31 (Greece) 2000": 1700.0,
    "Joseph and Ward32 (Australia) 2000": 1550.0,
    "Cybulski et al.33 (Poland) 2001": 1050.0,
    "Natale et al.25 (United States) 1998": 2162.0,
    "Bianconi et al.28 (Italy) 2000": 350.0,
    "Galperin et al.29 (Argentina) 2000": 600.0,
    "Hohnloser et al.3 (Germany) 2000": 600.0,
    "Villani et al.11 (Italy) 2000": 400.0,
}


# ---------------------------------------------------------------------
# Feature conversion helpers (Figure 6 of the paper)
# ---------------------------------------------------------------------
def _parse_comparison_treatment(treatment: str) -> float:
    """Map a free-text comparison treatment description to [0, 1].

    Implements the ``comparison_intensity`` feature described in
    Figure 6 of Kaul & Gordon (2024). The value encodes how
    aggressive the control arm is, with higher values representing
    more intensive comparators:

        - 0.0 if the comparator is a placebo or untreated control
        - 0.75 if an intravenous antiarrhythmic (diltiazem,
          verapamil, digoxin, procainamide) is used
        - 0.5 otherwise (missing, single oral drug, or any other
          case including a detected antiarrhythmic without an IV
          route)

    Args:
        treatment: Raw free-text description of the comparison
            treatment from the trial's Features dictionary. May be
            empty, None, or a missing-value sentinel.

    Returns:
        Intensity score in [0, 1]. Defaults to 0.5 when the input
        is empty or no rule matches, so the feature is never NA.
    """
    if not treatment:
        return 0.5
    t = treatment.lower().strip()
    if "placebo" in t or ("control" in t and "group" in t):
        return 0.0

    intensive_drugs = ["diltiazem", "verapamil", "digoxin", "procainamide"]
    is_iv = "iv" in t or "intravenous" in t
    drug_hits = sum(1 for d in intensive_drugs if d in t)

    if is_iv and drug_hits >= 1:
        return 0.75
    if drug_hits >= 1:
        return 0.5
    return 0.5


def _parse_duration_48h(duration: Any) -> float:
    """Threshold a duration expression at the 48-hour mark.

    Parses a free-text duration like ``"28 h"``, ``"7 d"``,
    ``"16 wk"``, or ``"35 mo"``, converts the first numeric value
    to hours, and compares against 48. Used for both
    ``af_duration_gt_48h`` and ``outcome_time_gt_48h`` per Figure 6
    of Kaul & Gordon (2024).

    Args:
        duration: Duration string with a unit suffix (min, h, d,
            wk, mo, y). May be None, a numeric value, ``"NA"``, or
            contain a range (e.g. ``"20 min to 48 h"``); only the
            first numeric token is used.

    Returns:
        ``-1.0`` if the parsed duration is <= 48 hours,
        ``+1.0`` if > 48 hours, and ``0.0`` when the input is
        missing or cannot be parsed.
    """
    if duration is None:
        return 0.0
    s = str(duration).strip().lower()
    if s in ("", "na", "none"):
        return 0.0
    m = re.search(r"(\d+\.?\d*)\s*(min|h|d|wk|mo|y)", s)
    if not m:
        return 0.0
    amount = float(m.group(1))
    unit = m.group(2)
    to_hours = {
        "min": 1 / 60, "h": 1.0, "d": 24.0,
        "wk": 24.0 * 7, "mo": 24.0 * 30, "y": 24.0 * 365,
    }
    return -1.0 if amount * to_hours.get(unit, 0) <= 48 else 1.0


def _to_float_or_na(value: Any) -> Optional[float]:
    """Convert a value to ``float``, preserving missingness.

    Used as a shared coercion step before domain-specific rescaling
    of continuous features (``mean_age``, ``mean_la_size``) so that
    downstream code can distinguish "unparseable or NA" from a
    legitimate zero.

    Args:
        value: Input value, which may already be numeric, a numeric
            string, None, or one of the common missing-value
            sentinels (``"NA"``, ``""``, ``"None"``, any casing).

    Returns:
        The value as a ``float``, or ``None`` if the input is
        missing or cannot be parsed as a number.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s.upper() in ("NA", "", "NONE"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_boolean(value: Any) -> float:
    """Convert a Yes/No/NA string to a signed ternary encoding.

    Handles the binary risk-of-bias features from Figure 6 of
    Kaul & Gordon (2024): ``adequate_concealment``,
    ``masked_patients``, ``masked_caregiver``, and
    ``masked_assessor``. The ternary encoding lets the model
    distinguish a positive finding from an explicit negative and
    from missing information, without introducing a separate
    indicator column.

    Args:
        value: Input value. Accepted truthy strings are ``"yes"``
            and ``"true"``; falsy strings are ``"no"`` and
            ``"false"`` (case-insensitive). Anything else — None,
            ``"NA"``, unknown — is treated as missing.

    Returns:
        ``1.0`` for truthy inputs, ``-1.0`` for falsy, and
        ``0.0`` for missing or unrecognized values.
    """
    if value is None:
        return 0.0
    s = str(value).strip().lower()
    if s in ("yes", "true"):
        return 1.0
    if s in ("no", "false"):
        return -1.0
    return 0.0


def _parse_percent(value: Any) -> float:
    """Normalize a percentage value into the [0, 1] interval.

    The raw data mixes percent-scale values (e.g. ``76`` meaning
    76%) with fraction-scale values (e.g. ``0.76``). This helper
    accepts either and returns a fraction, imputing 0.5 for
    missing inputs so the feature stays on the same scale as the
    other continuous features after rescaling.

    Args:
        value: Input value. May be numeric or string, on a percent
            scale (0-100) or fraction scale (0-1). Values greater
            than 1 are divided by 100.

    Returns:
        Fraction in [0, 1]. Returns ``0.5`` when ``value`` cannot
        be parsed, so the feature vector never contains NaN.
    """
    v = _to_float_or_na(value)
    if v is None:
        return 0.5
    return v / 100.0 if v > 1.0 else v


def _compute_log_relative_risk(
    events_treat: int, n_treat: int,
    events_ctrl: int, n_ctrl: int,
) -> Dict[str, float]:
    """Compute the log relative risk and its variance.

    The relative risk is ``(events_treat / n_treat) /
    (events_ctrl / n_ctrl)``; its log is the target effect size
    used by the conformal meta-analysis task. When either arm has
    zero events the standard Haldane-Anscombe correction (adding
    0.5 to event counts and 1.0 to sample sizes) is applied to
    keep the log finite and the variance positive.

    Args:
        events_treat: Number of successful events in the amiodarone
            arm.
        n_treat: Total patients in the amiodarone arm.
        events_ctrl: Number of successful events in the control arm.
        n_ctrl: Total patients in the control arm.

    Returns:
        Dictionary with two keys:

            - ``log_relative_risk``: ``log(p_treat / p_ctrl)``.
            - ``variance``: approximate sampling variance of the
              log relative risk, ``(1 - p_treat) / events_treat +
              (1 - p_ctrl) / events_ctrl``, used as the within-
              trial variance V in the meta-analysis.
    """
    e_t = float(events_treat)
    n_t = float(n_treat)
    e_c = float(events_ctrl)
    n_c = float(n_ctrl)
    if e_t == 0 or e_c == 0:
        e_t += 0.5
        n_t += 1.0
        e_c += 0.5
        n_c += 1.0
    p_t = e_t / n_t
    p_c = e_c / n_c
    return {
        "log_relative_risk": float(np.log(p_t / p_c)),
        "variance": float((1.0 - p_t) / e_t + (1.0 - p_c) / e_c),
    }


# ---------------------------------------------------------------------
# The dataset class
# ---------------------------------------------------------------------
FEATURE_COLUMNS = [
    "amiodarone_total_24h_mg",
    "comparison_intensity",
    "af_duration_gt_48h",
    "outcome_time_gt_48h",
    "mean_age",
    "mean_la_size",
    "fraction_male",
    "fraction_cv_disease",
    "followup_fraction",
    "adequate_concealment",
    "masked_patients",
    "masked_caregiver",
    "masked_assessor",
]


class AmiodaroneTrialDataset(BaseDataset):
    """Amiodarone clinical trial dataset for conformal meta-analysis.

    All trial data is bundled inside this module (AMIODARONE_RAW_TRIALS).
    No external download or data file is required. The dataset processes
    the embedded raw trials into a numeric CSV on first use.

    Args:
        root: Directory where the processed CSV will be stored.
        dataset_name: Optional name override. Defaults to
            "amiodarone_trials".
        config_path: Optional path to the YAML config. If None,
            uses the default in the configs directory.
        cache_dir: Optional cache directory.
        num_workers: Parallel workers. Defaults to 1.
        dev: Load only a small subset for development if True.

    Examples:
        >>> dataset = AmiodaroneTrialDataset(root="./data/amiodarone")
        >>> print(len(dataset.unique_patient_ids))  # 21
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        # Sanity-check the embedded trial data. These are structural
        # invariants baked into the paper's design (21 trials split
        # into 10 placebo-controlled "trusted" and 11 "untrusted"),
        # so failing fast here surfaces any accidental edit to
        # AMIODARONE_RAW_TRIALS before expensive processing.
        if len(AMIODARONE_RAW_TRIALS) != 21:
            raise ValueError(
                f"AMIODARONE_RAW_TRIALS must contain 21 trials, "
                f"got {len(AMIODARONE_RAW_TRIALS)}."
            )
        n_placebo = sum(
            1 for t in AMIODARONE_RAW_TRIALS
            if str(t["Features"].get("Comparison Treatment", ""))
            .strip().lower() == "placebo"
        )
        if n_placebo != 10:
            raise ValueError(
                f"Expected 10 placebo-controlled (trusted) trials, "
                f"found {n_placebo}. Check AMIODARONE_RAW_TRIALS "
                f"for edits to the 'Comparison Treatment' field."
            )
        missing_doses = [
            t["Name"] for t in AMIODARONE_RAW_TRIALS
            if t["Name"] not in MANUAL_AMIODARONE_DOSE_MG
        ]
        if missing_doses:
            raise ValueError(
                f"MANUAL_AMIODARONE_DOSE_MG is missing entries for "
                f"{len(missing_doses)} trial(s): {missing_doses}."
            )

        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = (
                Path(__file__).parent / "configs" / "amiodarone_trials.yaml"
            )

        csv_name = "amiodarone_trials-metadata-pyhealth.csv"
        if not os.path.exists(os.path.join(root, csv_name)):
            self.prepare_metadata(root)

        default_tables = ["amiodarone_trials"]

        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "amiodarone_trials",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Convert embedded raw trials to a PyHealth-compatible CSV.

        Applies feature conversion rules from Figure 6 of the paper,
        computes log relative risk and variance from event counts,
        and splits trials into "trusted" (placebo-controlled) and
        "untrusted" (other comparisons) groups.

        Args:
            root: Directory to save the processed CSV.
        """
        rows = []
        for raw in AMIODARONE_RAW_TRIALS:
            rows.append(_convert_trial(raw))
        df = pd.DataFrame(rows)
        df = _rescale_continuous(df)

        # Compute effects
        effects = df.apply(
            lambda r: _compute_log_relative_risk(
                r["events_amiodarone"], r["n_amiodarone"],
                r["events_control"], r["n_control"],
            ),
            axis=1,
            result_type="expand",
        )
        df = pd.concat([df, effects], axis=1)

        df["split"] = df["placebo_controlled"].apply(
            lambda p: "trusted" if p else "untrusted"
        )
        df.insert(0, "patient_id", df["trial_name"].apply(
            lambda n: re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_")
        ))
        df.insert(1, "visit_id", df["patient_id"].apply(
            lambda p: f"visit_{p}"
        ))

        os.makedirs(root, exist_ok=True)
        csv_path = os.path.join(
            root, "amiodarone_trials-metadata-pyhealth.csv"
        )
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved amiodarone trial metadata to {csv_path}")

    @property
    def default_task(self) -> "ConformalMetaAnalysisTask":
        """Returns the default task for this dataset.

        Configured for the amiodarone case study:
            - Target and observed effect are both ``log_relative_risk``
              (Task 2 in the paper: u is unverifiable, so the observed
              effect Y is used as the evaluation target).
            - ``prior_column=None`` because no prior exists in the raw
              data. The CMAPriorEncoder should be trained on the
              untrusted split first; its predictions are then written
              into the data as a ``prior_mean`` column, and the task
              is re-instantiated with ``prior_column="prior_mean"``.
            - Only the 10 placebo-controlled ("trusted") trials are
              emitted as samples. The 11 non-placebo trials are
              reserved for training the prior encoder.
        """
        from pyhealth.tasks.conformal_meta_analysis import (
            ConformalMetaAnalysisTask,
        )
        return ConformalMetaAnalysisTask(
            feature_columns=FEATURE_COLUMNS,
            target_column="log_relative_risk",
            observed_column="log_relative_risk",
            variance_column="variance",
            prior_column=None,
            split_column="split",
            split_value="trusted",
        )


# ---------------------------------------------------------------------
# Per-trial conversion used by prepare_metadata
# ---------------------------------------------------------------------
def _convert_trial(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one raw trial entry into a numeric feature dictionary.

    Applies all per-feature conversion rules from Figure 6 of
    Kaul & Gordon (2024): booleans to signed ternary, percentages
    to [0, 1], durations to 48 h thresholds, and comparison
    treatment to an intensity score. Continuous features
    (``mean_age``, ``mean_la_size``) are emitted as
    ``..._raw`` columns here and rescaled later by
    :func:`_rescale_continuous` once the dataset-wide min/max are
    known. The manually curated total amiodarone dose comes from
    :data:`MANUAL_AMIODARONE_DOSE_MG`, which overrides the paper's
    buggy LLM parser (Figure 7).

    Args:
        raw: One entry from :data:`AMIODARONE_RAW_TRIALS` with keys
            ``"Name"``, ``"Features"``, and ``"Results"``. The
            ``"Results"`` value is a four-tuple
            ``[events_treat, n_treat, events_ctrl, n_ctrl]``.

    Returns:
        Dictionary containing the trial name, all converted feature
        values, raw event counts, and a ``placebo_controlled``
        boolean used to assign the trusted/untrusted split. Ready
        to be collected into a DataFrame.
    """
    f = raw["Features"]
    results = raw["Results"]
    return {
        "trial_name": raw["Name"],
        "events_amiodarone": int(results[0]),
        "n_amiodarone": int(results[1]),
        "events_control": int(results[2]),
        "n_control": int(results[3]),
        "amiodarone_total_24h_mg": MANUAL_AMIODARONE_DOSE_MG.get(
            raw["Name"], 0.0
        ),
        "comparison_intensity": _parse_comparison_treatment(
            f.get("Comparison Treatment", "")
        ),
        "af_duration_gt_48h": _parse_duration_48h(
            f.get("Mean AF Duration")
        ),
        "outcome_time_gt_48h": _parse_duration_48h(
            f.get("Time to Outcome Measure")
        ),
        "mean_age_raw": _to_float_or_na(f.get("Mean Age")),
        "mean_la_size_raw": _to_float_or_na(
            f.get("Mean Left Atrium Size, mm")
        ),
        "fraction_male": _parse_percent(f.get("Fraction Male")),
        "fraction_cv_disease": _parse_percent(
            f.get("Fraction with CV Disease")
        ),
        "followup_fraction": _parse_percent(f.get("Follow-up Fraction")),
        "adequate_concealment": _parse_boolean(
            f.get("Adequate Concealment of Treatment")
        ),
        "masked_patients": _parse_boolean(f.get("Masked Patients")),
        "masked_caregiver": _parse_boolean(f.get("Masked Caregiver")),
        "masked_assessor": _parse_boolean(f.get("Masked Assessor")),
        "placebo_controlled": (
            str(f.get("Comparison Treatment", "")).strip().lower()
            == "placebo"
        ),
    }


def _rescale_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale continuous features to [-1, 1] using the dataset range.

    Applies a min-max rescaling across all 21 trials to
    ``mean_age_raw`` and ``mean_la_size_raw``, writing the rescaled
    values into new columns ``mean_age`` and ``mean_la_size``.
    Missing values (``None``) are imputed as ``0.0`` after
    rescaling, which corresponds to the midpoint of the [-1, 1]
    range. If a feature has zero range (constant or a single
    non-missing observation) the rescaled column is set to 0.0 for
    all rows. This matches the continuous-feature handling in
    Figure 6 of Kaul & Gordon (2024).

    Args:
        df: DataFrame containing the raw continuous columns
            ``mean_age_raw`` and ``mean_la_size_raw`` produced by
            :func:`_convert_trial`. Modified in place.

    Returns:
        The same DataFrame with two new columns, ``mean_age`` and
        ``mean_la_size``, both in [-1, 1] with no missing values.
    """
    for raw_col, scaled_col in [
        ("mean_age_raw", "mean_age"),
        ("mean_la_size_raw", "mean_la_size"),
    ]:
        values = df[raw_col].astype(float)
        valid = values.dropna()
        if len(valid) > 0:
            vmin, vmax = float(valid.min()), float(valid.max())
            rng = vmax - vmin
            if rng > 0:
                df[scaled_col] = 2.0 * (values - vmin) / rng - 1.0
            else:
                df[scaled_col] = 0.0
            df[scaled_col] = df[scaled_col].fillna(0.0)
        else:
            df[scaled_col] = 0.0
    return df