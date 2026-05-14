"""Generate a verifiable claims CSV from MIMIC-III Clinical Database Demo.

The MIMIC-III demo (physionet.org/content/mimiciii-demo/1.4/) is publicly
available without credentialed access and contains 100 patients with the same
schema as the full MIMIC-III release.

This script reads the demo CSVs, inspects actual lab values per admission,
and produces a claims.csv where every label (T/F) is deterministically correct
— derived from the real data, not hand-written.

Usage
-----
    python examples/generate_demo_claims.py \\
        --demo_root data/mimic-iii-demo/mimic-iii-clinical-database-demo-1.4 \\
        --output    data/demo_claims.csv \\
        --n_patients 20 \\
        --t_C_hours  72

Output columns: HADM_ID, claim, t_C, label
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Claim templates: (lab_name, threshold, direction, units, claim_text)
# direction: ">" → True if any value ABOVE threshold; "<" → True if any BELOW
# ---------------------------------------------------------------------------

# Each entry: (lab_name, threshold, direction, claim_text)
_TEMPLATES: List[Tuple[str, float, str, str]] = [
    ("Glucose",       150.0, ">",
     "The patient had at least one glucose level above 150 mg/dL."),
    ("Glucose",        80.0, "<",
     "The patient had at least one glucose level below 80 mg/dL."),
    ("Potassium",       5.0, ">",
     "The patient had at least one potassium level above 5.0 mEq/L."),
    ("Potassium",       3.2, "<",
     "The patient had at least one potassium level below 3.2 mEq/L."),
    ("Creatinine",      2.0, ">",
     "The patient had at least one creatinine level above 2.0 mg/dL."),
    ("Sodium",        145.0, ">",
     "The patient had at least one sodium level above 145 mEq/L."),
    ("Sodium",        130.0, "<",
     "The patient had at least one sodium level below 130 mEq/L."),
    ("Hematocrit",     25.0, "<",
     "The patient had at least one hematocrit reading below 25%."),
    ("White Blood Cells", 12.0, ">",
     "The patient had at least one white blood cell count above 12 K/uL."),
    ("Bicarbonate",    20.0, "<",
     "The patient had at least one bicarbonate level below 20 mEq/L."),
]


def generate_claims(
    demo_root: str,
    output_path: str,
    n_patients: int = 20,
    t_C_hours: float = 72.0,
) -> pd.DataFrame:
    """Generate labelled EHR claims from the MIMIC-III Clinical Database Demo.

    Reads ADMISSIONS, LABEVENTS, and D_LABITEMS from *demo_root*, applies each
    template in ``_TEMPLATES`` to derive deterministic T/F labels from actual
    lab values, lightly balances the class distribution, and writes a CSV.

    Args:
        demo_root: Path to the extracted demo directory containing
            ``ADMISSIONS.csv``, ``LABEVENTS.csv``, and ``D_LABITEMS.csv``.
        output_path: Destination path for the output CSV.
        n_patients: Maximum number of patients (admissions) to include.
        t_C_hours: Claim time horizon — only lab readings recorded within
            this many hours of admission are considered.

    Returns:
        DataFrame with columns HADM_ID, claim, t_C, label written to
        ``output_path``.
    """
    root = Path(demo_root)

    # Load and normalise column names to uppercase
    def _load(fname: str, **kwargs: object) -> pd.DataFrame:
        df = pd.read_csv(root / fname, low_memory=False, **kwargs)
        df.columns = [c.upper() for c in df.columns]
        return df

    adm = _load("ADMISSIONS.csv")
    adm["ADMITTIME"] = pd.to_datetime(adm["ADMITTIME"])
    adm = adm.dropna(subset=["HADM_ID"]).copy()
    adm["HADM_ID"] = adm["HADM_ID"].astype(int)

    labs = _load("LABEVENTS.csv")
    d_lab = _load("D_LABITEMS.csv")
    labs = labs.merge(d_lab[["ITEMID", "LABEL"]], on="ITEMID", how="left")
    labs = labs.dropna(subset=["HADM_ID", "VALUENUM"]).copy()
    labs["HADM_ID"] = labs["HADM_ID"].astype(int)
    labs["CHARTTIME"] = pd.to_datetime(labs["CHARTTIME"])

    # Pick the first n_patients admissions that have lab coverage
    hadm_ids = (
        adm[adm["HADM_ID"].isin(labs["HADM_ID"].unique())]
        .sort_values("HADM_ID")["HADM_ID"]
        .unique()[:n_patients]
    )

    records = []
    for hadm_id in hadm_ids:
        row = adm[adm["HADM_ID"] == hadm_id].iloc[0]
        admit_time = row["ADMITTIME"]
        cutoff = admit_time + pd.Timedelta(hours=t_C_hours)

        pt_labs = labs[
            (labs["HADM_ID"] == hadm_id) & (labs["CHARTTIME"] < cutoff)
        ]
        if pt_labs.empty:
            continue

        for lab_name, threshold, direction, claim_text in _TEMPLATES:
            vals = pt_labs[pt_labs["LABEL"] == lab_name]["VALUENUM"]
            if vals.empty:
                continue
            label = "T" if (
                (direction == ">" and (vals > threshold).any()) or
                (direction == "<" and (vals < threshold).any())
            ) else "F"
            records.append({
                "HADM_ID": hadm_id,
                "claim": claim_text,
                "t_C": t_C_hours,
                "label": label,
            })

    df = pd.DataFrame(records)

    # Balance: keep at most 3× more of the majority class per patient
    balanced = []
    for hid, grp in df.groupby("HADM_ID"):
        t = grp[grp["label"] == "T"]
        f = grp[grp["label"] == "F"]
        keep_f = min(len(f), max(len(t) * 3, 1))
        keep_t = min(len(t), max(len(f) * 3, 1))
        balanced.append(t.head(keep_t))
        balanced.append(f.head(keep_f))
    df = pd.concat(balanced, ignore_index=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    n_t = (df["label"] == "T").sum()
    n_f = (df["label"] == "F").sum()
    print(f"Generated {len(df)} claims for {df['HADM_ID'].nunique()} patients "
          f"(T={n_t}, F={n_f}) -> {output_path}")
    return df


def main() -> None:
    """Parse command-line arguments and invoke ``generate_claims``."""
    p = argparse.ArgumentParser(
        description="Generate verifiable claims from MIMIC-III demo data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--demo_root",
        required=True,
        help="Path to extracted MIMIC-III demo directory "
             "(contains ADMISSIONS.csv, LABEVENTS.csv, ...).",
    )
    p.add_argument(
        "--output",
        default="data/demo_claims.csv",
        help="Output path for the generated claims CSV.",
    )
    p.add_argument(
        "--n_patients",
        type=int,
        default=20,
        help="Number of patients to generate claims for.",
    )
    p.add_argument(
        "--t_C_hours",
        type=float,
        default=72.0,
        help="Claim timestamp: hours after admission.",
    )
    args = p.parse_args()
    generate_claims(args.demo_root, args.output, args.n_patients, args.t_C_hours)


if __name__ == "__main__":
    main()
