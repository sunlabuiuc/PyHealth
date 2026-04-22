"""Generate ~200 verifiable claims from full MIMIC-III (compressed CSVs).

Implements 10 lab-threshold templates and 5 medication templates inspired by
the paper's Appendix D slot-filling approach.  Labels are deterministically
derived from actual EHR values — no hand-labelling needed.

Label semantics
---------------
T  : claim is supported by the data (threshold met / medication given)
F  : claim is refuted by data (threshold never met / medication not given)
N  : claim cannot be verified — the relevant lab was never measured

Target distribution: ~40% T, ~25% F, ~35% N  (approximates paper's 50/15/35)

Data required (in --mimic3_root)
---------------------------------
ADMISSIONS.csv.gz   LABEVENTS.csv.gz    D_LABITEMS.csv.gz
INPUTEVENTS_MV.csv.gz   D_ITEMS.csv.gz

CHARTEVENTS is NOT required (vitals claims are omitted).

Usage
-----
    python examples/generate_claims_from_mimic3.py \\
        --mimic3_root  data/mimic-iii \\
        --output       data/full_claims.csv \\
        --n_admissions 25 \\
        --t_C_hours    72 \\
        --seed         42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Lab templates: (db_label, item_ids, threshold, direction, claim_text, units)
# direction: ">" → True when any reading is ABOVE threshold
#            "<" → True when any reading is BELOW threshold
# ---------------------------------------------------------------------------

LAB_TEMPLATES: List[Tuple[str, List[int], float, str, str, str]] = [
    # --- Glucose ---
    ("Glucose", [50931, 50809],
     180.0, ">",
     "The patient had at least one glucose level above 180 mg/dL.", "mg/dL"),
    ("Glucose", [50931, 50809],
     70.0,  "<",
     "The patient had a hypoglycemic episode (glucose below 70 mg/dL).", "mg/dL"),
    # --- Sodium ---
    ("Sodium", [50983],
     145.0, ">",
     "The patient had at least one sodium level above 145 mEq/L.", "mEq/L"),
    ("Sodium", [50983],
     130.0, "<",
     "The patient had at least one sodium level below 130 mEq/L (hyponatremia).",
     "mEq/L"),
    # --- Potassium ---
    ("Potassium", [50971],
     5.0, ">",
     "The patient had at least one potassium level above 5.0 mEq/L (hyperkalemia).",
     "mEq/L"),
    ("Potassium", [50971],
     3.5, "<",
     "The patient had at least one potassium level below 3.5 mEq/L (hypokalemia).",
     "mEq/L"),
    # --- Creatinine ---
    ("Creatinine", [50912],
     2.0, ">",
     "The patient's creatinine exceeded 2.0 mg/dL at some point.", "mg/dL"),
    # --- Hematocrit ---
    ("Hematocrit", [51221, 50810],
     25.0, "<",
     "The patient had at least one hematocrit reading below 25%.", "%"),
    # --- Hemoglobin ---
    ("Hemoglobin", [50811],
     8.0, "<",
     "The patient's hemoglobin dropped below 8.0 g/dL.", "g/dL"),
    # --- Platelet Count ---
    ("Platelet Count", [51265],
     100.0, "<",
     "The patient had thrombocytopenia (platelet count below 100 K/uL).", "K/uL"),
    # --- Bicarbonate ---
    ("Bicarbonate", [50882],
     20.0, "<",
     "The patient had at least one bicarbonate level below 20 mEq/L.", "mEq/L"),
    # --- BUN (Urea Nitrogen) ---
    ("Urea Nitrogen", [51006],
     40.0, ">",
     "The patient's BUN exceeded 40 mg/dL (elevated urea nitrogen).", "mg/dL"),
    # --- Lactate (often absent → good for N) ---
    ("Lactate", [50813],
     2.0, ">",
     "The patient had an elevated lactate level above 2.0 mmol/L.", "mmol/L"),
    # --- Bilirubin Total (often absent → good for N) ---
    ("Bilirubin, Total", [50885],
     2.0, ">",
     "The patient had elevated total bilirubin above 2.0 mg/dL.", "mg/dL"),
    # --- Albumin (often absent → good for N) ---
    ("Albumin", [50862],
     3.0, "<",
     "The patient had hypoalbuminemia (albumin below 3.0 g/dL).", "g/dL"),
    # --- WBC ---
    ("White Blood Cells", [51301],
     12.0, ">",
     "The patient had leukocytosis (white blood cell count above 12 K/uL).", "K/uL"),
    # --- Troponin T (often absent → good for N) ---
    ("Troponin T", [51003],
     0.1, ">",
     "The patient had an elevated troponin T level above 0.1 ng/mL.", "ng/mL"),
    # --- Calcium ---
    ("Calcium, Total", [50893],
     10.5, ">",
     "The patient had hypercalcemia (calcium above 10.5 mg/dL).", "mg/dL"),
    # --- Magnesium ---
    ("Magnesium", [50960],
     1.5, "<",
     "The patient had hypomagnesemia (magnesium below 1.5 mg/dL).", "mg/dL"),
    # --- Phosphate ---
    ("Phosphate", [50970],
     4.5, ">",
     "The patient had hyperphosphatemia (phosphate above 4.5 mg/dL).", "mg/dL"),
]

# ---------------------------------------------------------------------------
# Medication templates: (display_name, item_ids, claim_text)
# T if medication appears in INPUTEVENTS_MV for this admission; F otherwise.
# ---------------------------------------------------------------------------

MED_TEMPLATES: List[Tuple[str, List[int], str]] = [
    ("Heparin",
     [225975, 224145],
     "The patient received heparin during this admission."),
    ("Insulin (Regular)",
     [223258],
     "The patient was administered regular insulin."),
    ("Furosemide",
     [221794, 228340],
     "The patient received furosemide (Lasix) during this admission."),
    ("Propofol",
     [222168, 227210],
     "The patient was sedated with propofol."),
    ("Norepinephrine",
     [221906],
     "The patient required vasopressor support with norepinephrine."),
    ("Potassium Chloride",
     [225166],
     "The patient received potassium chloride supplementation."),
    ("Normal Saline",
     [225158],
     "The patient received normal saline (0.9% NaCl) fluid resuscitation."),
    ("Morphine Sulfate",
     [225154],
     "The patient received morphine sulfate for pain management."),
    ("Metoprolol",
     [225974],
     "The patient was given metoprolol for heart rate control."),
]


def _load_gz(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a gzip-compressed CSV and normalise all column names to uppercase.

    Args:
        path: Path to the .csv.gz file.
        **kwargs: Extra keyword arguments forwarded to ``pd.read_csv``.

    Returns:
        DataFrame with uppercased column names.
    """
    df = pd.read_csv(path, compression="gzip", low_memory=False, **kwargs)
    df.columns = [c.upper() for c in df.columns]
    return df


def load_labs_filtered(
    lab_path: Path,
    d_lab_path: Path,
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Load LABEVENTS rows for templates' ITEMIDs only (memory-efficient).

    Reads LABEVENTS.csv.gz in chunks and retains only the rows whose ITEMID
    appears in ``LAB_TEMPLATES``, avoiding loading all ~27 M rows at once.

    Args:
        lab_path: Path to ``LABEVENTS.csv.gz``.
        d_lab_path: Path to ``D_LABITEMS.csv.gz``.

    Returns:
        Tuple of:
          - DataFrame of filtered lab events with uppercased columns.
          - Dict mapping ITEMID (int) to LABEL string from ``D_LABITEMS``.
    """
    all_item_ids = set()
    for _, ids, *_ in LAB_TEMPLATES:
        all_item_ids.update(ids)

    d_lab = _load_gz(d_lab_path)
    item_to_label: Dict[int, str] = d_lab.set_index("ITEMID")["LABEL"].to_dict()

    # Read in chunks to avoid loading 27M rows at once
    chunk_size = 500_000
    kept: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        lab_path, compression="gzip", low_memory=False, chunksize=chunk_size
    ):
        chunk.columns = [c.upper() for c in chunk.columns]
        sub = chunk[chunk["ITEMID"].isin(all_item_ids)].dropna(
            subset=["HADM_ID", "VALUENUM"]
        )
        if not sub.empty:
            kept.append(sub)

    labs = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    if not labs.empty:
        labs["HADM_ID"] = labs["HADM_ID"].astype(int)
        labs["CHARTTIME"] = pd.to_datetime(labs["CHARTTIME"])
    return labs, item_to_label


def load_inputs_filtered(inp_path: Path) -> pd.DataFrame:
    """Load INPUTEVENTS_MV rows for templates' medication ITEMIDs only.

    Reads the file in chunks and keeps only rows whose ITEMID appears in
    ``MED_TEMPLATES``.

    Args:
        inp_path: Path to ``INPUTEVENTS_MV.csv.gz``.

    Returns:
        DataFrame with columns HADM_ID, ITEMID, AMOUNT, AMOUNTUOM, STARTTIME.
    """
    all_med_ids = set()
    for _, ids, _ in MED_TEMPLATES:
        all_med_ids.update(ids)

    chunk_size = 500_000
    kept: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        inp_path, compression="gzip", low_memory=False, chunksize=chunk_size
    ):
        chunk.columns = [c.upper() for c in chunk.columns]
        sub = chunk[chunk["ITEMID"].isin(all_med_ids)].dropna(subset=["HADM_ID"])
        if not sub.empty:
            kept.append(sub[["HADM_ID", "ITEMID", "AMOUNT", "AMOUNTUOM", "STARTTIME"]])

    inputs = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    if not inputs.empty:
        inputs["HADM_ID"] = inputs["HADM_ID"].astype(int)
    return inputs


def select_admissions(
    adm: pd.DataFrame,
    labs: pd.DataFrame,
    inputs: pd.DataFrame,
    n: int,
    seed: int,
) -> List[int]:
    """Pick n admissions with both lab AND input coverage for diversity.

    Prefers admissions that appear in both lab and medication event tables,
    falling back to lab-only admissions to reach the requested count.

    Args:
        adm: ADMISSIONS DataFrame (indexed or containing HADM_ID).
        labs: Filtered LABEVENTS DataFrame.
        inputs: Filtered INPUTEVENTS_MV DataFrame (may be empty).
        n: Number of admissions to select.
        seed: Random seed for reproducibility.

    Returns:
        List of up to n HADM_ID integers.
    """
    rng = random.Random(seed)
    lab_hadms = set(labs["HADM_ID"].unique())
    inp_hadms = set(inputs["HADM_ID"].unique()) if not inputs.empty else set()

    # Prefer admissions with both labs and inputs
    both = sorted(lab_hadms & inp_hadms)
    lab_only = sorted(lab_hadms - inp_hadms)

    candidates = both + lab_only
    rng.shuffle(candidates)
    return candidates[:n]


def generate_lab_claims(
    hadm_id: int,
    adm_row: pd.Series,
    labs: pd.DataFrame,
    t_C_hours: float,
) -> List[Dict[str, Any]]:
    """Return T/F/N records for all lab templates for one admission.

    Args:
        hadm_id: Hospital admission identifier.
        adm_row: Row from the ADMISSIONS table for this admission.
        labs: Filtered LABEVENTS DataFrame for all admissions.
        t_C_hours: Claim time horizon in hours after admission.

    Returns:
        List of record dicts, one per ``LAB_TEMPLATES`` entry, each containing
        HADM_ID, claim, t_C, label (T/F/N), and template metadata fields.
    """
    admit_time = adm_row["ADMITTIME"]
    cutoff = admit_time + pd.Timedelta(hours=t_C_hours)

    pt_labs = labs[
        (labs["HADM_ID"] == hadm_id)
        & (labs["CHARTTIME"] >= admit_time)
        & (labs["CHARTTIME"] < cutoff)
    ]

    records = []
    for db_label, item_ids, threshold, direction, claim_text, _units in LAB_TEMPLATES:
        row_vals = pt_labs[pt_labs["ITEMID"].isin(item_ids)]["VALUENUM"]

        if row_vals.empty:
            # Lab never measured → NEI
            label = "N"
        else:
            if direction == ">":
                label = "T" if (row_vals > threshold).any() else "F"
            else:
                label = "T" if (row_vals < threshold).any() else "F"

        records.append({
            "HADM_ID": hadm_id,
            "claim": claim_text,
            "t_C": t_C_hours,
            "label": label,
            "source": "lab",
            "template_lab": db_label,
            "template_threshold": threshold,
            "template_direction": direction,
            "n_values": len(row_vals),
            "min_value": float(row_vals.min()) if not row_vals.empty else None,
            "max_value": float(row_vals.max()) if not row_vals.empty else None,
        })

    return records


def generate_med_claims(
    hadm_id: int,
    inputs: pd.DataFrame,
    t_C_hours: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return T/F records for all medication templates.

    A claim is labelled T if the medication's ITEMID appears in INPUTEVENTS_MV
    for this admission, F otherwise.  No N label is produced because medication
    absence is treated as a definitive refutation.

    Args:
        hadm_id: Hospital admission identifier.
        inputs: Filtered INPUTEVENTS_MV DataFrame for all admissions.
        t_C_hours: Claim time horizon in hours (stored in output, not used to
            filter events because INPUTEVENTS_MV is pre-filtered by template).

    Returns:
        List of record dicts, one per ``MED_TEMPLATES`` entry.
    """
    if inputs.empty:
        pt_inputs_items: set = set()
    else:
        pt_inputs_items = set(inputs[inputs["HADM_ID"] == hadm_id]["ITEMID"].unique())

    records = []
    for med_name, item_ids, claim_text in MED_TEMPLATES:
        given = bool(set(item_ids) & pt_inputs_items)
        records.append({
            "HADM_ID": hadm_id,
            "claim": claim_text,
            "t_C": t_C_hours if t_C_hours else 999.0,
            "label": "T" if given else "F",
            "source": "medication",
            "template_lab": med_name,
            "template_threshold": None,
            "template_direction": None,
            "n_values": None,
            "min_value": None,
            "max_value": None,
        })
    return records


def balance_records(
    records: List[Dict[str, Any]],
    target_t: int,
    target_f: int,
    target_n: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Subsample records to approximate target T/F/N counts.

    Shuffles each label group independently, then truncates to the requested
    count.  Records with labels other than T, F, or N are discarded.

    Args:
        records: All generated claim records for one or more admissions.
        target_t: Maximum number of T (supported) records to keep.
        target_f: Maximum number of F (refuted) records to keep.
        target_n: Maximum number of N (not enough info) records to keep.
        rng: Seeded random instance used for shuffling.

    Returns:
        Subsampled list with at most target_t + target_f + target_n elements.
    """
    t_recs = [r for r in records if r["label"] == "T"]
    f_recs = [r for r in records if r["label"] == "F"]
    n_recs = [r for r in records if r["label"] == "N"]

    rng.shuffle(t_recs)
    rng.shuffle(f_recs)
    rng.shuffle(n_recs)

    return t_recs[:target_t] + f_recs[:target_f] + n_recs[:target_n]


def check_cui_coverage(cui_mapping_path: Optional[str]) -> None:
    """Warn about template ITEMIDs that have no CUI mapping entry.

    The ``full`` and ``no_gkg`` DOSSIER variants filter patient tables to rows
    with a matched ITEMID→CUI entry.  Claims about unmapped ITEMIDs will
    always produce empty patient tables, forcing an N prediction regardless of
    the true label.  Run this check after updating templates to catch gaps early.

    Args:
        cui_mapping_path: Path to ``mimic3_cui_mapping.csv``, or ``None`` to
            skip the check.
    """
    if cui_mapping_path is None or not Path(cui_mapping_path).exists():
        print("[CUI check] No mapping file provided — skipping coverage check.")
        return

    mapping = pd.read_csv(cui_mapping_path)
    # Column is "ID" in source repo format
    # (cui, pretty_name, types, ID, LABEL, source, Count)
    id_col = "ID" if "ID" in mapping.columns else "ITEMID"
    mapped_ids: set = set(
        pd.to_numeric(mapping[id_col], errors="coerce").dropna().astype(int).tolist()
    )

    all_template_ids: Dict[str, set] = {}
    for name, ids, *_ in LAB_TEMPLATES:
        all_template_ids.setdefault(name, set()).update(ids)
    for name, ids, _ in MED_TEMPLATES:
        all_template_ids.setdefault(name, set()).update(ids)

    missing: List[str] = []
    for tname, ids in sorted(all_template_ids.items()):
        unmapped = sorted(ids - mapped_ids)
        if unmapped:
            missing.append(f"  {tname}: ITEMIDs {unmapped} have no CUI mapping")

    if missing:
        print("[CUI check] WARNING — the following template ITEMIDs are not in "
              f"{cui_mapping_path}:")
        for m in missing:
            print(m)
        print("  Claims about these entities will produce empty patient tables in "
              "full/no_gkg variants, forcing N predictions. "
              "Add rows to the mapping file or remove the templates.")
    else:
        print(f"[CUI check] All {len(all_template_ids)} template entities are "
              "covered by the CUI mapping.")


def generate_claims(
    mimic3_root: str,
    output_path: str,
    n_admissions: int = 25,
    t_C_hours: float = 72.0,
    seed: int = 42,
    cui_mapping_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate labelled EHR claims from full MIMIC-III compressed CSVs.

    Loads ADMISSIONS, LABEVENTS, and INPUTEVENTS_MV, selects a diverse set of
    admissions, applies lab and medication templates to produce T/F/N claims,
    balances the label distribution, and writes two CSVs: a minimal four-column
    file for PyHealth and a richer version with template metadata.

    Args:
        mimic3_root: Directory containing ADMISSIONS.csv.gz, LABEVENTS.csv.gz,
            D_LABITEMS.csv.gz, INPUTEVENTS_MV.csv.gz, and D_ITEMS.csv.gz.
        output_path: Destination path for the minimal claims CSV.
        n_admissions: Number of admissions to sample claims from.
        t_C_hours: Claim time horizon — only lab/input events recorded within
            this many hours of admission are considered.
        seed: Random seed used for admission selection and balancing.
        cui_mapping_path: Optional path to ``mimic3_cui_mapping.csv``.  When
            provided, runs a coverage check before loading data.

    Returns:
        DataFrame with columns HADM_ID, claim, t_C, label written to
        ``output_path``.
    """
    root = Path(mimic3_root)
    rng = random.Random(seed)
    np.random.seed(seed)

    check_cui_coverage(cui_mapping_path)
    print("Loading admissions...")
    adm = _load_gz(root / "ADMISSIONS.csv.gz")
    adm["ADMITTIME"] = pd.to_datetime(adm["ADMITTIME"])
    adm = adm.dropna(subset=["HADM_ID"]).copy()
    adm["HADM_ID"] = adm["HADM_ID"].astype(int)
    adm = adm.set_index("HADM_ID")

    print("Loading lab events (filtering to template items, ~27M rows)...")
    labs, _item_to_label = load_labs_filtered(
        root / "LABEVENTS.csv.gz",
        root / "D_LABITEMS.csv.gz",
    )
    n_lab_adm = labs["HADM_ID"].nunique()
    print(f"  Kept {len(labs):,} lab rows for {n_lab_adm:,} admissions.")

    print("Loading input events (filtering to medication items)...")
    inputs = load_inputs_filtered(root / "INPUTEVENTS_MV.csv.gz")
    n_inp_adm = inputs["HADM_ID"].nunique()
    print(f"  Kept {len(inputs):,} input rows for {n_inp_adm:,} admissions.")

    print(f"Selecting {n_admissions} admissions...")
    hadm_ids = select_admissions(adm, labs, inputs, n_admissions, seed)
    print(f"  Selected HADM_IDs: {hadm_ids}")

    # Per-admission targets to reach ~200 total with right distribution
    per_adm_t = 3
    per_adm_f = 2
    per_adm_n = 3
    total_per_adm = per_adm_t + per_adm_f + per_adm_n

    all_records: List[dict] = []
    for hadm_id in hadm_ids:
        if hadm_id not in adm.index:
            continue
        adm_row = adm.loc[hadm_id]

        lab_recs = generate_lab_claims(hadm_id, adm_row, labs, t_C_hours)
        med_recs = generate_med_claims(hadm_id, inputs, t_C_hours)
        all_recs = lab_recs + med_recs

        balanced = balance_records(all_recs, per_adm_t, per_adm_f, per_adm_n, rng)
        all_records.extend(balanced)

    df = pd.DataFrame(all_records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle

    # Minimal columns for PyHealth EHRFactCheckingMIMIC3 task
    out_df = df[["HADM_ID", "claim", "t_C", "label"]].copy()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    # Also save the rich version for documentation
    rich_path = Path(output_path).with_stem(Path(output_path).stem + "_rich")
    df.to_csv(rich_path, index=False)

    n_t = (out_df["label"] == "T").sum()
    n_f = (out_df["label"] == "F").sum()
    n_n = (out_df["label"] == "N").sum()
    n_adm_out = out_df["HADM_ID"].nunique()
    print(f"\nGenerated {len(out_df)} claims across {n_adm_out} admissions")
    print(f"  T={n_t} ({100*n_t/len(out_df):.0f}%),  "
          f"F={n_f} ({100*n_f/len(out_df):.0f}%),  "
          f"N={n_n} ({100*n_n/len(out_df):.0f}%)")
    print(f"  -> {output_path}")
    print(f"  -> {rich_path}  (includes template metadata)")
    return out_df


def main() -> None:
    """Parse command-line arguments and invoke ``generate_claims``."""
    p = argparse.ArgumentParser(
        description="Generate verifiable claims from full MIMIC-III (compressed CSVs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mimic3_root",
        default="data/mimic-iii",
        help="Directory containing ADMISSIONS.csv.gz, LABEVENTS.csv.gz, etc.",
    )
    p.add_argument(
        "--output",
        default="data/full_claims.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--n_admissions",
        type=int,
        default=25,
        help="Number of admissions to sample claims from.",
    )
    p.add_argument(
        "--t_C_hours",
        type=float,
        default=72.0,
        help="Claim time: hours after admission (all evidence before this).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--cui_mapping_path",
        default=None,
        help=(
            "Optional path to mimic3_cui_mapping.csv. When provided, checks that "
            "every template ITEMID has a CUI entry "
            "(required for full/no_gkg variants)."
        ),
    )
    args = p.parse_args()
    generate_claims(
        args.mimic3_root,
        args.output,
        args.n_admissions,
        args.t_C_hours,
        args.seed,
        cui_mapping_path=args.cui_mapping_path,
    )


if __name__ == "__main__":
    main()
