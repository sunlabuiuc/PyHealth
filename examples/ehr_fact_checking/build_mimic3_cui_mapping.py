"""Generate a MIMIC-III ITEMID -> CUI mapping for DOSSIERPipeline.

Looks up each D_LABITEMS / D_ITEMS label in the pre-built umls_name_to_cui.pkl
cache and writes a two-column CSV (ID, cui) that DOSSIERPipeline uses to
populate the CUI column of the Lab and Vital tables.

Usage
-----
    python examples/build_mimic3_cui_mapping.py \\
        --mimic3_root data/mimic-iii \\
        --umls_dir    data/umls \\
        --out         data/umls/mimic3_cui_mapping.csv

Expected inputs
---------------
    <mimic3_root>/D_LABITEMS.csv  (or .csv.gz) — MIMIC-III lab dictionary
    <mimic3_root>/D_ITEMS.csv     (or .csv.gz) — MIMIC-III charting dictionary
    <umls_dir>/umls_name_to_cui.pkl            — built by build_umls_caches.py

Output
------
    CSV with columns: ID (int), cui (str, e.g. "C0017725")
    Rows with no match are omitted.

Lookup strategy (in order, first hit wins):
    1. "<label> measurement"   — preferred for lab items (gives test CUI)
    2. "<label> level"         — alternate clinical phrasing
    3. "<label>"               — plain name

Runtime: ~10 seconds.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _resolve_csv(directory: Path, name: str) -> Path:
    """Return path to *name* inside *directory*, falling back to the .gz variant.

    Args:
        directory: Directory to search in.
        name: Bare filename (e.g. ``"D_LABITEMS.csv"``).

    Returns:
        Path to the plain file if it exists, otherwise to ``name + ".gz"``.
        The returned path may not exist if neither variant is present.
    """
    p = directory / name
    if not p.exists():
        gz = directory / (name + ".gz")
        if gz.exists():
            return gz
    return p


def _lookup_cui(
    label: str,
    name_to_cui: Dict[str, str],
) -> Optional[str]:
    """Look up a UMLS CUI for an item label using a cascading suffix strategy.

    Tries ``"<label> measurement"``, ``"<label> level"``, then ``"<label>"``
    in order, returning the first match.

    Args:
        label: Raw item label string (e.g. ``"Glucose"``).
        name_to_cui: Mapping from lowercased preferred name to CUI.

    Returns:
        CUI string if found, otherwise ``None``.
    """
    key = label.strip().lower()
    for suffix in (" measurement", " level", ""):
        cui = name_to_cui.get(key + suffix)
        if cui:
            return cui
    return None


def main() -> None:
    """Parse command-line arguments and generate the ITEMID->CUI mapping CSV."""
    p = argparse.ArgumentParser(
        description="Build MIMIC-III ITEMID->CUI mapping for DOSSIERPipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mimic3_root", default="data/mimic-iii",
                   help="Path to MIMIC-III CSVs (D_LABITEMS.csv, D_ITEMS.csv)")
    p.add_argument("--umls_dir", default="data/umls",
                   help="Directory containing umls_name_to_cui.pkl")
    p.add_argument("--out", default="data/umls/mimic3_cui_mapping.csv",
                   help="Output CSV path (ID, cui)")
    args = p.parse_args()

    mimic_root = Path(args.mimic3_root)
    umls_dir   = Path(args.umls_dir)
    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    name_to_cui_pkl = umls_dir / "umls_name_to_cui.pkl"
    if not name_to_cui_pkl.exists():
        raise FileNotFoundError(
            f"{name_to_cui_pkl} not found.\n"
            "Run build_umls_caches.py first:\n"
            "  python examples/build_umls_caches.py "
            "--umls_dir data/umls/META --out_dir data/umls"
        )

    print(f"Loading {name_to_cui_pkl} ...")
    with name_to_cui_pkl.open("rb") as f:
        name_to_cui: dict = pickle.load(f)
    print(f"  {len(name_to_cui):,} UMLS preferred names loaded")

    rows: list[dict] = []

    # --- D_LABITEMS (lab items) ---
    lab_path = _resolve_csv(mimic_root, "D_LABITEMS.csv")
    if lab_path.exists():
        lab_df = pd.read_csv(lab_path, usecols=["ITEMID", "LABEL"], dtype=str)
        n_matched = 0
        for _, row in lab_df.iterrows():
            cui = _lookup_cui(str(row["LABEL"]), name_to_cui)
            if cui:
                rows.append({"ID": int(row["ITEMID"]), "cui": cui})
                n_matched += 1
        print(f"  D_LABITEMS: {n_matched}/{len(lab_df)} items matched a CUI")
    else:
        print(f"  WARNING: D_LABITEMS.csv not found at {lab_path.parent}; skipping")

    # --- D_ITEMS (charting / vital items) ---
    items_path = _resolve_csv(mimic_root, "D_ITEMS.csv")
    if items_path.exists():
        items_df = pd.read_csv(items_path, usecols=["ITEMID", "LABEL"], dtype=str)
        n_matched = 0
        for _, row in items_df.iterrows():
            cui = _lookup_cui(str(row["LABEL"]), name_to_cui)
            if cui:
                rows.append({"ID": int(row["ITEMID"]), "cui": cui})
                n_matched += 1
        print(f"  D_ITEMS:    {n_matched}/{len(items_df)} items matched a CUI")
    else:
        print(f"  WARNING: D_ITEMS.csv not found at {items_path.parent}; skipping")

    if not rows:
        raise RuntimeError("No CUI matches found. Check --mimic3_root and --umls_dir.")

    out_df = pd.DataFrame(rows).drop_duplicates("ID")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df):,} ITEMID->CUI rows -> {out_path}")
    print(f"\nUse with DOSSIERPipeline:")
    print(
        f"  DOSSIERPipeline(..., cui_mapping_path='{out_path}', "
        f"umls_dir='{args.umls_dir}')"
    )


if __name__ == "__main__":
    main()
