"""Process raw SemMedDB files into semmeddb_processed_10.csv for DOSSIER.

Mirrors the logic of source_repo/DOSSIER/notebooks/process_semmeddb.ipynb
but uses chunked reading to avoid loading the full ~20 GB PREDICATION file
into memory.

Steps
-----
1. Count citations per (SUBJECT_CUI, OBJECT_CUI, PREDICATE) from
   PREDICATION using chunked reads.
2. Keep only triples cited ≥ 10 times (same cutoff as paper).
3. Expand pipe-separated multi-CUI rows.
4. Drop molecular biology semantic types (aapp, gngm, celf, moft, genf).
5. Add SNOMED ISA hierarchy edges from MRHIER + MRCONSO.
6. Save semmeddb_processed_10.csv.

Usage
-----
    python examples/build_semmeddb_cache.py \\
        --semmeddb_dir  data/SemMedDB \\
        --umls_dir      data/umls/META \\
        --out_dir       data/SemMedDB

Expected inputs
---------------
    <semmeddb_dir>/semmedVER43_2024_R_PREDICATION.csv.gz   (~3 GB)
    <umls_dir>/MRHIER_SNOMED.RRF    (pre-filtered; build with build_umls_caches.py)
    <umls_dir>/MRCONSO.RRF          (2 GB — filtered to SNOMEDCT_US on-the-fly)

Runtime: ~30–60 min depending on disk speed.
Output:  semmeddb_processed_10.csv (~300–500 MB) in --out_dir.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Semantic types to drop (molecular/genetic, per paper)
_DROP_SEMTYPES = {"aapp", "gngm", "celf", "moft", "genf"}

# Citation cutoff
_CUTOFF = 10

# PREDICATION columns (no header; notebook drops cols 12,13,14)
_PRED_COLS = [
    "PREDICATION_ID", "SENTENCE_ID", "PMID", "PREDICATE",
    "SUBJECT_CUI", "SUBJECT_NAME", "SUBJECT_SEMTYPE", "SUBJECT_NOVELTY",
    "OBJECT_CUI", "OBJECT_NAME", "OBJECT_SEMTYPE", "OBJECT_NOVELTY",
]


def _is_valid_cui(s: str) -> bool:
    """Return True if *s* looks like a single UMLS CUI (contains 'C', no pipe)."""
    return isinstance(s, str) and "C" in s and "|" not in s


def _expand_pipes(df: pd.DataFrame) -> pd.DataFrame:
    """Expand rows with pipe-separated SUBJECT_CUI or OBJECT_CUI.

    SemMedDB sometimes encodes multiple CUIs in a single cell separated by
    ``|``.  This function creates one row per (subject, object) CUI pair so
    every row has exactly one CUI per field.

    Args:
        df: DataFrame with at least SUBJECT_CUI and OBJECT_CUI columns.

    Returns:
        DataFrame where all pipe-separated CUI cells have been expanded.
    """
    multi_s = df["SUBJECT_CUI"].str.contains("|", regex=False)
    multi_o = df["OBJECT_CUI"].str.contains("|", regex=False)
    good = df[~multi_s & ~multi_o]
    pipe_rows = df[multi_s | multi_o]
    if pipe_rows.empty:
        return good

    expanded = []
    for _, row in pipe_rows.iterrows():
        s_cui = row["SUBJECT_CUI"]
        o_cui = row["OBJECT_CUI"]
        subjects = s_cui.split("|") if "|" in s_cui else [s_cui]
        objects = o_cui.split("|") if "|" in o_cui else [o_cui]
        for sn_idx, sc in enumerate(subjects):
            for oc in objects:
                new_row = row.copy()
                new_row["SUBJECT_CUI"] = sc
                # Take the matching subject name; pipe-expand of names can differ
                if "|" in str(row.get("SUBJECT_NAME", "")):
                    names = str(row["SUBJECT_NAME"]).split("|")
                    new_row["SUBJECT_NAME"] = names[min(sn_idx, len(names) - 1)]
                new_row["OBJECT_CUI"] = oc
                expanded.append(new_row)

    if expanded:
        return pd.concat([good, pd.DataFrame(expanded)], ignore_index=True)
    return good


def build_predication(pred_path: Path) -> pd.DataFrame:
    """Build the deduplicated predication DataFrame from the SemMedDB file.

    Uses two passes over the file: the first counts citations per
    (SUBJECT_CUI, OBJECT_CUI, PREDICATE) triple; the second collects full
    rows only for triples that meet the ``_CUTOFF`` threshold.

    Args:
        pred_path: Path to the SemMedDB PREDICATION CSV or gzipped CSV.

    Returns:
        DataFrame with columns from ``_PRED_COLS`` plus ``n_refs``, deduplicated
        on (SUBJECT_CUI, OBJECT_CUI, PREDICATE).
    """
    print(f"Reading {pred_path} ({pred_path.stat().st_size // 1_000_000} MB) ...")
    chunk_size = 500_000

    # Pass 1: count citations per triple
    counts: defaultdict = defaultdict(int)
    total_rows = 0
    for chunk in pd.read_csv(
        pred_path, header=None, encoding="ISO-8859-1",
        usecols=[0, 3, 4, 8], chunksize=chunk_size,
        on_bad_lines="skip",
    ):
        chunk.columns = ["PREDICATION_ID", "PREDICATE", "SUBJECT_CUI", "OBJECT_CUI"]
        chunk = chunk.dropna()
        # Quick CUI validity check (no pipe, contains C)
        mask = (
            chunk["SUBJECT_CUI"].str.contains("C", regex=False) &
            chunk["OBJECT_CUI"].str.contains("C", regex=False)
        )
        chunk = chunk[mask]
        triples = zip(
            chunk["PREDICATE"], chunk["SUBJECT_CUI"], chunk["OBJECT_CUI"]
        )
        for pred, sc, oc in triples:
            counts[(sc, oc, pred)] += 1
        total_rows += len(chunk)

    print(f"  Pass 1 done: {total_rows:,} valid rows, {len(counts):,} unique triples")

    # Keep triples with >= cutoff citations
    valid_triples = {k for k, v in counts.items() if v >= _CUTOFF}
    print(f"  Triples with >= {_CUTOFF} citations: {len(valid_triples):,}")
    del counts

    # Pass 2: collect full rows for valid triples
    rows = []
    for chunk in pd.read_csv(
        pred_path, header=None, names=_PRED_COLS + ["_c12", "_c13", "_c14"],
        encoding="ISO-8859-1", usecols=list(range(12)),
        chunksize=chunk_size, on_bad_lines="skip",
    ):
        chunk = chunk.dropna(subset=["SUBJECT_CUI", "OBJECT_CUI"])
        chunk = _expand_pipes(chunk)
        # Filter valid CUIs
        valid_s = chunk["SUBJECT_CUI"].apply(_is_valid_cui)
        valid_o = chunk["OBJECT_CUI"].apply(_is_valid_cui)
        chunk = chunk[valid_s & valid_o]
        # Drop molecular semtypes
        chunk = chunk[
            ~chunk["SUBJECT_SEMTYPE"].isin(_DROP_SEMTYPES) &
            ~chunk["OBJECT_SEMTYPE"].isin(_DROP_SEMTYPES)
        ]
        # Filter to valid triples
        key_col = list(
            zip(chunk["SUBJECT_CUI"], chunk["OBJECT_CUI"], chunk["PREDICATE"])
        )
        chunk = chunk[[k in valid_triples for k in key_col]]
        rows.append(chunk[_PRED_COLS])

    df = pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["SUBJECT_CUI", "OBJECT_CUI", "PREDICATE"]
    )
    df["n_refs"] = _CUTOFF
    print(f"  Pass 2 done: {len(df):,} deduplicated triples")
    return df


def build_snomed_hierarchy(
    mrhier_path: Path,
    mrconso_path: Path,
) -> pd.DataFrame:
    """Build ISA edges from SNOMED hierarchy in MRHIER + MRCONSO.

    Args:
        mrhier_path: Path to the pre-filtered ``MRHIER_SNOMED.RRF`` file.
        mrconso_path: Path to ``MRCONSO.RRF`` (filtered to SNOMEDCT_US on-the-fly).

    Returns:
        DataFrame with columns SUBJECT_CUI, OBJECT_CUI, PREDICATE (``"ISA"``),
        SUBJECT_NAME, OBJECT_NAME, SUBJECT_NOVELTY, OBJECT_NOVELTY, and n_refs.
    """
    print(
        f"Building SNOMED hierarchy from "
        f"{mrhier_path.name} + {mrconso_path.name} ..."
    )

    # atom → (CUI, str_label) from MRCONSO — SNOMEDCT_US only
    print("  Loading atom→CUI from MRCONSO (SNOMEDCT_US rows)...")
    atom_rows = []
    for chunk in pd.read_csv(
        mrconso_path, sep="|", header=None, usecols=[0, 7, 11, 14],
        dtype=str, encoding="utf-8", chunksize=500_000,
    ):
        sub = chunk[chunk[11] == "SNOMEDCT_US"][[0, 7, 14]]
        atom_rows.append(sub)
    atom_df = pd.concat(atom_rows, ignore_index=True)
    atom_df.columns = ["cui", "atom", "str_label"]
    atom_df = atom_df.drop_duplicates("atom")
    atom_to_cui = atom_df.set_index("atom")["cui"].to_dict()
    atom_to_name = atom_df.set_index("atom")["str_label"].to_dict()
    print(f"  {len(atom_to_cui):,} SNOMED atoms loaded")

    # MRHIER_SNOMED.RRF: col 0=CUI, col 6=PATH (dot-separated atom IDs)
    print("  Expanding hierarchy paths...")
    hier_rows = []
    for chunk in pd.read_csv(
        mrhier_path, sep="|", header=None, usecols=[0, 6],
        dtype=str, encoding="utf-8", chunksize=500_000,
    ):
        chunk = chunk.dropna(subset=[6])
        chunk[6] = chunk[6].str.split(".")
        sub = chunk.explode(6)
        sub = sub[sub[6].isin(atom_to_cui)]
        sub = sub.rename(columns={0: "SUBJECT_CUI", 6: "atom"})
        sub["OBJECT_CUI"] = sub["atom"].map(atom_to_cui)
        hier_rows.append(sub[["SUBJECT_CUI", "OBJECT_CUI"]])

    hier_df = pd.concat(hier_rows, ignore_index=True).drop_duplicates(
        subset=["SUBJECT_CUI", "OBJECT_CUI"]
    )
    # Remove generic concepts from paper
    hier_df = hier_df[~hier_df["OBJECT_CUI"].isin(["C2720507", "C0037088"])]

    # Add manual BP entries (from paper notebook)
    extra = pd.DataFrame([
        ("C0428881", "C0871470"),
        ("C0428881", "C1306620"),
        ("C0428884", "C0428883"),
        ("C0428884", "C1305849"),
    ], columns=["SUBJECT_CUI", "OBJECT_CUI"])
    hier_df = pd.concat([hier_df, extra], ignore_index=True).drop_duplicates(
        subset=["SUBJECT_CUI", "OBJECT_CUI"]
    )

    hier_df["PREDICATE"] = "ISA"
    hier_df["SUBJECT_NAME"] = hier_df["SUBJECT_CUI"].map(atom_to_name)
    hier_df["OBJECT_NAME"]  = hier_df["OBJECT_CUI"].map(atom_to_name)
    hier_df["SUBJECT_NOVELTY"] = 1
    hier_df["OBJECT_NOVELTY"]  = 1
    hier_df["n_refs"] = _CUTOFF
    print(f"  {len(hier_df):,} ISA edges from SNOMED hierarchy")
    return hier_df


def main() -> None:
    """Parse command-line arguments and run the SemMedDB processing pipeline."""
    p = argparse.ArgumentParser(
        description="Process SemMedDB into semmeddb_processed_10.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--semmeddb_dir", default="data/SemMedDB",
                   help="Directory with semmedVER43_*_PREDICATION.csv[.gz]")
    p.add_argument("--umls_dir", default="data/umls/META",
                   help="UMLS META directory (needs MRHIER_SNOMED.RRF + MRCONSO.RRF)")
    p.add_argument("--out_dir", default="data/SemMedDB",
                   help="Output directory")
    p.add_argument("--cutoff", type=int, default=10,
                   help="Minimum citation count for a triple to be kept")
    p.add_argument(
        "--unzip", action="store_true",
        help=(
            "Decompress PREDICATION.csv.gz to a plain .csv before processing. "
            "Recommended: reading plain CSV is ~2x faster than streaming gz. "
            "Requires ~15 GB extra disk space."
        ),
    )
    args = p.parse_args()

    semmeddb_dir = Path(args.semmeddb_dir)
    umls_dir     = Path(args.umls_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find PREDICATION file — prefer plain .csv (faster), fall back to .csv.gz
    pred_csv = next(semmeddb_dir.glob("semmedVER43_*_PREDICATION.csv"), None)
    # exclude any .csv.gz files matched by the plain .csv glob on some systems
    if pred_csv and pred_csv.suffix == ".gz":
        pred_csv = None
    pred_gz  = next(semmeddb_dir.glob("semmedVER43_*_PREDICATION.csv.gz"), None)

    if pred_csv is None and pred_gz is None:
        raise FileNotFoundError(
            f"No PREDICATION.csv or PREDICATION.csv.gz found in {semmeddb_dir}"
        )

    if args.unzip and pred_gz is not None and pred_csv is None:
        pred_csv = pred_gz.with_suffix("")  # strip .gz → .csv
        print(f"Decompressing {pred_gz.name} -> {pred_csv.name} "
              f"({pred_gz.stat().st_size // 1_000_000} MB gz) ...")
        with gzip.open(pred_gz, "rb") as src, pred_csv.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"  Done. Plain CSV: {pred_csv.stat().st_size // 1_000_000} MB")

    pred_path = pred_csv if pred_csv is not None else pred_gz
    print(f"Using PREDICATION file: {pred_path.name} "
          f"({pred_path.stat().st_size // 1_000_000} MB)")

    mrhier_path  = umls_dir / "MRHIER_SNOMED.RRF"
    mrconso_path = umls_dir / "MRCONSO.RRF"
    for f in [mrhier_path, mrconso_path]:
        if not f.exists():
            raise FileNotFoundError(
                f"{f} not found.\n"
                "Run build_umls_caches.py first to build MRHIER_SNOMED.RRF."
            )

    # Step 1: Process PREDICATION
    df_pred = build_predication(pred_path)

    # Step 2: SNOMED hierarchy
    df_hier = build_snomed_hierarchy(mrhier_path, mrconso_path)

    # Step 3: Combine
    out_cols = [
        "PREDICATION_ID", "PREDICATE",
        "SUBJECT_CUI", "SUBJECT_NAME", "SUBJECT_SEMTYPE", "SUBJECT_NOVELTY",
        "OBJECT_CUI", "OBJECT_NAME", "OBJECT_SEMTYPE", "OBJECT_NOVELTY",
        "n_refs",
    ]
    df_hier["PREDICATION_ID"] = np.arange(
        int(df_pred["PREDICATION_ID"].max()) + 1,
        int(df_pred["PREDICATION_ID"].max()) + 1 + len(df_hier),
    )
    df_hier["SUBJECT_SEMTYPE"] = ""
    df_hier["OBJECT_SEMTYPE"]  = ""

    df_final = pd.concat(
        [df_pred[[c for c in out_cols if c in df_pred.columns]],
         df_hier[[c for c in out_cols if c in df_hier.columns]]],
        ignore_index=True,
    ).drop_duplicates(subset=["SUBJECT_CUI", "OBJECT_CUI", "PREDICATE"])

    out_path = out_dir / f"semmeddb_processed_{args.cutoff}.csv"
    df_final.to_csv(out_path, index=False)
    size_mb = out_path.stat().st_size / 1_000_000
    print(f"\nDone. {len(df_final):,} rows → {out_path} ({size_mb:.0f} MB)")
    print(f"\nUse with DOSSIERPipeline:")
    print(
        f"  DOSSIERPipeline(..., semmeddb_path='{out_path}', "
        "prompt_variant='no_umls')"
    )


if __name__ == "__main__":
    main()
