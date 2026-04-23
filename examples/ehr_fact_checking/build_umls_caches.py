"""Build compact UMLS cache files from raw Metathesaurus RRF files.

Run this once after extracting UMLS to build three pickle caches that
DOSSIERPipeline uses for local (offline) entity tagging:

    umls_cat_mapping.pkl   — CUI → list[str] semantic type names
    snomed_umls_mapping.pkl — SNOMED concept code → CUI
    umls_name_to_cui.pkl   — lowercase English name → CUI (English preferred)

Usage
-----
    python examples/build_umls_caches.py \\
        --umls_dir data/umls/META \\
        --out_dir  data/umls

Expected inputs in --umls_dir:
    MRSTY.RRF    (~200 MB)   semantic type assignments
    MRCONSO.RRF  (~2.1 GB)   concept names across vocabularies

Runtime: ~3-5 min (MRCONSO is 2 GB; read in chunks).
Output:  three .pkl files in --out_dir, total ~50-200 MB.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


def build_cat_mapping(mrsty_path: Path) -> Dict[str, List[str]]:
    """Build CUI -> list[semantic_type_name] mapping from MRSTY.RRF.

    MRSTY columns (no header, pipe-separated):
        0: CUI, 1: TUI, 2: STN, 3: STY (semantic type name), 4: ATUI, 5: CVF

    Args:
        mrsty_path: Path to ``MRSTY.RRF``.

    Returns:
        Dict mapping each CUI string to a list of its semantic type names.
    """
    print(f"Reading {mrsty_path} ({mrsty_path.stat().st_size // 1_000_000} MB)...")
    df = pd.read_csv(
        mrsty_path, sep="|", header=None, usecols=[0, 3],
        dtype=str, encoding="utf-8",
    )
    df.columns = ["CUI", "STY"]
    mapping: dict = defaultdict(list)
    for cui, sty in zip(df["CUI"], df["STY"]):
        mapping[cui].append(sty)
    result = dict(mapping)
    print(f"  umls_cat_mapping: {len(result):,} CUIs")
    return result


def build_snomed_mapping(mrconso_path: Path) -> Dict[str, str]:
    """Build SNOMED concept code -> CUI mapping from MRCONSO.RRF.

    Filters SAB == ``SNOMEDCT_US``, keeps col 0 (CUI) and col 13 (CODE).
    Matches source repo ``constants.py`` exactly.

    Args:
        mrconso_path: Path to ``MRCONSO.RRF``.

    Returns:
        Dict mapping SNOMED concept code string to CUI string.
        Only the first CUI seen per code is kept.
    """
    print(f"Reading MRCONSO for SNOMED mapping (chunked)...")
    mapping: dict = {}
    chunk_size = 500_000
    seen = set()
    for chunk in pd.read_csv(
        mrconso_path, sep="|", header=None, usecols=[0, 11, 13],
        dtype=str, encoding="utf-8", chunksize=chunk_size,
    ):
        sub = chunk[chunk[11] == "SNOMEDCT_US"][[0, 13]].dropna()
        for cui, code in zip(sub[0], sub[13]):
            if code not in seen:
                mapping[code] = cui
                seen.add(code)
    print(f"  snomed_umls_mapping: {len(mapping):,} SNOMED codes")
    return mapping


def build_name_to_cui(mrconso_path: Path) -> Dict[str, str]:
    """Build lowercase English preferred name -> CUI mapping from MRCONSO.RRF.

    Filters: LAT=ENG (col 1), ISPREF=Y (col 6).
    Keeps first CUI per lowercased name.

    MRCONSO columns::

        0:CUI  1:LAT  2:TS  3:LUI  4:STT  5:SUI  6:ISPREF  7:AUI
        8:SAUI 9:SCUI 10:SDUI 11:SAB 12:TTY 13:CODE 14:STR 15:SRL 16:SUPPRESS

    Args:
        mrconso_path: Path to ``MRCONSO.RRF``.

    Returns:
        Dict mapping lowercased English preferred name to CUI string.
    """
    print("Reading MRCONSO for name->CUI mapping (chunked)...")
    mapping: dict = {}
    chunk_size = 500_000
    for chunk in pd.read_csv(
        mrconso_path, sep="|", header=None,
        usecols=[0, 1, 6, 14],
        dtype=str, encoding="utf-8", chunksize=chunk_size,
    ):
        # Filter English preferred terms
        sub = chunk[(chunk[1] == "ENG") & (chunk[6] == "Y")][[0, 14]].dropna()
        for cui, name in zip(sub[0], sub[14]):
            key = name.strip().lower()
            if key and key not in mapping:
                mapping[key] = cui
    print(f"  umls_name_to_cui: {len(mapping):,} English preferred names")
    return mapping


def main() -> None:
    """Parse command-line arguments and build all three UMLS cache pickles."""
    p = argparse.ArgumentParser(
        description="Build UMLS cache pickles from raw RRF files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--umls_dir", default="data/umls/META",
                   help="Directory containing MRSTY.RRF and MRCONSO.RRF")
    p.add_argument("--out_dir",  default="data/umls",
                   help="Output directory for .pkl cache files")
    args = p.parse_args()

    umls_dir = Path(args.umls_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mrsty   = umls_dir / "MRSTY.RRF"
    mrconso = umls_dir / "MRCONSO.RRF"

    for f in [mrsty, mrconso]:
        if not f.exists():
            raise FileNotFoundError(
                f"{f} not found. Extract UMLS zip first:\n"
                "  python examples/build_umls_caches.py --help"
            )

    cat_mapping    = build_cat_mapping(mrsty)
    snomed_mapping = build_snomed_mapping(mrconso)
    name_to_cui    = build_name_to_cui(mrconso)

    outputs = {
        "umls_cat_mapping.pkl":    cat_mapping,
        "snomed_umls_mapping.pkl": snomed_mapping,
        "umls_name_to_cui.pkl":    name_to_cui,
    }
    for fname, obj in outputs.items():
        dest = out_dir / fname
        with dest.open("wb") as f:
            pickle.dump(obj, f, protocol=4)
        size_mb = dest.stat().st_size / 1_000_000
        print(f"  Saved {dest}  ({size_mb:.1f} MB)")

    print("\nDone. Pass --umls_dir to DOSSIERPipeline to use local UMLS:")
    print(f"  DOSSIERPipeline(..., umls_dir='{out_dir}', prompt_variant='no_gkg')")


if __name__ == "__main__":
    main()
