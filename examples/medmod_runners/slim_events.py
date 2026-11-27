"""Write a slim copy of MIMIC-IV v1.0 for the MedMod IHM task.

    python -m examples.medmod_runners.slim_events --data-root /path/to/mimic_data

reads ``<data-root>/mimiciv/1.0`` and writes ``<data-root>/mimiciv_medmod_slim/1.0``
(override either with ``--ehr-root`` / ``--out``).

Why: PyHealth caches a dataset by converting every table to parquet and then
sorting the whole event table by patient (``BaseDataset._event_transform``),
which Dask does with a temporary copy on disk. With all of chartevents and
labevents this needs more free space than a laptop usually has
(P2POutOfDiskError). This script streams the two event tables once and keeps
only rows that MedMod's IHM pipeline could use, so PyHealth works on a much
smaller table. Point ``--ehr-root`` (or ``MedModMIMIC4Dataset(ehr_root=...)``)
at ``--out`` afterwards.

A row is kept only if ALL of these hold (each is a row MedMod drops anyway):

1. itemid is one of the 114 ready itemids
   (bench/scripts/extract_subjects_iv.py:91-96, bench/preprocessing.py:141-164)
2. hadm_id is not empty (bench/scripts/validate_events.py:56-57)
3. value is not empty (bench/subject.py:30; MedMod's full pandas-1.1.5 NA rule
   is still applied later by the task, so strings like "NA" are kept here)
4. charttime is within [intime - 1 h, intime + 49 h] of ANY ICU stay of the same
   subject. MedMod only uses rows with -1e-6 < Hours < 48 + 1e-6 of the stay they
   belong to (create_in_hospital_mortality.py:47-48); a row can belong to a stay
   through its own stay_id or by falling inside that stay's [intime, outtime]
   (bench/subject.py:39-45), and in both cases its Hours are measured from that
   stay's intime, so the 1 h margins keep every row that can reach a window.

Not affected: cohort filters, labels, LOS (from icustays.los), diagnoses, CXR.
Rows outside every 48 h window never reach a sample. One approximation that
already existed widens slightly: the task infers each subject's value type
(for same-timestamp tie-breaks only) from the rows it sees.

Values are copied as raw text (no NA parsing, no float conversion), and the
output keeps only the columns the v1.0 config reads. patients, admissions,
icustays, diagnoses_icd and d_icd_diagnoses are symlinked, not copied.
"""

from __future__ import annotations

import argparse
import gzip
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pyhealth.datasets.medmod.extraction import read_itemid_to_variable_map

MARGIN_BEFORE = pd.Timedelta(hours=1)
MARGIN_AFTER = pd.Timedelta(hours=49)

COLUMNS = {
    "icu/chartevents.csv.gz": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "charttime",
        "itemid",
        "value",
        "valuenum",
    ],
    "hosp/labevents.csv.gz": [
        "subject_id",
        "hadm_id",
        "charttime",
        "itemid",
        "value",
        "valuenum",
    ],
}
LINKED = (
    "core/patients.csv.gz",
    "core/admissions.csv.gz",
    "icu/icustays.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "hosp/d_icd_diagnoses.csv.gz",
)


def _resolve(root: Path, rel: str) -> Path:
    """Accept .csv.gz or .csv, like PyHealth's ``_csv_tsv_gz_path``."""
    p = root / rel
    if p.exists():
        return p
    alt = root / rel[:-3] if rel.endswith(".gz") else root / f"{rel}.gz"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Neither {p} nor {alt} exists")


def load_windows(icustays_csv: str | Path) -> pd.DataFrame:
    """One row per ICU stay: subject_id (str), lo, hi (timestamps)."""
    stays = pd.read_csv(
        icustays_csv, dtype=str, keep_default_na=False, usecols=["subject_id", "intime"]
    )
    intime = pd.to_datetime(stays["intime"], format="%Y-%m-%d %H:%M:%S")
    return pd.DataFrame(
        {
            "subject_id": stays["subject_id"],
            "lo": intime - MARGIN_BEFORE,
            "hi": intime + MARGIN_AFTER,
        }
    )


def filter_chunk(
    chunk: pd.DataFrame, itemids: set, windows: pd.DataFrame
) -> pd.DataFrame:
    """Rules 1-4 on one chunk of raw-string rows; returns kept rows, same order."""
    keep = (
        chunk["itemid"].isin(itemids)
        & (chunk["hadm_id"] != "")
        & (chunk["value"] != "")
    )
    sub = chunk[keep]
    if sub.empty:
        return sub
    t = pd.to_datetime(sub["charttime"], format="%Y-%m-%d %H:%M:%S")
    probe = pd.DataFrame(
        {"row": sub.index, "subject_id": sub["subject_id"].values, "t": t.values}
    )
    m = probe.merge(windows, on="subject_id", how="inner")
    in_win = m.loc[(m["t"] >= m["lo"]) & (m["t"] <= m["hi"]), "row"].unique()
    return sub.loc[sub.index.isin(in_win)]


@dataclass
class TableReport:
    table: str
    rows_in: int = 0
    rows_out: int = 0

    def __str__(self) -> str:
        pct = 100.0 * self.rows_out / self.rows_in if self.rows_in else 0.0
        return (
            f"{self.table}: kept {self.rows_out:,} of {self.rows_in:,} rows "
            f"({pct:.1f}%)"
        )


def slim_table(
    src: Path,
    dst: Path,
    columns: list,
    itemids: set,
    windows: pd.DataFrame,
    chunksize: int = 2_000_000,
    progress: bool = True,
) -> TableReport:
    report = TableReport(table=src.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    start = time.time()
    reader = pd.read_csv(
        src,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        usecols=columns,
        chunksize=chunksize,
    )
    try:
        # One continuous gzip stream (not one member per chunk), header once.
        with gzip.open(tmp, "wt", newline="") as fh:
            pd.DataFrame(columns=columns).to_csv(fh, index=False)
            for i, chunk in enumerate(reader):
                chunk = chunk[columns]
                report.rows_in += len(chunk)
                out = filter_chunk(chunk, itemids, windows)
                report.rows_out += len(out)
                out.to_csv(fh, header=False, index=False)
                if progress:
                    print(
                        f"  {src.name}: chunk {i + 1}, {report.rows_in:,} rows read, "
                        f"{report.rows_out:,} kept, {time.time() - start:.0f}s",
                        flush=True,
                    )
    except BaseException:
        tmp.unlink(missing_ok=True)  # never leave a half-written table behind
        raise
    os.replace(tmp, dst)
    return report


def build_slim_copy(
    ehr_root: str | Path,
    out: str | Path,
    itemid_map: str | Path | None = None,
    chunksize: int = 2_000_000,
    progress: bool = True,
) -> list:
    ehr_root, out = Path(ehr_root).resolve(), Path(out).resolve()
    if out == ehr_root:
        raise ValueError("--out must differ from --ehr-root")
    itemids = {str(i) for i in read_itemid_to_variable_map(itemid_map)["itemid"]}
    windows = load_windows(_resolve(ehr_root, "icu/icustays.csv.gz"))

    for rel in LINKED:
        src = _resolve(ehr_root, rel)
        dst = out / src.relative_to(ehr_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)

    reports = []
    for rel, cols in COLUMNS.items():
        src = _resolve(ehr_root, rel)
        dst = out / rel  # always written as .csv.gz
        reports.append(
            slim_table(src, dst, cols, itemids, windows, chunksize, progress)
        )
        if progress:
            print(reports[-1], flush=True)
    return reports


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", help="local MIMIC folder that contains mimiciv/1.0")
    ap.add_argument(
        "--ehr-root", help="MIMIC-IV v1.0 folder (default: <data-root>/mimiciv/1.0)"
    )
    ap.add_argument(
        "--out",
        help="where to write the slim copy "
        "(default: <data-root>/mimiciv_medmod_slim/1.0)",
    )
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args(argv)
    if not args.data_root and not (args.ehr_root and args.out):
        ap.error("give --data-root, or both --ehr-root and --out")
    root = Path(args.data_root) if args.data_root else None
    ehr_root = Path(args.ehr_root) if args.ehr_root else root / "mimiciv" / "1.0"
    out = Path(args.out) if args.out else root / "mimiciv_medmod_slim" / "1.0"
    if not ehr_root.is_dir():
        ap.error(f"MIMIC-IV folder not found: {ehr_root}")
    reports = build_slim_copy(ehr_root, out, chunksize=args.chunksize)
    print("\nDone.")
    for r in reports:
        print(r)


if __name__ == "__main__":
    main()
