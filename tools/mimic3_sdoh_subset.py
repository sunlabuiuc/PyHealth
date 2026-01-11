import argparse
import gzip
from pathlib import Path

import pandas as pd


def write_gzip_csv(df: pd.DataFrame, path: Path, header: bool) -> None:
    mode = "wt" if header else "at"
    with gzip.open(path, mode) as f:
        df.to_csv(f, index=False, header=header)


def filter_csv(
    source: Path,
    dest: Path,
    column: str,
    allowed: set,
    chunksize: int = 200_000,
) -> None:
    header_written = False
    for chunk in pd.read_csv(source, chunksize=chunksize):
        filtered = chunk[chunk[column].astype(str).isin(allowed)]
        if filtered.empty:
            continue
        write_gzip_csv(filtered, dest, header=not header_written)
        header_written = True
    if not header_written:
        write_gzip_csv(pd.DataFrame(columns=pd.read_csv(source, nrows=0).columns), dest, header=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small MIMIC-III subset for SDOH evaluation.")
    parser.add_argument("--mimic-root", required=True, help="Path to full MIMIC-III root.")
    parser.add_argument("--label-csv-path", required=True, help="Path to sdoh_icd9_dataset.csv.")
    parser.add_argument("--output-root", required=True, help="Output folder for the subset.")
    parser.add_argument("--note-chunksize", type=int, default=200_000)
    parser.add_argument("--max-hadm", type=int, default=0, help="Limit to first N HADM_IDs.")
    args = parser.parse_args()

    mimic_root = Path(args.mimic_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.label_csv_path)
    hadm_list = labels["HADM_ID"].astype(str).unique().tolist()
    if args.max_hadm and args.max_hadm > 0:
        hadm_list = hadm_list[: args.max_hadm]
    hadm_ids = set(hadm_list)
    subject_ids = set(labels["SUBJECT_ID"].astype(str).unique().tolist())

    admissions_src = mimic_root / "ADMISSIONS.csv.gz"
    icustays_src = mimic_root / "ICUSTAYS.csv.gz"
    patients_src = mimic_root / "PATIENTS.csv.gz"
    noteevents_src = mimic_root / "NOTEEVENTS.csv.gz"

    print("Filtering ADMISSIONS...")
    filter_csv(admissions_src, output_root / "ADMISSIONS.csv.gz", "HADM_ID", hadm_ids)

    print("Filtering ICUSTAYS...")
    filter_csv(icustays_src, output_root / "ICUSTAYS.csv.gz", "HADM_ID", hadm_ids)

    print("Filtering PATIENTS...")
    filter_csv(patients_src, output_root / "PATIENTS.csv.gz", "SUBJECT_ID", subject_ids)

    print("Filtering NOTEEVENTS...")
    filter_csv(
        noteevents_src,
        output_root / "NOTEEVENTS.csv.gz",
        "HADM_ID",
        hadm_ids,
        chunksize=args.note_chunksize,
    )

    print("Subset written to:", output_root)


if __name__ == "__main__":
    main()
