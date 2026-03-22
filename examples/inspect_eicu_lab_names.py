import os
from collections import Counter
import gzip
import csv

ROOT = "/home/medukonis/Documents/Illinois/Spring_2026/CS598_Deep_Learning_For_Healthcare/Project/Datasets/eicu-collaborative-research-database-2.0"


def find_lab_file(root: str) -> str:
    candidates = [
        os.path.join(root, "lab.csv.gz"),
        os.path.join(root, "lab.csv"),
        os.path.join(root, "lab.parquet"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not find lab.csv.gz, lab.csv, or lab.parquet in dataset root.")


def inspect_csv(path: str, max_rows: int = 200000):
    counter = Counter()
    opener = gzip.open if path.endswith(".gz") else open

    with opener(path, "rt", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if "labname" not in reader.fieldnames:
            raise ValueError(f"'labname' column not found. Columns: {reader.fieldnames}")

        for i, row in enumerate(reader):
            name = row.get("labname")
            if name:
                counter[name] += 1
            if i + 1 >= max_rows:
                break

    return counter


def inspect_parquet(path: str):
    import polars as pl

    df = pl.read_parquet(path, columns=["labname"])
    counts = (
        df.group_by("labname")
        .len()
        .sort("len", descending=True)
    )
    return counts


def main():
    lab_path = find_lab_file(ROOT)
    print(f"Using lab file: {lab_path}\n")

    if lab_path.endswith(".parquet"):
        counts = inspect_parquet(lab_path)
        print("Top lab names:\n")
        print(counts.head(50))
    else:
        counter = inspect_csv(lab_path)
        print("Top lab names:\n")
        for name, count in counter.most_common(50):
            print(f"{name}\t{count}")


if __name__ == "__main__":
    main()
