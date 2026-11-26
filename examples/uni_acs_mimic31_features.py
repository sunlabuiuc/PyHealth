from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path.home() / "Desktop" / "mimiciv-3.1"
DERIVED_PATH = DATA_PATH / "derived"

LAB_NAMES = {
    "Creatinine": "creatinine",
    "Urea Nitrogen": "bun",
    "Sodium": "sodium",
    "Potassium": "potassium",
    "Chloride": "chloride",
    "Bicarbonate": "bicarbonate",
    "Glucose": "glucose",
    "White Blood Cells": "wbc",
    "Hemoglobin": "hemoglobin",
    "Platelet Count": "platelets",
    "Lactate": "lactate",
    "Bilirubin, Total": "bilirubin",
}


def load_cohort(derived_path=DERIVED_PATH):
    cohort = pd.read_csv(
        derived_path / "sepsis_icd_cohort.csv",
        parse_dates=["admittime", "dischtime"],
    )

    # Predict at 24 hours after admission
    cohort["cutoff"] = cohort["admittime"] + pd.Timedelta(hours=24)

    # Exclude admissions ending before the prediction time
    cohort = cohort[cohort["dischtime"] > cohort["cutoff"]].copy()

    print("Eligible admissions:", len(cohort))
    print("Mortality:", round(cohort["hospital_mortality"].mean(), 4))

    return cohort


def get_lab_items(data_path=DATA_PATH):
    items = pd.read_csv(
        data_path / "hosp" / "d_labitems.csv.gz"
    )

    # Use blood measurements only
    items = items[items["fluid"] == "Blood"].copy()
    items["feature"] = items["label"].map(LAB_NAMES)
    items = items.dropna(subset=["feature"])

    print("\nSelected lab items:")
    print(items[["itemid", "label", "feature"]].to_string(index=False))

    return dict(zip(items["itemid"], items["feature"]))


def extract_labs(cohort, item_map, *, data_path=DATA_PATH, require_available=False):
    """Compute admission-level lab means, optionally requiring timely results."""
    admission_ids = set(cohort["hadm_id"])
    item_ids = set(item_map)

    times = cohort.set_index("hadm_id")[["admittime", "cutoff"]]

    results = []

    columns = ["hadm_id", "itemid", "charttime", "valuenum"]
    if require_available:
        columns.append("storetime")

    reader = pd.read_csv(
        data_path / "hosp" / "labevents.csv.gz",
        usecols=columns,
        chunksize=200000,
    )

    for i, chunk in enumerate(reader, start=1):
        chunk = chunk[
            chunk["hadm_id"].isin(admission_ids)
            & chunk["itemid"].isin(item_ids)
        ].copy()

        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        chunk = chunk.join(times, on="hadm_id")

        chunk = chunk[
            (chunk["charttime"] >= chunk["admittime"])
            & (chunk["charttime"] <= chunk["cutoff"])
        ].copy()

        if require_available:
            storetime = pd.to_datetime(chunk["storetime"], format="ISO8601", errors="coerce")
            chunk = chunk[storetime.le(chunk["cutoff"]) & np.isfinite(chunk["valuenum"])]

        chunk = chunk.dropna(subset=["valuenum"])

        if chunk.empty:
            continue

        chunk["feature"] = chunk["itemid"].map(item_map)

        # Aggregate within each chunk; combine later
        summary = (
            chunk.groupby(["hadm_id", "feature"])["valuenum"]
            .agg(["sum", "count"])
            .reset_index()
        )
        results.append(summary)

        if i % 100 == 0:
            print(f"Processed {i} chunks...")

    if not results:
        raise ValueError("No matching laboratory measurements found.")

    all_results = pd.concat(results, ignore_index=True)

    combined = (
        all_results.groupby(["hadm_id", "feature"])[["sum", "count"]]
        .sum()
        .reset_index()
    )

    combined["value"] = combined["sum"] / combined["count"]

    labs = combined.pivot(
        index="hadm_id",
        columns="feature",
        values="value",
    ).reset_index()

    return labs


def main():
    print("MIMIC-IV 3.1: First-24-hour sepsis feature extraction")

    cohort = load_cohort()
    item_map = get_lab_items()
    labs = extract_labs(cohort, item_map)

    data = cohort.merge(
        labs,
        on="hadm_id",
        how="left",
        validate="one_to_one",
    )

    data["gender_male"] = (data["gender"] == "M").astype(int)

    features = ["age", "gender_male"] + list(LAB_NAMES.values())

    for col in features:
        if col not in data:
            data[col] = np.nan

    output = data[
        ["subject_id", "hadm_id", "anchor_year_group",
         "hospital_mortality"] + features
    ].copy()

    print("\nFinal dataset shape:", output.shape)
    print("Mortality:", round(output["hospital_mortality"].mean(), 4))

    print("\nFeature missingness:")
    print(output[features].isna().mean().round(3))

    print("\nAnchor year groups:")
    print(output["anchor_year_group"].value_counts().sort_index())

    output_path = DERIVED_PATH / "sepsis_24h_features.csv"
    output.to_csv(output_path, index=False)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
