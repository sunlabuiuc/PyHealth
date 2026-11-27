"""ConVIRT pretraining samples from MedMod's own phenotyping extraction.

    python -m examples.medmod_runners.pretrain_samples --data-root /path/to/mimic_data
    python -m examples.medmod_runners.pretrain_samples --data-root ... --check 20

MedMod pretrains ConVIRT with `--task` left at its default, phenotyping
(src/parser.py:13; scripts/pretrain/simclr.sh), on the paired train split, and
uses val for early stopping. This builds those samples from MedMod's
extraction in ``<data-root>/medmod_extracted/phenotyping`` (listfiles, and the
episode files of both train and val in ``train/``), paired with the images of
``<data-root>/cohorts/pretrain_cohort.csv`` (latest AP image with intime <=
StudyDateTime <= outtime and a CheXpert row: utils.py:200-211, 280-286,
342-365). Stays whose resized image is missing are dropped (one train image
is missing on PhysioNet).

EHR, as MedMod's loader (src/data/datasets.py, src/data/utils.py):
* rows with 0 < Hours < full_stay_time, strict on both sides
  (``_read_timeseries``, datasets.py:137-160); full_stay_time =
  round((outtime - intime) in hours, 3) from root/all_stays.csv
  (utils.py:194-195), passed as ``upper`` (utils.py:209-210, datasets.py:280-
  287); MedMod's two dummy rows if none is left (datasets.py:155-160);
* discretized with end = the listfile's period_length (datasets.py:196, 201),
  1-hour bins, previous-value imputation, masks (preprocessing.py:247-252):
  bins = int(period_length + 1 - 1e-6) (preprocessing.py:57), varying per stay;
* standardized with the phenotyping normalizer ph_ts1.0 (preprocessing.py:
  254-266) by ``MedModEHRSequenceProcessor``, whose collate zero-pads a batch
  at the end and returns the true lengths (utils.py:160-170, 399-406).

Images: get_preset("convirt", train=True) for train, get_preset("convirt") for
val (utils.py:114-129), loaded per batch from ``resized/<dicom_id>.jpg``.

``--check N`` rebuilds N random stays independently (pandas reader + MedMod's
own preprocessing.py Discretizer and Normalizer, with the normalizer's
statistics read by our restricted unpickler) and compares them in float32.

Aggregates only: counts and length statistics, no IDs or rows.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pyhealth.datasets.medmod.discretizer import (
    PH_NORMALIZER,
    Discretizer,
    medmod_ehr_funcs,
)
from pyhealth.datasets.medmod.extraction import MEDMOD_DUMMY_ROWS

SPLITS = ("train", "val")
VERIFIED = {"train": 7755, "val": 882}  # paper Table 2 (7,756 / 882) minus 1


def read_timeseries(
    path: Path, lower: float, upper: float
) -> Tuple[np.ndarray, List[str], bool]:
    """``_read_timeseries`` (datasets.py:137-160), line for line. Returns the
    rows, the header and whether MedMod's dummy rows were used."""
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(",")
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(",")
            t = float(mas[0])
            if t < lower:
                continue
            elif (t > lower) & (t < upper):
                ret.append(np.array(mas))
            elif t > upper:
                break
    if ret:
        return np.stack(ret), header, False
    return np.stack([np.array(r) for r in MEDMOD_DUMMY_ROWS]), header, True


def read_listfile(path: Path) -> Dict[str, Tuple[str, float]]:
    """stay_id -> (episode file name, period_length), parsed as MedMod does
    (``line.split(',')``, ``float(mas[1])``; datasets.py:96-121)."""
    out = {}
    with open(path) as f:
        f.readline()
        for line in f:
            mas = line.strip().split(",")
            out[str(int(float(mas[2])))] = (mas[0], float(mas[1]))
    return out


def full_stay_times(all_stays: Path) -> Dict[str, float]:
    """stay_id -> round((outtime - intime) in hours, 3) (utils.py:194-195)."""
    stays = pd.read_csv(all_stays, usecols=["stay_id", "intime", "outtime"])
    hours = (
        (pd.to_datetime(stays["outtime"]) - pd.to_datetime(stays["intime"]))
        .apply(lambda x: np.round(x.total_seconds() / 60 / 60, 3))
    )
    return dict(zip(stays["stay_id"].astype(str), hours))


def load_raw(data_root: Path) -> Tuple[List[Dict], Dict]:
    """Raw samples (``ehr`` unnormalized float64) and a funnel of counts."""
    ext = data_root / "medmod_extracted"
    pheno = ext / "phenotyping"
    image_dir = data_root / "mimic-cxr-jpg" / "2.0.0" / "resized"
    upper = full_stay_times(ext / "root" / "all_stays.csv")
    cohort = pd.read_csv(
        data_root / "cohorts" / "pretrain_cohort.csv",
        usecols=["split", "subject_id", "stay_id", "dicom_id"],
    )
    disc = Discretizer()
    samples, funnel = [], {}
    for split in SPLITS:
        listfile = read_listfile(pheno / f"{split}_listfile.csv")
        part = cohort[cohort["split"] == split]
        f = funnel[split] = {
            "listfile_stays": len(listfile),
            "paired_cohort": len(part),
            "not_in_listfile": 0,
            "image_missing": 0,
            "dummy_rows": 0,
            "kept": 0,
        }
        for row in part.itertuples(index=False):
            stay_id = str(row.stay_id)
            if stay_id not in listfile:
                f["not_in_listfile"] += 1
                continue
            image = image_dir / f"{row.dicom_id}.jpg"
            if not image.is_file():  # missing on PhysioNet
                f["image_missing"] += 1
                continue
            name, period_length = listfile[stay_id]
            X, header, dummy = read_timeseries(
                pheno / "train" / name, 0, upper[stay_id]
            )
            f["dummy_rows"] += int(dummy)
            ehr = disc.transform(X, header=header, end=period_length)
            samples.append(
                {
                    "patient_id": str(row.subject_id),
                    "visit_id": stay_id,
                    "ehr": ehr,
                    "image_path": str(image),
                    "split": split,
                    # kept for --check only, removed before the dataset is built
                    "_episode": str(pheno / "train" / name),
                    "_period_length": period_length,
                    "_upper": upper[stay_id],
                }
            )
            f["kept"] += 1
    return samples, funnel


def build_sample_dataset(raw: List[Dict]):
    """PyHealth SampleDataset: ehr (bins x 76, varying), image_path, split."""
    from pyhealth.datasets import create_sample_dataset
    from pyhealth.processors.medmod_ehr_processor import MedModEHRSequenceProcessor

    clean = [{k: v for k, v in s.items() if not k.startswith("_")} for s in raw]
    return create_sample_dataset(
        clean,
        input_schema={
            "ehr": (MedModEHRSequenceProcessor, {"normalizer": str(PH_NORMALIZER)}),
            "image_path": "raw",
        },
        output_schema={},
        dataset_name="medmod_pretraining",
        task_name="MedModConVIRTPretraining",
        in_memory=False,
    )


def independent_tensor(sample: Dict, medmod_src: Path) -> np.ndarray:
    """The same stay rebuilt without our code path: pandas reads the episode
    file as text, MedMod's own Discretizer and Normalizer (preprocessing.py)
    do the rest; the normalizer's means/stds come from our restricted loader
    (no raw pickle.load), then float32 as MedMod's trainer casts."""
    spec = importlib.util.spec_from_file_location(
        "medmod_preprocessing", medmod_src / "data" / "preprocessing.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    disc = mod.Discretizer(
        timestep=1.0,
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
        config_path=str(medmod_src / "data" / "resources" / "discretizer_config.json"),
    )
    df = pd.read_csv(sample["_episode"], dtype=str, keep_default_na=False)
    hours = df["Hours"].astype(float)
    df = df[(hours > 0) & (hours < sample["_upper"])]
    rows = df.values.tolist()
    if not rows:
        rows = [list(r) for r in MEDMOD_DUMMY_ROWS]
    data, new_header = disc.transform(
        rows, header=list(df.columns), end=sample["_period_length"]
    )
    # continuous columns as ehr_funcs picks them (preprocessing.py:253-254)
    fields = [i for i, x in enumerate(new_header.split(",")) if x.find("->") == -1]
    norm = mod.Normalizer(fields=fields)
    _, ours = medmod_ehr_funcs()
    norm._means, norm._stds = ours.means, ours.stds
    return norm.transform(data).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--check", type=int, default=0, help="rebuild N random stays")
    ap.add_argument("--medmod-src", help="MedMod's src/ folder (needed with --check)")
    args = ap.parse_args()
    if args.check and not args.medmod_src:
        ap.error("--check needs --medmod-src")

    raw, funnel = load_raw(Path(args.data_root))
    print("funnel (counts):")
    for split, f in funnel.items():
        print(f"  {split}: {f}")
    counts = {s: sum(r["split"] == s for r in raw) for s in SPLITS}
    print(f"kept: {counts}  expected {VERIFIED}  "
          f"{'OK' if counts == VERIFIED else 'DIFFERS'}")
    for split in SPLITS:
        bins = np.array([r["ehr"].shape[0] for r in raw if r["split"] == split])
        print(f"  {split} bins: min {bins.min()}  median {int(np.median(bins))}  "
              f"p90 {int(np.percentile(bins, 90))}  max {bins.max()}")
    period_ok = all(
        r["ehr"].shape[0] == int(r["_period_length"] + 1.0 - 1e-6) for r in raw
    )
    print(f"bins == int(period_length + 1 - 1e-6) for every stay: {period_ok}")

    dataset = build_sample_dataset(raw)
    print(f"SampleDataset: {len(dataset)} samples")

    if args.check:
        rng = np.random.default_rng(0)
        picks = rng.choice(len(raw), size=min(args.check, len(raw)), replace=False)
        identical, max_diff = 0, 0.0
        for i in picks:
            ours = dataset[int(i)]["ehr"].numpy()
            ref = independent_tensor(raw[int(i)], Path(args.medmod_src))
            same_shape = ours.shape == ref.shape
            diff = float(np.abs(ours - ref).max()) if same_shape else float("inf")
            identical += int(same_shape and np.array_equal(ours, ref))
            max_diff = max(max_diff, diff)
        print(f"check: {identical}/{len(picks)} stays bit-identical in float32 "
              f"(max |diff| {max_diff:g})")


if __name__ == "__main__":
    main()
