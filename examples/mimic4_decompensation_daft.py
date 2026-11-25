# Authors: Christopher Perez, Kai Cheung (kaicc2), Sarah Lee
# Paper: MedMod: Multimodal Benchmark for Medical Prediction Tasks with Electronic
#   Health Records and Chest X-Ray Scans (Elsharief et al., CHIL 2025)
# Paper link: https://proceedings.mlr.press/v287/elsharief25a.html
# Description: DAFT on MIMIC-IV + MIMIC-CXR decompensation and mortality, ablations.

"""DAFT on MIMIC-IV + MIMIC-CXR: decompensation and mortality, with ablations.

Reproduces part of MedMod (Elsharief et al., CHIL 2025,
https://proceedings.mlr.press/v287/elsharief25a.html) in PyHealth:

- task: :class:`~pyhealth.tasks.MedModDecompensationMIMIC4` (death in the next 24 h,
  predicted every few hours) and the upstream
  :class:`~pyhealth.tasks.EHRCXRMortalityMIMIC4`
- model: :class:`~pyhealth.models.DAFT`

Ablations (none of them are in the MedMod paper):

- ``direction``: ``cxr_to_ehr`` (MedMod's implementation) vs ``ehr_to_cxr`` (the
  original DAFT paper)
- ``missing_cxr``: stays without an X-ray skip the transform, or use a learned
  stand-in for the image
- ``layer_after``: which ResNet stage the transform is attached to
- an EHR-only baseline (DAFT without the image input is a two-layer LSTM)

By default the script writes a small synthetic MIMIC-IV + MIMIC-CXR tree (random
values and images, no patient data) and runs the whole pipeline on it in about a
minute on a CPU, so it runs without credentialed data access. Those numbers say
nothing about performance. For the real data (credentialed PhysioNet access), pass
the roots. These are the settings of the results below; run it once per seed
(``--seed 0``, ``1`` and ``2``) and average the test scores:

    python examples/mimic4_decompensation_daft.py \\
        --ehr-root /path/to/mimiciv/3.1 --cxr-root /path/to/mimic-cxr-jpg/2.1.0 \\
        --task mortality --paired-only --split 0.7 0.1 0.2 --seed 0 \\
        --image-size 224 --backbone resnet34 --hidden 256 \\
        --epochs 15 --lr 3e-4 --batch-size 32 --pretrained

For the decompensation table, use the same command with ``--task decompensation``
(the task's defaults: a prediction every 4 h, a 48 h window, a 24 h horizon).

Results on the real data (MIMIC-IV 3.1, paired stays, 15 epochs, lr 3e-4, batch 32,
ResNet-34 from ImageNet weights, best epoch by validation AUROC, mean +- std over
three seeds, each a different 70/10/20 patient split). They were produced with these
tasks, models and settings and the PyHealth ``Trainer``, by a runner in our project
repository that also trains :class:`~pyhealth.models.MedFuse`; this script covers
the EHR-only and DAFT rows, not the MedFuse ones. Small differences from the table are expected across
hardware (the runs used an Apple GPU).

In-hospital mortality (``EHRCXRMortalityMIMIC4``, 4,689 / 655 / 1,338 stays for
seed 0):

    model                  test AUROC      test AUPRC      MedMod (AUROC / AUPRC)
    EHR-only LSTM          0.822 +- 0.020  0.480 +- 0.048  0.829 / 0.502
    MedFuse                0.815 +- 0.017  0.478 +- 0.047  0.819 / 0.482
    DAFT, cxr_to_ehr       0.818 +- 0.022  0.496 +- 0.048  0.826 / 0.494
    DAFT, ehr_to_cxr       0.821 +- 0.016  0.480 +- 0.040  -

Decompensation (``MedModDecompensationMIMIC4``, every 4 h, about 16,800 test
samples per split, 2.3 to 3.0% positive):

    model                  test AUROC      test AUPRC
    EHR-only LSTM          0.919 +- 0.009  0.456 +- 0.063
    MedFuse                0.916 +- 0.013  0.447 +- 0.049
    DAFT, cxr_to_ehr       0.922 +- 0.011  0.464 +- 0.052
    DAFT, ehr_to_cxr       0.923 +- 0.011  0.461 +- 0.065

Findings: with the same training for every model, the X-ray adds little. On
mortality, EHR-only and DAFT are within the spread of MedMod's numbers. On
decompensation, no model clearly beats EHR-only once the image must come from
before the prediction time. MedMod's decompensation numbers are not comparable: it
samples every hour (Sec. 4.1), and its code (``get_recent_cxr`` in
``src/data/utils.py``, github.com/nyuad-cai/MedMod) pairs one X-ray per stay with
every hour, so an early hour can see a later image.
"""

from __future__ import annotations

import argparse
import gzip
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pyhealth.datasets import MIMIC4Dataset, PatientSplit, get_dataloader
from pyhealth.models import DAFT
from pyhealth.tasks import EHRCXRMortalityMIMIC4, MedModDecompensationMIMIC4
from pyhealth.trainer import Trainer

ADMIT_TIME = pd.Timestamp("2150-01-01 08:00:00")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# heart rate, systolic blood pressure, GCS eye opening
CHART_ITEMS = {220045: (85.0, "bpm"), 220179: (120.0, "mmHg"), 220739: (3.0, None)}
GLUCOSE_ITEM = 50931

# name, DAFT options; None means the EHR-only baseline
ABLATIONS = [
    ("ehr_only", None),
    ("daft_cxr_to_ehr", {"direction": "cxr_to_ehr"}),
    ("daft_ehr_to_cxr", {"direction": "ehr_to_cxr"}),
    ("daft_learned_missing", {"direction": "cxr_to_ehr", "missing_cxr": "learned"}),
    ("daft_layer_after_2", {"direction": "cxr_to_ehr", "layer_after": 2}),
]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Writes a gzipped CSV, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as file:
        frame.to_csv(file, index=False)


def write_synthetic_mimic(root: Path, n_patients: int, seed: int) -> tuple[Path, Path]:
    """Writes random tables and images in the MIMIC-IV and MIMIC-CXR layout.

    Every patient has one 72-hour ICU stay; one in three dies at hour 60, and
    their vital signs drift so the labels can be learned. Two thirds of the stays
    have an AP image at hour 6, the rest only at hour 66 (never usable).

    Args:
        root: Directory to write to.
        n_patients: Number of patients.
        seed: Seed of the random values.

    Returns:
        The EHR root and the CXR root.
    """
    rng = np.random.default_rng(seed)
    ehr_root, cxr_root = root / "mimic-iv", root / "mimic-cxr"
    rows: dict[str, list[dict]] = {
        k: [] for k in ("patients", "admissions", "stays", "charts", "labs", "images")
    }
    for index in range(n_patients):
        subject_id, hadm_id, stay_id = 10_000_000 + index, 20_000_000 + index, index
        died = index % 3 == 0
        admit = ADMIT_TIME + pd.Timedelta(days=index)
        outtime = admit + pd.Timedelta(hours=72)
        death = admit + pd.Timedelta(hours=60)
        rows["patients"].append(
            {
                "subject_id": subject_id,
                "gender": "F",
                "anchor_age": 60,
                "anchor_year": 2150,
                "anchor_year_group": "2008 - 2010",
                "dod": None,
            }
        )
        rows["admissions"].append(
            {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "admittime": admit.strftime(TIME_FORMAT),
                "dischtime": outtime.strftime(TIME_FORMAT),
                "deathtime": death.strftime(TIME_FORMAT) if died else None,
                "admission_type": "URGENT",
                "admission_location": "EMERGENCY ROOM",
                "discharge_location": "DIED" if died else "HOME",
                "insurance": "Other",
                "language": "ENGLISH",
                "marital_status": "SINGLE",
                "race": "OTHER",
                "hospital_expire_flag": int(died),
            }
        )
        rows["stays"].append(
            {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "stay_id": stay_id,
                "first_careunit": "Medical Intensive Care Unit (MICU)",
                "last_careunit": "Medical Intensive Care Unit (MICU)",
                "intime": admit.strftime(TIME_FORMAT),
                "outtime": outtime.strftime(TIME_FORMAT),
                "los": 3.0,
            }
        )
        for hour in range(0, 72, 2):
            time = (admit + pd.Timedelta(hours=hour, minutes=5)).strftime(TIME_FORMAT)
            drift = 0.4 * hour if died else 0.0
            for itemid, (mean, unit) in CHART_ITEMS.items():
                value = 4.0 if itemid == 220739 else mean + rng.normal(0, 5) + drift
                rows["charts"].append(
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "stay_id": stay_id,
                        "charttime": time,
                        "storetime": time,
                        "itemid": itemid,
                        "value": str(value),
                        "valuenum": value,
                        "valueuom": unit,
                        "warning": 0,
                    }
                )
            rows["labs"].append(
                {
                    "labevent_id": len(rows["labs"]),
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "specimen_id": 0,
                    "itemid": GLUCOSE_ITEM,
                    "order_provider_id": None,
                    "charttime": time,
                    "storetime": time,
                    "value": "110",
                    "valuenum": 110.0,
                    "valueuom": "mg/dL",
                    "ref_range_lower": 70,
                    "ref_range_upper": 100,
                    "flag": None,
                    "priority": None,
                    "comments": None,
                }
            )
        image_time = admit + pd.Timedelta(hours=66 if index % 3 == 1 else 6)
        study_id, dicom_id = 50_000_000 + index, f"synthetic-{index}"
        path = (
            cxr_root
            / "files"
            / f"p{str(subject_id)[:2]}"
            / f"p{subject_id}"
            / f"s{study_id}"
            / f"{dicom_id}.jpg"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8)).save(path)
        rows["images"].append(
            {
                "dicom_id": dicom_id,
                "subject_id": subject_id,
                "study_id": study_id,
                "PerformedProcedureStepDescription": "CHEST (PORTABLE AP)",
                "ViewPosition": "AP",
                "Rows": 64,
                "Columns": 64,
                "StudyDate": image_time.strftime("%Y%m%d"),
                "StudyTime": image_time.strftime("%H%M%S"),
                "ProcedureCodeSequence_CodeMeaning": "CHEST (PORTABLE AP)",
                "ViewCodeSequence_CodeMeaning": "antero-posterior",
                "PatientOrientationCodeSequence_CodeMeaning": "Erect",
            }
        )

    write_csv(pd.DataFrame(rows["patients"]), ehr_root / "hosp" / "patients.csv.gz")
    write_csv(pd.DataFrame(rows["admissions"]), ehr_root / "hosp" / "admissions.csv.gz")
    write_csv(pd.DataFrame(rows["stays"]), ehr_root / "icu" / "icustays.csv.gz")
    write_csv(pd.DataFrame(rows["charts"]), ehr_root / "icu" / "chartevents.csv.gz")
    write_csv(pd.DataFrame(rows["labs"]), ehr_root / "hosp" / "labevents.csv.gz")
    write_csv(
        pd.DataFrame(
            [
                {
                    "itemid": GLUCOSE_ITEM,
                    "label": "Glucose",
                    "fluid": "Blood",
                    "category": "Chemistry",
                }
            ]
        ),
        ehr_root / "hosp" / "d_labitems.csv.gz",
    )
    write_csv(
        pd.DataFrame(rows["images"]), cxr_root / "mimic-cxr-2.0.0-metadata.csv.gz"
    )
    return ehr_root, cxr_root


def run_ablations(train, val, test, args: argparse.Namespace) -> pd.DataFrame:
    """Trains every ablation on the same split and returns the test scores."""
    loaders = [
        get_dataloader(part, batch_size=args.batch_size, shuffle=part is train)
        for part in (train, val, test)
    ]
    results = []
    for name, options in ABLATIONS:
        torch.manual_seed(args.seed)
        common = {
            "ehr_hidden_dim": args.hidden,
            "cxr_backbone": args.backbone,
            "cxr_pretrained": args.pretrained,
        }
        if options is None:  # no image input: a two-layer LSTM
            model = DAFT(train, cxr_feature_key="no_image", cxr_mask_key=None, **common)
        else:
            model = DAFT(train, **common, **options)
        trainer = Trainer(model, metrics=["roc_auc", "pr_auc"], enable_logging=False)
        trainer.train(
            loaders[0],
            loaders[1],
            epochs=args.epochs,
            optimizer_params={"lr": args.lr},
            monitor="roc_auc",
        )
        scores = trainer.evaluate(loaders[2])
        results.append(
            {"ablation": name, "auroc": scores["roc_auc"], "auprc": scores["pr_auc"]}
        )
    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ehr-root", help="MIMIC-IV root; synthetic data if unset")
    parser.add_argument("--cxr-root", help="MIMIC-CXR-JPG root")
    parser.add_argument(
        "--task", choices=["decompensation", "mortality"], default="decompensation"
    )
    parser.add_argument("--patients", type=int, default=45)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=(0.6, 0.2, 0.2),
        metavar=("TRAIN", "VAL", "TEST"),
        help="patient split ratios (the results above use 0.7 0.1 0.2)",
    )
    parser.add_argument(
        "--paired-only",
        action="store_true",
        help="keep only samples with an X-ray (the paired cohort of the results)",
    )
    return parser.parse_args()


def main() -> None:
    """Builds the task, runs the ablations and prints the results."""
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        ehr_root, cxr_root, cache = args.ehr_root, args.cxr_root, None
        if ehr_root is None:
            print("Synthetic data: the scores below say nothing about performance.")
            ehr_root, cxr_root = write_synthetic_mimic(
                Path(tmp), args.patients, args.seed
            )
            cache = str(Path(tmp) / "cache")
        dataset = MIMIC4Dataset(
            ehr_root=str(ehr_root),
            cxr_root=str(cxr_root),
            ehr_tables=["chartevents", "labevents"],
            cxr_tables=["metadata"],
            cache_dir=cache,
            num_workers=1,
        )
        if args.task == "decompensation":
            task = MedModDecompensationMIMIC4(
                image_size=args.image_size, paired_only=args.paired_only
            )
        else:
            task = EHRCXRMortalityMIMIC4(
                image_size=args.image_size, paired_only=args.paired_only
            )
        train, val, test = dataset.set_task(
            task, split=PatientSplit(ratios=tuple(args.split), seed=args.seed)
        )
        label = next(iter(task.output_schema))
        for name, part in (("train", train), ("val", val), ("test", test)):
            positives = sum(float(s[label]) for s in part)
            images = sum(float(s["cxr_mask"]) for s in part)
            print(
                f"{name}: {len(part)} samples, {positives:.0f} positive, "
                f"{images:.0f} with an image"
            )
        table = run_ablations(train, val, test, args)
        print(table.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
