"""MedFuse on EHR + CXR tasks: smoke test on synthetic MIMIC-IV-shaped data.

SMOKE TEST ONLY. MIMIC-CXR is not used here: this script writes a small
synthetic MIMIC-IV + MIMIC-CXR tree to a temporary directory (random values
and images, no patient data) and runs the real pipeline on it, from
``MIMIC4Dataset`` through ``set_task`` to training
:class:`~pyhealth.models.MedFuse`. The metrics printed below say nothing about
performance. No credentialed data is needed, but the phenotyping task
downloads the public ICD to CCS mappings the first time it runs.

To run on the real data (credentialed PhysioNet access to MIMIC-IV and
MIMIC-CXR-JPG), pass their roots:

    python examples/ehr_cxr_medfuse_mimic4.py \
        --ehr-root /path/to/mimic-iv/2.2 \
        --cxr-root /path/to/mimic-cxr-jpg/2.0.0 \
        --image-size 224 --epochs 10

For reference, Hayat et al. (2022, MLHC) report for MedFuse (OPTIMAL) on the
partial test set: in-hospital mortality 0.874 AUROC / 0.567 AUPRC.

Paper:
    Hayat et al. "MedFuse: Multi-modal fusion with clinical time-series data
    and chest X-ray images." MLHC 2022. https://arxiv.org/abs/2207.07027
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
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import MedFuse
from pyhealth.tasks import EHRCXRMortalityMIMIC4, EHRCXRPhenotypingMIMIC4

ADMIT_TIME = pd.Timestamp("2150-01-01 08:00:00")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# Heart rate, systolic blood pressure and GCS eye opening.
CHART_ITEMS = {220045: (85.0, "bpm"), 220179: (120.0, "mmHg"), 220739: (3.0, None)}
GLUCOSE_ITEM = 50931


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Writes a gzipped CSV, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as file:
        frame.to_csv(file, index=False)


def write_synthetic_mimic(root: Path, n_patients: int, seed: int) -> tuple[Path, Path]:
    """Writes random tables and images in the MIMIC-IV and MIMIC-CXR layout.

    Every patient has one 72-hour ICU stay. Two thirds of the stays have an AP
    image in the first 48 hours; the others have only an image taken after
    48 hours, which the mortality task must ignore.

    Args:
        root: Directory to write to.
        n_patients: Number of patients.
        seed: Seed of the random values.

    Returns:
        The EHR root and the CXR root.
    """
    rng = np.random.default_rng(seed)
    ehr_root, cxr_root = root / "mimic-iv", root / "mimic-cxr"
    patients, admissions, stays, charts, labs, diagnoses, images = (
        [], [], [], [], [], [], []
    )
    for index in range(n_patients):
        subject_id = 10000000 + index
        hadm_id = 20000000 + index
        stay_id = 30000000 + index
        died = int(index % 3 == 0)
        admit = ADMIT_TIME + pd.Timedelta(days=index)
        outtime = admit + pd.Timedelta(hours=72)
        patients.append(
            {"subject_id": subject_id, "gender": "F", "anchor_age": 60,
             "anchor_year": 2150, "anchor_year_group": "2008 - 2010", "dod": None}
        )
        admissions.append(
            {"subject_id": subject_id, "hadm_id": hadm_id,
             "admittime": admit.strftime(TIME_FORMAT),
             "dischtime": (outtime + pd.Timedelta(hours=24)).strftime(TIME_FORMAT),
             "deathtime": None, "admission_type": "URGENT",
             "admission_location": "EMERGENCY ROOM", "discharge_location": "HOME",
             "insurance": "Other", "language": "ENGLISH", "marital_status": "SINGLE",
             "race": "OTHER", "hospital_expire_flag": died}
        )
        stays.append(
            {"subject_id": subject_id, "hadm_id": hadm_id, "stay_id": stay_id,
             "first_careunit": "Medical Intensive Care Unit (MICU)",
             "last_careunit": "Medical Intensive Care Unit (MICU)",
             "intime": admit.strftime(TIME_FORMAT),
             "outtime": outtime.strftime(TIME_FORMAT), "los": 3.0}
        )
        for hour in range(0, 72, 4):
            time = (admit + pd.Timedelta(hours=hour, minutes=5)).strftime(TIME_FORMAT)
            for itemid, (mean, unit) in CHART_ITEMS.items():
                value = 4.0 if itemid == 220739 else mean + rng.normal(0, 5) + 10 * died
                charts.append(
                    {"subject_id": subject_id, "hadm_id": hadm_id, "stay_id": stay_id,
                     "charttime": time, "itemid": itemid, "value": str(value),
                     "valuenum": value, "valueuom": unit}
                )
            labs.append(
                {"subject_id": subject_id, "hadm_id": hadm_id,
                 "itemid": GLUCOSE_ITEM, "charttime": time, "storetime": time,
                 "value": "110", "valuenum": 110.0, "valueuom": "mg/dL", "flag": None}
            )
        for code in ("99591", "4280") if died else ("4280",):
            diagnoses.append(
                {"subject_id": subject_id, "hadm_id": hadm_id, "seq_num": 1,
                 "icd_code": code, "icd_version": 9, "dischtime": None}
            )

        # One stay in three has its only image after the prediction time.
        image_time = admit + pd.Timedelta(hours=60 if index % 3 == 1 else 30)
        study_id, dicom_id = 50000000 + index, f"synthetic-{index}"
        path = (
            cxr_root / "files" / f"p{str(subject_id)[:2]}" / f"p{subject_id}"
            / f"s{study_id}" / f"{dicom_id}.jpg"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        images.append(
            {"dicom_id": dicom_id, "subject_id": subject_id, "study_id": study_id,
             "PerformedProcedureStepDescription": "CHEST (PORTABLE AP)",
             "ViewPosition": "AP", "Rows": 64, "Columns": 64,
             "StudyDate": image_time.strftime("%Y%m%d"),
             "StudyTime": image_time.strftime("%H%M%S"),
             "ProcedureCodeSequence_CodeMeaning": "CHEST (PORTABLE AP)",
             "ViewCodeSequence_CodeMeaning": "antero-posterior",
             "PatientOrientationCodeSequence_CodeMeaning": "Erect"}
        )

    write_csv(pd.DataFrame(patients), ehr_root / "hosp" / "patients.csv.gz")
    write_csv(pd.DataFrame(admissions), ehr_root / "hosp" / "admissions.csv.gz")
    write_csv(pd.DataFrame(stays), ehr_root / "icu" / "icustays.csv.gz")
    write_csv(pd.DataFrame(charts), ehr_root / "icu" / "chartevents.csv.gz")
    write_csv(pd.DataFrame(labs), ehr_root / "hosp" / "labevents.csv.gz")
    write_csv(
        pd.DataFrame(
            [{"itemid": GLUCOSE_ITEM, "label": "Glucose", "fluid": "Blood",
              "category": "Chemistry"}]
        ),
        ehr_root / "hosp" / "d_labitems.csv.gz",
    )
    write_csv(
        pd.DataFrame(diagnoses).drop(columns="dischtime"),
        ehr_root / "hosp" / "diagnoses_icd.csv.gz",
    )
    write_csv(pd.DataFrame(images), cxr_root / "mimic-cxr-2.0.0-metadata.csv.gz")
    return ehr_root, cxr_root


def run_task(dataset, task, args) -> None:
    """Trains MedFuse on one task and prints test metrics."""
    train, val, test = dataset.set_task(
        task, split=PatientSplit(ratios=(0.6, 0.2, 0.2), seed=args.seed)
    )
    masks = [float(sample["cxr_mask"]) for sample in train]
    print(
        f"\n{task.task_name}: {len(train)}/{len(val)}/{len(test)} train/val/test "
        f"stays, {sum(masks):.0f}/{len(masks)} training stays with an image"
    )

    torch.manual_seed(args.seed)
    model = MedFuse(
        dataset=train,
        ehr_hidden_dim=16,
        ehr_num_layers=1,
        cxr_backbone="resnet18",
        cxr_pretrained=args.pretrained,
        fusion_hidden_dim=16,
        projection_dim=16,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = get_dataloader(train, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}: last batch loss {loss.item():.4f}")

    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in get_dataloader(test, batch_size=args.batch_size):
            output = model(**batch)
            y_true.append(output["y_true"].cpu().numpy())
            y_prob.append(output["y_prob"].cpu().numpy())
    y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
    print(f"test outputs {y_prob.shape}; mean probability {y_prob.mean():.3f}")
    if y_prob.shape[1] == 1 and len(np.unique(y_true)) > 1:
        scores = binary_metrics_fn(
            y_true.ravel(), y_prob.ravel(), metrics=["roc_auc", "pr_auc"]
        )
        print(f"smoke-test metrics (not meaningful): {scores}")


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ehr-root", help="MIMIC-IV root; synthetic data if unset")
    parser.add_argument("--cxr-root", help="MIMIC-CXR-JPG root")
    parser.add_argument("--patients", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Loads the data and runs both tasks."""
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        ehr_root, cxr_root = args.ehr_root, args.cxr_root
        if ehr_root is None:
            print("SMOKE TEST on synthetic data; results are not meaningful.")
            ehr_root, cxr_root = write_synthetic_mimic(
                Path(tmp), args.patients, args.seed
            )
        dataset = MIMIC4Dataset(
            ehr_root=str(ehr_root),
            cxr_root=str(cxr_root),
            ehr_tables=["chartevents", "labevents", "diagnoses_icd"],
            cxr_tables=["metadata"],
            cache_dir=str(Path(tmp) / "cache") if args.ehr_root is None else None,
        )
        run_task(dataset, EHRCXRMortalityMIMIC4(image_size=args.image_size), args)
        run_task(dataset, EHRCXRPhenotypingMIMIC4(image_size=args.image_size), args)


if __name__ == "__main__":
    main()
