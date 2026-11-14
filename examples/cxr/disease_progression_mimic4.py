"""Disease progression between chest X-rays: smoke test on fabricated data.

SMOKE TEST ONLY. This script writes a tiny fabricated MIMIC-IV, MIMIC-CXR and
Chest ImaGenome tree to a temporary directory (random values and images, no
patient data) and runs the real pipeline on it, from ``MIMIC4ImaGenomeDataset`` through
``set_task`` to a batch of samples. It trains no model, and nothing it prints
means anything clinically.

To run on the real data, you need credentialed PhysioNet access to MIMIC-IV,
MIMIC-CXR-JPG and Chest ImaGenome. Unzip ``silver_dataset/scene_graph.zip``,
then pass the three roots:

    python examples/cxr/disease_progression_mimic4.py \
        --ehr-root /path/to/mimic-iv/2.2 \
        --cxr-root /path/to/mimic-cxr-jpg/2.0.0 \
        --imagenome-root /path/to/chest-imagenome/1.0.0

For reference, Liu et al. (2025) report 3982/560/1137 train/val/test samples
and, for DiPro on this task, 0.466 macro-F1 and 0.664 AUROC.

Paper:
    Liu et al. "Multimodal Disease Progression Modeling via Spatiotemporal
    Disentanglement and Multiscale Alignment." NeurIPS 2025.
    https://arxiv.org/abs/2510.11112
"""

from __future__ import annotations

import argparse
import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pyhealth.datasets import GoldTestSplit, MIMIC4ImaGenomeDataset, get_dataloader
from pyhealth.tasks import DiseaseProgressionMIMIC4

ADMIT_TIME = pd.Timestamp("2150-01-01 08:00:00")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
STAY_HOURS = 120
IMAGE_HOURS = (10.5, 30.25, 50.75)
HEART_RATE = 220045
GLUCOSE = 50931
MICU = "Medical Intensive Care Unit (MICU)"
STATUSES = ("worsened", "improved", "no change")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Writes a gzipped CSV, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as file:
        frame.to_csv(file, index=False)


def write_ehr(root: Path, n_patients: int, rng: np.random.Generator) -> None:
    """Writes one ICU stay per patient in the MIMIC-IV layout.

    Lengths of stay differ because identical ones would all be IQR outliers.
    """
    patients, admissions, stays, charts, labs = [], [], [], [], []
    for index in range(n_patients):
        subject_id, hadm_id = 10000000 + index, 20000000 + index
        admit = ADMIT_TIME + pd.Timedelta(days=index)
        hours = STAY_HOURS + index
        outtime = admit + pd.Timedelta(hours=hours)
        patients.append(
            {"subject_id": subject_id, "gender": "F", "anchor_age": 60,
             "anchor_year": 2150, "anchor_year_group": "2008 - 2010", "dod": None}
        )
        admissions.append(
            {"subject_id": subject_id, "hadm_id": hadm_id,
             "admittime": admit.strftime(TIME_FORMAT),
             "dischtime": outtime.strftime(TIME_FORMAT), "deathtime": None,
             "admission_type": "URGENT", "admission_location": "EMERGENCY ROOM",
             "discharge_location": "HOME", "insurance": "Other",
             "language": "ENGLISH", "marital_status": "SINGLE", "race": "OTHER",
             "hospital_expire_flag": 0}
        )
        stays.append(
            {"subject_id": subject_id, "hadm_id": hadm_id,
             "stay_id": 30000000 + index, "first_careunit": MICU,
             "last_careunit": MICU, "intime": admit.strftime(TIME_FORMAT),
             "outtime": outtime.strftime(TIME_FORMAT), "los": hours / 24}
        )
        for hour in range(0, STAY_HOURS, 4):
            time = (admit + pd.Timedelta(hours=hour, minutes=5)).strftime(TIME_FORMAT)
            rate = float(rng.normal(85, 5))
            charts.append(
                {"subject_id": subject_id, "hadm_id": hadm_id,
                 "stay_id": 30000000 + index, "charttime": time,
                 "itemid": HEART_RATE, "value": str(rate), "valuenum": rate,
                 "valueuom": "bpm"}
            )
            labs.append(
                {"subject_id": subject_id, "hadm_id": hadm_id, "itemid": GLUCOSE,
                 "charttime": time, "storetime": time, "value": "110",
                 "valuenum": 110.0, "valueuom": "mg/dL", "flag": None,
                 "ref_range_lower": 70.0, "ref_range_upper": 100.0}
            )
    write_csv(pd.DataFrame(patients), root / "hosp" / "patients.csv.gz")
    write_csv(pd.DataFrame(admissions), root / "hosp" / "admissions.csv.gz")
    write_csv(pd.DataFrame(stays), root / "icu" / "icustays.csv.gz")
    write_csv(pd.DataFrame(charts), root / "icu" / "chartevents.csv.gz")
    write_csv(pd.DataFrame(labs), root / "hosp" / "labevents.csv.gz")
    write_csv(
        pd.DataFrame(
            [{"itemid": GLUCOSE, "label": "Glucose", "fluid": "Blood",
              "category": "Chemistry"}]
        ),
        root / "hosp" / "d_labitems.csv.gz",
    )


def write_images(root: Path, n_patients: int, rng: np.random.Generator) -> None:
    """Writes three AP images per stay in the MIMIC-CXR-JPG layout."""
    rows = []
    for index in range(n_patients):
        subject_id = 10000000 + index
        admit = ADMIT_TIME + pd.Timedelta(days=index)
        for order, hour in enumerate(IMAGE_HOURS):
            study_id = 50000000 + 10 * index + order
            dicom_id = f"fabricated-{index}-{order}"
            time = admit + pd.Timedelta(hours=hour)
            path = (
                root / "files" / f"p{str(subject_id)[:2]}" / f"p{subject_id}"
                / f"s{study_id}" / f"{dicom_id}.jpg"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = rng.integers(0, 256, (64, 64), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            rows.append(
                {"dicom_id": dicom_id, "subject_id": subject_id, "study_id": study_id,
                 "PerformedProcedureStepDescription": "CHEST (PORTABLE AP)",
                 "ViewPosition": "AP", "Rows": 64, "Columns": 64,
                 "StudyDate": time.strftime("%Y%m%d"),
                 "StudyTime": time.strftime("%H%M%S"),
                 "ProcedureCodeSequence_CodeMeaning": "CHEST (PORTABLE AP)",
                 "ViewCodeSequence_CodeMeaning": "antero-posterior",
                 "PatientOrientationCodeSequence_CodeMeaning": "Erect"}
            )
    write_csv(pd.DataFrame(rows), root / "mimic-cxr-2.0.0-metadata.csv.gz")


def write_imagenome(root: Path, n_patients: int, rng: np.random.Generator) -> None:
    """Writes scene graphs per image, and gold labels for the last patient."""
    graphs = root / "silver_dataset" / "scene_graph"
    gold_dir = root / "gold_dataset"
    graphs.mkdir(parents=True)
    gold_dir.mkdir()
    gold_rows = ["patient_id\tcurrent_image_id\tprevious_image_id\tbbox\tlabel_name\tcomparison"]
    for index in range(n_patients):
        subject_id = 10000000 + index
        admit = ADMIT_TIME + pd.Timedelta(days=index)
        for order, hour in enumerate(IMAGE_HOURS):
            dicom_id = f"fabricated-{index}-{order}"
            relationships = []
            if order:
                previous = f"fabricated-{index}-{order - 1}"
                status = str(rng.choice(STATUSES))
                relationships.append(
                    {"subject_id": f"{dicom_id}_left lung",
                     "object_id": f"{previous}_left lung",
                     "bbox_name": "left lung",
                     "relationship_names": [f"comparison|yes|{status}"],
                     "attributes": ["anatomicalfinding|yes|lung opacity"]}
                )
                if index == n_patients - 1:
                    gold_rows.append(
                        f"{subject_id}\t{dicom_id}\t{previous}\tleft lung\t"
                        f"lung opacity\t{status}"
                    )
            graph = {
                "patient_id": subject_id,
                "study_id": 50000000 + 10 * index + order,
                "StudyDateTime": str(admit + pd.Timedelta(hours=hour)),
                "StudyOrder": order + 1,
                "viewpoint": "AP",
                "gender": "F",
                "age_decile": "60-70",
                "objects": [],
                "relationships": relationships,
            }
            (graphs / f"{dicom_id}_SceneGraph.json").write_text(json.dumps(graph))
    (gold_dir / "gold_object_comparison_with_coordinates.txt").write_text(
        "\n".join(gold_rows) + "\n"
    )


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ehr-root", help="MIMIC-IV root; fabricated data if unset")
    parser.add_argument("--cxr-root", help="MIMIC-CXR-JPG root")
    parser.add_argument("--imagenome-root", help="chest-imagenome/1.0.0 root")
    parser.add_argument("--patients", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    roots = (args.ehr_root, args.cxr_root, args.imagenome_root)
    if any(roots) and not all(roots):
        parser.error("--ehr-root, --cxr-root and --imagenome-root go together")
    return args


def main() -> None:
    """Loads the data, builds the samples and prints one batch."""
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        ehr_root, cxr_root = args.ehr_root, args.cxr_root
        imagenome_root = args.imagenome_root
        if ehr_root is None:
            print("SMOKE TEST on fabricated data; results are not meaningful.")
            root, rng = Path(tmp), np.random.default_rng(args.seed)
            ehr_root = root / "mimic-iv"
            cxr_root = root / "mimic-cxr"
            imagenome_root = root / "chest-imagenome"
            write_ehr(ehr_root, args.patients, rng)
            write_images(cxr_root, args.patients, rng)
            write_imagenome(imagenome_root, args.patients, rng)
        dataset = MIMIC4ImaGenomeDataset(
            imagenome_root=str(imagenome_root),
            imagenome_tables=[
                "silver_images", "silver_comparisons", "gold_comparisons"
            ],
            ehr_root=str(ehr_root),
            cxr_root=str(cxr_root),
            ehr_tables=["chartevents", "labevents"],
            cxr_tables=["metadata"],
            cache_dir=str(Path(tmp) / "cache") if args.ehr_root is None else None,
        )
        task = DiseaseProgressionMIMIC4(image_size=args.image_size)
        gold = task.gold_subjects(dataset)
        train, val, test = dataset.set_task(
            task, split=GoldTestSplit(gold, seed=args.seed)
        )
        print(f"{len(gold)} gold subjects")
        print(f"{len(train)}/{len(val)}/{len(test)} train/val/test samples")

        batch = next(iter(get_dataloader(train, batch_size=args.batch_size)))
        # The time_image processor gives each sample (images, hours, "image").
        images = torch.stack([sample[0] for sample in batch["cxr"]])
        hours = torch.stack([sample[1] for sample in batch["cxr"]])
        print(f"ehr {tuple(batch['ehr'].shape)}")
        print(f"cxr {tuple(images.shape)}, hours since admission {tuple(hours.shape)}")
        print(f"progression {tuple(batch['progression'].shape)}")
        print(f"from_gold in test: {[int(s['from_gold']) for s in test]}")


if __name__ == "__main__":
    main()
