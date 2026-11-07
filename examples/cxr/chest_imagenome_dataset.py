"""Load Chest ImaGenome scene graphs and gold comparisons with PyHealth.

SMOKE TEST ON FABRICATED DATA. Nothing here comes from the real dataset, and the
printed rows mean nothing clinically. Chest ImaGenome is credentialed: real use
needs a PhysioNet account, CITI training and the signed data use agreement, from
https://physionet.org/content/chest-imagenome/1.0.0/

To run on real data, unzip ``silver_dataset/scene_graph.zip`` and pass the
``chest-imagenome/1.0.0`` directory as ``--root``. Otherwise this script writes
two tiny fabricated patients to a temporary directory and loads those.

    python examples/cxr/chest_imagenome_dataset.py
    python examples/cxr/chest_imagenome_dataset.py --root /data/chest-imagenome/1.0.0
"""

import argparse
import json
import tempfile
from pathlib import Path

from pyhealth.datasets import ChestImaGenomeDataset


def box(name: str) -> dict:
    """Returns a fabricated anatomical box."""
    return {
        "bbox_name": name,
        "original_x1": 100,
        "original_y1": 200,
        "original_x2": 900,
        "original_y2": 1500,
    }


def write_fabricated_files(root: Path) -> None:
    """Writes two patients in the layout that the real download has."""
    graphs = root / "silver_dataset" / "scene_graph"
    gold = root / "gold_dataset"
    graphs.mkdir(parents=True)
    gold.mkdir()
    region = "left lung"
    for dicom_id, patient, study, stamp, order, relationships in (
        ("first-image", 900001, 500001, "2150-01-02 03:04:05", 1, []),
        (
            "second-image",
            900001,
            500002,
            "2150-02-02 03:04:05",
            2,
            [
                {
                    "subject_id": f"second-image_{region}",
                    "object_id": f"first-image_{region}",
                    "bbox_name": region,
                    "relationship_names": ["comparison|yes|worsened"],
                    "attributes": ["anatomicalfinding|yes|lung opacity"],
                }
            ],
        ),
        ("only-image", 900002, 500003, "2151-05-06 07:08:09", 1, []),
    ):
        graph = {
            "patient_id": patient,
            "study_id": study,
            "StudyDateTime": stamp,
            "StudyOrder": order,
            "viewpoint": "AP",
            "gender": "F",
            "age_decile": "60-70",
            "objects": [box(region)],
            "relationships": relationships,
        }
        (graphs / f"{dicom_id}_SceneGraph.json").write_text(json.dumps(graph))
    (gold / "gold_object_comparison_with_coordinates.txt").write_text(
        "patient_id\tcurrent_image_id\tprevious_image_id\tbbox\tlabel_name\tcomparison\n"
        f"900001\tsecond-image\tfirst-image\t{region}\tlung opacity\tworsened\n"
    )


def show(root: str, cache_dir: str) -> None:
    """Loads the dataset and prints what a task would read for one patient."""
    dataset = ChestImaGenomeDataset(root=root, cache_dir=cache_dir)
    dataset.stats()

    patient = dataset.get_patient(dataset.unique_patient_ids[0])
    print(f"Patient {patient.patient_id}")
    for image in patient.get_events("silver_images"):
        print(f"  image {image['dicom_id']} in study {image['study_id']}")
    for row in patient.get_events("silver_comparisons"):
        print(
            f"  silver: study {row['previous_study_id']} -> {row['study_id']}, "
            f"{row['bbox_name']}: {row['findings']} {row['comparisons']}"
        )
    for row in patient.get_events("gold_comparisons"):
        print(
            f"  gold: {row['previous_dicom_id']} -> {row['dicom_id']}, "
            f"{row['bbox_name']}: {row['finding']} {row['comparison']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help="chest-imagenome/1.0.0 directory")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        if args.root:
            show(args.root, str(Path(tmp) / "cache"))
            return
        print("SMOKE TEST: loading fabricated files, not real data.")
        root = Path(tmp) / "chest-imagenome"
        write_fabricated_files(root)
        show(str(root), str(Path(tmp) / "cache"))


if __name__ == "__main__":
    main()
