"""PyHealth dataset for Chest ImaGenome (scene graphs for MIMIC-CXR).

Dataset link:
    https://physionet.org/content/chest-imagenome/1.0.0/

Dataset paper: (please cite if you use this dataset)
    Joy T. Wu, Nkechinyere N. Agu, Ismini Lourentzou, Arjun Sharma, Joseph A.
    Paguio, et al. "Chest ImaGenome Dataset for Clinical Reasoning." NeurIPS
    2021. https://arxiv.org/abs/2108.00316

Adapted from:
    DiPro, commit 64b12ec (MIT License), https://github.com/Chenliu-svg/DiPro
    Copyright (c) 2025 LIU Chen @ The Hong Kong Polytechnic University
    Chen Liu, Wenfang Yao, Kejing Yin, William K. Cheung, and Jing Qin.
    "Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement
    and Multiscale Alignment." NeurIPS 2025. https://arxiv.org/abs/2510.11112

Description:
    Flattens the silver scene graphs (one JSON per frontal chest X-ray) and the
    gold object comparisons into tables keyed by MIMIC-CXR ``subject_id``,
    ``study_id`` and ``dicom_id``. Which fields are kept, and how a comparison
    links two images, follow DiPro's ``process_ImaGenome.py`` and ``get_cxr.py``.
    Access requires PhysioNet credentials and the data use agreement.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .base_dataset import BaseDataset
from .metadata_csv import (
    metadata_cache_dir,
    metadata_csv_exists,
    resolve_metadata_root,
)

logger = logging.getLogger(__name__)

_DATASET_KEY = "chest_imagenome"

IMAGES_CSV = "chest_imagenome-pyhealth-silver-images.csv"
OBJECTS_CSV = "chest_imagenome-pyhealth-silver-objects.csv"
COMPARISONS_CSV = "chest_imagenome-pyhealth-silver-comparisons.csv"
GOLD_COMPARISONS_CSV = "chest_imagenome-pyhealth-gold-comparisons.csv"

LIST_SEPARATOR = ";"
"""Separates the values in the multi-valued ``silver_comparisons`` columns."""

_SCENE_GRAPH_SUFFIX = "_SceneGraph.json"
_SCENE_GRAPH_DIRS = (
    "silver_dataset/scene_graph",
    "silver_dataset/scene_graph/scene_graph",  # the path DiPro's comment shows
)
_GOLD_COMPARISON_FILE = "gold_dataset/gold_object_comparison_with_coordinates.txt"
_GOLD_COLUMNS = {
    "patient_id": "subject_id",
    "current_image_id": "dicom_id",
    "previous_image_id": "previous_dicom_id",
    "bbox": "bbox_name",
    "label_name": "finding",
    "comparison": "comparison",
}

_COMPARISON_RELATIONS = (
    "comparison|yes|worsened",
    "comparison|yes|improved",
    "comparison|yes|no change",
)
_FINDING_CATEGORIES = ("anatomicalfinding", "disease")

_IMAGE_COLUMNS = (
    "subject_id",
    "study_id",
    "dicom_id",
    "study_datetime",
    "study_order",
    "viewpoint",
    "gender",
    "age_decile",
)
_OBJECT_COLUMNS = (
    "subject_id",
    "study_id",
    "dicom_id",
    "study_datetime",
    "bbox_name",
    "original_x1",
    "original_y1",
    "original_x2",
    "original_y2",
)
_COMPARISON_COLUMNS = (
    "subject_id",
    "study_id",
    "dicom_id",
    "previous_study_id",
    "previous_dicom_id",
    "study_datetime",
    "bbox_name",
    "findings",
    "finding_polarities",
    "comparisons",
)
_GOLD_OUTPUT_COLUMNS = (
    "subject_id",
    "study_id",
    "dicom_id",
    "previous_study_id",
    "previous_dicom_id",
    "study_datetime",
    "bbox_name",
    "finding",
    "comparison",
)

# Mislabeled boxes that DiPro's correct_scenegraph fixes, keyed by image, box
# name and original top-left corner. None drops the box (it duplicates another
# one under the wrong name); a string renames it.
_OBJECT_CORRECTIONS: dict[tuple[str, str, int, int], str | None] = {
    (
        "3e953c67-28b78460-4f783437-379841d3-a79ec530",
        "left mid lung zone",
        1500,
        886,
    ): None,
    (
        "8bd72ea4-f6e5470f-3ddbc81e-a1da43d8-b6e39876",
        "left lower lung zone",
        777,
        586,
    ): None,
    (
        "38bbebbd-b67908dc-387567bf-e3f965b7-77a18536",
        "right apical zone",
        1146,
        532,
    ): "left apical zone",
    (
        "38bbebbd-b67908dc-387567bf-e3f965b7-77a18536",
        "right upper lung zone",
        1159,
        518,
    ): "left upper lung zone",
    (
        "038c0310-98178aef-3414521f-1d34c309-8b94b78e",
        "left lower lung zone",
        641,
        1732,
    ): "right lower lung zone",
    (
        "038c0310-98178aef-3414521f-1d34c309-8b94b78e",
        "left mid lung zone",
        777,
        1309,
    ): "right mid lung zone",
}


def _find_scene_graphs(root: Path) -> list[Path]:
    """Lists the unzipped scene graph files under ``root``."""
    for folder in _SCENE_GRAPH_DIRS:
        paths = sorted((root / folder).glob(f"*{_SCENE_GRAPH_SUFFIX}"))
        if paths:
            return paths
    raise FileNotFoundError(
        f"No *{_SCENE_GRAPH_SUFFIX} files in {root}/{_SCENE_GRAPH_DIRS[0]}; "
        "unzip silver_dataset/scene_graph.zip there first."
    )


def _format_datetime(value: object) -> str:
    """Formats ``StudyDateTime`` for the config, or "" if it cannot be parsed.

    Only strings are parsed: pandas reads a number as nanoseconds since 1970,
    which would give a wrong date instead of a blank one.
    """
    if not isinstance(value, str):
        return ""
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d %H:%M:%S")


def _image_id(node_id: str) -> str:
    """Returns the image in a ``<dicom_id>_<bbox_name>`` node id."""
    return node_id.split("_")[0]


def _object_rows(graph: dict, dicom_id: str) -> Iterable[dict]:
    """Yields the anatomical boxes of one scene graph, corrected as in DiPro."""
    for obj in graph.get("objects", []):
        key = (dicom_id, obj["bbox_name"], obj["original_x1"], obj["original_y1"])
        bbox_name = _OBJECT_CORRECTIONS.get(key, obj["bbox_name"])
        if bbox_name is None:
            continue
        yield {
            "bbox_name": bbox_name,
            "original_x1": obj["original_x1"],
            "original_y1": obj["original_y1"],
            "original_x2": obj["original_x2"],
            "original_y2": obj["original_y2"],
        }


def _comparison_rows(graph: dict) -> Iterable[dict]:
    """Yields the region-level comparisons of one scene graph.

    In a relationship, ``subject_id`` is the region on the current image and
    ``object_id`` the same region on the previous image. Relationships with no
    worsened, improved or no change label are skipped, as in DiPro.
    """
    for relationship in graph.get("relationships", []):
        comparisons = [
            name.split("|")[2]
            for name in relationship["relationship_names"]
            if name in _COMPARISON_RELATIONS
        ]
        if not comparisons:
            continue
        findings = [
            attribute.split("|")
            for attribute in relationship["attributes"]
            if attribute.split("|")[0] in _FINDING_CATEGORIES
        ]
        yield {
            "dicom_id": _image_id(relationship["subject_id"]),
            "previous_dicom_id": _image_id(relationship["object_id"]),
            "bbox_name": relationship["bbox_name"],
            "findings": LIST_SEPARATOR.join(parts[-1] for parts in findings),
            "finding_polarities": LIST_SEPARATOR.join(
                parts[1] if len(parts) == 3 else "" for parts in findings
            ),
            "comparisons": LIST_SEPARATOR.join(comparisons),
        }


def _read_gold_comparisons(path: Path) -> pd.DataFrame:
    """Reads the gold comparison file, or an empty table if it is missing."""
    if not path.exists():
        logger.warning("%s not found; gold_comparisons will be empty", path)
        return pd.DataFrame(columns=list(_GOLD_COLUMNS.values()))
    gold = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(_GOLD_COLUMNS) - set(gold.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    gold = gold[list(_GOLD_COLUMNS)].rename(columns=_GOLD_COLUMNS)
    return gold.drop_duplicates()  # keeps the file order, which DiPro relies on


def prepare_chest_imagenome(root: str | Path) -> Path:
    """Writes the tables that the dataset config reads.

    Reads every scene graph in ``silver_dataset/scene_graph/`` and the gold
    comparison file in ``gold_dataset/``. A scene graph that cannot be parsed,
    or that lacks a field read here such as ``patient_id`` or ``study_id``, is
    skipped with a warning, and a missing gold file gives an empty
    ``gold_comparisons`` table. Rows are streamed to disk, since the full
    dataset has one scene graph per frontal image.

    Args:
        root: The ``chest-imagenome/1.0.0`` directory, with
            ``silver_dataset/scene_graph.zip`` already unzipped.

    Returns:
        The directory holding the tables: ``root``, or a folder under
        ``~/.cache/pyhealth/chest_imagenome`` when ``root`` is read-only.

    Raises:
        FileNotFoundError: If there are no unzipped scene graphs.
        ValueError: If the gold file lacks a column DiPro reads.

    Examples:
        >>> prepare_chest_imagenome("data/chest-imagenome/1.0.0")  # doctest: +SKIP
    """
    root = Path(root)
    paths = _find_scene_graphs(root)
    out = root if os.access(root, os.W_OK) else metadata_cache_dir(root, _DATASET_KEY)
    out.mkdir(parents=True, exist_ok=True)
    tmp = {name: out / f"{name}.tmp" for name in (IMAGES_CSV, OBJECTS_CSV)}
    pending = out / f"{COMPARISONS_CSV}.pending"

    # dicom_id -> (subject_id, study_id, study_datetime)
    images: dict[str, tuple[str, str, str]] = {}
    undated = 0
    with (
        open(tmp[IMAGES_CSV], "w", newline="") as images_file,
        open(tmp[OBJECTS_CSV], "w", newline="") as objects_file,
        open(pending, "w", newline="") as pending_file,
    ):
        image_writer = csv.DictWriter(images_file, _IMAGE_COLUMNS)
        object_writer = csv.DictWriter(objects_file, _OBJECT_COLUMNS)
        pending_writer = csv.DictWriter(
            pending_file, [c for c in _COMPARISON_COLUMNS if c != "previous_study_id"]
        )
        for writer in (image_writer, object_writer, pending_writer):
            writer.writeheader()
        for path in paths:
            try:
                with open(path) as handle:
                    graph = json.load(handle)
            except (OSError, ValueError) as error:
                logger.warning("Skipping unreadable scene graph %s: %s", path, error)
                continue
            dicom_id = path.name.removesuffix(_SCENE_GRAPH_SUFFIX)
            try:
                ids = {
                    "subject_id": str(graph["patient_id"]),
                    "study_id": str(graph["study_id"]),
                    "study_datetime": _format_datetime(graph.get("StudyDateTime")),
                }
                objects = list(_object_rows(graph, dicom_id))
                comparisons = list(_comparison_rows(graph))
            except (KeyError, TypeError, IndexError, AttributeError) as error:
                logger.warning(
                    "Skipping scene graph %s with a missing or malformed field: %r",
                    path,
                    error,
                )
                continue
            undated += not ids["study_datetime"]
            images[dicom_id] = (
                ids["subject_id"],
                ids["study_id"],
                ids["study_datetime"],
            )
            image_writer.writerow({
                **ids,
                "dicom_id": dicom_id,
                "study_order": graph.get("StudyOrder", ""),
                "viewpoint": graph.get("viewpoint", ""),
                "gender": graph.get("gender", ""),
                "age_decile": graph.get("age_decile", ""),
            })
            for row in objects:
                object_writer.writerow({**ids, "dicom_id": dicom_id, **row})
            for row in comparisons:
                pending_writer.writerow({**ids, **row})

    if undated:
        logger.warning("%d scene graphs have no usable StudyDateTime", undated)

    def study_of(dicom_id: str) -> str:
        return images.get(dicom_id, ("", "", ""))[1]

    # The previous image's study is only known once every graph has been read.
    tmp[COMPARISONS_CSV] = out / f"{COMPARISONS_CSV}.tmp"
    with (
        open(pending, newline="") as pending_file,
        open(tmp[COMPARISONS_CSV], "w", newline="") as comparisons_file,
    ):
        writer = csv.DictWriter(comparisons_file, _COMPARISON_COLUMNS)
        writer.writeheader()
        for row in csv.DictReader(pending_file):
            row["previous_study_id"] = study_of(row["previous_dicom_id"])
            writer.writerow(row)
    pending.unlink()

    gold = _read_gold_comparisons(root / _GOLD_COMPARISON_FILE)
    gold["study_id"] = gold["dicom_id"].map(study_of)
    gold["previous_study_id"] = gold["previous_dicom_id"].map(study_of)
    gold["study_datetime"] = gold["dicom_id"].map(
        lambda dicom_id: images.get(dicom_id, ("", "", ""))[2]
    )
    tmp[GOLD_COMPARISONS_CSV] = out / f"{GOLD_COMPARISONS_CSV}.tmp"
    gold[list(_GOLD_OUTPUT_COLUMNS)].to_csv(tmp[GOLD_COMPARISONS_CSV], index=False)

    # All four tables appear together, so a partial run is redone.
    for name, path in tmp.items():
        os.replace(path, out / name)
    return out


class ChestImaGenomeDataset(BaseDataset):
    """Chest ImaGenome silver scene graphs and gold object comparisons.

    Four tables, all keyed by MIMIC-CXR ``subject_id`` and timestamped with the
    current study's ``StudyDateTime``. ``study_id`` and ``dicom_id`` join to
    :class:`~pyhealth.datasets.MIMIC4CXRDataset`:

    - ``silver_images``: one row per frontal image (``study_id``, ``dicom_id``,
      ``study_order``, ``viewpoint``, ``gender``, ``age_decile``).
    - ``silver_objects``: one row per anatomical box (``dicom_id``,
      ``bbox_name`` and the ``original_x1/y1/x2/y2`` pixel coordinates), with
      the box fixes that DiPro applies.
    - ``silver_comparisons``: one row per region compared between two images
      (``dicom_id`` and ``study_id`` are the current image;
      ``previous_dicom_id`` and ``previous_study_id`` the earlier one, whose
      study is blank when it has no scene graph). ``bbox_name`` is the region;
      ``findings``, ``finding_polarities`` and ``comparisons`` hold values
      joined by :data:`LIST_SEPARATOR`. Polarities (``yes``/``no``) line up
      with findings. More than one comparison means the labels are ambiguous.
    - ``gold_comparisons``: the same image pairing, one row per manually
      validated ``bbox_name``, ``finding`` and ``comparison``. Studies and the
      timestamp are blank for an image with no scene graph.

    The tables are built once from the raw files and written into ``root``, or
    under ``~/.cache/pyhealth/chest_imagenome`` when ``root`` is read-only.
    Delete them to rebuild. Images are not loaded, and no progression labels
    are derived; ambiguity handling belongs to the task.

    Args:
        root: The ``chest-imagenome/1.0.0`` directory, with
            ``silver_dataset/scene_graph.zip`` already unzipped. Access needs
            PhysioNet credentials and the signed data use agreement.
        tables: Tables to load. Defaults to all four.
        dataset_name: Name of the dataset. Defaults to ``"chest_imagenome"``.
        config_path: Path to a YAML config. Defaults to the bundled one.
        **kwargs: Passed to :class:`~pyhealth.datasets.BaseDataset`
            (e.g. ``cache_dir``, ``dev``).

    Examples:
        >>> from pyhealth.datasets import ChestImaGenomeDataset
        >>> dataset = ChestImaGenomeDataset(
        ...     root="data/chest-imagenome/1.0.0"
        ... )  # doctest: +SKIP
        >>> patient = dataset.get_patient("10000032")  # doctest: +SKIP
        >>> patient.get_events("silver_comparisons")[0]["comparisons"]  # doctest: +SKIP
        'worsened'
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        names = (IMAGES_CSV, OBJECTS_CSV, COMPARISONS_CSV, GOLD_COMPARISONS_CSV)
        if not all(metadata_csv_exists(root, _DATASET_KEY, name) for name in names):
            logger.info("Building the Chest ImaGenome tables from %s", root)
            prepare_chest_imagenome(root)
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "chest_imagenome.yaml")
        super().__init__(
            root=resolve_metadata_root(root, _DATASET_KEY, GOLD_COMPARISONS_CSV),
            tables=tables
            or [
                "silver_images",
                "silver_objects",
                "silver_comparisons",
                "gold_comparisons",
            ],
            dataset_name=dataset_name or "chest_imagenome",
            config_path=config_path,
            **kwargs,
        )
