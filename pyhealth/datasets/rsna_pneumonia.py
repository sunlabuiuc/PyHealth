"""RSNA Pneumonia Detection Challenge DICOM dataset."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .base_dataset import BaseDataset
from .imaging_utils import _resolve_metadata_dir

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE_DIR = (
    "mdai_rsna_project_x9N20BZa_images_2018-07-20-153330"
)
_DEFAULT_MAPPING = "pneumonia-challenge-dataset-mappings_2018.json"


def _as_int(value: Any, default: int = 0) -> int:
    """Convert a metadata value to ``int`` with a fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class RSNAPneumoniaDataset(BaseDataset):
    """RSNA Pneumonia Detection Challenge chest X-ray dataset.

    The dataset maps the challenge mapping JSON to one event per DICOM image.
    Labels come from ``subset_init_label``:

    - 1: lung opacity / pneumonia positive
    - 0 or 2: negative

    By default ``patient_id`` is the study UID, so metadata generation is
    fast. Set ``read_dicom_patient_id=True`` to read the DICOM ``PatientID``
    and view/geometry attributes; this requires pydicom and scans every file.

    Args:
        root: Dataset root containing the mapping JSON and image directory.
        image_dir: Optional DICOM image directory.
        mapping_file: Optional path to the challenge mapping JSON.
        dataset_name: Optional dataset name.
        config_path: Optional table configuration path.
        cache_dir: Directory for processed caches.
        metadata_dir: Explicit metadata output directory.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        read_dicom_patient_id: Read PatientID and DICOM geometry attributes.
        refresh_metadata: Regenerate the metadata CSV.

    Examples:
        >>> from pyhealth.datasets import RSNAPneumoniaDataset
        >>> dataset = RSNAPneumoniaDataset(  # doctest: +SKIP
        ...     root="/path/to/RSNAPneumoniaDetection2018"
        ... )
        >>> dataset.stats()  # doctest: +SKIP
    """

    default_dataset_name = "rsna_pneumonia"
    default_table = "images"

    def __init__(
        self,
        root: str,
        image_dir: str | None = None,
        mapping_file: str | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        metadata_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
        read_dicom_patient_id: bool = False,
        refresh_metadata: bool = False,
    ) -> None:
        """Build metadata and initialize the RSNA Pneumonia dataset.

        Args:
            root: Dataset root directory.
            image_dir: Optional DICOM image directory.
            mapping_file: Optional mapping JSON path.
            dataset_name: Optional dataset name.
            config_path: Optional table configuration path.
            cache_dir: Directory for processed caches.
            metadata_dir: Explicit metadata output directory.
            num_workers: Number of workers for dataset processing.
            dev: Whether to limit the dataset to 1000 patients.
            read_dicom_patient_id: Read DICOM PatientID and geometry fields.
            refresh_metadata: Regenerate the metadata CSV.
        """
        source_root = Path(root).resolve()
        self.source_root = source_root
        self.image_dir = (
            Path(image_dir).resolve()
            if image_dir is not None
            else (source_root / _DEFAULT_IMAGE_DIR).resolve()
        )
        self.mapping_path = (
            Path(mapping_file).resolve()
            if mapping_file is not None
            else (source_root / _DEFAULT_MAPPING).resolve()
        )
        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"RSNA image directory not found: {self.image_dir}. "
                "Download the RSNA Pneumonia Detection Challenge data from "
                f"Kaggle and place it under {source_root}."
            )
        if not self.mapping_path.exists():
            raise FileNotFoundError(
                f"RSNA mapping file not found: {self.mapping_path}. "
                f"Expected {_DEFAULT_MAPPING} under {source_root}."
            )

        self.read_dicom_patient_id = bool(read_dicom_patient_id)
        name = dataset_name or self.default_dataset_name
        self.metadata_dir = (
            Path(metadata_dir)
            if metadata_dir is not None
            else _resolve_metadata_dir(cache_dir, name)
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = (
            self.metadata_dir / "rsna-pneumonia-metadata-pyhealth.csv"
        )

        if refresh_metadata or not self.metadata_path.exists():
            self.prepare_metadata()

        if config_path is None:
            config_path = str(
                Path(__file__).parent
                / "configs"
                / "rsna_pneumonia.yaml"
            )

        super().__init__(
            root=str(self.metadata_dir),
            tables=[self.default_table],
            dataset_name=name,
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def prepare_metadata(self) -> None:
        """Create the RSNA image metadata CSV from the mapping JSON."""
        with open(self.mapping_path) as handle:
            mapping = json.load(handle)

        rows = []
        for entry in mapping:
            study_id = str(entry.get("StudyInstanceUID", "") or "")
            series_id = str(entry.get("SeriesInstanceUID", "") or "")
            sop_id = str(entry.get("SOPInstanceUID", "") or "")
            image_path = str(
                self.image_dir
                / study_id
                / series_id
                / f"{sop_id}.dcm"
            )
            subset_label = _as_int(entry.get("subset_init_label", 0))
            orig_labels = entry.get("orig_labels") or []
            row = {
                "patient_id": study_id or sop_id,
                "record_id": sop_id or str(entry.get("img_id", "")),
                "image_path": image_path,
                "study_id": study_id,
                "series_id": series_id,
                "sop_id": sop_id,
                "modality": "CR",
                "view_position": "",
                "rows": "",
                "columns": "",
                "label": 1 if subset_label == 1 else 0,
                "subset_init_label": subset_label,
                "orig_labels": "|".join(str(label) for label in orig_labels),
                "fold": _as_int(entry.get("subset_group", 0)),
            }
            if self.read_dicom_patient_id:
                header = self._read_dicom_header(image_path)
                patient_id = header.pop("patient_id", None)
                if patient_id:
                    row["patient_id"] = patient_id
                row.update(header)
            rows.append(row)

        pd.DataFrame(rows).to_csv(self.metadata_path, index=False)
        logger.info(
            "Wrote %d RSNA images to %s", len(rows), self.metadata_path
        )

    @property
    def default_task(self):
        """Return the default RSNA pneumonia classification task."""
        from ..tasks.rsna_pneumonia_classification import (
            RSNAPneumoniaClassification,
        )

        return RSNAPneumoniaClassification()

    @staticmethod
    def _read_dicom_header(image_path: str) -> dict:
        """Read selected DICOM attributes, falling back to empty values."""
        try:
            import pydicom
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "read_dicom_patient_id=True requires pydicom. "
                "Install it with `pip install pyhealth[imaging]`."
            ) from exc

        try:
            dataset = pydicom.dcmread(
                image_path,
                stop_before_pixels=True,
                specific_tags=[
                    "PatientID",
                    "Modality",
                    "ViewPosition",
                    "Rows",
                    "Columns",
                ],
            )
        except (OSError, ValueError, pydicom.errors.InvalidDicomError):
            return {}

        return {
            "patient_id": str(
                getattr(dataset, "PatientID", "") or ""
            )
            or None,
            "modality": str(getattr(dataset, "Modality", "") or "")
            or "CR",
            "view_position": str(
                getattr(dataset, "ViewPosition", "") or ""
            ),
            "rows": _as_int(getattr(dataset, "Rows", 0)),
            "columns": _as_int(getattr(dataset, "Columns", 0)),
        }
