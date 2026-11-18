"""CMMD (Chinese Mammography Database) dataset."""

import logging
import os
from pathlib import Path

import pandas as pd
from filelock import FileLock

from .base_dataset import BaseDataset
from .imaging.io import _resolve_metadata_dir

logger = logging.getLogger(__name__)

_DEFAULT_CLINICAL = "CMMD_clinicaldata_revision"
_METADATA_FILE = "cmmd-metadata-pyhealth.csv"
_MARKER_FILE = "cmmd-metadata-key.txt"
_VIEWS = {"cranio-caudal": "CC", "medio-lateral oblique": "MLO"}
_COLUMNS = [
    "patient_id", "record_id", "image_path", "laterality", "view", "cohort",
    "has_label", "label", "classification", "abnormality", "subtype", "age",
]


def _read_clinical_table(path: Path) -> pd.DataFrame:
    """Read the CMMD clinical sheet from ``.xlsx`` (needs openpyxl) or ``.csv``."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"Reading {path.name} needs the optional package openpyxl "
            "(`pip install openpyxl`). Alternatively, save the sheet as CSV "
            "and pass clinical_file='CMMD_clinicaldata_revision.csv'."
        ) from exc
    return pd.read_excel(path, engine="openpyxl")


def _read_header(path: str) -> dict | None:
    """Read patient id, laterality and view from a CMMD DICOM header."""
    import pydicom

    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=[
                "PatientID", "ImageLaterality", "Laterality",
                "ViewCodeSequence", "ViewPosition", "SOPInstanceUID",
            ],
        )
    except (OSError, ValueError, pydicom.errors.InvalidDicomError) as exc:
        logger.warning("Could not read DICOM header from %s: %s", path, exc)
        return None
    laterality = str(
        getattr(ds, "ImageLaterality", "") or getattr(ds, "Laterality", "") or ""
    ).upper()
    view = str(getattr(ds, "ViewPosition", "") or "").upper()
    sequence = getattr(ds, "ViewCodeSequence", None)
    if not view and sequence:
        meaning = str(getattr(sequence[0], "CodeMeaning", "") or "").lower()
        view = _VIEWS.get(meaning, meaning)
    return {
        "patient_id": str(getattr(ds, "PatientID", "") or ""),
        "record_id": str(getattr(ds, "SOPInstanceUID", "") or ""),
        "laterality": laterality,
        "view": view,
    }


class CMMDDataset(BaseDataset):
    """CMMD: The Chinese Mammography Database.

    The dataset has one event per full-field mammogram DICOM (one breast,
    one CC or MLO view). The clinical sheet has one row per patient and
    breast (``ID1``, ``LeftRight``) with the pathology-confirmed
    ``classification`` (Benign/Malignant); 97 patients have a row for each
    breast and 30 of them have different labels per side. Each image takes
    the label of its own breast, matched on the DICOM ``ImageLaterality``.
    Images of a breast with no row (the contralateral breast of a one-sided
    case) are kept with ``has_label=0`` and are skipped by the task.

    The collection has two cohorts: D1 (benign and malignant, 1,107 rows)
    and D2 (malignant with molecular ``subtype`` plus 12 benign, 765 rows),
    stored in ``cohort``. There is no official split: split by
    ``patient_id`` (for example with :class:`~pyhealth.datasets.PatientSplit`)
    so both views and both breasts of a patient stay together.

    Differences from CLIMB: CLIMB gives every image of a patient the label
    of each clinical row, ignoring laterality, so contralateral breasts are
    mislabelled; here labels are per breast side. CLIMB also splits on
    the first 75% of sheet rows, which puts D1 first and leaves a test set
    of 465 malignant and 3 benign rows; a random patient split avoids
    that cohort shift.

    Getting the data:
        - Source: https://www.cancerimagingarchive.net/collection/cmmd/
          (The Cancer Imaging Archive, doi 10.7937/tcia.eqde-4b16).
        - Access: open, no account needed.
        - Download: the DICOMs (about 23 GB, 5,202 images) with the NBIA
          Data Retriever and the collection's ``.tcia`` manifest, and
          ``CMMD_clinicaldata_revision.xlsx`` from the collection page.
        - ``root``: a folder holding the clinical sheet and the DICOMs in any
          sub-folder layout (the NBIA tree ``CMMD/<ID1>/...`` works). Patient
          id, laterality and view are read from the DICOM headers, which
          needs ``pydicom``. Reading the ``.xlsx`` needs the optional
          ``openpyxl`` package; a CSV export of the sheet works without it.
          Derived metadata is written to ``metadata_dir`` (by default under
          the PyHealth cache), never into ``root``.

    Licence: Creative Commons Attribution 4.0 (CC BY 4.0).

    Citation:
        Cui, C., Li, L., Cai, H., Fan, Z., Zhang, L., Dan, T., Li, J., &
        Wang, J. (2021). The Chinese Mammography Database (CMMD): An online
        mammography database with biopsy confirmed types for machine
        diagnosis of breast. The Cancer Imaging Archive.
        https://doi.org/10.7937/tcia.eqde-4b16

    Benchmarked in CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB).

    Args:
        root: Folder with the clinical sheet and the DICOMs.
        clinical_file: Clinical sheet path, absolute or relative to
            ``root``. Defaults to ``CMMD_clinicaldata_revision.xlsx``, or the
            ``.csv`` of the same name when only that exists.
        dataset_name: Optional dataset name. Default ``"cmmd"``.
        config_path: Optional table configuration path.
        cache_dir: Directory for processed caches.
        metadata_dir: Directory for the derived metadata CSV. Defaults to
            ``<cache>/metadata/<dataset_name>``.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        refresh_metadata: Regenerate the metadata CSV, e.g. after adding
            DICOMs to ``root``.

    Examples:
        >>> from pyhealth.datasets import CMMDDataset
        >>> dataset = CMMDDataset(root="/path/to/cmmd")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
    """

    default_dataset_name = "cmmd"
    default_table = "images"

    def __init__(
        self,
        root: str,
        clinical_file: str | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        metadata_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
        refresh_metadata: bool = False,
    ) -> None:
        self.source_root = Path(root).expanduser().resolve()
        if clinical_file is None:
            xlsx = self.source_root / f"{_DEFAULT_CLINICAL}.xlsx"
            csv = self.source_root / f"{_DEFAULT_CLINICAL}.csv"
            self.clinical_path = csv if csv.exists() and not xlsx.exists() else xlsx
        else:
            path = Path(clinical_file).expanduser()
            self.clinical_path = (
                path if path.is_absolute() else self.source_root / path
            ).resolve()
        if not self.clinical_path.exists():
            raise FileNotFoundError(
                f"CMMD clinical sheet not found: {self.clinical_path}. Download "
                f"{_DEFAULT_CLINICAL}.xlsx from "
                "https://www.cancerimagingarchive.net/collection/cmmd/."
            )

        name = dataset_name or self.default_dataset_name
        self.metadata_dir = (
            Path(metadata_dir).expanduser()
            if metadata_dir is not None
            else _resolve_metadata_dir(cache_dir, name)
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / _METADATA_FILE
        marker = self.metadata_dir / _MARKER_FILE
        key = f"root={self.source_root}|clinical={self.clinical_path}"
        with FileLock(str(self.metadata_dir / (_METADATA_FILE + ".lock"))):
            current = marker.read_text().strip() if marker.exists() else None
            if refresh_metadata or not self.metadata_path.exists() or current != key:
                self.prepare_metadata()
                marker.write_text(key)

        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "cmmd.yaml")
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
        """Join every DICOM under ``root`` to the clinical row of its breast.

        Examples:
            >>> dataset = CMMDDataset(root="/path/to/cmmd")  # doctest: +SKIP
            >>> dataset.prepare_metadata()  # doctest: +SKIP
        """
        try:
            import pydicom  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CMMDDataset reads DICOM headers and needs pydicom. Install it "
                "with `pip install pyhealth[imaging]`."
            ) from exc

        clinical = _read_clinical_table(self.clinical_path)
        clinical.columns = [str(c).strip() for c in clinical.columns]
        labels = {}
        for _, row in clinical.iterrows():
            side = str(row["LeftRight"]).strip().upper()[:1]
            labels[(str(row["ID1"]).strip(), side)] = row

        rows, unlabeled = [], 0
        for dirpath, _, filenames in os.walk(self.source_root):
            for filename in sorted(filenames):
                if not filename.lower().endswith(".dcm"):
                    continue
                path = str(Path(dirpath) / filename)
                header = _read_header(path)
                if header is None:
                    continue
                patient_id = header["patient_id"] or Path(dirpath).name
                side = header["laterality"][:1]
                clinical_row = labels.get((patient_id, side))
                record = {
                    "patient_id": patient_id,
                    "record_id": header["record_id"] or path,
                    "image_path": path,
                    "laterality": side,
                    "view": header["view"],
                    "cohort": patient_id.split("-")[0],
                    "has_label": int(clinical_row is not None),
                    "label": "",
                    "classification": "",
                    "abnormality": "",
                    "subtype": "",
                    "age": "",
                }
                if clinical_row is None:
                    unlabeled += 1
                else:
                    classification = str(clinical_row["classification"]).strip()
                    subtype = clinical_row.get("subtype")
                    record.update(
                        {
                            "label": int(classification.lower() == "malignant"),
                            "classification": classification,
                            "abnormality": clinical_row.get("abnormality", ""),
                            "subtype": "" if pd.isna(subtype) else subtype,
                            "age": clinical_row.get("Age", ""),
                        }
                    )
                rows.append(record)
        if not rows:
            raise FileNotFoundError(f"No CMMD DICOMs found under {self.source_root}.")
        tmp_path = self.metadata_path.with_suffix(".csv.tmp")
        pd.DataFrame(rows, columns=_COLUMNS).to_csv(tmp_path, index=False)
        os.replace(tmp_path, self.metadata_path)
        logger.info(
            "Wrote %d CMMD images to %s (%d without a clinical row for their "
            "breast)",
            len(rows),
            self.metadata_path,
            unlabeled,
        )

    @property
    def default_task(self):
        """Return the default malignancy classification task.

        Examples:
            >>> dataset = CMMDDataset(root="/path/to/cmmd")  # doctest: +SKIP
            >>> dataset.default_task.task_name  # doctest: +SKIP
            'CMMDMalignancyClassification'
        """
        from ..tasks.cmmd_classification import CMMDMalignancyClassification

        return CMMDMalignancyClassification()
