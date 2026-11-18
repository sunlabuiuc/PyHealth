"""KiTS23 kidney tumour CT dataset."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from filelock import FileLock

from .base_dataset import BaseDataset
from .imaging.io import _resolve_metadata_dir

logger = logging.getLogger(__name__)

_CLINICAL_JSON = "kits23.json"
_METADATA_CSV = "kits23-metadata-pyhealth.csv"
_MARKER = "kits23-metadata-source.txt"

#: Clinical fields of ``kits23.json`` copied into the metadata CSV.
KITS23_CLINICAL_FIELDS = (
    "tumor_histologic_subtype",
    "tumor_isup_grade",
    "pathology_t_stage",
    "pathology_n_stage",
    "pathology_m_stage",
    "pathologic_size",
    "radiographic_size",
    "gender",
    "age_at_nephrectomy",
    "vital_status",
)


def _malignant_code(value: Any) -> int:
    """Return 1 malignant, 0 benign, -1 unknown for a ``malignant`` entry."""
    if value is True:
        return 1
    if value is False:
        return 0
    return -1


class KiTS23Dataset(BaseDataset):
    """KiTS23: contrast-enhanced abdominal CTs of kidney tumour patients.

    The 2023 Kidney Tumor Segmentation Challenge training set holds 489
    preoperative CTs of patients who had a partial or radical nephrectomy at
    the M Health Fairview / University of Minnesota medical centre, each with
    a kidney / tumour / cyst segmentation and the clinical and pathology
    record of ``kits23.json``. This class reads that file into one ``cases``
    event per patient (one CT each):

    - ``patient_id`` and ``record_id``: the case id, e.g. ``case_00010``.
    - ``image_path``: ``case_XXXXX/imaging.nii.gz``, or empty when the
      imaging has not been downloaded; ``has_imaging`` says which.
    - ``segmentation_path``: ``case_XXXXX/segmentation.nii.gz`` (labels 1
      kidney, 2 tumour, 3 cyst), or empty when missing.
    - ``malignant``: 1 malignant, 0 benign, -1 unknown (7 cases have no
      label). Of the 482 labelled cases 439 are malignant and 43 benign.
    - The clinical fields of :data:`KITS23_CLINICAL_FIELDS`, such as the
      histologic subtype, ISUP grade and TNM stage.

    The challenge test set's labels are not public, so all 489 cases are
    training data and there is no official split.

    Getting the data:
        - Source: https://kits-challenge.org/kits23/ and
          https://github.com/neheller/kits23 (imaging on Hugging Face,
          ``neheller/KiTS-Challenge-Imaging``).
        - Access: open, no account needed.
        - Licence: the data are CC BY-NC-SA 4.0 (non-commercial, share
          alike; the repository code is MIT). Contact the KiTS organisers
          for commercial use.
        - Download: ``git clone https://github.com/neheller/kits23``, then
          ``pip install -e kits23`` and run ``kits23_download_data``, which
          puts ``imaging.nii.gz`` (about 42 GB in total) in each
          ``dataset/case_XXXXX/`` folder. A single case can be fetched from
          ``https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/case_XXXXX.nii.gz``
          and saved as ``dataset/case_XXXXX/imaging.nii.gz``.
        - ``root``: the repository's ``dataset/`` folder holding
          ``kits23.json`` and the ``case_XXXXX/`` folders (the repository
          root also works).

    The metadata CSV is written to ``metadata_dir`` (by default
    ``<cache_dir>/metadata/kits23``), never into ``root``. It is rebuilt
    when ``root``, ``kits23.json`` or the set of downloaded CTs changes.

    Citation:
        Heller, N., Isensee, F., Maier-Hein, K. H., et al. The state of the
        art in kidney and kidney tumor segmentation in contrast-enhanced CT
        imaging: Results of the KiTS19 challenge. Medical Image Analysis 67,
        101821, 2021. The KiTS23 dataset is described at
        https://kits-challenge.org/kits23/.

    Benchmarked in CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB).
    Deviations from CLIMB: CLIMB labels KiTS23 as benign / malignant (two
    entries in its label list) but sets ``NUM_CLASSES = 4``, giving a four-way
    head for a binary task; here the task is binary. CLIMB splits on the
    first 75% of cases in ``kits23.json`` order; here every sample carries
    its case id as ``patient_id`` so a patient split (stratified, since only
    9% of cases are benign) cannot leak. Cases with an unknown ``malignant``
    value are dropped by the task rather than counted as benign.

    Args:
        root: The KiTS23 ``dataset/`` folder, or the repository root.
        dataset_name: Optional dataset name. Defaults to ``"kits23"``.
        config_path: Optional table configuration path.
        cache_dir: Directory for PyHealth caches.
        metadata_dir: Explicit directory for the metadata CSV.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        refresh_metadata: Rebuild the metadata CSV even if it is current.

    Examples:
        >>> from pyhealth.datasets import KiTS23Dataset
        >>> from pyhealth.tasks import KiTS23MalignancyClassification
        >>> dataset = KiTS23Dataset(root="/data/kits23/dataset")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> samples = dataset.set_task(KiTS23MalignancyClassification())  # doctest: +SKIP
    """

    default_dataset_name = "kits23"
    default_table = "cases"

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        metadata_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
        refresh_metadata: bool = False,
    ) -> None:
        source_root = Path(root).expanduser().resolve()
        if not (source_root / _CLINICAL_JSON).exists() and (
            source_root / "dataset" / _CLINICAL_JSON
        ).exists():
            source_root = source_root / "dataset"
        self.source_root = source_root
        self.clinical_path = source_root / _CLINICAL_JSON
        if not self.clinical_path.exists():
            raise FileNotFoundError(
                f"{_CLINICAL_JSON} not found under {source_root}. Clone "
                "https://github.com/neheller/kits23 and pass its dataset/ "
                "folder as root."
            )

        name = dataset_name or self.default_dataset_name
        self.metadata_dir = (
            Path(metadata_dir).expanduser()
            if metadata_dir is not None
            else _resolve_metadata_dir(cache_dir, name)
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / _METADATA_CSV
        marker = self.metadata_dir / _MARKER
        source_key = self._source_key()
        with FileLock(str(self.metadata_path) + ".lock"):
            current = marker.read_text().strip() if marker.exists() else None
            if (
                refresh_metadata
                or not self.metadata_path.exists()
                or current != source_key
            ):
                self.prepare_metadata()
                marker.write_text(source_key)

        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "kits23.yaml")
        super().__init__(
            root=str(self.metadata_dir),
            tables=[self.default_table],
            dataset_name=name,
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def _source_key(self) -> str:
        """Hash of the inputs the metadata CSV depends on."""
        stat = self.clinical_path.stat()
        parts = [str(self.source_root), f"{stat.st_size}:{stat.st_mtime_ns}"]
        parts.extend(
            str(path) for path in sorted(self.source_root.glob("case_*/*.nii.gz"))
        )
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

    def prepare_metadata(self) -> None:
        """Write one row per case of ``kits23.json`` to the metadata CSV.

        Examples:
            >>> dataset = KiTS23Dataset(root="/data/kits23/dataset")  # doctest: +SKIP
            >>> dataset.prepare_metadata()  # doctest: +SKIP
        """
        with open(self.clinical_path) as handle:
            cases = json.load(handle)
        rows = []
        for case in cases:
            case_id = str(case["case_id"])
            folder = self.source_root / case_id
            image = folder / "imaging.nii.gz"
            segmentation = folder / "segmentation.nii.gz"
            row = {
                "patient_id": case_id,
                "record_id": case_id,
                "image_path": str(image) if image.exists() else "",
                "segmentation_path": (
                    str(segmentation) if segmentation.exists() else ""
                ),
                "has_imaging": int(image.exists()),
                "malignant": _malignant_code(case.get("malignant")),
            }
            for field in KITS23_CLINICAL_FIELDS:
                value = case.get(field)
                row[field] = "" if value is None else value
            rows.append(row)
        columns = [
            "patient_id", "record_id", "image_path", "segmentation_path",
            "has_imaging", "malignant", *KITS23_CLINICAL_FIELDS,
        ]
        pd.DataFrame(rows, columns=columns).to_csv(self.metadata_path, index=False)
        logger.info(
            "Wrote %d KiTS23 cases (%d with imaging) to %s",
            len(rows),
            sum(row["has_imaging"] for row in rows),
            self.metadata_path,
        )

    @property
    def default_task(self):
        """Return the default benign / malignant classification task.

        Examples:
            >>> KiTS23Dataset(root="/data/kits23").default_task  # doctest: +SKIP
        """
        from ..tasks.kits23_classification import KiTS23MalignancyClassification

        return KiTS23MalignancyClassification()
