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
          https://github.com/neheller/kits23 (the CTs are hosted on Hugging
          Face, https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging).
        - Access: open, no account, registration or agreement needed.
        - Files: the GitHub repository (about 0.5 GB) holds ``kits23.json``
          and each case's ``segmentation.nii.gz`` (plus ``instances/``
          annotator masks). The 489 CTs, ``images/case_XXXXX.nii.gz`` on
          Hugging Face (about 42 GB in total, 86 MB on average),
          are fetched separately.
        - Download::

              git clone https://github.com/neheller/kits23
              cd kits23
              pip3 install -e .
              kits23_download_data     # all CTs into dataset/case_XXXXX/imaging.nii.gz

          To try a few cases, skip ``kits23_download_data`` and fetch single
          CTs instead::

              wget -O dataset/case_00000/imaging.nii.gz \
                  https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/case_00000.nii.gz

          Nothing needs unpacking (``.nii.gz`` is read directly). Cases
          without ``imaging.nii.gz`` are kept in the metadata with
          ``has_imaging = 0`` and skipped by the tasks.
        - ``root`` layout (the repository root also works)::

              kits23/dataset/                 # root
                  kits23.json
                  case_00000/
                      imaging.nii.gz          # from Hugging Face
                      segmentation.nii.gz     # from the git clone
                      instances/
                  case_00001/
                  ...

        - Licence: the images and segmentations are CC BY-NC-SA 4.0
          (non-commercial, attribution, share alike; the repository code is
          MIT). Contact the KiTS organisers for commercial use.

    The metadata CSV is written to ``metadata_dir``, never into ``root``.
    By default this is ``<cache_dir>/metadata/kits23-<hash>``, where
    ``<hash>`` is the first 12 hex digits of the SHA-1 of the resolved
    ``root``, so two roots sharing one ``cache_dir`` get separate CSVs. An
    explicit ``metadata_dir`` is used as given. The CSV is rebuilt when
    ``root``, ``kits23.json`` or the set of downloaded CTs changes.

    Citation:
        Heller, N., Isensee, F., Trofimova, D., et al. The KiTS21 Challenge:
        Automatic segmentation of kidneys, renal tumors, and renal cysts in
        corticomedullary-phase CT. arXiv:2307.01984, 2023 (the paper the
        KiTS23 repository asks users to cite). The KiTS23 challenge itself is
        described at https://kits-challenge.org/kits23/. Predecessor
        challenge: Heller, N., Isensee, F., Maier-Hein, K. H., et al. The
        state of the art in kidney and kidney tumor segmentation in
        contrast-enhanced CT imaging: Results of the KiTS19 challenge.
        Medical Image Analysis 67, 101821, 2021.

    Benchmarked in CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB).
    Deviations from CLIMB: CLIMB labels KiTS23 as benign / malignant (two
    entries in its label list) but sets ``NUM_CLASSES = 4``, giving a four-way
    head for a binary task; here the task is binary. CLIMB splits on the
    first 75% of cases in ``kits23.json`` order; here every sample carries
    its case id as ``patient_id`` so a patient split cannot leak. Only 9% of
    cases are benign, and PyHealth's ``split_by_patient`` is not stratified,
    so on a small subset a split may hold one class only, which leaves ROC
    AUC undefined. Cases with an unknown ``malignant``
    value are dropped by the task rather than counted as benign.

    Args:
        root: The KiTS23 ``dataset/`` folder, or the repository root.
        dataset_name: Optional dataset name. Defaults to ``"kits23"``.
        config_path: Optional table configuration path.
        cache_dir: Directory for PyHealth caches.
        metadata_dir: Explicit directory for the metadata CSV. Defaults to
            ``<cache_dir>/metadata/kits23-<sha1(root)[:12]>``.
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
            else _resolve_metadata_dir(
                cache_dir, f"{name}-{self._root_digest(source_root)}"
            )
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

    @staticmethod
    def _root_digest(source_root: Path) -> str:
        """Short SHA-1 of the resolved root, keying the default metadata dir."""
        return hashlib.sha1(str(source_root).encode("utf-8")).hexdigest()[:12]

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
