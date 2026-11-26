"""PAD-UFES-20 smartphone skin-lesion dataset."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from filelock import FileLock

from .base_dataset import BaseDataset
from .imaging.io import _resolve_metadata_dir

if TYPE_CHECKING:
    from ..tasks.pad_ufes_20_classification import PADUFES20Classification

logger = logging.getLogger(__name__)

PAD_UFES_20_METADATA_CSV = "metadata.csv"
_DERIVED_CSV = "pad_ufes_20-metadata-pyhealth.csv"
_REQUIRED_COLUMNS = ("patient_id", "lesion_id", "img_id", "diagnostic")
PAD_UFES_20_CLINICAL_COLUMNS: tuple[str, ...] = (
    "smoke",
    "drink",
    "background_father",
    "background_mother",
    "age",
    "pesticide",
    "gender",
    "skin_cancer_history",
    "cancer_history",
    "has_piped_water",
    "has_sewage_system",
    "fitspatrick",
    "region",
    "diameter_1",
    "diameter_2",
    "itch",
    "grew",
    "hurt",
    "changed",
    "bleed",
    "elevation",
)
"""The 21 clinical columns of ``metadata.csv``, all kept as event attributes."""


def _find_images(image_root: Path) -> dict[str, str]:
    """Map each PNG file name under ``image_root`` to its absolute path.

    The official release splits the images over ``imgs_part_1/2/3``
    sub-folders, so the search is recursive.
    """
    found: dict[str, str] = {}
    for dirpath, _, files in os.walk(image_root):
        for name in files:
            if name.lower().endswith(".png"):
                found.setdefault(name, str(Path(dirpath, name).resolve()))
    return found


class PADUFES20Dataset(BaseDataset):
    """PAD-UFES-20 clinical (smartphone) skin-lesion dataset.

    2,298 photos of 1,641 skin lesions from 1,373 patients, collected by the
    Dermatological and Surgical Assistance Program (PAD) of the Federal
    University of Espirito Santo (UFES), Brazil. Each image has one of six
    diagnoses: actinic keratosis (ACK), basal cell carcinoma (BCC), melanoma
    (MEL), nevus (NEV), squamous cell carcinoma including Bowen's disease
    (SCC) and seborrheic keratosis (SEK). About 58% are biopsy-proven
    (``biopsed``). These are clinical photos taken with smartphones, not
    dermoscopy images.

    One event is created per image. ``patient_id`` is the PAD patient id
    (``PAT_xxx``), so :class:`~pyhealth.datasets.PatientSplit` keeps every
    image and lesion of a patient in one split. Every other column of
    ``metadata.csv`` is an event attribute: ``image_id`` (the ``img_id``
    column), ``path``, ``lesion_id``, ``diagnostic``, ``biopsed`` and all
    21 clinical columns: ``smoke``, ``drink``, ``background_father``,
    ``background_mother``, ``age``, ``pesticide``, ``gender``,
    ``skin_cancer_history``, ``cancer_history``, ``has_piped_water``,
    ``has_sewage_system``, ``fitspatrick``, ``region``, ``diameter_1``,
    ``diameter_2``, ``itch``, ``grew``, ``hurt``, ``changed``, ``bleed``
    and ``elevation``. Many are empty for some patients, and a clinical
    column absent from ``metadata.csv`` is kept as an empty attribute.
    Most PNGs are RGBA, so image tasks should convert to RGB (the bundled
    :class:`~pyhealth.tasks.PADUFES20Classification` does).

    The derived metadata CSV is written to a metadata directory, never into
    ``root``. By default that directory is under the PyHealth cache, keyed
    by a hash of ``root``, the source ``metadata.csv`` and the image file
    names. Rows whose image file is missing are dropped with a warning.

    Getting the data:
        - Source: https://data.mendeley.com/datasets/zr7vgbcyr2/1
          (Mendeley Data, DOI 10.17632/zr7vgbcyr2.1).
        - Access: open. No account, registration or data-use agreement.
        - Download: four files, 3.6 GB in total: ``metadata.csv``
          (316 KB) and ``imgs_part_1.zip`` (1.2 GB), ``imgs_part_2.zip``
          (1.1 GB) and ``imgs_part_3.zip`` (1.2 GB). Either click
          "Download All" on the Mendeley Data page (one zip of the same
          files; unzip it and move ``metadata.csv`` into ``root`` and the
          three image archives into ``root/images/``), or fetch each file
          from its permanent public link::

              BASE=https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files
              mkdir -p pad_ufes_20/images && cd pad_ufes_20
              wget -O metadata.csv $BASE/fa850265-57da-48f0-ba3e-998b3e44b1f6/file_downloaded
              wget -O images/imgs_part_1.zip $BASE/1cc2f71f-20a2-412d-b746-a9b9bc20c966/file_downloaded
              wget -O images/imgs_part_2.zip $BASE/559a60ed-5504-475d-996c-6a8bc253b5e7/file_downloaded
              wget -O images/imgs_part_3.zip $BASE/34dcdf8e-e5f1-4b35-aa0b-5135051ff852/file_downloaded

        - Unpack: unzip each image archive inside ``images/``
          (``cd images && for f in imgs_part_*.zip; do unzip -q "$f"; done``).
          Each creates an ``imgs_part_<n>/`` folder of PNGs. The zips can
          then be deleted.
        - ``root`` layout (PNGs may also sit directly in ``images/``)::

              pad_ufes_20/              <- root
                  metadata.csv
                  images/
                      imgs_part_1/PAT_100_393_595.png ...
                      imgs_part_2/...
                      imgs_part_3/...

    Licence:
        CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/):
        commercial use and derivatives allowed, with attribution.

    Citation:
        Pacheco AGC, Lima GR, Salomao AS, et al. PAD-UFES-20: a skin lesion
        dataset composed of patient data and clinical images collected from
        smartphones. Data in Brief 32, 106221 (2020).
        https://doi.org/10.1016/j.dib.2020.106221

    Args:
        root: Folder holding ``metadata.csv`` and ``images/``.
        dataset_name: Optional dataset name. Defaults to ``"pad_ufes_20"``.
        config_path: Optional YAML config. Defaults to the bundled
            ``configs/pad_ufes_20.yaml``.
        cache_dir: Directory for PyHealth caches. Also the default parent of
            the metadata directory.
        metadata_dir: Explicit directory for the derived metadata CSV.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        refresh_metadata: Regenerate the derived metadata CSV.

    Examples:
        >>> from pyhealth.datasets import PADUFES20Dataset
        >>> from pyhealth.tasks import PADUFES20Classification
        >>> dataset = PADUFES20Dataset(root="/data/pad_ufes_20")  # doctest: +SKIP
        >>> samples = dataset.set_task(PADUFES20Classification())  # doctest: +SKIP
    """

    default_dataset_name = "pad_ufes_20"
    default_table = "pad_ufes_20"

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
        self.source_root = Path(root).expanduser().resolve()
        self.metadata_csv = self.source_root / PAD_UFES_20_METADATA_CSV
        self.image_root = self.source_root / "images"
        if not self.metadata_csv.is_file():
            raise FileNotFoundError(
                f"{PAD_UFES_20_METADATA_CSV} not found under {self.source_root}. "
                "Download PAD-UFES-20 from "
                "https://data.mendeley.com/datasets/zr7vgbcyr2/1"
            )
        if not self.image_root.is_dir():
            raise FileNotFoundError(
                f"images/ folder not found under {self.source_root}. Unzip "
                "images/imgs_part_{1,2,3}.zip from the PAD-UFES-20 archive."
            )

        name = dataset_name or self.default_dataset_name
        self._images = _find_images(self.image_root)
        digest = hashlib.sha1(str(self.source_root).encode("utf-8"))
        digest.update(self.metadata_csv.read_bytes())
        digest.update("\n".join(sorted(self._images)).encode("utf-8"))
        source_key = digest.hexdigest()[:10]
        self.metadata_dir = (
            Path(metadata_dir).expanduser()
            if metadata_dir is not None
            else _resolve_metadata_dir(cache_dir, name) / source_key
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / _DERIVED_CSV
        with FileLock(str(self.metadata_path) + ".lock"):
            if refresh_metadata or not self.metadata_path.exists():
                self.prepare_metadata()

        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "pad_ufes_20.yaml")
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
        """Write the derived metadata CSV with one row per available image.

        Raises:
            ValueError: If ``metadata.csv`` lacks a required column.
            FileNotFoundError: If none of the listed images is found.

        Examples:
            >>> ds = PADUFES20Dataset(root="/data/pad_ufes_20")  # doctest: +SKIP
            >>> ds.prepare_metadata()  # doctest: +SKIP
        """
        meta = pd.read_csv(self.metadata_csv)
        missing_cols = [c for c in _REQUIRED_COLUMNS if c not in meta.columns]
        if missing_cols:
            raise ValueError(f"{self.metadata_csv} is missing columns {missing_cols}.")
        for column in (*PAD_UFES_20_CLINICAL_COLUMNS, "biopsed"):
            if column not in meta.columns:
                meta[column] = None
        meta["path"] = meta["img_id"].map(self._images)
        n_missing = int(meta["path"].isna().sum())
        if n_missing == len(meta):
            raise FileNotFoundError(
                f"None of the PNGs listed in {self.metadata_csv} were found "
                f"under {self.image_root}."
            )
        if n_missing:
            logger.warning(
                "PAD-UFES-20: dropping %d of %d rows whose image is missing.",
                n_missing,
                len(meta),
            )
        meta = meta.dropna(subset=["path"]).rename(columns={"img_id": "image_id"})
        meta["lesion_id"] = meta["lesion_id"].astype(str)
        tmp_path = self.metadata_path.with_suffix(".csv.tmp")
        meta.to_csv(tmp_path, index=False)
        os.replace(tmp_path, self.metadata_path)
        logger.info("Wrote %d PAD-UFES-20 images to %s", len(meta), self.metadata_path)

    @property
    def default_task(self) -> PADUFES20Classification:
        """Return the default six-class diagnosis task.

        Returns:
            A :class:`~pyhealth.tasks.PADUFES20Classification` instance.

        Examples:
            >>> ds = PADUFES20Dataset(root="/data/pad_ufes_20")  # doctest: +SKIP
            >>> ds.default_task.task_name  # doctest: +SKIP
            'PADUFES20Classification'
        """
        from ..tasks.pad_ufes_20_classification import PADUFES20Classification

        return PADUFES20Classification()
