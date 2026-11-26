"""CBIS-DDSM curated screening mammography dataset."""

import logging
import os
import re
from pathlib import Path

import pandas as pd
from filelock import FileLock

from ..utils import download_file
from .base_dataset import BaseDataset
from .imaging.io import _extract_zip, _resolve_metadata_dir
from .splitter import SubjectSplit

logger = logging.getLogger(__name__)

_NBIA_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
_CSV_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/{}.csv"
_CSV_STEMS = {
    ("mass", "train"): "mass_case_description_train_set",
    ("mass", "test"): "mass_case_description_test_set",
    ("calc", "train"): "calc_case_description_train_set",
    ("calc", "test"): "calc_case_description_test_set",
}
_METADATA_FILE = "cbis-ddsm-metadata-pyhealth.csv"
_MARKER_FILE = "cbis-ddsm-metadata-key.txt"
# DICOM UIDs are dot-separated digit groups (PS3.5 9.1).
_UID_PATTERN = re.compile(r"[0-9]+(\.[0-9]+)*")


def _series_uid_from_path(image_file_path: str) -> str:
    """Return the SeriesInstanceUID part of a CBIS-DDSM CSV image path.

    CSV paths look like ``Mass-Training_P_00001_LEFT_CC/<StudyUID>/
    <SeriesUID>/000000.dcm``. The file name differs between the CSVs and
    the NBIA downloads, so the series UID is the stable key.
    """
    parts = str(image_file_path).strip().split("/")
    return parts[2] if len(parts) > 2 else ""


class CBISDDSMDataset(BaseDataset):
    """CBIS-DDSM (Curated Breast Imaging Subset of DDSM) mammography dataset.

    The dataset has one event per annotated lesion row of the four official
    case-description CSVs (mass/calcification x train/test). Each row points
    to a full-field mammogram of one breast in one view (CC or MLO). Several
    lesions can share an image, so the image-level tasks group events by
    ``image_id`` (``<patient>_<side>_<view>``). Rows whose full-mammogram
    DICOM is not on disk are skipped and counted in the log.

    Official split and leakage: CBIS-DDSM ships a train/test split, stored in
    the ``split`` attribute. It is not patient-disjoint: mass and
    calcification rows of the same patient (sometimes of the same image)
    fall in different splits, so 31 patients appear in both. Set
    ``drop_overlap_patients=True`` to remove those patients; the split is
    then patient-disjoint and :meth:`official_split` returns it as a
    :class:`~pyhealth.datasets.SubjectSplit`.

    Differences from CLIMB: CLIMB trains and tests on the official split
    as is, with the 31 shared patients (and 29 images present in both
    splits) left in. The default here matches it; ``drop_overlap_patients``
    is the leak-free option. For BI-RADS, see
    :class:`~pyhealth.tasks.CBISDDSMBIRADSClassification`.

    Getting the data:
        - Source page: https://www.cancerimagingarchive.net/collection/cbis-ddsm/
          (The Cancer Imaging Archive, collection doi
          10.7937/K9/TCIA.2016.7O02S9CY). The paper is cited below.
        - Access: open. No account, registration or data-use agreement is
          needed. You must follow the TCIA Data Usage Policy and cite the
          data set and the paper.
        - Case-description CSVs (about 2 MB in total):
          ``mass_case_description_train_set.csv`` (755 KB),
          ``mass_case_description_test_set.csv`` (212 KB),
          ``calc_case_description_train_set.csv`` (904 KB) and
          ``calc_case_description_test_set.csv`` (186 KB). Pass
          ``download=True`` to fetch them, or run, for each file,
          ``wget https://www.cancerimagingarchive.net/wp-content/uploads/<name>.csv``.
        - DICOMs (163.5 GB for the whole collection, 6,775 series, 10,239
          images): install the NBIA Data Retriever
          (https://www.cancerimagingarchive.net/tcia-data-retriever/),
          download a ``.tcia`` manifest from the "Data Access" table of the
          source page, open it in the Data Retriever, choose ``root`` as the
          destination and start the download. Only the full mammograms are
          used, so the four "full_mammogram_images" manifests of the
          "Detailed Description" section are enough (the
          ``CBIS-DDSM-All-...tcia`` manifest also brings the ROI masks and
          crops, which are ignored). For a few series,
          :meth:`download_series` fetches them over the NBIA REST API.
        - Unpacking: nothing to unpack. The CSVs are plain files, the Data
          Retriever writes DICOM folders and :meth:`download_series`
          extracts its zips itself.
        - ``root`` layout. CSVs can be at any depth; each full-mammogram
          series is found by the name of its folder (the SeriesInstanceUID)
          or through the Data Retriever's ``metadata.csv``::

              root/
                mass_case_description_train_set.csv
                mass_case_description_test_set.csv
                calc_case_description_train_set.csv
                calc_case_description_test_set.csv
                # then the DICOMs, in any one of these layouts:
                dicom/<SeriesUID>/*.dcm               # download_series()
                <case>/<StudyUID>/<SeriesUID>/*.dcm   # CSV image paths
                <any>/metadata.csv                    # Data Retriever output,
                <any>/CBIS-DDSM/...                   # found via metadata.csv

          Derived metadata is written to ``metadata_dir`` (by default under
          the PyHealth cache), never into ``root``.
        - Licence: Creative Commons Attribution 3.0 (CC BY 3.0). Commercial
          use and derivatives are allowed with attribution; there are no NC
          or ND terms.

    Citation:
        Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M., &
        Rubin, D. L. (2017). A curated mammography data set for use in
        computer-aided detection and diagnosis research. Scientific Data, 4,
        170177. https://doi.org/10.1038/sdata.2017.177

        Data set: Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D.
        (2016). Curated Breast Imaging Subset of Digital Database for
        Screening Mammography (CBIS-DDSM). The Cancer Imaging Archive.
        https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY

    Benchmarked in CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB).

    Args:
        root: Folder with the case-description CSVs and the DICOMs.
        download: Download the four case-description CSVs when missing.
        drop_overlap_patients: Drop patients that appear in both the official
            train and test CSVs (31 patients). Default False keeps the
            official split unchanged.
        dataset_name: Optional dataset name. Default ``"cbis_ddsm"``.
        config_path: Optional table configuration path.
        cache_dir: Directory for processed caches.
        metadata_dir: Directory for the derived metadata CSV. Defaults to
            ``<cache>/metadata/<dataset_name>`` (with a ``_no_overlap``
            suffix when ``drop_overlap_patients=True``).
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        refresh_metadata: Regenerate the metadata CSV, e.g. after adding
            DICOMs to ``root``.

    Examples:
        >>> from pyhealth.datasets import CBISDDSMDataset
        >>> dataset = CBISDDSMDataset(root="/path/to/cbis-ddsm")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
    """

    default_dataset_name = "cbis_ddsm"
    default_table = "lesions"

    def __init__(
        self,
        root: str,
        download: bool = False,
        drop_overlap_patients: bool = False,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        metadata_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
        refresh_metadata: bool = False,
    ) -> None:
        self.source_root = Path(root).expanduser().resolve()
        self.drop_overlap_patients = bool(drop_overlap_patients)
        if download:
            self._download_csvs()

        name = dataset_name or self.default_dataset_name
        self.metadata_dir = (
            Path(metadata_dir).expanduser()
            if metadata_dir is not None
            else _resolve_metadata_dir(
                cache_dir, f"{name}_no_overlap" if self.drop_overlap_patients else name
            )
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / _METADATA_FILE
        marker = self.metadata_dir / _MARKER_FILE
        key = (
            f"root={self.source_root}"
            f"|drop_overlap_patients={int(self.drop_overlap_patients)}"
        )
        with FileLock(str(self.metadata_dir / (_METADATA_FILE + ".lock"))):
            current = marker.read_text().strip() if marker.exists() else None
            if refresh_metadata or not self.metadata_path.exists() or current != key:
                self.prepare_metadata()
                marker.write_text(key)

        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "cbis_ddsm.yaml")
        super().__init__(
            root=str(self.metadata_dir),
            tables=[self.default_table],
            dataset_name=name,
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def _download_csvs(self) -> None:
        """Fetch the four case-description CSVs into ``root``."""
        self.source_root.mkdir(parents=True, exist_ok=True)
        for stem in _CSV_STEMS.values():
            if not list(self.source_root.rglob(f"{stem}.csv")):
                download_file(
                    _CSV_URL.format(stem), str(self.source_root / f"{stem}.csv")
                )

    def _find_csv(self, stem: str) -> Path:
        """Locate one case-description CSV under ``root``."""
        matches = sorted(self.source_root.rglob(f"{stem}.csv"))
        if not matches:
            raise FileNotFoundError(
                f"{stem}.csv not found under {self.source_root}. Pass "
                "download=True or download it from "
                "https://www.cancerimagingarchive.net/collection/cbis-ddsm/."
            )
        return matches[0]

    def _series_index(self) -> dict[str, str]:
        """Map each SeriesInstanceUID on disk to its first DICOM file."""
        index: dict[str, str] = {}
        for dirpath, _, filenames in os.walk(self.source_root):
            dicoms = sorted(f for f in filenames if f.lower().endswith(".dcm"))
            if dicoms:
                index.setdefault(
                    Path(dirpath).name, str(Path(dirpath) / dicoms[0])
                )
            if "metadata.csv" in filenames:
                meta_path = Path(dirpath) / "metadata.csv"
                try:
                    meta = pd.read_csv(meta_path)
                except (OSError, ValueError) as exc:
                    logger.warning("Could not read %s: %s", meta_path, exc)
                    continue
                if not {"Series UID", "File Location"} <= set(meta.columns):
                    continue
                for uid, location in zip(meta["Series UID"], meta["File Location"]):
                    folder = (Path(dirpath) / str(location)).resolve()
                    files = sorted(folder.glob("*.dcm")) if folder.is_dir() else []
                    if files:
                        index[str(uid)] = str(files[0])
        return index

    def prepare_metadata(self) -> None:
        """Write one row per lesion whose full-mammogram DICOM is on disk.

        Examples:
            >>> dataset = CBISDDSMDataset(root="/path/to/cbis-ddsm")  # doctest: +SKIP
            >>> dataset.prepare_metadata()  # doctest: +SKIP
        """
        index = self._series_index()
        rows, missing = [], 0
        for (kind, split), stem in _CSV_STEMS.items():
            df = pd.read_csv(self._find_csv(stem))
            df.columns = [str(c).strip().replace("_", " ") for c in df.columns]
            for _, row in df.iterrows():
                series_uid = _series_uid_from_path(row["image file path"])
                if series_uid not in index:
                    missing += 1
                    continue
                patient_id = str(row["patient id"]).strip()
                side = str(row["left or right breast"]).strip()
                view = str(row["image view"]).strip()
                abnormality_id = int(row["abnormality id"])
                image_id = f"{patient_id}_{side}_{view}"
                rows.append(
                    {
                        "patient_id": patient_id,
                        "record_id": f"{kind}_{split}_{image_id}_{abnormality_id}",
                        "image_id": image_id,
                        "image_path": index[series_uid],
                        "series_uid": series_uid,
                        "laterality": side,
                        "view": view,
                        "abnormality_id": abnormality_id,
                        "abnormality_type": kind,
                        "assessment": int(row["assessment"]),
                        "pathology": str(row["pathology"]).strip(),
                        "breast_density": row["breast density"],
                        "subtlety": row["subtlety"],
                        "split": split,
                    }
                )
        columns = [
            "patient_id", "record_id", "image_id", "image_path", "series_uid",
            "laterality", "view", "abnormality_id", "abnormality_type",
            "assessment", "pathology", "breast_density", "subtlety", "split",
        ]
        metadata = pd.DataFrame(rows, columns=columns)
        splits_per_patient = metadata.groupby("patient_id")["split"].nunique()
        overlap = set(splits_per_patient[splits_per_patient > 1].index)
        if overlap:
            if self.drop_overlap_patients:
                metadata = metadata[~metadata["patient_id"].isin(overlap)]
                logger.info(
                    "Dropped %d patients present in both official splits",
                    len(overlap),
                )
            else:
                logger.warning(
                    "%d CBIS-DDSM patients appear in both the official train "
                    "and test CSVs. Pass drop_overlap_patients=True for a "
                    "patient-disjoint split.",
                    len(overlap),
                )
        if metadata.empty:
            raise FileNotFoundError(
                f"No CBIS-DDSM full-mammogram DICOMs found under "
                f"{self.source_root} ({missing} CSV rows without an image)."
            )
        tmp_path = self.metadata_path.with_suffix(".csv.tmp")
        metadata.to_csv(tmp_path, index=False)
        os.replace(tmp_path, self.metadata_path)
        logger.info(
            "Wrote %d CBIS-DDSM lesions to %s (%d CSV rows without an image)",
            len(metadata),
            self.metadata_path,
            missing,
        )

    def official_split(self) -> SubjectSplit:
        """Return the official train/test split as a patient split.

        The split is only patient-disjoint after dropping the patients
        present in both official CSVs, so this needs
        ``drop_overlap_patients=True`` whenever such patients are loaded.
        To keep every image instead, build the parts with the task's
        ``split`` argument (``split="train"`` / ``split="test"``).

        Returns:
            A :class:`~pyhealth.datasets.SubjectSplit` of (train, test)
            patient ids for ``set_task(task, split=...)``.

        Raises:
            ValueError: If a loaded patient is in both splits, or a split
                has no patients.

        Examples:
            >>> from pyhealth.datasets import CBISDDSMDataset
            >>> from pyhealth.tasks import CBISDDSMPathologyClassification
            >>> dataset = CBISDDSMDataset(  # doctest: +SKIP
            ...     root="/path/to/cbis-ddsm", drop_overlap_patients=True
            ... )
            >>> train, test = dataset.set_task(  # doctest: +SKIP
            ...     CBISDDSMPathologyClassification(),
            ...     split=dataset.official_split(),
            ... )
        """
        metadata = pd.read_csv(self.metadata_path, dtype={"patient_id": str})
        train = set(metadata.loc[metadata["split"] == "train", "patient_id"])
        test = set(metadata.loc[metadata["split"] == "test", "patient_id"])
        overlap = train & test
        if overlap:
            raise ValueError(
                f"{len(overlap)} patients are in both official splits. Load "
                "the dataset with drop_overlap_patients=True, or use the "
                "task's split='train'/'test' argument."
            )
        if not train or not test:
            raise ValueError("Both official splits need at least one patient.")
        return SubjectSplit((sorted(train), sorted(test)))

    def download_series(self, series_uids: list[str]) -> None:
        """Download DICOM series into ``root/dicom/<SeriesUID>/`` (NBIA REST).

        Meant for small subsets; use the NBIA Data Retriever for the full
        collection. Call :meth:`prepare_metadata` (or reload with
        ``refresh_metadata=True``) afterwards to index the new files.

        Args:
            series_uids: SeriesInstanceUIDs, e.g. from the CSV image paths.

        Raises:
            ValueError: If a UID is not a dotted-digit DICOM UID. The UID
                becomes a folder name, so anything else (``/``, ``..``) is
                rejected before any path is built.

        Examples:
            >>> dataset = CBISDDSMDataset(root="/path/to/cbis-ddsm")  # doctest: +SKIP
            >>> dataset.download_series(  # doctest: +SKIP
            ...     ["1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515"]
            ... )
        """
        for uid in series_uids:
            if not isinstance(uid, str) or not _UID_PATTERN.fullmatch(uid):
                raise ValueError(f"Not a DICOM SeriesInstanceUID: {uid!r}")
            target = self.source_root / "dicom" / uid
            if target.is_dir() and any(target.glob("*.dcm")):
                continue
            archive = self.source_root / "dicom" / f"{uid}.zip"
            archive.parent.mkdir(parents=True, exist_ok=True)
            download_file(
                f"{_NBIA_URL}/getImage?SeriesInstanceUID={uid}", str(archive)
            )
            _extract_zip(archive, target)
            archive.unlink()

    @property
    def default_task(self):
        """Return the default image-level pathology task.

        Examples:
            >>> dataset = CBISDDSMDataset(root="/path/to/cbis-ddsm")  # doctest: +SKIP
            >>> dataset.default_task.task_name  # doctest: +SKIP
            'CBISDDSMPathologyClassification'
        """
        from ..tasks.cbis_ddsm_classification import (
            CBISDDSMPathologyClassification,
        )

        return CBISDDSMPathologyClassification()
