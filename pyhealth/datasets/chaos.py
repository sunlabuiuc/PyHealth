"""CHAOS CT/MRI DICOM series dataset."""

import logging
from pathlib import Path

import pandas as pd

from .base_dataset import BaseDataset
from .imaging.io import _download_archives, _resolve_metadata_dir

logger = logging.getLogger(__name__)


def _count_files(path: Path, pattern: str) -> int:
    """Count files under a directory, returning 0 when it does not exist."""
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.rglob(pattern))


class CHAOSDataset(BaseDataset):
    """CHAOS abdominal CT/MRI dataset.

    Each event represents one DICOM series:

    - CT cases have one ``DICOM_anon`` series.
    - MR ``T1DUAL`` cases have ``InPhase`` and ``OutPhase`` series sharing one
      ground-truth directory.
    - MR ``T2SPIR`` cases have one series.

    The loader only records paths, series metadata and mask directory paths.
    Pixel data is read later by ``DicomSeriesProcessor``.

    Args:
        root: Root directory containing ``Train_Sets`` and ``Test_Sets``.
        dataset_name: Optional dataset name.
        config_path: Optional table configuration path.
        cache_dir: Directory for processed caches.
        metadata_dir: Explicit metadata output directory.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        download: Download and extract CHAOS when it is missing.
            Archive checksums are not verified automatically.
        refresh_metadata: Regenerate the metadata CSV.

    Examples:
        >>> from pyhealth.datasets import CHAOSDataset
        >>> dataset = CHAOSDataset(  # doctest: +SKIP
        ...     root="/path/to/CHAOS", download=True
        ... )
        >>> dataset.stats()  # doctest: +SKIP
    """

    default_dataset_name = "chaos"
    default_table = "series"
    download_urls = (
        "https://zenodo.org/api/records/3362845/files/CHAOS_Train_Sets.zip/content",
        "https://zenodo.org/api/records/3362845/files/CHAOS_Test_Sets.zip/content",
    )

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        metadata_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
        download: bool = False,
        refresh_metadata: bool = False,
    ) -> None:
        """Build metadata and initialize the CHAOS dataset.

        Args:
            root: Root directory containing the CHAOS split folders.
            dataset_name: Optional dataset name.
            config_path: Optional table configuration path.
            cache_dir: Directory for processed caches.
            metadata_dir: Explicit metadata output directory.
            num_workers: Number of workers for dataset processing.
            dev: Whether to limit the dataset to 1000 patients.
            download: Download and extract CHAOS when it is missing.
            refresh_metadata: Regenerate the metadata CSV.
        """
        source_root = Path(root).expanduser().resolve()
        if download and not (source_root / "Train_Sets").is_dir():
            _download_archives(
                self.download_urls,
                source_root,
                archive_type="zip",
                expected=("Train_Sets", "Test_Sets"),
            )
        if not (source_root / "Train_Sets").is_dir():
            raise FileNotFoundError(
                f"CHAOS Train_Sets not found under {source_root}. "
                "Pass download=True to download CHAOS."
            )

        name = dataset_name or self.default_dataset_name
        self.source_root = source_root
        self.metadata_dir = (
            Path(metadata_dir).expanduser()
            if metadata_dir is not None
            else _resolve_metadata_dir(cache_dir, name)
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / "chaos-metadata-pyhealth.csv"

        if refresh_metadata or not self.metadata_path.exists():
            self.prepare_metadata()

        if config_path is None:
            config_path = str(
                Path(__file__).parent / "configs" / "chaos.yaml"
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
        """Create the CHAOS series metadata CSV."""
        rows = []
        for split in ("Train_Sets", "Test_Sets"):
            split_root = self.source_root / split
            rows.extend(self._ct_rows(split, split_root / "CT"))
            rows.extend(self._mr_rows(split, split_root / "MR"))

        train_patients = {
            row["patient_id"] for row in rows if row["split"] == "Train_Sets"
        }
        test_patients = {
            row["patient_id"] for row in rows if row["split"] == "Test_Sets"
        }
        overlap = sorted(train_patients & test_patients)
        if overlap:
            raise ValueError(
                f"CHAOS Train_Sets and Test_Sets share patient IDs: {overlap}."
            )

        pd.DataFrame(rows).to_csv(self.metadata_path, index=False)
        logger.info(
            "Wrote %d CHAOS series to %s", len(rows), self.metadata_path
        )

    def _ct_rows(self, split: str, root: Path) -> list[dict]:
        """Build rows for CT cases."""
        if not root.is_dir():
            return []
        rows = []
        for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            series_dir = case_dir / "DICOM_anon"
            if not series_dir.is_dir():
                continue
            case = case_dir.name
            rows.append(
                self._metadata_row(
                    split=split,
                    patient_id=f"ct_{case}",
                    record_id=f"ct_{case}",
                    series_dir=series_dir,
                    mask_dir=case_dir / "Ground",
                    modality="CT",
                    sequence="CT",
                    phase="",
                )
            )
        return rows

    def _mr_rows(self, split: str, root: Path) -> list[dict]:
        """Build rows for MR cases and T1DUAL phases."""
        if not root.is_dir():
            return []
        rows = []
        for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            case = case_dir.name
            for sequence in ("T1DUAL", "T2SPIR"):
                sequence_dir = case_dir / sequence
                dicom_root = sequence_dir / "DICOM_anon"
                if not dicom_root.is_dir():
                    continue
                mask_dir = sequence_dir / "Ground"
                phases = sorted(
                    path.name
                    for path in dicom_root.iterdir()
                    if path.is_dir()
                    and path.name in {"InPhase", "OutPhase"}
                )
                if sequence == "T1DUAL" and phases:
                    for phase in phases:
                        rows.append(
                            self._metadata_row(
                                split=split,
                                patient_id=f"mr_{case}",
                                record_id=(
                                    f"mr_{case}_t1dual_{phase.lower()}"
                                ),
                                series_dir=dicom_root / phase,
                                mask_dir=mask_dir,
                                modality="MR",
                                sequence=sequence,
                                phase=phase,
                            )
                        )
                else:
                    rows.append(
                        self._metadata_row(
                            split=split,
                            patient_id=f"mr_{case}",
                            record_id=f"mr_{case}_{sequence.lower()}",
                            series_dir=dicom_root,
                            mask_dir=mask_dir,
                            modality="MR",
                            sequence=sequence,
                            phase="",
                        )
                    )
        return rows

    @staticmethod
    def _metadata_row(
        split: str,
        patient_id: str,
        record_id: str,
        series_dir: Path,
        mask_dir: Path,
        modality: str,
        sequence: str,
        phase: str,
    ) -> dict:
        """Build one metadata row."""
        has_mask = mask_dir.is_dir()
        return {
            "patient_id": patient_id,
            "record_id": record_id,
            "series_path": str(series_dir.resolve()),
            "mask_dir": str(mask_dir.resolve()) if has_mask else "",
            "modality": modality,
            "sequence": sequence,
            "phase": phase,
            "split": split,
            "n_slices": _count_files(series_dir, "*.dcm"),
            "n_masks": _count_files(mask_dir, "*.png") if has_mask else 0,
            "has_mask": has_mask,
        }

    @property
    def default_task(self):
        """Return the default CHAOS segmentation task.

        Official task 4 (abdominal organs, CT + MRI) is the challenge's
        combined multi-organ task.  Use the other CHAOS task classes for
        single-modality or liver-only experiments.
        """
        from ..tasks.chaos_segmentation import (
            CHAOSAbdominalSegmentationCTMRI,
        )

        return CHAOSAbdominalSegmentationCTMRI()
