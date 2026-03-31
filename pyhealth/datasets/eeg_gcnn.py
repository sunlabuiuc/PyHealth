import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class EEGGCNNDataset(BaseDataset):
    """EEG-GCNN dataset pooling TUAB normal-subset and MPI LEMON controls.

    This dataset supports the EEG-GCNN paper (Wagh & Varatharajah, ML4H @
    NeurIPS 2020) which distinguishes "normal-appearing" patient EEGs (from
    TUAB) from truly healthy EEGs (from MPI LEMON).

    **TUAB (normal subset):** The Temple University EEG Abnormal Corpus
    provides EDF recordings labelled normal/abnormal. Only the *normal*
    recordings are used here — these are the "patient" class (label 0).

    **MPI LEMON:** The Leipzig Study for Mind-Body-Emotion Interactions
    provides BrainVision EEG recordings from healthy controls — these form
    the "healthy" class (label 1).

    Paper:
        Wagh, N. & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
        Electroencephalogram-based Neurological Disease Diagnosis using a
        Domain-guided Graph Convolutional Neural Network. *Proceedings of
        Machine Learning for Health (ML4H) at NeurIPS 2020*, PMLR 136.
        https://proceedings.mlr.press/v136/wagh20a.html

    Authors' code: https://github.com/neerajwagh/eeg-gcnn

    Args:
        root: Root directory containing TUAB and/or LEMON data.
            Expected structure for TUAB::

                <root>/train/normal/01_tcp_ar/<subject_dirs>/*.edf
                <root>/eval/normal/01_tcp_ar/<subject_dirs>/*.edf

            Expected structure for LEMON::

                <root>/lemon/<subject_dirs>/*.vhdr

        dataset_name: Name of the dataset. Defaults to ``"eeg_gcnn"``.
        config_path: Path to the YAML config. Defaults to the built-in
            ``eeg_gcnn.yaml``.
        subset: Which data source(s) to load. One of ``"tuab"``,
            ``"lemon"``, or ``"both"`` (default).
        dev: If ``True``, limit to a small subset for quick iteration.

    Attributes:
        task: Optional task name after ``set_task()`` is called.
        samples: Sample list after task is set.
        patient_to_index: Maps patient IDs to sample indices.
        visit_to_index: Maps visit/record IDs to sample indices.

    Examples:
        >>> from pyhealth.datasets import EEGGCNNDataset
        >>> from pyhealth.tasks import EEGGCNNDiseaseDetection
        >>> dataset = EEGGCNNDataset(
        ...     root="/data/eeg-gcnn/",
        ... )
        >>> dataset.stats()
        >>> sample_dataset = dataset.set_task(EEGGCNNDiseaseDetection())
        >>> sample = sample_dataset[0]
        >>> print(sample["psd_features"].shape)  # (8, 6)
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        subset: Optional[str] = "both",
        **kwargs,
    ) -> None:
        if config_path is None:
            config_path = (
                Path(__file__).parent / "configs" / "eeg_gcnn.yaml"
            )

        self.root = root

        if subset == "tuab":
            tables = ["tuab"]
        elif subset == "lemon":
            tables = ["lemon"]
        elif subset == "both":
            tables = ["tuab", "lemon"]
        else:
            raise ValueError(
                "subset must be one of 'tuab', 'lemon', or 'both'"
            )

        self.prepare_metadata()

        root_path = Path(root)
        cache_dir = Path.home() / ".cache" / "pyhealth" / "eeg_gcnn"

        use_cache = False
        for table in tables:
            shared_csv = root_path / f"eeg_gcnn-{table}-pyhealth.csv"
            cache_csv = cache_dir / f"eeg_gcnn-{table}-pyhealth.csv"
            if not shared_csv.exists() and cache_csv.exists():
                use_cache = True
                break

        if use_cache:
            logger.info("Using cached metadata from %s", cache_dir)
            root = str(cache_dir)

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "eeg_gcnn",
            config_path=config_path,
            **kwargs,
        )

    def prepare_metadata(self) -> None:
        """Build and save metadata CSVs for TUAB normal subset and LEMON.

        Writes:
            - ``<root>/eeg_gcnn-tuab-pyhealth.csv``
            - ``<root>/eeg_gcnn-lemon-pyhealth.csv``

        TUAB filenames: ``<subject_id>_<record>_<tcp>.edf``
        LEMON filenames: ``sub-<id>.vhdr`` (BrainVision header)
        """
        root = Path(self.root)
        cache_dir = Path.home() / ".cache" / "pyhealth" / "eeg_gcnn"

        # --- TUAB normal subset ---
        shared_csv = root / "eeg_gcnn-tuab-pyhealth.csv"
        cache_csv = cache_dir / "eeg_gcnn-tuab-pyhealth.csv"

        if not shared_csv.exists() and not cache_csv.exists():
            tuab_rows = []
            for split in ("train", "eval"):
                normal_dir = root / split / "normal" / "01_tcp_ar"
                if not normal_dir.is_dir():
                    logger.debug(
                        "TUAB normal dir not found: %s", normal_dir
                    )
                    continue
                for edf in sorted(normal_dir.rglob("*.edf")):
                    parts = edf.stem.split("_")
                    patient_id = f"tuab_{parts[0]}"
                    record_id = parts[1] if len(parts) > 1 else "0"
                    tuab_rows.append(
                        {
                            "patient_id": patient_id,
                            "record_id": record_id,
                            "signal_file": str(edf),
                            "source": "tuab",
                            "label": 0,
                        }
                    )

            if tuab_rows:
                df = pd.DataFrame(tuab_rows)
                df.sort_values(
                    ["patient_id", "record_id"],
                    inplace=True,
                    na_position="last",
                )
                df.reset_index(drop=True, inplace=True)
                self._write_csv(df, shared_csv, cache_dir, "tuab")

        # --- LEMON healthy controls ---
        shared_csv = root / "eeg_gcnn-lemon-pyhealth.csv"
        cache_csv = cache_dir / "eeg_gcnn-lemon-pyhealth.csv"

        if not shared_csv.exists() and not cache_csv.exists():
            lemon_rows = []
            lemon_dir = root / "lemon"
            if lemon_dir.is_dir():
                for subject_dir in sorted(lemon_dir.iterdir()):
                    if not subject_dir.is_dir():
                        continue
                    for vhdr in sorted(subject_dir.glob("*.vhdr")):
                        patient_id = f"lemon_{subject_dir.name}"
                        record_id = vhdr.stem
                        lemon_rows.append(
                            {
                                "patient_id": patient_id,
                                "record_id": record_id,
                                "signal_file": str(vhdr),
                                "source": "lemon",
                                "label": 1,
                            }
                        )

            if lemon_rows:
                df = pd.DataFrame(lemon_rows)
                df.sort_values(
                    ["patient_id", "record_id"],
                    inplace=True,
                    na_position="last",
                )
                df.reset_index(drop=True, inplace=True)
                self._write_csv(df, shared_csv, cache_dir, "lemon")

    @staticmethod
    def _write_csv(
        df: "pd.DataFrame",
        shared_path: Path,
        cache_dir: Path,
        table_name: str,
    ) -> None:
        """Write CSV to shared location, falling back to cache."""
        try:
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(shared_path, index=False)
            logger.info("Wrote %s metadata to %s", table_name, shared_path)
        except (PermissionError, OSError):
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / shared_path.name
            df.to_csv(cache_path, index=False)
            logger.info(
                "Wrote %s metadata to cache: %s", table_name, cache_path
            )

    @property
    def default_task(self):
        """Returns the default task for the EEG-GCNN dataset.

        Returns:
            EEGGCNNDiseaseDetection: The default task instance.
        """
        from pyhealth.tasks import EEGGCNNDiseaseDetection

        return EEGGCNNDiseaseDetection()
