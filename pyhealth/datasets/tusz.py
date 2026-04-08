import os
import logging
import pandas as pd
from pathlib import Path

from typing import Optional
from .base_dataset import BaseDataset
from pyhealth.tasks import TUSZTask

logger = logging.getLogger(__name__)

class TUSZDataset(BaseDataset):
    """Base EEG dataset for the TUH Seizure Corpus (TUSZ)

    Dataset is available at ......
    
    This corpus contains EEG recordings with seizure annotations.

    Args:
        root: root directory of the raw data.
        dataset_name: optional name of the dataset.
        config_path: optional config file name, defaults to "tusz.yaml".
        subset: which split to use: "train", "eval", or "both".
        **kwargs: other arguments passed to BaseDataset.

    Examples:
        >>> from pyhealth.datasets import TUSZDataset
        >>> from pyhealth.tasks import TUSZTask
        >>> dataset = TUSZDataset(root="/srv/local/data/TUH/tuh_eeg_seizure/v1.5.2/")
        >>> dataset.stats()
        >>> sample_dataset = dataset.set_task(TUSZTask())
        >>> sample = sample_dataset[0]
        >>> print(sample['signal'].shape)
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        subset: Optional[str] = 'both',
        **kwargs
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "tusz.yaml"

        self.root = root

        if subset in ['train', 'eval']:
            logger.info(f"Using subset: {subset}")
            tables = [subset]
        elif subset == 'both':
            logger.info("Using both train and eval subsets")
            tables = ["train", "eval"]
        else:
            raise ValueError("subset must be one of 'train', 'eval', or 'both'")

        self.prepare_metadata()

        # Determine where the CSVs are located (shared directory or cache)
        root_path = Path(root)
        cache_dir = Path.home() / ".cache" / "pyhealth" / "tusz"

        use_cache = False
        for table in tables:
            shared_csv = root_path / f"tusz-{table}-pyhealth.csv"
            cache_csv = cache_dir / f"tusz-{table}-pyhealth.csv"
            if not shared_csv.exists() and cache_csv.exists():
                use_cache = True
                break

        if use_cache:
            logger.info(f"Using cached metadata from {cache_dir}")
            root = str(cache_dir)

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "tusz",
            config_path=config_path,
            **kwargs
        )

    def prepare_metadata(self) -> None:
        """Build and save processed metadata CSVs for TUSZ train/eval separately."""

        root = Path(self.root)
        cache_dir = Path.home() / ".cache" / "pyhealth" / "tusz"

        train_rows: list[dict] = []
        eval_rows: list[dict] = []

        for split in ("train", "eval"):
            shared_csv = root / f"tusz-{split}-pyhealth.csv"
            cache_csv = cache_dir / f"tusz-{split}-pyhealth.csv"
            if shared_csv.exists() or cache_csv.exists():
                continue

            split_dir = root / split
            if not split_dir.is_dir():
                logger.warning("Split directory not found: %s", split_dir)
                continue

            for subject_dir in split_dir.iterdir():
                if not subject_dir.is_dir() or subject_dir.name.startswith("."):
                    continue

                for edf_path in subject_dir.glob("*.edf"):
                    stem = edf_path.stem

                    if split == "train":
                        parts = stem.split("_")
                        record_id = parts[-1]

                        train_rows.append(
                            {
                                "patient_id": subject_dir.name,
                                "record_id": record_id,
                                "signal_file": str(edf_path),
                            }
                        )

                    else:  # eval
                        parts = stem.split("_")
                        segment_id = parts[-1]  # adjust as needed
                        eval_rows.append(
                            {
                                "patient_id": subject_dir.name,
                                "segment_id": segment_id,
                                "signal_file": str(edf_path),
                            }
                        )

        cache_dir.mkdir(parents=True, exist_ok=True)

        if train_rows:
            train_df = pd.DataFrame(train_rows)
            train_df.sort_values(["patient_id", "record_id"], inplace=True, na_position="last")
            train_df.reset_index(drop=True, inplace=True)

            train_csv_shared = root / "tusz-train-pyhealth.csv"
            train_csv_cache = cache_dir / "tusz-train-pyhealth.csv"
            try:
                train_csv_shared.parent.mkdir(parents=True, exist_ok=True)
                train_df.to_csv(train_csv_shared, index=False)
                logger.info(f"Wrote train metadata to {train_csv_shared}")
            except (PermissionError, OSError):
                train_df.to_csv(train_csv_cache, index=False)
                logger.info(f"Wrote train metadata to cache: {train_csv_cache}")

        if eval_rows:
            eval_df = pd.DataFrame(eval_rows)
            eval_df.sort_values(["patient_id", "segment_id"], inplace=True, na_position="last")
            eval_df.reset_index(drop=True, inplace=True)

            eval_csv_shared = root / "tusz-eval-pyhealth.csv"
            eval_csv_cache = cache_dir / "tusz-eval-pyhealth.csv"
            try:
                eval_csv_shared.parent.mkdir(parents=True, exist_ok=True)
                eval_df.to_csv(eval_csv_shared, index=False)
                logger.info(f"Wrote eval metadata to {eval_csv_shared}")
            except (PermissionError, OSError):
                eval_df.to_csv(eval_csv_cache, index=False)
                logger.info(f"Wrote eval metadata to cache: {eval_csv_cache}")

    @property
    def default_task(self) -> TUSZTask:
        """Returns the default task for TUSZ dataset."""
        return TUSZTask()