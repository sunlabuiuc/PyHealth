import os
import logging
import pandas as pd
from pathlib import Path

from typing import Optional
from .base_dataset import BaseDataset
from pyhealth.tasks import EEGEventsTUEV

logger = logging.getLogger(__name__)

class TUEVDataset(BaseDataset):
    """Base EEG dataset for the TUH EEG Events Corpus

    Dataset is available at https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

    This corpus is a subset of TUEG that contains annotations of EEG segments as one of six classes: (1) spike and sharp wave (SPSW), (2) generalized periodic epileptiform discharges (GPED), (3) periodic lateralized epileptiform discharges (PLED), (4) eye movement (EYEM), (5) artifact (ARTF) and (6) background (BCKG).

    Files are named in the form of bckg_032_a_.edf in the eval partition:
        bckg: this file contains background annotations.
		032: a reference to the eval index	
		a_.edf: EEG files are split into a series of files starting with a_.edf, a_1.ef, ... These represent pruned EEGs, so the original EEG is split into these segments, and uninteresting parts of the original recording were deleted.
    or in the form of 00002275_00000001.edf in the train partition:
        00002275: a reference to the train index. 
		0000001: indicating that this is the first file associated with this patient.

    Args:
        root: root directory of the raw data. *You can choose to use the path to Cassette portion or the Telemetry portion.*
        dataset_name: name of the dataset.
        config_path: Optional configuration file name, defaults to "tuev.yaml".
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "EEG_events").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, record_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import TUEVDataset
        >>> from pyhealth.tasks import EEGEventsTUEV
        >>> dataset = TUEVDataset(
        ...         root="/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf/",
        ...     )
        >>> dataset.stats()
        >>> sample_dataset = dataset.set_task(EEGEventsTUEV())
        >>> sample = sample_dataset[0]
        >>> print(sample['signal'].shape)  # (16, 1280)

        For a complete example, see `examples/conformal_eeg/tuev_eeg_quickstart.ipynb`.
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
            from pathlib import Path
            config_path = Path(__file__).parent / "configs" / "tuev.yaml"

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
            
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "tuev",
            config_path=config_path,
            **kwargs
        )
        
    def prepare_metadata(self) -> None:
        """Build and save processed metadata CSVs for TUEV train/eval separately.

        This writes:
        - <root>/tuev-train-pyhealth.csv
        - <root>/tuev-eval-pyhealth.csv

        Train filenames look like: 00002275_00000001.edf
        - subject_id = 00002275
        - record_id  = 00000001

        Eval filenames look like: bckg_032_a_.edf
        - label_kind = bckg (or spsw/gped/pled/eyem/artf depending on file)
        - eval_index = 032
        - segment_id = a_ / a_1 / ...
        """
        root = Path(self.root)

        train_rows: list[dict] = []
        eval_rows: list[dict] = []

        for split in ("train", "eval"):
            if os.path.exists(root / f"tuev-{split}-pyhealth.csv"):
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

                    else:
                        parts = stem.split("_")
                        label = parts[0]
                        segment_id = "_".join(parts[2:])

                        eval_rows.append(
                            {
                                "patient_id": subject_dir.name,
                                "label": label,
                                "segment_id": segment_id,
                                "signal_file": str(edf_path),
                            }
                        )

        root.mkdir(parents=True, exist_ok=True)

        # Write train metadata
        if train_rows:
            train_df = pd.DataFrame(train_rows)
            train_df.sort_values(
                ["patient_id", "record_id"], inplace=True, na_position="last"
            )
            train_df.reset_index(drop=True, inplace=True)
            train_csv = root / "tuev-train-pyhealth.csv"
            train_df.to_csv(train_csv, index=False)


        # Write eval metadata
        if eval_rows:
            eval_df = pd.DataFrame(eval_rows)
            eval_df.sort_values(
                ["patient_id", "segment_id", "label"],
                inplace=True,
                na_position="last",
            )
            eval_df.reset_index(drop=True, inplace=True)
            eval_csv = root / "tuev-eval-pyhealth.csv"
            eval_df.to_csv(eval_csv, index=False)
    
    @property
    def default_task(self) -> EEGEventsTUEV:
        """Returns the default task for the BMD-HS dataset: BMDHSDiseaseClassification.
        
        Returns:
            BMDHSDiseaseClassification: The default task instance.
        """
        return EEGEventsTUEV()