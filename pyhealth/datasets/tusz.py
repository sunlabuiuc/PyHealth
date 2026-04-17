"""
PyHealth task for extracting features with STFT and Frequency Bands using the Temple University Hospital (TUH) EEG Seizure Corpus (TUSZ) dataset V2.0.5.

Dataset link:
    https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml

Dataset paper:
    Vinit Shah, Eva von Weltin, Silvia Lopez, et al., “The Temple University Hospital Seizure Detection Corpus,” arXiv preprint arXiv:1801.08085, 2018. Available: https://arxiv.org/abs/1801.08085

Dataset paper link:
    https://arxiv.org/abs/1801.08085

Author:
    Fernando Kenji Sakabe (fks@illinois.edu), 
    Jesica Hirsch (jesicah2@illinois.edu), 
    Jung-Jung Hsieh (jhsieh8@illinois.edu)
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Optional
from pyhealth.tasks import TUSZTask

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class TUSZDataset(BaseDataset):
    """Base EEG dataset for the TUH Seizure Corpus (TUSZ)

    Dataset is available at https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml.
    
    This corpus contains EEG recordings with seizure annotations (start time, stop, channel and seizure type).

    File Structure:
        edf/
        ├── train/                             # train set
        ├── dev/                               # development set (test/validation)
        ├── eval/                              # evaluation set (test/validation)
            └── <subject_id>/                  # e.g., aaaaaajy
                └── sXXX_YYYY/                 # session ID + year
                        └── <montage_type>/    # e.g., 02_tcp_le
                            ├── *.edf          # raw EEG signal
                            ├── *.csv          # annotations (events)
                            ├── *.csv_bi       # binary labels
                            └── *.tse / *.lbl  # (sometimes present)

    Args:
        root        : root directory of the raw data.
        dataset_name: optional name of the dataset.
        config_path : optional config file name, defaults to "tusz.yaml".
        subset      : which split to use: "train", "eval", or "both".
        use_cache   : optional boolean to determine whether to use cached results or not
        **kwargs    : other arguments passed to BaseDataset.

    Attributes:
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, record_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id/session_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import TUSZDataset
        >>> from pyhealth.tasks import TUSZTask
        >>> dataset = TUSZDataset(root = 'tuh_eeg_v2.0.5', subset = 'train')
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        subset: Optional[str] = 'train',
        use_cache: Optional[bool] = True,
        **kwargs
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "tusz.yaml"

        self.root = root
        self.root_path = Path(root)
        self.dataset_name = dataset_name or "tusz"
        self.cache_dir = Path.home() / ".cache" / "pyhealth" / self.dataset_name
        tables = self.__set_tables(subset)
        self.final_tables = tables

        self.use_cache = use_cache and self.__use_cache()
        if not self.use_cache:
            self.prepare_metadata()

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=self.dataset_name,
            config_path=config_path,
            **kwargs
        )
    
    def prepare_metadata(self) -> None:
        """Build and save processed metadata CSVs for TUSZ train/eval/dev separately."""
        
        for table in self.final_tables:
            self.__create_csv(table)

    @property
    def default_task(self) -> TUSZTask:
        """Returns the default task for TUSZ dataset."""
        return TUSZTask()

    def __set_tables(self, subset):
        """Returns tables from the subset if valid."""
        if subset in ['train', 'eval', 'dev']:
            return [ subset ]
        if ',' in subset:
            return subset.split(',')
        if subset == 'all':
            return ['train', 'eval', 'dev']
        raise ValueError("subset must be one of None, 'train', 'dev', 'eval', or 'all'")

    def __use_cache(self):
        """Returns whether or not to use cached results."""
        for table in self.final_tables:
            cache_csv = self.__get_cache_csv_name(table)
            if not cache_csv.exists():
                return False
        return True

    def __get_data_csv_name(self, data_type):
        """Returns where to store current result."""
        return self.root_path / f"{self.dataset_name}-{data_type}-pyhealth.csv"

    def __get_cache_csv_name(self, data_type):
        """Returns where to store cached results."""
        return self.cache_dir / f"{self.dataset_name}-{data_type}-pyhealth.csv"

    def __create_csv(self, data_type):
        """Saves patient records into csvs."""
        shared_csv = self.__get_data_csv_name(data_type)
        cache_csv = self.__get_cache_csv_name(data_type)

        if self.use_cache and cache_csv.exists():
            return

        split_dir = self.root_path / data_type if data_type else self.root_path
        if not split_dir.is_dir():
            logger.warning("Split directory not found: %s", split_dir)
            return

        data_rows = [
            {
                "patient_id": subject_dir.name,
                "record_id": "_".join(edf_path.stem.split("_")[1:]),
                "signal_file": str(edf_path),
            }
            for subject_dir in split_dir.iterdir() 
            if subject_dir.is_dir() and not subject_dir.name.startswith(".")
            for edf_path in subject_dir.rglob("*.edf")
        ]

        if not data_rows:
            logger.warning("No data rows: %s", split_dir)
            return

        df = pd.DataFrame(data_rows)
        df.sort_values(["patient_id", "record_id"], inplace=True, na_position="last")
        df.reset_index(drop=True, inplace=True)

        try:
            shared_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(shared_csv, index=False)
            logger.info(f"Wrote train metadata to {shared_csv}")
        except (PermissionError, OSError):
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_csv, index=False)
            logger.info(f"Wrote train metadata to cache: {cache_csv}")
