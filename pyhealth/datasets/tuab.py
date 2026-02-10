import os
import logging
import pandas as pd
from pathlib import Path

from typing import Optional
from .base_dataset import BaseDataset
from pyhealth.tasks import EEGAbnormalTUAB

logger = logging.getLogger(__name__)



class TUABDataset(BaseDataset):
    """Base EEG dataset for the TUH Abnormal EEG Corpus

    Dataset is available at https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

    The TUAB dataset (or Temple University Hospital EEG Abnormal Corpus) is a collection of EEG data acquired at the Temple University Hospital. 
    
    The dataset contains both normal and abnormal EEG readings.

    Files are named in the form aaaaamye_s001_t000.edf. This includes the subject identifier ("aaaaamye"), the session number ("s001") and a token number ("t000"). EEGs are split into a series of files starting with *t000.edf, *t001.edf, ...

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data. *You can choose to use the path to Cassette portion or the Telemetry portion.*
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "EEG_abnormal").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, record_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import TUABDataset
        >>> dataset = TUABDataset(
        ...         root="/srv/local/data/TUH/tuh_eeg_abnormal/v3.0.0/edf/",
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
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
            config_path = Path(__file__).parent / "configs" / "tuab.yaml"

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
        cache_dir = Path.home() / ".cache" / "pyhealth" / "tuab"
        
        # Check if CSVs exist in cache and not in shared location
        use_cache = False
        for table in tables:
            shared_csv = root_path / f"tuab-{table}-pyhealth.csv"
            cache_csv = cache_dir / f"tuab-{table}-pyhealth.csv"
            if not shared_csv.exists() and cache_csv.exists():
                use_cache = True
                break
        
        # Use cache directory as root if CSVs are there
        if use_cache:
            logger.info(f"Using cached metadata from {cache_dir}")
            root = str(cache_dir)
            
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "tuab",
            config_path=config_path,
            **kwargs
        )
        
        
    def prepare_metadata(self) -> None:
        """Build and save processed metadata CSVs for TUAB train/eval separately.

        This writes:
        - <root>/tuab-train-pyhealth.csv
        - <root>/tuab-eval-pyhealth.csv

        Train and eval filenames look like: aaaaalkt_s001_t000.edf
        - subject_id = aaaaalkt
        - session_id = s001
        - token_id  = t000
        
        We define record_id as session_id + token_id.

        The label is derived from the directory:
        - abnormal -> 1
        - normal   -> 0
        """
        root = Path(self.root)
        cache_dir = Path.home() / ".cache" / "pyhealth" / "tuab"

        for split in ("train", "eval"):
            # Check if metadata exists in either shared location or cache
            shared_csv = root / f"tuab-{split}-pyhealth.csv"
            cache_csv = cache_dir / f"tuab-{split}-pyhealth.csv"
            if shared_csv.exists() or cache_csv.exists():
                continue
            
            rows: list[dict] = []
            for label in ("normal", "abnormal"):
                label = 1 if label == "abnormal" else 0
                edf_dir = root / split / label/ "01_tcp_ar"
                
                if not edf_dir.is_dir():
                    logger.warning("EDF directory not found: %s", edf_dir)
                    continue
                
                for edf_path in sorted(edf_dir.glob("*.edf")):
                    stem = edf_path.stem
                    parts = stem.split("_")
                    
                    if len(parts) != 3:
                        logger.warning("Invalid filename format: %s", edf_path)
                        continue
                    
                    subject_id = parts[0]
                    session_id = parts[1]
                    token_id = parts[2]
                    
                    record_id = f'{session_id}_{token_id}'
                    
                    rows.append(
                        {
                            "patient_id": subject_id,
                            "record_id": record_id,
                            "signal_file": str(edf_path),
                            "label": label,
                        }
                    )
            
            if not rows:
                continue
            # Setup cache directory as fallback for metadata CSVs
            cache_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(rows)
            
            df.sort_values(
                ["patient_id", "session_id", "token_id"],
                inplace=True,
                na_position="last",
            )
            df.reset_index(drop=True, inplace=True)
            
            # Try shared location first, fall back to cache if no write permission
            csv_shared = root / f"tuab-{split}-pyhealth.csv"
            csv_cache = cache_dir / f"tuab-{split}-pyhealth.csv"
            
            try:
                csv_shared.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_shared, index=False)
                logger.info(f"Wrote {split} metadata to {csv_shared}")
            except (PermissionError, OSError):
                df.to_csv(csv_cache, index=False)
                logger.info(f"Wrote {split} metadata to cache: {csv_cache}")
                
                
            
    
    @property
    def default_task(self) -> EEGAbnormalTUAB:
        """Returns the default task for the TUAB dataset: EEGAbnormalTUAB.
        
        Returns:
            EEGAbnormalTUAB: The default task instance.
        """
        return EEGAbnormalTUAB()
        

