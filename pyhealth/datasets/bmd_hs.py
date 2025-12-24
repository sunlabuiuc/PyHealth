import logging
from pathlib import Path
from typing import Optional
import narwhals as pl

from pyhealth.tasks.base_task import BaseTask
from pyhealth.tasks.bmd_hs_disease_classification import BMDHSDiseaseClassification
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class BMDHSDataset(BaseDataset):
    """BUET Multi-disease Heart Sound (BMD-HS) Dataset
    Repository (current main branch) available at:
    https://github.com/sani002/BMD-HS-Dataset

    BMD-HS is a curated collection of phonocardiogram (PCG/heart-sound) recordings
    designed for automated cardiovascular disease research. It includes
    multi-label annotations for common valvular conditions: Aortic Stenosis (AS),
    Aortic Regurgitation (AR), Mitral Regurgitation (MR), Mitral Stenosis (MS),
    a Multi-Disease (MD) label for co-existing conditions, and Normal (N)—with
    accompanying patient-level metadata. The dataset also provides a training
    CSV mapping patient IDs to labels and up to eight 20-second recordings
    per patient captured at different auscultation positions.

    Citations:
    ----------
    If you use this dataset, please cite:
    Ali, S. N., Zahin, A., Shuvo, S. B., Nizam, N. B., Nuhash, S. I. S. K., Razin, S. S.,
    Sani, S. M. S., Rahman, F., Nizam, N. B., Azam, F. B., Hossen, R., Ohab, S., Noor, N., & Hasan, T. (2024).
    BUET Multi-disease Heart Sound Dataset: A Comprehensive Auscultation Dataset for Developing
    Computer-Aided Diagnostic Systems. arXiv:2409.00724. https://arxiv.org/abs/2409.00724

    Args:
        root: Root directory containing the repository files (e.g., the cloned or extracted repo).
        dataset_name: Optional dataset name, defaults to "bmd_hs".
        config_path: Optional configuration file name, defaults to "bmd_hs.yaml".

    Attributes:
        root: Root directory containing the dataset files.
        dataset_name: Name of the dataset.
        config_path: Path to configuration file.

    Expected Files & Structure (main branch):
    -----------------------------------------
    - train/                 # .wav audio files (20 s, ~4 kHz), up to 8 per patient/positions
    - train.csv              # labels and recording filenames per patient
        • patient_id
        • AS, AR, MR, MS     # 0 = absent, 1 = present
        • N                  # 0 = disease, 1 = normal (healthy indicator)
        • recording_1 … recording_8  # filenames for position-wise recordings
    - additional_metadata.csv
        • patient_id, Age, Gender (M/F), Smoker (0/1), Lives (U/F)

    Example:
        >>> from pyhealth.datasets import BMDHSDataset
        >>> dataset = BMDHSDataset(root=".../BMD-HS-Dataset/")
        >>> dataset.stats()

    Note:
        This loader assumes the repository's current layout (train/, train.csv,
        additional_metadata.csv) and multi-label schema as described above. Set
        `root` to the repository directory that includes these files and folders.
    """


    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        recordings_path: Optional[str] = None,
        **kwargs
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "bmd_hs.yaml"

        default_tables = ["diagnoses", "recordings", "metadata"]
        self.recordings_path = Path(recordings_path) if recordings_path else (Path(root) / "train")
        
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "bmd_hs",
            config_path=config_path,
            **kwargs
        )
        
    def preprocess_recordings(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Preprocess the recordings table by prepending the recordings_path to recording filenames."""
        import os
        recording_columns = [f"recording_{i}" for i in range(1, 9)]

        for col in recording_columns:
            if col in df.collect_schema().names():
                df = df.with_columns(
                    pl.concat_str([pl.lit(str(self.recordings_path)), pl.lit(os.sep), pl.col(col), pl.lit('.wav')]).alias(col)
                )

        return df
    
    @property
    def default_task(self) -> BMDHSDiseaseClassification:
        """Returns the default task for the BMD-HS dataset: BMDHSDiseaseClassification.
        
        Returns:
            BMDHSDiseaseClassification: The default task instance.
        """
        return BMDHSDiseaseClassification()