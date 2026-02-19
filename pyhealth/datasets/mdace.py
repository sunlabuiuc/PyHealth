"""
MDACE is the first publicly available code evidence dataset, which is built on a subset of the MIMIC-III clinical records.
The dataset – annotated by professional medical coders – consists of 302 Inpatient charts with 3,934 evidence spans and 52 Profee charts with 5,563 evidence spans.

The dataset can be found at
https://github.com/3mcloud/MDACE.git


@inproceedings{cheng-etal-2023-mdace,
    title = "{MDACE}: {MIMIC} Documents Annotated with Code Evidence",
    author = "Cheng, Hua  and
      Jafari, Rana  and
      Russell, April  and
      Klopfer, Russell  and
      Lu, Edmond  and
      Striner, Benjamin  and
      Gormley, Matthew",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.416/",
    doi = "10.18653/v1/2023.acl-long.416",
    pages = "7534--7550",
    abstract = "We introduce a dataset for evidence/rationale extraction on an extreme multi-label classification task over long medical documents. One such task is Computer-Assisted Coding (CAC) which has improved significantly in recent years, thanks to advances in machine learning technologies. Yet simply predicting a set of final codes for a patient encounter is insufficient as CAC systems are required to provide supporting textual evidence to justify the billing codes. A model able to produce accurate and reliable supporting evidence for each code would be a tremendous benefit. However, a human annotated code evidence corpus is extremely difficult to create because it requires specialized knowledge. In this paper, we introduce MDACE, the first publicly available code evidence dataset, which is built on a subset of the MIMIC-III clinical records. The dataset {--} annotated by professional medical coders {--} consists of 302 Inpatient charts with 3,934 evidence spans and 52 Profee charts with 5,563 evidence spans. We implemented several evidence extraction methods based on the EffectiveCAN model (Liu et al., 2021) to establish baseline performance on this dataset. MDACE can be used to evaluate code evidence extraction methods for CAC systems, as well as the accuracy and interpretability of deep learning models for multi-label classification. We believe that the release of MDACE will greatly improve the understanding and application of deep learning technologies for medical coding and document classification."
}


"""
import logging
import warnings
from typing import List, Optional, Dict

import polars as pl
from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


class MDACEDataset(BaseDataset):
    """
    Dataset class for handling MDACE (MIMIC Documents Annotated with Code Evidence) data.

    The MDACE dataset contains MIMIC-III clinical records annotated with code evidence
    by professional medical coders. It includes inpatient and profee charts.

    This class expects the following tables in the root directory:
        - mdace_notes.csv (or .parquet): Contains the clinical notes (discharged summaries).
        - mdace_inpatient_annotations.csv: Contains annotations for inpatient charts.
        - mdace_profee_annotations.csv: Contains annotations for profee charts.

    Paper: Cheng, Hua, et al. "MDACE: MIMIC Documents Annotated with Code Evidence."
           Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics. 2023.

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.
    """

    def __init__(
        self,
        root: str,
        tables: List[str] = None,
        dataset_name: Optional[str] = "mdace",
        config_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initializes the MDACEDataset with the specified parameters.

        Args:
            root (str): The root directory where the dataset is stored.
            tables (List[str], optional): A list of tables to include.
                                          Defaults to ["mdace_notes", "mdace_inpatient_annotations", "mdace_profee_annotations"].
            dataset_name (str, optional): The name of the dataset. Defaults to "mdace".
            config_path (str, optional): The path to the configuration file. If not provided, a default config is used.
            **kwargs: Additional keyword arguments to be passed to the BaseDataset class.
        """
        if tables is None:
            tables = [
                "mdace_notes",
                "mdace_inpatient_annotations",
                "mdace_profee_annotations",
            ]

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            **kwargs
        )

    def preprocess_mdace_notes(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Table-specific preprocess function for 'mdace_notes'.

        Standardizes column names and types for the notes table.

        Args:
            df (pl.LazyFrame): Input dataframe.

        Returns:
            pl.LazyFrame: Preprocessed dataframe.
        """
        # Ensure note_id is a string as per original script logic
        if "note_id" in df.columns:
            df = df.with_columns(pl.col("note_id").cast(pl.Utf8))
        return df

    def _preprocess_annotations(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Shared preprocessing logic for annotation tables.
        Renames columns to PyHealth standard and normalizes code types.
        """
        # Rename columns to standard PyHealth/Project convention
        # Mapping based on original mdace.py
        rename_map = {
            "NOTE_ID": "note_id",
            "start_index": "start",
            "end_index": "end",
            "TAG": "label",
        }
        
        # Only rename columns that exist
        existing_cols = df.columns
        actual_rename = {k: v for k, v in rename_map.items() if k in existing_cols}
        df = df.rename(actual_rename)

        # Cast note_id to string
        df = df.with_columns(pl.col("note_id").cast(pl.Utf8))

        # Normalize code_type values if the column exists
        if "code_type" in df.columns:
            df = df.with_columns(
                pl.col("code_type")
                .str.replace("ICD-9-CM", "icd9cm", literal=True)
                .str.replace("ICD-10-CM", "icd10cm", literal=True)
                .str.replace("ICD-10-PCS", "icd10pcs", literal=True)
                .str.replace("CPT", "cpt", literal=True)
                .str.replace("ICD-9-PCS", "icd9pcs", literal=True)
            )
        
        return df

    def preprocess_mdace_inpatient_annotations(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Table-specific preprocess function for 'mdace_inpatient_annotations'.

        Args:
            df (pl.LazyFrame): Input dataframe.

        Returns:
            pl.LazyFrame: Preprocessed dataframe.
        """
        return self._preprocess_annotations(df)

    def preprocess_mdace_profee_annotations(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Table-specific preprocess function for 'mdace_profee_annotations'.

        Args:
            df (pl.LazyFrame): Input dataframe.

        Returns:
            pl.LazyFrame: Preprocessed dataframe.
        """
        return self._preprocess_annotations(df)

if __name__ == "__main__":
    # Test case to verify the class structure
    # Note: This assumes dummy csv files exist in 'data/mdace_test' for testing purposes.
    # You would need to create these dummy files to run this block successfully.
    print("This module provides the MDACEDataset class.")