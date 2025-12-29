"""COSMIC dataset for PyHealth.

This module provides the COSMICDataset class for loading and processing
COSMIC (Catalogue Of Somatic Mutations In Cancer) data for machine learning tasks.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class COSMICDataset(BaseDataset):
    """COSMIC dataset for cancer somatic mutation analysis.

    COSMIC (Catalogue Of Somatic Mutations In Cancer) is the world's largest
    and most comprehensive resource for exploring the impact of somatic
    mutations in human cancer. This dataset enables mutation pathogenicity
    prediction and cancer gene analysis tasks.

    Dataset is available at:
    https://cancer.sanger.ac.uk/cosmic/download

    Note:
        COSMIC requires registration and license agreement for data access.

    Args:
        root: Root directory of the raw data containing the COSMIC files.
        tables: Optional list of additional tables to load beyond defaults.
        dataset_name: Optional name of the dataset. Defaults to "cosmic".
        config_path: Optional path to the configuration file. If not provided,
            uses the default config in the configs directory.

    Attributes:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import COSMICDataset
        >>> dataset = COSMICDataset(root="/path/to/cosmic")
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> print(samples[0])
    """

    def __init__(
        self,
        root: str,
        tables: List[str] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "cosmic.yaml"

        # Prepare standardized CSV if not exists
        pyhealth_csv = os.path.join(root, "cosmic-pyhealth.csv")
        if not os.path.exists(pyhealth_csv):
            logger.info("Preparing COSMIC metadata...")
            self.prepare_metadata(root)

        default_tables = ["mutations"]
        tables = default_tables + (tables or [])

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "cosmic",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Prepare metadata for the COSMIC dataset.

        Converts raw COSMIC TSV/CSV files to standardized CSV format.

        Args:
            root: Root directory containing the COSMIC files.
        """
        # Try to find the raw COSMIC file
        possible_files = [
            "CosmicMutantExportCensus.tsv",
            "CosmicMutantExportCensus.tsv.gz",
            "CosmicMutantExport.tsv",
            "CosmicMutantExport.tsv.gz",
            "cosmic_mutations.tsv",
            "cosmic_mutations.csv",
        ]

        raw_file = None
        for fname in possible_files:
            fpath = os.path.join(root, fname)
            if os.path.exists(fpath):
                raw_file = fpath
                break

        if raw_file is None:
            logger.warning(
                f"No raw COSMIC file found in {root}. "
                "Please download from https://cancer.sanger.ac.uk/cosmic/download "
                "and place CosmicMutantExportCensus.tsv in the root directory."
            )
            # Create empty placeholder
            pd.DataFrame(
                columns=[
                    "sample_id",
                    "gene_name",
                    "hgvsc",
                    "hgvsp",
                    "mutation_description",
                    "fathmm_prediction",
                    "primary_site",
                    "primary_histology",
                    "mutation_somatic_status",
                ]
            ).to_csv(os.path.join(root, "cosmic-pyhealth.csv"), index=False)
            return

        logger.info(f"Processing COSMIC file: {raw_file}")

        # Read the raw file
        sep = "\t" if ".tsv" in raw_file else ","
        if raw_file.endswith(".gz"):
            df = pd.read_csv(raw_file, sep=sep, compression="gzip", low_memory=False)
        else:
            df = pd.read_csv(raw_file, sep=sep, low_memory=False)

        # Standardize column names (COSMIC uses various naming conventions)
        column_mapping = {
            "ID_SAMPLE": "sample_id",
            "GENE_NAME": "gene_name",
            "Gene name": "gene_name",
            "HGVSC": "hgvsc",
            "HGVSP": "hgvsp",
            "MUTATION_DESCRIPTION": "mutation_description",
            "Mutation Description": "mutation_description",
            "FATHMM_PREDICTION": "fathmm_prediction",
            "FATHMM prediction": "fathmm_prediction",
            "PRIMARY_SITE": "primary_site",
            "Primary site": "primary_site",
            "PRIMARY_HISTOLOGY": "primary_histology",
            "Primary histology": "primary_histology",
            "MUTATION_SOMATIC_STATUS": "mutation_somatic_status",
            "Mutation somatic status": "mutation_somatic_status",
        }

        # Rename columns that exist
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)

        # Select columns that exist in our schema
        output_cols = [
            "sample_id",
            "gene_name",
            "hgvsc",
            "hgvsp",
            "mutation_description",
            "fathmm_prediction",
            "primary_site",
            "primary_histology",
            "mutation_somatic_status",
        ]
        available_cols = [c for c in output_cols if c in df.columns]

        # If sample_id doesn't exist, create from index
        if "sample_id" not in df.columns:
            df["sample_id"] = df.index.astype(str)
            available_cols = ["sample_id"] + [c for c in available_cols if c != "sample_id"]

        df_out = df[available_cols]

        # Save to standardized CSV
        output_path = os.path.join(root, "cosmic-pyhealth.csv")
        df_out.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_out)} mutations to {output_path}")

    @property
    def default_task(self):
        """Returns the default task for this dataset.

        Returns:
            MutationPathogenicityPrediction: The default prediction task.
        """
        from pyhealth.tasks import MutationPathogenicityPrediction

        return MutationPathogenicityPrediction()
