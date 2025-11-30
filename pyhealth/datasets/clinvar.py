"""ClinVar dataset for PyHealth.

This module provides the ClinVarDataset class for loading and processing
ClinVar variant data for machine learning tasks.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class ClinVarDataset(BaseDataset):
    """ClinVar dataset for variant classification.

    ClinVar is a freely accessible, public archive of reports of the relationships
    among human variations and phenotypes, with supporting evidence. This dataset
    enables variant pathogenicity prediction tasks.

    Dataset is available at:
    https://ftp.ncbi.nlm.nih.gov/pub/clinvar/

    Args:
        root: Root directory of the raw data containing the ClinVar files.
        tables: Optional list of additional tables to load beyond defaults.
        dataset_name: Optional name of the dataset. Defaults to "clinvar".
        config_path: Optional path to the configuration file. If not provided,
            uses the default config in the configs directory.

    Attributes:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import ClinVarDataset
        >>> dataset = ClinVarDataset(root="/path/to/clinvar")
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
            config_path = Path(__file__).parent / "configs" / "clinvar.yaml"

        # Prepare standardized CSV if not exists
        pyhealth_csv = os.path.join(root, "clinvar-pyhealth.csv")
        if not os.path.exists(pyhealth_csv):
            logger.info("Preparing ClinVar metadata...")
            self.prepare_metadata(root)

        default_tables = ["variants"]
        tables = default_tables + (tables or [])

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "clinvar",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Prepare metadata for the ClinVar dataset.

        Converts raw ClinVar variant_summary.txt to standardized CSV format.

        Args:
            root: Root directory containing the ClinVar files.
        """
        # Try to find the raw ClinVar file
        possible_files = [
            "variant_summary.txt",
            "variant_summary.txt.gz",
            "clinvar_variant_summary.txt",
            "clinvar.vcf",
        ]

        raw_file = None
        for fname in possible_files:
            fpath = os.path.join(root, fname)
            if os.path.exists(fpath):
                raw_file = fpath
                break

        if raw_file is None:
            logger.warning(
                f"No raw ClinVar file found in {root}. "
                "Please download from https://ftp.ncbi.nlm.nih.gov/pub/clinvar/ "
                "and place variant_summary.txt in the root directory."
            )
            # Create empty placeholder
            pd.DataFrame(
                columns=[
                    "gene_symbol",
                    "clinical_significance",
                    "review_status",
                    "chromosome",
                    "position",
                    "reference_allele",
                    "alternate_allele",
                    "variant_type",
                    "assembly",
                ]
            ).to_csv(os.path.join(root, "clinvar-pyhealth.csv"), index=False)
            return

        logger.info(f"Processing ClinVar file: {raw_file}")

        # Read the raw file
        if raw_file.endswith(".gz"):
            df = pd.read_csv(raw_file, sep="\t", compression="gzip", low_memory=False)
        else:
            df = pd.read_csv(raw_file, sep="\t", low_memory=False)

        # Standardize column names
        column_mapping = {
            "GeneSymbol": "gene_symbol",
            "ClinicalSignificance": "clinical_significance",
            "ReviewStatus": "review_status",
            "Chromosome": "chromosome",
            "PositionVCF": "position",
            "ReferenceAlleleVCF": "reference_allele",
            "AlternateAlleleVCF": "alternate_allele",
            "Type": "variant_type",
            "Assembly": "assembly",
        }

        # Select and rename columns that exist
        available_cols = [c for c in column_mapping.keys() if c in df.columns]
        df_out = df[available_cols].rename(
            columns={k: v for k, v in column_mapping.items() if k in available_cols}
        )

        # Filter for GRCh38 assembly if assembly column exists
        if "assembly" in df_out.columns:
            df_out = df_out[df_out["assembly"] == "GRCh38"]

        # Save to standardized CSV
        output_path = os.path.join(root, "clinvar-pyhealth.csv")
        df_out.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_out)} variants to {output_path}")

    @property
    def default_task(self):
        """Returns the default task for this dataset.

        Returns:
            VariantClassificationClinVar: The default classification task.
        """
        from pyhealth.tasks import VariantClassificationClinVar

        return VariantClassificationClinVar()
