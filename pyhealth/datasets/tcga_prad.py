"""TCGA-PRAD dataset for PyHealth.

This module provides the TCGAPRADDataset class for loading and processing
TCGA Prostate Adenocarcinoma (PRAD) data for machine learning tasks.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class TCGAPRADDataset(BaseDataset):
    """TCGA Prostate Adenocarcinoma (PRAD) dataset.

    The Cancer Genome Atlas (TCGA) PRAD dataset contains multi-omics data
    for prostate adenocarcinoma patients, including somatic mutations,
    clinical data, and survival outcomes. This dataset enables cancer
    survival prediction and mutation analysis tasks.

    Dataset is available at:
    https://portal.gdc.cancer.gov/projects/TCGA-PRAD

    Args:
        root: Root directory of the raw data containing the TCGA-PRAD files.
        tables: Optional list of additional tables to load beyond defaults.
        dataset_name: Optional name of the dataset. Defaults to "tcga_prad".
        config_path: Optional path to the configuration file. If not provided,
            uses the default config in the configs directory.

    Attributes:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import TCGAPRADDataset
        >>> dataset = TCGAPRADDataset(root="/path/to/tcga_prad")
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
            config_path = Path(__file__).parent / "configs" / "tcga_prad.yaml"

        # Prepare standardized CSVs if not exists
        mutations_csv = os.path.join(root, "tcga_prad_mutations-pyhealth.csv")
        clinical_csv = os.path.join(root, "tcga_prad_clinical-pyhealth.csv")

        if not os.path.exists(mutations_csv) or not os.path.exists(clinical_csv):
            logger.info("Preparing TCGA-PRAD metadata...")
            self.prepare_metadata(root)

        default_tables = ["mutations", "clinical"]
        tables = default_tables + (tables or [])

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "tcga_prad",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Prepare metadata for the TCGA-PRAD dataset.

        Converts raw TCGA MAF and clinical files to standardized CSV format.

        Args:
            root: Root directory containing the TCGA-PRAD files.
        """
        # Process mutations file
        TCGAPRADDataset._prepare_mutations(root)
        # Process clinical file
        TCGAPRADDataset._prepare_clinical(root)

    @staticmethod
    def _prepare_mutations(root: str) -> None:
        """Prepare mutations data from MAF file."""
        # Try to find the raw mutations file
        possible_files = [
            "PRAD_mutations.csv",
            "TCGA.PRAD.mutect.maf",
            "TCGA.PRAD.mutect.maf.gz",
            "PRAD.maf",
            "PRAD.maf.gz",
            "mutations.maf",
        ]

        raw_file = None
        for fname in possible_files:
            fpath = os.path.join(root, fname)
            if os.path.exists(fpath):
                raw_file = fpath
                break

        output_path = os.path.join(root, "tcga_prad_mutations-pyhealth.csv")

        if raw_file is None:
            logger.warning(
                f"No raw TCGA-PRAD mutations file found in {root}. "
                "Please download from GDC portal or use TCGAmutations R package."
            )
            # Create empty placeholder
            pd.DataFrame(
                columns=[
                    "patient_id",
                    "hugo_symbol",
                    "variant_classification",
                    "variant_type",
                    "hgvsc",
                    "hgvsp",
                    "tumor_sample_barcode",
                ]
            ).to_csv(output_path, index=False)
            return

        logger.info(f"Processing TCGA-PRAD mutations file: {raw_file}")

        # Read the raw file
        if raw_file.endswith(".gz"):
            df = pd.read_csv(
                raw_file, sep="\t", compression="gzip", comment="#", low_memory=False
            )
        elif raw_file.endswith(".maf"):
            df = pd.read_csv(raw_file, sep="\t", comment="#", low_memory=False)
        else:
            df = pd.read_csv(raw_file, low_memory=False)

        # Standardize column names
        column_mapping = {
            "Hugo_Symbol": "hugo_symbol",
            "Variant_Classification": "variant_classification",
            "Variant_Type": "variant_type",
            "HGVSc": "hgvsc",
            "HGVSp_Short": "hgvsp",
            "HGVSp": "hgvsp",
            "Tumor_Sample_Barcode": "tumor_sample_barcode",
        }

        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)

        # Extract patient_id from tumor_sample_barcode (first 12 characters)
        if "tumor_sample_barcode" in df.columns:
            df["patient_id"] = df["tumor_sample_barcode"].str[:12]
        else:
            df["patient_id"] = df.index.astype(str)

        # Select output columns
        output_cols = [
            "patient_id",
            "hugo_symbol",
            "variant_classification",
            "variant_type",
            "hgvsc",
            "hgvsp",
            "tumor_sample_barcode",
        ]
        available_cols = [c for c in output_cols if c in df.columns]
        df_out = df[available_cols]

        df_out.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_out)} mutations to {output_path}")

    @staticmethod
    def _prepare_clinical(root: str) -> None:
        """Prepare clinical data file."""
        # Try to find the raw clinical file
        possible_files = [
            "PRAD_clinical.csv",
            "clinical.tsv",
            "clinical.csv",
            "nationwidechildrens.org_clinical_patient_prad.txt",
        ]

        raw_file = None
        for fname in possible_files:
            fpath = os.path.join(root, fname)
            if os.path.exists(fpath):
                raw_file = fpath
                break

        output_path = os.path.join(root, "tcga_prad_clinical-pyhealth.csv")

        if raw_file is None:
            logger.warning(
                f"No raw TCGA-PRAD clinical file found in {root}. "
                "Please download from GDC portal."
            )
            # Create empty placeholder
            pd.DataFrame(
                columns=[
                    "patient_id",
                    "age_at_diagnosis",
                    "gleason_score",
                    "vital_status",
                    "days_to_death",
                    "tumor_stage",
                ]
            ).to_csv(output_path, index=False)
            return

        logger.info(f"Processing TCGA-PRAD clinical file: {raw_file}")

        # Read the raw file
        sep = "\t" if raw_file.endswith(".tsv") or raw_file.endswith(".txt") else ","
        df = pd.read_csv(raw_file, sep=sep, low_memory=False)

        # Standardize column names (TCGA uses various naming conventions)
        column_mapping = {
            "submitter_id": "patient_id",
            "bcr_patient_barcode": "patient_id",
            "case_id": "patient_id",
            "age_at_diagnosis": "age_at_diagnosis",
            "age_at_initial_pathologic_diagnosis": "age_at_diagnosis",
            "gleason_score": "gleason_score",
            "primary_gleason_grade": "gleason_score",
            "vital_status": "vital_status",
            "days_to_death": "days_to_death",
            "tumor_stage": "tumor_stage",
            "ajcc_pathologic_stage": "tumor_stage",
            "pathologic_stage": "tumor_stage",
        }

        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)

        # If patient_id doesn't exist, create from index
        if "patient_id" not in df.columns:
            df["patient_id"] = df.index.astype(str)

        # Select output columns
        output_cols = [
            "patient_id",
            "age_at_diagnosis",
            "gleason_score",
            "vital_status",
            "days_to_death",
            "tumor_stage",
        ]
        available_cols = [c for c in output_cols if c in df.columns]
        df_out = df[available_cols].drop_duplicates(subset=["patient_id"])

        df_out.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_out)} clinical records to {output_path}")

    @property
    def default_task(self):
        """Returns the default task for this dataset.

        Returns:
            CancerSurvivalPrediction: The default prediction task.
        """
        from pyhealth.tasks import CancerSurvivalPrediction

        return CancerSurvivalPrediction()
