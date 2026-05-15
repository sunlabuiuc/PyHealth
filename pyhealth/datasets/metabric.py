"""METABRIC dataset for PyHealth.

METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) is a
landmark breast cancer study combining clinical and genomic data for ~2,000
patients with long-term follow-up.

The dataset is publicly available from:
  - cBioPortal: https://www.cbioportal.org/study/summary?id=brca_metabric
    Download "All clinical data" (data_clinical_patient.txt) and optionally
    "CNA data" or "mRNA expression" tables.
  - Kaggle (pre-processed): search "METABRIC breast cancer clinical"
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class METABRICDataset(BaseDataset):
    """METABRIC breast cancer clinical dataset.

    Each patient has a single clinical record with demographics, treatment
    indicators, tumour characteristics, and two survival endpoints:

    - **Overall Survival (OS)**: time ``OS_MONTHS`` and status ``OS_STATUS``
      (0 = living, 1 = died from cancer or unknown cause).
    - **Relapse-Free Survival (RFS)**: time ``RFS_MONTHS`` and status
      ``RFS_STATUS`` (0 = no relapse, 1 = relapse or death).

    Args:
        root: Directory containing the processed ``metabric_clinical.csv``
            (or the raw cBioPortal ``data_clinical_patient.txt``).
        tables: Additional tables to load beyond the default ``["metabric"]``.
        dataset_name: Optional dataset name; defaults to ``"metabric"``.
        config_path: Optional path to a YAML config; defaults to the bundled
            ``configs/metabric.yaml``.
        **kwargs: Passed through to :class:`~pyhealth.datasets.BaseDataset`.

    Examples:
        >>> from pyhealth.datasets import METABRICDataset
        >>> dataset = METABRICDataset(root="/path/to/metabric")
        >>> dataset.stats()
        >>> samples = dataset.set_task(task)
    """

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "metabric.yaml"

        processed_csv = os.path.join(root, "metabric_clinical.csv")
        if not os.path.exists(processed_csv):
            logger.info(
                "metabric_clinical.csv not found — attempting to prepare from raw data."
            )
            self.prepare_metadata(root)

        default_tables = ["metabric"]
        tables = default_tables + (tables or [])

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "metabric",
            config_path=str(config_path),
            **kwargs,
        )

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Convert raw cBioPortal download to the processed CSV.

        Looks for ``data_clinical_patient.txt`` (tab-separated, with comment
        header rows) inside ``root`` and writes
        ``metabric_clinical.csv``.

        Args:
            root: Directory to search for the raw file and write output.
        """
        raw_candidates = [
            "data_clinical_patient.txt",
            "METABRIC_RNA_Mutation.csv",
            "metabric.csv",
        ]
        raw_file: Optional[str] = None
        for fname in raw_candidates:
            candidate = os.path.join(root, fname)
            if os.path.exists(candidate):
                raw_file = candidate
                break

        output_path = os.path.join(root, "metabric_clinical.csv")

        if raw_file is None:
            logger.warning(
                f"No raw METABRIC file found in {root}. "
                "Please download 'data_clinical_patient.txt' from "
                "https://www.cbioportal.org/study/summary?id=brca_metabric "
                "and place it in the root directory."
            )
            # Write an empty placeholder so BaseDataset doesn't crash
            pd.DataFrame(
                columns=[
                    "PATIENT_ID",
                    "AGE_AT_DIAGNOSIS",
                    "OS_MONTHS",
                    "OS_STATUS",
                    "RFS_MONTHS",
                    "RFS_STATUS",
                    "INFERRED_MENOPAUSAL_STATE",
                    "TUMOR_SIZE",
                    "TUMOR_STAGE",
                    "NPI",
                    "CELLULARITY",
                    "CHEMOTHERAPY",
                    "ER_IHC",
                    "HER2_SNP6",
                    "HORMONE_THERAPY",
                    "INTCLUST",
                    "ONCOTREE_CODE",
                    "RADIO_THERAPY",
                    "THREEGENE",
                    "GRADE",
                    "TYPE_OF_BREAST_SURGERY",
                    "PR_STATUS",
                    "HER2_STATUS",
                ]
            ).to_csv(output_path, index=False)
            return

        logger.info(f"Processing METABRIC raw file: {raw_file}")

        if raw_file.endswith(".txt"):
            # cBioPortal format: skip lines starting with '#', tab-separated
            df = pd.read_csv(raw_file, sep="\t", comment="#", low_memory=False)
        else:
            df = pd.read_csv(raw_file, low_memory=False)

        # Normalise column names to upper-case and replace spaces/hyphens
        df.columns = (
            df.columns.str.upper()
            .str.strip()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )

        # Common alternative column name mappings
        rename = {
            "PATIENT_IDENTIFIER": "PATIENT_ID",
            "OVERALL_SURVIVAL_STATUS": "OS_STATUS",
            "OVERALL_SURVIVAL_MONTHS": "OS_MONTHS",
            "RELAPSE_FREE_STATUS": "RFS_STATUS",
            "RELAPSE_FREE_STATUS_MONTHS": "RFS_MONTHS",
            "AGE_AT_INITIAL_PATHOLOGIC_DIAGNOSIS": "AGE_AT_DIAGNOSIS",
            "INFERRED_MENOPAUSAL_STATE": "INFERRED_MENOPAUSAL_STATE",
            "NOTTINGHAM_PROGNOSTIC_INDEX": "NPI",
            "3_GENE_CLASSIFIER_SUBTYPE": "THREEGENE",
            "TYPE_OF_BREAST_SURGERY": "TYPE_OF_BREAST_SURGERY",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Parse OS_STATUS: cBioPortal stores "0:LIVING" / "1:DECEASED"
        if "OS_STATUS" in df.columns:
            df["OS_STATUS"] = (
                df["OS_STATUS"]
                .astype(str)
                .str.extract(r"^(\d+)", expand=False)
                .astype(float)
            )
        if "RFS_STATUS" in df.columns:
            df["RFS_STATUS"] = (
                df["RFS_STATUS"]
                .astype(str)
                .str.extract(r"^(\d+)", expand=False)
                .astype(float)
            )

        if "PATIENT_ID" not in df.columns:
            df["PATIENT_ID"] = df.index.astype(str)

        df = df.drop_duplicates(subset=["PATIENT_ID"])
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} METABRIC records to {output_path}")
