''' Name: Szymon Szymura
    NetID: szymura2
    Paper title: BulkRNABert: Cancer prognosis from bulk RNA-seq based language models
    Paper link: https://www.biorxiv.org/content/10.1101/2024.06.18.599483v2.full
    Description: This is the code for a custom dataset of bulk RNA-seq data from TCGA. For simplicity data is obtained from cBioPortal datasets website: 
    https://www.cbioportal.org/datasets instead of original TCGA portal. Data from cBioPortal is already pre-processed and organized into a folder for 
    each study, which contains multiple files, including RNA-seq expression as well as patient information and survival along with other data (methylation, CNA)
    While cBioPortal file of RNA-seq data is not a raw counts and rather, quantified expression, it should be compatible with downstream PyHealth models. 
    This code reads in RNA expression and clinical data files from each folder and combines them based on patient ID. Then it selects top x genes shared across cancers
    (folders), subsets each expression file, concatanates into filnal dataset normalizes gene expression (subtract mean and divide by standard deviation).
    Files to review:
    - pyhealth/datasets/cbioportal_bulk_rna.py
    - pyhealth/tasks/bulk_rna_classification.py
    - pyhealth/tasks/bulk_rna_survival.py
    - tests/test_cbioportal_bulk_rna_dataset.py
    - tests/test_bulk_rna_tasks.py
    - examples/cbioportal_bulk_rna_showcase.ipynb
    - docs/api/datasets/pyhealth.datasets.cbioportal_bulk_rna.rst
    - docs/api/tasks/pyhealth.tasks.bulk_rna_cancer_classification.rst
    - docs/api/tasks/pyhealth.tasks.bulk_rna_survival_prediction.rst
'''

import json
import os
from pathlib import Path
from typing import List, Optional
import logging

import numpy as np
import pandas as pd

try:
    from pyhealth.datasets import BaseDataset
except ImportError:
    from pyhealth.datasets.base_dataset import BaseDataset

logging.getLogger("distributed").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

os.environ["LITDATA_DISABLE_VERSION_CHECK"] = "1"


class CBioPortalBulkRNADataset(BaseDataset):
    """cBioPortal bulk RNA-seq dataset.

    This dataset loads bulk RNA-seq expression data and clinical patient
    data from cBioPortal study folders and prepares them for PyHealth tasks.

    The dataset performs the following steps:
        1. Reads raw expression and clinical files.
        2. Derives patient IDs from sample identifiers.
        3. Finds common genes across studies.
        4. Standardizes expression values.
        5. Selects top variable genes.
        6. Merges expression with clinical data.
        7. Writes a PyHealth-compatible CSV file.

    Args:
        root (str): Root folder containing study subfolders.
        study_dirs (List[str]): List of study folder names.
        top_k_genes (int): Number of most variable genes to retain.
        tables (Optional[List[str]]): Tables to load (default: ["samples"]).
        dataset_name (Optional[str]): Dataset name.
        config_path (Optional[str]): Path to YAML config file.
        **kwargs: Additional arguments passed to BaseDataset.

    Attributes:
        study_dirs (List[str]): Study folder names.
        top_k_genes (int): Number of selected genes.

    Example:
        >>> dataset = CBioPortalBulkRNADataset(
        ...     root="/path/to/data",
        ...     study_dirs=["brca_tcga", "luad_tcga"],
        ...     top_k_genes=1000,
        ... )
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        study_dirs: List[str],
        top_k_genes: int = 5000,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:

        self.study_dirs = study_dirs
        self.top_k_genes = top_k_genes

        if config_path is None:
            config_path = (
                Path(__file__).parent / "configs" / "cbioportal_bulk_rna.yaml"
            )
    
        samples_csv = os.path.join(root, "cbioportal_bulk_rna_samples-pyhealth.csv")

        if not os.path.exists(samples_csv):
            self.prepare_metadata(
                root=root,
                study_dirs=study_dirs,
                top_k_genes=top_k_genes,
            )

        super().__init__(
            root=root,
            tables=tables or ["samples"],
            dataset_name=dataset_name or "cbioportal_bulk_rna",
            config_path=str(config_path),
            **kwargs,
        )

    @staticmethod
    def prepare_metadata(
        root: str,
        study_dirs: List[str],
        top_k_genes: int = 5000,
    ) -> None:
        """Prepare dataset by reading raw files and writing processed CSV.
        Args:
            root (str): Root directory containing study folders.
            study_dirs (List[str]): List of study folder names.
            top_k_genes (int): Number of genes to retain.
        Returns:
            None
        Raises:
            FileNotFoundError: If required files are missing.
            ValueError: If no common genes are found.
        Example:
            >>> CBioPortalBulkRNADataset.prepare_metadata(
            ... root="/data",
            ... study_dirs=["brca_tcga"],
            ... top_k_genes=1000)
        """

        loaded = []
        gene_sets = []

        # read each study first
        for study_dir in study_dirs:
            study_path = os.path.join(root, study_dir)

            expr_path = os.path.join(study_path, "data_mrna_seq_v2_rsem.txt")
            clinical_path = os.path.join(study_path, "data_clinical_patient.txt")

            if not os.path.exists(expr_path):
                raise FileNotFoundError(f"Missing expression file: {expr_path}")
            if not os.path.exists(clinical_path):
                raise FileNotFoundError(f"Missing clinical file: {clinical_path}")

            expr_df = CBioPortalBulkRNADataset._load_expression(expr_path)
            clin_df = CBioPortalBulkRNADataset._load_clinical_patient(clinical_path)

            genes = set(expr_df.columns) - {"sample_id", "patient_id"}
            gene_sets.append(genes)

            loaded.append(
                {
                    "study_id": study_dir,
                    "expr_df": expr_df,
                    "clin_df": clin_df,
                }
            )

        common_genes = sorted(set.intersection(*gene_sets))
        if len(common_genes) == 0:
            raise ValueError("No overlapping genes found across studies.")

        logger.info("Found %d common genes across studies", len(common_genes))

        merged_studies = []

        # subset to common genes and merge with clinical
        for item in loaded:
            study_id = item["study_id"]
            expr_df = item["expr_df"][["sample_id", "patient_id"] + common_genes].copy()
            clin_df = item["clin_df"].copy()

            merged = expr_df.merge(clin_df, on="patient_id", how="inner")

            merged["cancer_type"] = study_id

            # subtype may or may not exist
            if "subtype" not in merged.columns:
                merged["subtype"] = pd.NA

            merged_studies.append(merged)

        all_df = pd.concat(merged_studies, ignore_index=True)

        # keep one sample per patient
        all_df = all_df.sort_values(["patient_id", "sample_id"])
        all_df = all_df.drop_duplicates(subset=["patient_id"], keep="first")

        # ---------- global feature preprocessing ----------
        X = all_df[common_genes].to_numpy(dtype=float)

        # standardize gene-wise
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-6
        X = (X - mean) / std

        # top variable genes
        var = X.var(axis=0)
        keep_idx = np.argsort(var)[-top_k_genes:]
        keep_idx = np.sort(keep_idx)

        selected_genes = [common_genes[i] for i in keep_idx]
        X = X[:, keep_idx]

        logger.info("Retained %d most variable genes", len(selected_genes))

        # write selected genes to a txt file for reproducibility
        genes_txt = os.path.join(root, "cbioportal_bulk_rna_selected_genes.txt")
        with open(genes_txt, "w", encoding="utf-8") as f:
            for gene in selected_genes:
                f.write(f"{gene}\n")

        # convert each row to serialized vector
        all_df["expression_json"] = [json.dumps(row.tolist()) for row in X]

        # final columns
        keep_cols = [
            "patient_id",
            "sample_id",
            "cancer_type",
            "subtype",
            "os_months",
            "dfs_months",
            "os_status",
            "dfs_status",
            "expression_json",
        ]

        for col in keep_cols:
            if col not in all_df.columns:
                all_df[col] = pd.NA

        final_df = all_df[keep_cols].copy()

        output_path = os.path.join(root, "cbioportal_bulk_rna_samples-pyhealth.csv")
        final_df.to_csv(output_path, index=False)

        logger.info("Saved %d processed samples to %s", len(final_df), output_path)

    @staticmethod
    def _load_expression(expr_path: str) -> pd.DataFrame:
        """Load and preprocess RNA-seq expression data.
        Args:
            expr_path (str): Path to the expression file.
        Returns:
            pd.DataFrame: DataFrame with one row per sample and gene expression values.
        Raises:
            FileNotFoundError: If the expression file does not exist.
        Example:
            >>> df = CBioPortalBulkRNADataset._load_expression("data.txt")
        """

        df = pd.read_csv(expr_path, sep="\t", comment="#", low_memory=False)

        # cBioPortal commonly uses first column for gene symbol
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "gene_symbol"})

        df = df.dropna(subset=["gene_symbol"])
        df["gene_symbol"] = df["gene_symbol"].astype(str)
        df = df.drop_duplicates(subset=["gene_symbol"], keep="first")

        # transpose into sample x gene
        df = df.set_index("gene_symbol").transpose().reset_index()
        df = df.rename(columns={"index": "sample_id"})

        # derive patient_id from TCGA-style sample barcode
        # first 12 chars usually identify patient
        df["patient_id"] = df["sample_id"].astype(str).str[:12]

        gene_cols = [c for c in df.columns if c not in {"sample_id", "patient_id"}]
        for col in gene_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df[gene_cols] = df[gene_cols].fillna(0.0)

        return df

    @staticmethod
    def _load_clinical_patient(clinical_path: str) -> pd.DataFrame:
        """Load and standardize cBioPortal clinical patient data.
        This method reads a cBioPortal patient-level clinical file, renames
        selected columns to a standardized format used by this dataset, validates
        the presence of the patient identifier column, and removes duplicated 
        patients records.
        Supported standardized columns include survival, recurrence, and selected
        demographic or treatment-related attributes when present in the source file.
        
        Args:
            clinical_path (str): Path to the cBioPortal clinical patient file.
        Returns:
            pd.DataFrame: Standardized patient-level clinical dataframe containing
            at least the ``patient_id``` columns and any available renamed clinical fields.
        Raises:
            ValueError: If the input file does not contain a ```PATIENT_ID`` column 
            after loading and standarization.
        Example:
            >>> df = CBioPortalBulkRNADataser.__load_clinical_patient(
            ... "/path/to/data_clinical_patient.txt")
        """

        df = pd.read_csv(clinical_path, sep="\t", comment="#", low_memory=False)

        rename_map = {
            "PATIENT_ID": "patient_id",
            "OS_MONTHS": "os_months",
            "DFS_MONTHS": "dfs_months",
            "OS_STATUS": "os_status",
            "DFS_STATUS": "dfs_status",
            "SEX": "sex",
            "RACE": "race",
            "ETHNICITY": "ethnicity",
            "SUBTYPE": "subtype",
            "RADIATION_THERAPY": "radiation_therapy",
            "CANCER_TYPE": "cancer_type_raw",
        }

        existing = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=existing)

        if "patient_id" not in df.columns:
            raise ValueError(f"Clinical patient file missing PATIENT_ID: {clinical_path}")

        df = df.drop_duplicates(subset=["patient_id"], keep="first")
        return df

    @property
    def default_task(self):

        from pyhealth.tasks import BulkRNACancerClassification

        return BulkRNACancerClassification()