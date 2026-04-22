"""TCGA Pan-Cancer Bulk RNA-seq dataset for PyHealth.

This module provides the TCGARNASeqDataset class for loading and processing
bulk RNA-seq data from The Cancer Genome Atlas (TCGA) for cancer type
classification and survival analysis tasks, as used in BulkRNABert.

Paper: Gélard et al., "BulkRNABert: Cancer prognosis from bulk RNA-seq
based language models", bioRxiv 2024.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# 33 TCGA cohort abbreviations
TCGA_COHORTS = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
]

RUNTIME_CONFIG_NAME = "tcga_rnaseq_pyhealth_config.yaml"


def _infer_gene_columns_from_tokenized_csv(path: str) -> List[str]:
    """Return gene column names from a tokenized RNA-seq CSV header."""
    header = pd.read_csv(path, nrows=0)
    skip = {"patient_id", "cohort"}
    return [c for c in header.columns if c.lower() not in skip]


def _write_runtime_config(root: str, gene_names: List[str]) -> str:
    """Write a PyHealth ``DatasetConfig``-compatible YAML under ``root``.

    ``BaseDataset`` expects ``version``, ``file_path``, ``patient_id``,
    ``timestamp``, and ``attributes`` per table.

    Args:
        root: Dataset root directory (output file lives here too).
        gene_names: Gene symbols as they appear in the tokenized CSV columns.

    Returns:
        Absolute path to the written YAML file.
    """
    rnaseq_attrs = ["cohort"] + [g.lower() for g in gene_names]
    clinical_attrs = [
        "cohort",
        "vital_status",
        "days_to_death",
        "days_to_last_follow_up",
    ]
    cfg = {
        "version": "1.0",
        "tables": {
            "rnaseq": {
                "file_path": "tcga_rnaseq_tokenized-pyhealth.csv",
                "patient_id": "patient_id",
                "timestamp": None,
                "attributes": rnaseq_attrs,
                "join": [],
            },
            "clinical": {
                "file_path": "tcga_rnaseq_clinical-pyhealth.csv",
                "patient_id": "patient_id",
                "timestamp": None,
                "attributes": clinical_attrs,
                "join": [],
            },
        },
    }
    out_path = os.path.join(root, RUNTIME_CONFIG_NAME)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return out_path


class TCGARNASeqDataset(BaseDataset):
    """TCGA Pan-Cancer Bulk RNA-seq dataset for cancer prognosis.

    Loads bulk RNA-seq gene expression data (TPM) from The Cancer Genome
    Atlas across up to 33 cancer cohorts, along with clinical metadata
    for survival analysis. Implements the preprocessing pipeline from
    BulkRNABert: log10(1+x) transformation, max-normalization, and
    discretization into B expression bins.

    This dataset supports two downstream tasks:
        - Pan-cancer or cohort-specific cancer type classification
        - Survival time prediction (time-to-event with right-censoring)

    Dataset available at: https://portal.gdc.cancer.gov/

    A machine-readable ``DatasetConfig`` YAML is written to
    ``{root}/tcga_rnaseq_pyhealth_config.yaml`` on init (unless
    ``config_path`` is provided) so ``BaseDataset`` can load tables via
    the standard schema.

    Args:
        root: Root directory containing ``rna_seq.csv`` and
            ``clinical.csv`` files.
        n_bins: Number of expression bins for tokenization. Defaults to 64.
        n_genes: Number of genes to retain. If None, uses all common genes.
        tables: Optional additional tables to load.
        dataset_name: Optional dataset name override.
        config_path: Optional path to a valid ``DatasetConfig`` YAML. If
            ``None``, a config is generated under ``root`` from gene names.

    Attributes:
        n_bins: Number of discretization bins.
        n_genes: Number of genes after filtering.
        gene_names: List of gene names after filtering.

    Examples:
        >>> from pyhealth.datasets import TCGARNASeqDataset
        >>> dataset = TCGARNASeqDataset(root="/path/to/tcga_rnaseq")
        >>> samples = dataset.set_task()
        >>> print(samples[0])
    """

    def __init__(
        self,
        root: str,
        n_bins: int = 64,
        n_genes: Optional[int] = None,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.n_bins = n_bins
        self.n_genes = n_genes
        self.gene_names: List[str] = []

        # Prepare preprocessed CSVs if not already done
        rnaseq_out = os.path.join(root, "tcga_rnaseq_tokenized-pyhealth.csv")
        clinical_out = os.path.join(root, "tcga_rnaseq_clinical-pyhealth.csv")

        if not os.path.exists(rnaseq_out) or not os.path.exists(clinical_out):
            logger.info("Preparing TCGA RNA-seq metadata")
            self._prepare_metadata(root, n_bins, n_genes, rnaseq_out, clinical_out)

        gene_file = os.path.join(root, "tcga_rnaseq_genes.txt")
        if os.path.exists(gene_file):
            with open(gene_file, encoding="utf-8") as f:
                self.gene_names = [line.strip() for line in f if line.strip()]
        if not self.gene_names and os.path.exists(rnaseq_out):
            self.gene_names = _infer_gene_columns_from_tokenized_csv(rnaseq_out)

        if config_path is None:
            config_path = _write_runtime_config(root, self.gene_names)
        elif not os.path.isfile(config_path):
            raise FileNotFoundError(f"config_path does not exist: {config_path}")

        default_tables = ["rnaseq", "clinical"]
        tables = default_tables + (tables or [])

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "tcga_rnaseq",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def _prepare_metadata(
        root: str,
        n_bins: int,
        n_genes: Optional[int],
        rnaseq_out: str,
        clinical_out: str,
    ) -> None:
        """Prepare and preprocess RNA-seq and clinical CSVs.

        Applies log10(1 + x) transformation, max-normalization, and linear
        binning to TPM expression values, then saves tokenized output.
        """
        rnaseq_raw = os.path.join(root, "rna_seq.csv")
        clinical_raw = os.path.join(root, "clinical.csv")

        if not os.path.exists(rnaseq_raw):
            logger.warning(
                f"rna_seq.csv not found in {root}. "
                "Please download TCGA RNA-seq TPM data from "
                "https://portal.gdc.cancer.gov/ and save as rna_seq.csv "
                "with rows=samples, columns=genes, plus a 'patient_id' column."
            )
            TCGARNASeqDataset._create_placeholder_csvs(
                rnaseq_out, clinical_out, n_bins
            )
            return

        logger.info("Loading RNA-seq expression matrix")
        df = pd.read_csv(rnaseq_raw)
        if "patient_id" in df.columns:
            df = df.set_index("patient_id", drop=True)

        gene_cols = [c for c in df.columns if c.lower() != "cohort"]
        cohort_col = df["cohort"] if "cohort" in df.columns else None
        expr = df[gene_cols].astype(float)

        if n_genes is not None and n_genes < len(gene_cols):
            variances = expr.var(axis=0)
            top_genes = variances.nlargest(n_genes).index.tolist()
            expr = expr[top_genes]

        gene_names = expr.columns.tolist()

        gene_file = os.path.join(root, "tcga_rnaseq_genes.txt")
        with open(gene_file, "w", encoding="utf-8") as f:
            f.write("\n".join(gene_names))

        logger.info("Applying log10(1+x) transformation")
        expr_log = np.log10(1.0 + expr.values)

        logger.info("Applying max-normalization")
        row_max = expr_log.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        expr_norm = expr_log / row_max

        logger.info("Discretizing into %s bins.", n_bins)
        expr_binned = np.floor(expr_norm * n_bins).astype(int)
        expr_binned = np.clip(expr_binned, 0, n_bins - 1)

        out_df = pd.DataFrame(expr_binned, index=expr.index, columns=gene_names)
        out_df.index.name = "patient_id"
        out_df = out_df.reset_index()
        if cohort_col is not None:
            aligned = cohort_col.reindex(out_df["patient_id"]).values
            out_df.insert(1, "cohort", aligned)
        out_df.to_csv(rnaseq_out, index=False)
        logger.info("Saved tokenized RNA-seq to %s", rnaseq_out)

        if os.path.exists(clinical_raw):
            clin = pd.read_csv(clinical_raw)
            rename = {
                "bcr_patient_barcode": "patient_id",
                "submitter_id": "patient_id",
                "vital_status": "vital_status",
                "days_to_death": "days_to_death",
                "days_to_last_follow_up": "days_to_last_follow_up",
                "project_id": "cohort",
            }
            clin = clin.rename(
                columns={k: v for k, v in rename.items() if k in clin.columns}
            )
            if "patient_id" not in clin.columns:
                clin.insert(0, "patient_id", clin.index.astype(str))
            clin.to_csv(clinical_out, index=False)
            logger.info("Saved clinical data to %s", clinical_out)
        else:
            logger.warning(
                "clinical.csv not found in %s. "
                "Survival tasks will not be available without clinical data.",
                root,
            )
            TCGARNASeqDataset._create_placeholder_clinical(clinical_out)

    @staticmethod
    def _create_placeholder_csvs(
        rnaseq_out: str, clinical_out: str, n_bins: int
    ) -> None:
        """Create minimal placeholder CSVs when raw data is unavailable.

        One synthetic row keeps ``BaseDataset`` table scans well-defined.

        Args:
            rnaseq_out: Output path for tokenized RNA-seq CSV.
            clinical_out: Output path for clinical CSV.
            n_bins: Number of bins (reserved for API compatibility).
        """
        del n_bins
        genes = ("GENE0", "GENE1")
        row = {"patient_id": "TCGA-PLACEHOLDER", "cohort": "BRCA"}
        for g in genes:
            row[g] = 0
        pd.DataFrame([row]).to_csv(rnaseq_out, index=False)
        gene_file = os.path.join(
            os.path.dirname(rnaseq_out), "tcga_rnaseq_genes.txt"
        )
        with open(gene_file, "w", encoding="utf-8") as f:
            f.write("\n".join(genes))
        pd.DataFrame(
            [
                {
                    "patient_id": "TCGA-PLACEHOLDER",
                    "cohort": "BRCA",
                    "vital_status": "alive",
                    "days_to_death": None,
                    "days_to_last_follow_up": 365.0,
                }
            ]
        ).to_csv(clinical_out, index=False)

    @staticmethod
    def _create_placeholder_clinical(clinical_out: str) -> None:
        """Create an empty placeholder clinical CSV.

        Args:
            clinical_out: Output path for clinical CSV.
        """
        pd.DataFrame(
            columns=[
                "patient_id",
                "cohort",
                "vital_status",
                "days_to_death",
                "days_to_last_follow_up",
            ]
        ).to_csv(clinical_out, index=False)

    @property
    def default_task(self):
        """Returns the default cancer type classification task.

        Returns:
            TCGACancerTypeTask: The default classification task.
        """
        from pyhealth.tasks import TCGACancerTypeTask

        return TCGACancerTypeTask()
