"""PyHealth dataset for TCGA bulk RNA-seq gene expression data.

This module provides ``TCGARNASeqDataset``, which loads TCGA RNA-seq gene
expression and clinical data, applies the BulkRNABert preprocessing pipeline
(log10(1+TPM) followed by per-sample max-normalization), and writes a single
merged CSV that BaseDataset can ingest.

Dataset paper:
    Gélard et al. (2024). BulkRNABert: Cancer prognosis from bulk RNA-seq
    based language models. https://doi.org/10.1101/2024.06.18.599483
"""

import json
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

_META_COLS = ["patient_id", "sample_id", "cohort"]
_PROCESSED_FILENAME = "tcga_rnaseq_pyhealth.csv"


class TCGARNASeqDataset(BaseDataset):
    """Dataset class for TCGA bulk RNA-seq gene expression data.

    Loads ``gene_expression.csv`` and ``clinical.csv`` from ``root``,
    applies log10(1+TPM) max-normalization, serializes the gene expression
    vector for each sample into a single JSON column, merges clinical
    annotations, and writes the result to ``tcga_rnaseq_pyhealth.csv``.
    Subsequent instantiations reuse the cached processed file.

    The gene expression preprocessing pipeline follows BulkRNABert
    (Gélard et al., 2024):

    1. ``log10(1 + TPM)`` applied element-wise.
    2. Per-sample max-normalization: divide each value by the row maximum
       (a row maximum of 0 is replaced by 1 to avoid division by zero).

    Input files expected in ``root``:

    - ``gene_expression.csv``: columns ``patient_id``, ``sample_id``,
      ``cohort``, and one column per gene with raw TPM values.
    - ``clinical.csv``: columns ``patient_id``, ``cohort``,
      ``survival_time``, ``event``.

    Attributes:
        root (str): Root directory containing the input CSV files.
        dataset_name (str): Fixed to ``"tcga_rnaseq"``.

    Examples:
        >>> dataset = TCGARNASeqDataset(root="/path/to/tcga")
        >>> samples = dataset.set_task(TCGARNASeqCancerTypeClassification())
    """

    def __init__(
        self,
        root: str,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initializes the TCGARNASeqDataset.

        Verifies that the required input files are present, runs the
        preprocessing pipeline (writing ``tcga_rnaseq_pyhealth.csv`` if it
        does not already exist), then delegates to ``BaseDataset.__init__``.

        Args:
            root (str): Root directory containing ``gene_expression.csv``
                and ``clinical.csv``.
            config_path (Optional[str]): Path to the YAML configuration file.
                Defaults to the bundled ``configs/tcga_rnaseq.yaml``.
            **kwargs: Additional keyword arguments forwarded to
                ``BaseDataset.__init__`` (e.g. ``dev``, ``num_workers``).

        Raises:
            FileNotFoundError: If ``gene_expression.csv`` or ``clinical.csv``
                is not found in ``root``.

        Examples:
            >>> dataset = TCGARNASeqDataset(root="/path/to/tcga")
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "tcga_rnaseq.yaml"
            )

        self._verify_data(root)
        self._preprocess(root)

        super().__init__(
            root=root,
            tables=["rnaseq"],
            dataset_name="tcga_rnaseq",
            config_path=config_path,
            **kwargs,
        )

    def _verify_data(self, root: str) -> None:
        """Checks that required input files exist in the root directory.

        Args:
            root (str): Root directory to check.

        Raises:
            FileNotFoundError: If ``gene_expression.csv`` is missing from
                ``root``.
            FileNotFoundError: If ``clinical.csv`` is missing from ``root``.

        Examples:
            >>> dataset._verify_data("/path/to/tcga")
        """
        ge_path = os.path.join(root, "gene_expression.csv")
        if not os.path.isfile(ge_path):
            raise FileNotFoundError(
                f"Required file not found: {ge_path}. "
                "Please ensure gene_expression.csv is present in the root directory."
            )

        clin_path = os.path.join(root, "clinical.csv")
        if not os.path.isfile(clin_path):
            raise FileNotFoundError(
                f"Required file not found: {clin_path}. "
                "Please ensure clinical.csv is present in the root directory."
            )

    def _preprocess(self, root: str) -> None:
        """Runs the full preprocessing pipeline and writes the processed CSV.

        If ``tcga_rnaseq_pyhealth.csv`` already exists in ``root``, this
        method returns immediately (cache guard). Otherwise it:

        1. Reads ``gene_expression.csv`` and ``clinical.csv``.
        2. Identifies gene columns (all columns except ``patient_id``,
           ``sample_id``, and ``cohort``).
        3. Normalizes gene expression via :meth:`_normalize`.
        4. Serializes each row's gene vector to a JSON string in a single
           ``gene_expression`` column (individual gene columns are dropped).
        5. Left-merges with clinical data on ``patient_id``.
        6. Writes the result to ``tcga_rnaseq_pyhealth.csv``.

        Args:
            root (str): Root directory for reading inputs and writing output.

        Examples:
            >>> dataset._preprocess("/path/to/tcga")
        """
        out_path = os.path.join(root, _PROCESSED_FILENAME)
        if os.path.isfile(out_path):
            logger.info(
                f"Processed file already exists at {out_path}, skipping preprocessing."
            )
            return

        logger.info("Preprocessing TCGA RNA-seq data...")

        ge_df = pd.read_csv(os.path.join(root, "gene_expression.csv"))
        clin_df = pd.read_csv(os.path.join(root, "clinical.csv"))

        gene_cols: List[str] = [c for c in ge_df.columns if c not in _META_COLS]

        ge_df = self._normalize(ge_df, gene_cols)

        ge_df["gene_expression"] = ge_df[gene_cols].apply(
            lambda row: json.dumps(row.tolist()), axis=1
        )
        ge_df = ge_df.drop(columns=gene_cols)

        merged = ge_df.merge(
            clin_df[["patient_id", "survival_time", "event"]],
            on="patient_id",
            how="left",
        )

        merged.to_csv(out_path, index=False)
        logger.info(f"Processed CSV written to {out_path}")

    def _normalize(self, df: pd.DataFrame, gene_cols: List[str]) -> pd.DataFrame:
        """Applies log10(1+TPM) max-normalization to gene expression columns.

        If all values in ``gene_cols`` are already in [0, 1], the dataframe
        is returned unchanged (idempotency guard for pre-normalized inputs).
        Otherwise, the two-step pipeline from BulkRNABert is applied:

        1. ``log10(1 + x)`` applied element-wise to every gene column.
        2. Each row is divided by its maximum value.  Rows with a maximum of
           0 are divided by 1 to avoid ``NaN``.

        Args:
            df (pd.DataFrame): DataFrame containing gene expression data.
            gene_cols (List[str]): Column names corresponding to gene features.

        Returns:
            pd.DataFrame: DataFrame with gene columns normalized to [0, 1].

        Examples:
            >>> normalized_df = dataset._normalize(df, gene_cols)
            >>> normalized_df[gene_cols].max().max() <= 1.0
            True
        """
        if df[gene_cols].max().max() <= 1.0:
            logger.info("Gene expression appears already normalized, skipping.")
            return df

        df = df.copy()
        df[gene_cols] = np.log10(1 + df[gene_cols].values)

        row_max = df[gene_cols].max(axis=1).replace(0, 1)
        df[gene_cols] = df[gene_cols].div(row_max, axis=0)

        return df
