"""GDSC (Genomics of Drug Sensitivity in Cancer) Dataset.

Paper:
    Yang, W. et al. (2013). Genomics of Drug Sensitivity in Cancer (GDSC):
    a resource for therapeutic biomarker discovery in cancer cells.
    Nucleic Acids Research, 41(D1), D955-D961.

    Tao, Y. et al. (2020). Predicting Drug Sensitivity of Cancer Cell Lines
    via Collaborative Filtering with Contextual Attention. MLHC 2020.

This module wraps pre-processed GDSC data as a PyHealth dataset for
cancer drug sensitivity prediction tasks. Each "patient" corresponds to
a cancer cell line characterised by its binary gene expression profile.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pyhealth.datasets.sample_dataset import create_sample_dataset


class GDSCDataset:
    """GDSC dataset for cancer drug sensitivity prediction.

    Loads pre-processed GDSC data from CSV files and converts it into
    a PyHealth ``SampleDataset`` via :meth:`set_task`. Follows the PyHealth
    ``dataset.set_task()`` convention.

    Each *patient* corresponds to one cancer cell line identified by its
    COSMIC ID. The single *visit* for each patient represents its genomic
    measurement (gene expression + drug sensitivity).

    **Source data layout** (``data_dir``):

    .. code-block:: text

        exp_gdsc.csv          Binary gene expression    (1014 cell lines x 3000 genes)
        gdsc.csv              Binary drug sensitivity   ( 846 cell lines x  260 drugs)
        drug_info_gdsc.csv    Drug metadata with target pathways and drug names
        exp_emb_gdsc.csv      Gene2Vec embeddings       (3001 x 200)

    Args:
        data_dir (str): Path to the directory containing the CSV files.
            Defaults to ``"originalData"``.
        seed (int): Random seed (used by downstream splitters). Defaults
            to ``2019``.

    Attributes:
        gene_names (List[str]): Ordered gene column names (3000 genes).
        drug_ids (List[str]): Ordered drug column names (260 drugs).
        drug_names (List[str]): Human-readable drug names resolved from
            ``drug_info_gdsc.csv`` ``Name`` column.
        drug_pathway_ids (List[int]): Integer pathway ID per drug (length 260).
        gene_embeddings (np.ndarray): Pre-trained Gene2Vec matrix of shape
            ``(3001, 200)``; row 0 is the zero-padding vector.

    Examples:
        >>> from pyhealth.datasets import GDSCDataset
        >>> dataset = GDSCDataset(data_dir="originalData")
        >>> dataset.summary()
        GDSC Dataset Summary
          Cell lines:       846
          ...
        >>> sample_ds = dataset.set_task()
        >>> len(sample_ds)
        846
        >>> sample = sample_ds[0]
        >>> "gene_indices" in sample
        True
    """

    dataset_name: str = "GDSC"

    def __init__(self, data_dir: str = "originalData", seed: int = 2019) -> None:
        self.data_dir = data_dir
        self.seed = seed
        self._load_data()

    # ------------------------------------------------------------------
    # Internal data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load and align all CSV data files."""
        self.exp = pd.read_csv(
            os.path.join(self.data_dir, "exp_gdsc.csv"), index_col=0
        )
        self.tgt = pd.read_csv(
            os.path.join(self.data_dir, "gdsc.csv"), index_col=0
        )
        self.drug_info = pd.read_csv(
            os.path.join(self.data_dir, "drug_info_gdsc.csv"), index_col=0
        )
        self.gene_embeddings = np.loadtxt(
            os.path.join(self.data_dir, "exp_emb_gdsc.csv"), delimiter=","
        )

        # Restrict to cell lines present in both expression and sensitivity data
        self.common_samples = sorted(set(self.exp.index) & set(self.tgt.index))
        self.exp = self.exp.loc[self.common_samples]
        self.tgt = self.tgt.loc[self.common_samples]

        # Gene and drug IDs (before mapping functions)
        self.gene_names: List[str] = list(self.exp.columns)
        self.drug_ids: List[str] = list(self.tgt.columns)

        self._build_pathway_mapping()
        self._build_id_to_name_mapping()

    def _build_pathway_mapping(self) -> None:
        """Map each drug column to an integer pathway ID."""
        id2pw = dict(zip(self.drug_info.index, self.drug_info["Target pathway"]))
        self.drug_pathways = [id2pw.get(int(c), "Unknown") for c in self.tgt.columns]
        unique_pathways = sorted(set(self.drug_pathways))
        self.pathway2id: Dict[str, int] = {pw: i for i, pw in enumerate(unique_pathways)}
        self.drug_pathway_ids: List[int] = [self.pathway2id[pw] for pw in self.drug_pathways]

    def _build_id_to_name_mapping(self) -> None:
        """Map numeric drug IDs to drug names via drug_info_gdsc.csv ``Name`` column."""
        self.id_to_name: Dict[str, str] = {}
        for drug_id, row in self.drug_info.iterrows():
            self.id_to_name[str(drug_id)] = row["Name"]

        self.drug_names: List[str] = [
            self.id_to_name.get(str(int(drug_id)), f"UNKNOWN_{drug_id}")
            for drug_id in self.drug_ids
        ]

    # ------------------------------------------------------------------
    # PyHealth task interface
    # ------------------------------------------------------------------

    def set_task(self, task=None):
        """Apply a task to produce a model-ready dataset.

        Follows the PyHealth ``dataset.set_task()`` convention: iterates
        over every cell line, calls ``task(patient)``, and collects the
        returned sample dicts into a dataset.

        Args:
            task: An object with ``task_name``, ``input_schema``,
                ``output_schema`` attributes and a
                ``__call__(patient) -> List[dict]`` method conforming to the
                :class:`~pyhealth.tasks.BaseTask` interface. If ``None``,
                :class:`~pyhealth.tasks.DrugSensitivityPredictionGDSC` is
                used with default settings.

        Returns:
            Dataset ready for a standard PyTorch
            :class:`~torch.utils.data.DataLoader`.
        """
        if task is None:
            from pyhealth.tasks.drug_sensitivity_gdsc import (
                DrugSensitivityPredictionGDSC,
            )
            task = DrugSensitivityPredictionGDSC()

        samples: List[Dict] = []
        for cell_line in self.common_samples:
            patient = {
                "patient_id": str(cell_line),
                "gene_expression": self.exp.loc[cell_line].values,
                "drug_sensitivity": self.tgt.loc[cell_line].values.astype(float),
                "drug_pathway_ids": self.drug_pathway_ids,
            }
            samples.extend(task(patient))

        return create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name=self.dataset_name,
            task_name=task.task_name,
            in_memory=True,
        )

    # ------------------------------------------------------------------
    # Model-configuration accessors
    # ------------------------------------------------------------------

    def get_gene_embeddings(self) -> np.ndarray:
        """Return the pre-trained Gene2Vec embedding matrix.

        Returns:
            np.ndarray: Shape ``(3001, 200)``. Row 0 is the zero-padding
            vector; rows 1-3000 correspond to 1-indexed gene positions used
            in the ``gene_indices`` sample field.
        """
        return self.gene_embeddings

    def get_pathway_info(self) -> Dict:
        """Return drug pathway metadata for model initialisation.

        Returns:
            dict: Keys are ``pathway2id``, ``id2pathway``,
            ``num_pathways``, and ``drug_pathway_ids``.
        """
        return {
            "pathway2id": self.pathway2id,
            "id2pathway": {v: k for k, v in self.pathway2id.items()},
            "num_pathways": len(self.pathway2id),
            "drug_pathway_ids": self.drug_pathway_ids,
        }

    def get_overlap_drugs(self, other_dataset) -> Tuple[List[int], List[int], List[str]]:
        """Find drugs overlapping with another dataset by drug name.

        Compares by drug *name* rather than numeric ID so that GDSC (which
        uses numeric COSMIC drug IDs as column headers) and CCLE (which uses
        drug names) can be matched for cross-dataset evaluation.

        Args:
            other_dataset: Another dataset instance with a ``drug_names``
                attribute (or falling back to ``drug_ids``).

        Returns:
            tuple: ``(self_indices, other_indices, overlap_names)`` where each
            element of ``self_indices`` / ``other_indices`` is the integer
            column position of the shared drug in the respective dataset, and
            ``overlap_names`` is the sorted list of shared drug name strings.
        """
        self_names = set(self.drug_names)
        other_names = set(
            other_dataset.drug_names
            if hasattr(other_dataset, "drug_names")
            else other_dataset.drug_ids
        )
        overlap = sorted(self_names & other_names)

        self_indices = [self.drug_names.index(d) for d in overlap]
        other_indices = [
            (
                other_dataset.drug_names.index(d)
                if hasattr(other_dataset, "drug_names")
                else other_dataset.drug_ids.index(d)
            )
            for d in overlap
        ]
        return self_indices, other_indices, overlap

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print dataset summary statistics to stdout."""
        total_pairs = len(self.common_samples) * len(self.tgt.columns)
        tested = self.tgt.notnull().sum().sum()
        sensitive = (self.tgt == 1).sum().sum()
        resistant = (self.tgt == 0).sum().sum()

        print("GDSC Dataset Summary")
        print(f"  Cell lines:       {len(self.common_samples)}")
        print(f"  Drugs:            {len(self.tgt.columns)}")
        print(f"  Genes:            {len(self.exp.columns)}")
        print(f"  Active genes/cell:{int(self.exp.sum(axis=1).mean())}")
        print(f"  Total pairs:      {total_pairs}")
        print(f"  Tested pairs:     {int(tested)} ({tested / total_pairs:.1%})")
        print(
            f"  Missing pairs:    {int(total_pairs - tested)}"
            f" ({(total_pairs - tested) / total_pairs:.1%})"
        )
        print(f"  Sensitive:        {int(sensitive)} ({sensitive / tested:.1%})")
        print(f"  Resistant:        {int(resistant)} ({resistant / tested:.1%})")
        print(f"  Pathways:         {len(self.pathway2id)}")
        print(f"  Embedding shape:  {self.gene_embeddings.shape}")
