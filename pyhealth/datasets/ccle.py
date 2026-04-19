"""CCLE (Cancer Cell Line Encyclopedia) Dataset.

Paper:
    Barretina, J. et al. (2012). The Cancer Cell Line Encyclopedia enables
    predictive modelling of anticancer drug sensitivity.
    Nature, 483(7391), 603-607.

This module wraps pre-processed CCLE data as a PyHealth dataset for cancer
drug sensitivity prediction. Follows the same conventions as
:class:`~pyhealth.datasets.GDSCDataset` so both datasets can be used
interchangeably (and jointly for cross-dataset evaluation).

Unlike GDSC — where drug column headers are numeric COSMIC IDs — CCLE uses
drug *names* directly as column headers. Cross-dataset drug matching is
therefore done by name via :meth:`get_overlap_drugs`.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pyhealth.datasets.sample_dataset import create_sample_dataset


class CCLEDataset:
    """CCLE dataset for cancer drug sensitivity prediction.

    Loads pre-processed CCLE data from CSV files and converts it into a
    PyHealth dataset via :meth:`set_task`. Follows the PyHealth
    ``dataset.set_task()`` convention.

    Each *patient* corresponds to one cancer cell line. The single *visit*
    represents its genomic measurement (gene expression + drug sensitivity).

    **Source data layout** (``data_dir``):

    .. code-block:: text

        exp_ccle.csv          Binary gene expression    (cell lines x genes)
        ccle.csv              Binary drug sensitivity   (cell lines x drugs)
        drug_info_ccle.csv    Drug metadata with target pathways
        exp_emb_ccle.csv      Gene2Vec embeddings       (n_genes+1 x emb_dim)

    Unlike GDSC, CCLE drug columns are identified by drug **name** (not
    numeric ID). :meth:`get_overlap_drugs` handles name-based cross-dataset
    matching automatically.

    Args:
        data_dir (str): Path to the directory containing the CSV files.
            Defaults to ``"ccleData"``.
        seed (int): Random seed (used by downstream splitters). Defaults
            to ``2019``.

    Attributes:
        gene_names (List[str]): Ordered gene column names.
        drug_ids (List[str]): Ordered drug column names (drug names).
        drug_names (List[str]): Same as ``drug_ids`` for CCLE (drug names
            are used directly as column headers).
        drug_pathway_ids (List[int]): Integer pathway ID per drug.
        gene_embeddings (np.ndarray): Pre-trained Gene2Vec matrix; row 0 is
            the zero-padding vector.

    Examples:
        >>> from pyhealth.datasets import CCLEDataset
        >>> dataset = CCLEDataset(data_dir="ccleData")
        >>> sample_ds = dataset.set_task()
        >>> "gene_indices" in sample_ds[0]
        True
    """

    dataset_name: str = "CCLE"

    def __init__(self, data_dir: str = "ccleData", seed: int = 2019) -> None:
        self.data_dir = data_dir
        self.seed = seed
        self._load_data()

    # ------------------------------------------------------------------
    # Internal data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load and align all CSV data files."""
        try:
            self.exp = pd.read_csv(
                os.path.join(self.data_dir, "exp_ccle.csv"), index_col=0
            )
            self.tgt = pd.read_csv(
                os.path.join(self.data_dir, "ccle.csv"), index_col=0
            )
            self.drug_info = pd.read_csv(
                os.path.join(self.data_dir, "drug_info_ccle.csv"), index_col=0
            )
            self.gene_embeddings = np.loadtxt(
                os.path.join(self.data_dir, "exp_emb_ccle.csv"), delimiter=","
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"CCLE data files not found in {self.data_dir}. "
                "Ensure exp_ccle.csv, ccle.csv, drug_info_ccle.csv, and "
                "exp_emb_ccle.csv exist."
            ) from exc

        # Find common cell lines using case/punctuation-insensitive matching to
        # handle CCLE naming inconsistencies between tables (e.g. "22Rv1" vs
        # "22RV1", "42-MG-BA" vs "42MGBA").
        def _norm(s):
            return str(s).upper().replace("-", "").replace(" ", "").replace(".", "")

        exp_norm = {_norm(i): i for i in self.exp.index}
        tgt_norm = {_norm(i): i for i in self.tgt.index}
        common_norm = sorted(set(exp_norm) & set(tgt_norm))
        # Use the expression-side label as the canonical cell-line ID.
        self.common_samples = [exp_norm[k] for k in common_norm]
        self.exp = self.exp.loc[self.common_samples]
        self.tgt = self.tgt.loc[[tgt_norm[k] for k in common_norm]]
        self.tgt.index = self.common_samples

        # Preprocessed CCLE labels are inverted (~75% "1") relative to the
        # paper's 24.8% sensitive prior and GDSC's "1 = sensitive" convention.
        # Flip so "1 = sensitive" is consistent across both datasets.
        observed = self.tgt.notnull()
        self.tgt = self.tgt.where(~observed, 1 - self.tgt)

        # Gene and drug IDs
        self.gene_names: List[str] = list(self.exp.columns)
        # For CCLE, column headers ARE drug names (not numeric IDs)
        self.drug_ids: List[str] = list(self.tgt.columns)
        self.drug_names: List[str] = self.drug_ids

        self._build_pathway_mapping()

    def _build_pathway_mapping(self) -> None:
        """Map each drug column (drug name) to an integer pathway ID.

        Uses case-insensitive name matching against ``drug_info_ccle.csv``
        index to handle minor casing discrepancies.
        """
        id2pw: Dict[str, str] = {}
        for idx in self.drug_info.index:
            id2pw[str(idx)] = self.drug_info.loc[idx, "Target pathway"]

        self.drug_pathways: List[str] = []
        for drug_name in self.tgt.columns:
            name_str = str(drug_name)
            if name_str in id2pw:
                pw = id2pw[name_str]
            else:
                # Case-insensitive fallback
                matches = [k for k in id2pw if k.lower() == name_str.lower()]
                pw = id2pw[matches[0]] if matches else "Unknown"
            self.drug_pathways.append(pw)

        unique_pathways = sorted(set(self.drug_pathways))
        self.pathway2id: Dict[str, int] = {pw: i for i, pw in enumerate(unique_pathways)}
        self.drug_pathway_ids: List[int] = [
            self.pathway2id[pw] for pw in self.drug_pathways
        ]

    # ------------------------------------------------------------------
    # PyHealth task interface
    # ------------------------------------------------------------------

    def set_task(self, task=None):
        """Apply a task to produce a model-ready dataset.

        Args:
            task: An object with ``task_name``, ``input_schema``,
                ``output_schema`` attributes and a
                ``__call__(patient) -> List[dict]`` method. If ``None``,
                :class:`~pyhealth.tasks.DrugSensitivityPredictionCCLE` is
                used with default settings.

        Returns:
            Dataset ready for a standard PyTorch
            :class:`~torch.utils.data.DataLoader`.
        """
        if task is None:
            from pyhealth.tasks.drug_sensitivity_ccle import (
                DrugSensitivityPredictionCCLE,
            )
            task = DrugSensitivityPredictionCCLE()

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
            np.ndarray: Row 0 is the zero-padding vector; subsequent rows
            correspond to 1-indexed gene positions used in ``gene_indices``.
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

        Args:
            other_dataset: Another dataset instance with a ``drug_names``
                attribute (or falling back to ``drug_ids``).

        Returns:
            tuple: ``(self_indices, other_indices, overlap_names)``
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

        print("CCLE Dataset Summary")
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
