"""TCGA RNA-seq embedding dataset + factory for 5-cohort cancer classification.

This module pairs pre-computed :class:`~pyhealth.models.BulkRNABert` encoder
outputs (``.npy``) with TCGA cancer-type labels resolved from the GDC file
mapping CSV. It exposes two entry points:

* :class:`TCGARNASeqEmbeddingDataset` — a full :class:`~pyhealth.datasets.BaseDataset`
  subclass that merges the ``.npy`` matrix and the identifier / mapping CSVs
  into a single cache CSV and then loads it through the standard YAML-driven
  loader. Pair with :class:`~pyhealth.tasks.TCGACancerClassification5Cohort`
  via ``dataset.set_task(task)``. Use this path for full-pipeline workflows.

* :func:`load_tcga_cancer_classification_5cohort` — a thin shortcut that
  returns an :class:`~pyhealth.datasets.InMemorySampleDataset` directly,
  skipping the BaseDataset event-dataframe round-trip. Useful for small
  experiments and unit tests.

Inputs expected on disk (both entry points):

* ``embeddings_path`` — ``.npy`` file produced by
  ``examples/bulk_rna_bert/tcga_rnaseq_extract_embeddings_bulk_rna_bert.py``. Row ``i``
  of this file must correspond to row ``i`` of ``identifier_csv``.
* ``identifier_csv`` — the preprocessed TCGA CSV (``tcga_preprocessed.csv``)
  used during pre-training. Only the ``identifier`` column is consumed here.
* ``mapping_csv`` — ``tcga_file_mapping.csv`` from the TCGA GDC metadata
  dump. The joiner filters to ``sample_type == "Primary Tumor"`` and
  ``project`` in the five target cohorts, then keys on
  ``identifier == file_name.split(".")[0]``.

Author: Yohei Shibata (NetID: yoheis2)
Paper: BulkRNABert: Cancer prognosis from bulk RNA-seq based language models
       (Gelard et al., PMLR 259, 2025)
Paper link: https://proceedings.mlr.press/v259/gelard25a.html
Description: BaseDataset wrapper + shortcut factory that pair BulkRNABert
    ``.npy`` embeddings with 5-cohort TCGA cancer-type labels resolved from
    the GDC file-mapping CSV.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from pyhealth.tasks.tcga_cancer_classification_5cohort import (
    LABEL_MAP,
    TCGACancerClassification5Cohort,
)

from .base_dataset import BaseDataset
from .sample_dataset import InMemorySampleDataset, create_sample_dataset

logger = logging.getLogger(__name__)

MERGED_CSV_NAME = "tcga_rnaseq_embedding.csv"


def _build_identifier_to_label(mapping_csv: str | Path) -> dict[str, int]:
    """Parse ``tcga_file_mapping.csv`` and return ``identifier -> int label``.

    The identifier is defined as ``file_name.split(".")[0]`` to match how
    the preprocessed CSV is keyed during pre-training. Only rows whose
    ``project`` is in :data:`LABEL_MAP` and whose ``sample_type`` equals
    ``"Primary Tumor"`` are retained.
    """
    identifier_to_label: dict[str, int] = {}
    with open(mapping_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["project"] in LABEL_MAP and row["sample_type"] == "Primary Tumor":
                key = row["file_name"].split(".")[0]
                identifier_to_label[key] = LABEL_MAP[row["project"]]
    return identifier_to_label


def _identifier_to_cohort(mapping_csv: str | Path) -> dict[str, str]:
    """Like :func:`_build_identifier_to_label` but returns the raw cohort tag."""
    out: dict[str, str] = {}
    with open(mapping_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["project"] in LABEL_MAP and row["sample_type"] == "Primary Tumor":
                key = row["file_name"].split(".")[0]
                out[key] = row["project"]
    return out


def _select_rows(
    embeddings: np.ndarray,
    identifier_csv: str | Path,
    identifier_to_label: dict[str, int],
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Slice ``embeddings`` to rows whose identifier is in the label map."""
    with open(identifier_csv) as f:
        header = f.readline().rstrip("\n").split(",")
        if header[-1] != "identifier":
            raise ValueError(
                f"{identifier_csv}: last column must be 'identifier', got "
                f"{header[-1]!r}"
            )
        indices: List[int] = []
        labels: List[int] = []
        identifiers: List[str] = []
        for row_idx, line in enumerate(f):
            identifier = line.rstrip("\n").rsplit(",", 1)[-1]
            label = identifier_to_label.get(identifier)
            if label is not None:
                indices.append(row_idx)
                labels.append(label)
                identifiers.append(identifier)

    if not indices:
        raise ValueError(
            "No rows in the identifier CSV matched any label in "
            "tcga_file_mapping.csv. Check that the two files cover the "
            "same cohort."
        )

    if max(indices) >= embeddings.shape[0]:
        raise ValueError(
            f"identifier CSV has row index {max(indices)} but embeddings "
            f"has only {embeddings.shape[0]} rows — mismatched files?"
        )

    selected = embeddings[np.asarray(indices)]
    return selected, labels, identifiers


def _build_merged_csv(
    embeddings_path: str | Path,
    identifier_csv: str | Path,
    mapping_csv: str | Path,
    out_csv: str | Path,
) -> None:
    """Materialize ``{patient_id, cohort, embedding_json}`` rows to ``out_csv``.

    The embedding is serialized as a JSON list of floats so the whole table
    fits in a single CSV and can be loaded through the standard BaseDataset
    YAML path without bespoke readers.
    """
    embeddings = np.load(embeddings_path).astype(np.float32)
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D (n_samples, embed_dim), got shape "
            f"{embeddings.shape}"
        )
    identifier_to_cohort = _identifier_to_cohort(mapping_csv)
    with open(identifier_csv) as f:
        header = f.readline().rstrip("\n").split(",")
        if header[-1] != "identifier":
            raise ValueError(
                f"{identifier_csv}: last column must be 'identifier', got "
                f"{header[-1]!r}"
            )
        rows: List[Tuple[str, str, str]] = []
        for row_idx, line in enumerate(f):
            identifier = line.rstrip("\n").rsplit(",", 1)[-1]
            cohort = identifier_to_cohort.get(identifier)
            if cohort is None:
                continue
            if row_idx >= embeddings.shape[0]:
                raise ValueError(
                    f"identifier CSV has row index {row_idx} but embeddings "
                    f"has only {embeddings.shape[0]} rows — mismatched files?"
                )
            emb_json = json.dumps(embeddings[row_idx].tolist())
            rows.append((identifier, cohort, emb_json))

    if not rows:
        raise ValueError(
            "No rows in the identifier CSV matched any label in the mapping "
            "CSV. Check that the two files cover the same cohort."
        )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "cohort", "embedding_json"])
        writer.writerows(rows)
    logger.info("Wrote merged TCGA embedding CSV to %s (%d rows)",
                out_csv, len(rows))


class TCGARNASeqEmbeddingDataset(BaseDataset):
    """BaseDataset wrapper over pre-computed BulkRNABert TCGA embeddings.

    The dataset materializes (once, lazily) a merged CSV with columns
    ``patient_id``, ``cohort``, ``embedding_json`` under ``root``, then
    loads it through the standard :class:`~pyhealth.datasets.BaseDataset`
    YAML path. Pair with :class:`~pyhealth.tasks.TCGACancerClassification5Cohort`:

    .. code-block:: python

        from pyhealth.datasets import TCGARNASeqEmbeddingDataset
        from pyhealth.tasks import TCGACancerClassification5Cohort

        dataset = TCGARNASeqEmbeddingDataset(
            root="/path/to/tcga",
            embeddings_path="/path/to/tcga_discrete_refinit_step600.npy",
            identifier_csv="/path/to/tcga_preprocessed.csv",
            mapping_csv="/path/to/tcga_file_mapping.csv",
        )
        samples = dataset.set_task(TCGACancerClassification5Cohort())

    Args:
        root: Directory used to host the generated merged CSV
            (``tcga_rnaseq_embedding.csv``). A pre-existing CSV in this
            directory will be reused.
        embeddings_path: Path to the ``.npy`` embedding matrix. Ignored
            if the merged CSV already exists at ``root``.
        identifier_csv: Path to ``tcga_preprocessed.csv`` (only the
            ``identifier`` column is consumed).
        mapping_csv: Path to ``tcga_file_mapping.csv``.
        tables: Tables to load. Defaults to ``["rnaseq_embedding"]`` which
            matches the shipped YAML.
        dataset_name: Optional override for :attr:`BaseDataset.dataset_name`.
        config_path: Optional override for the YAML config. Defaults to
            ``configs/tcga_rnaseq_embedding.yaml`` shipped alongside this
            module.
        **kwargs: Forwarded to :meth:`BaseDataset.__init__`.
    """

    def __init__(
        self,
        root: str | Path,
        embeddings_path: Optional[str | Path] = None,
        identifier_csv: Optional[str | Path] = None,
        mapping_csv: Optional[str | Path] = None,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        merged_csv = root_path / MERGED_CSV_NAME
        if not merged_csv.exists():
            if embeddings_path is None or identifier_csv is None or mapping_csv is None:
                raise ValueError(
                    f"{merged_csv} does not exist and one of "
                    "embeddings_path / identifier_csv / mapping_csv was not "
                    "provided. Pass all three to build the merged CSV."
                )
            _build_merged_csv(
                embeddings_path, identifier_csv, mapping_csv, merged_csv
            )

        if config_path is None:
            config_path = str(
                Path(__file__).parent / "configs" / "tcga_rnaseq_embedding.yaml"
            )

        super().__init__(
            root=str(root_path),
            tables=list(tables or ["rnaseq_embedding"]),
            dataset_name=dataset_name or "tcga_rnaseq_embedding",
            config_path=config_path,
            **kwargs,
        )

    @property
    def default_task(self) -> TCGACancerClassification5Cohort:
        """The 5-cohort classification task is the canonical pairing."""
        return TCGACancerClassification5Cohort()


def load_tcga_cancer_classification_5cohort(
    embeddings_path: str | Path,
    identifier_csv: str | Path,
    mapping_csv: str | Path,
    dataset_name: str = "TCGA_BulkRNABert_Embeddings",
) -> InMemorySampleDataset:
    """Build an in-memory SampleDataset of pre-computed BulkRNABert embeddings.

    Thin shortcut around :class:`TCGARNASeqEmbeddingDataset` that bypasses
    the event-dataframe materialization and returns samples directly. Used
    by unit tests and short experiments where the BaseDataset caching
    overhead is not justified.

    Args:
        embeddings_path: Path to the ``.npy`` file holding all TCGA
            embeddings in the same row order as ``identifier_csv``. Shape
            ``(n_total_samples, embed_dim)``.
        identifier_csv: ``tcga_preprocessed.csv`` used during pre-training.
            Its last column must be ``identifier``.
        mapping_csv: ``tcga_file_mapping.csv`` from the TCGA GDC metadata
            dump.
        dataset_name: Name attached to the returned dataset.

    Returns:
        An :class:`InMemorySampleDataset` whose samples are dicts
        ``{"patient_id": identifier, "embedding": np.ndarray,
        "label": int}`` with schema defined by
        :class:`~pyhealth.tasks.TCGACancerClassification5Cohort`.
    """
    embeddings = np.load(embeddings_path).astype(np.float32)
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D (n_samples, embed_dim), got shape "
            f"{embeddings.shape}"
        )
    logger.info(
        "Loaded BulkRNABert embeddings from %s (shape=%s)",
        embeddings_path,
        embeddings.shape,
    )

    identifier_to_label = _build_identifier_to_label(mapping_csv)
    logger.info(
        "Built identifier->label map from %s (%d entries)",
        mapping_csv,
        len(identifier_to_label),
    )

    selected_embeddings, labels, identifiers = _select_rows(
        embeddings, identifier_csv, identifier_to_label
    )
    logger.info(
        "Selected %d samples across %d cohorts",
        len(labels),
        len(set(labels)),
    )

    task = TCGACancerClassification5Cohort()
    samples = [
        {
            "patient_id": identifier,
            "embedding": selected_embeddings[i],
            "label": int(labels[i]),
        }
        for i, identifier in enumerate(identifiers)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name=dataset_name,
        task_name=task.task_name,
        in_memory=True,
    )


def stratified_split_indices(
    labels: List[int] | np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(train_idx, val_idx, test_idx)`` stratified by label.

    PyHealth's built-in ``split_by_sample`` does not preserve class
    proportions; this helper mirrors the per-class shuffle used in the
    reference BulkRNABert downstream pipeline and extends it to a
    three-way split so that early-stopping and best-checkpoint selection
    can run on a held-out validation set disjoint from the final test set
    (matching the dominant PyHealth example convention of
    ``split_by_patient([0.7, 0.1, 0.2])``).

    Per-class allocation::

        n_test  = max(1, int(n * test_ratio))  if n >= 2 else 0
        n_val   = max(1, int(n * val_ratio))   if (n - n_test) >= 2 else 0
        n_train = n - n_test - n_val

    The ``max(1, ...)`` rule guarantees every class that has at least two
    samples is represented in the test split; the same rule is then applied
    to val after reserving test. Classes with fewer than two samples are
    routed entirely to train (val may be empty for very small classes),
    which is consistent with the 2-way behavior this function replaced.

    Args:
        labels: Integer labels of shape ``(n_samples,)``.
        val_ratio: Fraction of each class routed to the validation split.
        test_ratio: Fraction of each class routed to the test split.
        seed: Seed for the NumPy generator used in the per-class shuffle.

    Returns:
        Three int64 arrays of indices into the original sample order, each
        shuffled. ``val_idx`` may be empty if every class has fewer than
        two samples left after the test split.
    """
    labels_arr = np.asarray(labels)
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError(
            f"val_ratio ({val_ratio}) and test_ratio ({test_ratio}) must be "
            "non-negative"
        )
    if val_ratio + test_ratio >= 1:
        raise ValueError(
            f"val_ratio + test_ratio ({val_ratio + test_ratio}) must be < 1 "
            "so that train receives a positive fraction"
        )
    rng = np.random.default_rng(seed)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for lbl in np.unique(labels_arr):
        idx = np.where(labels_arr == lbl)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(n * test_ratio)) if n >= 2 else 0
        n_val = max(1, int(n * val_ratio)) if (n - n_test) >= 2 else 0
        test_idx.extend(idx[:n_test].tolist())
        val_idx.extend(idx[n_test:n_test + n_val].tolist())
        train_idx.extend(idx[n_test + n_val:].tolist())

    train_arr = np.array(train_idx, dtype=np.int64)
    val_arr = np.array(val_idx, dtype=np.int64)
    test_arr = np.array(test_idx, dtype=np.int64)
    rng.shuffle(train_arr)
    rng.shuffle(val_arr)
    rng.shuffle(test_arr)
    return train_arr, val_arr, test_arr


__all__ = [
    "TCGARNASeqEmbeddingDataset",
    "load_tcga_cancer_classification_5cohort",
    "stratified_split_indices",
]
