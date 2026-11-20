"""TCGA bulk RNA-seq survival dataset.

Thin loader over cached 256-d BulkRNABert embeddings joined with TCGA
survival labels. The data is one CSV, ``tcga_rna_seq.csv``, hosted at
`cstein2/TCGABulkBertRNAEmbeddings
<https://huggingface.co/datasets/cstein2/TCGABulkBertRNAEmbeddings>`_.

The CSV has columns:

    patient_id : str    TCGA patient barcode, e.g. ``TCGA-XX-0001``
    cohort     : str    TCGA cohort, e.g. ``TCGA-BRCA``
    duration   : float  Survival time in days; positive
    event      : int    1 = death observed, 0 = censored
    embedding  : str    JSON-serialized list of 256 floats

Data provenance:

- **RNA-seq and survival labels** come from TCGA via the NCI Genomic
  Data Commons (GDC, https://portal.gdc.cancer.gov/). Only open-access
  tier data is used; no controlled-access fields are included.
- **256-d embeddings** are the mean over genes of the frozen BulkRNABert
  encoder's layer-4 output. Checkpoint and encoder code from Gélard et
  al., "BulkRNABert: Cancer prognosis from bulk RNA-seq based language
  models" (ML4H 2025, https://arxiv.org/abs/2502.06407), via the
  reference implementation at
  https://github.com/instadeepai/multiomics-open-research.

The dataset is a thin loader; it does not generate the CSV. To regenerate
the embeddings from raw TCGA, see the accompanying research repository's
``pull_data.py`` and encoder parity validation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

_CSV_NAME = "tcga_rna_seq.csv"

HF_TCGA_RNA_SEQ_REPO = "cstein2/TCGABulkBertRNAEmbeddings"
HF_TCGA_RNA_SEQ_REVISION = "c1cca7f308c5d92b3337c6c36bdb3e857ba7991e"


class TCGARNASeqDataset(BaseDataset):
    """TCGA bulk RNA-seq survival dataset, from frozen BulkRNABert embeddings.

    Each patient is one row: TCGA cohort, survival time in days, event
    indicator, and a 256-d embedding from the BulkRNABert encoder (the mean
    over genes of the last transformer layer, cached by the reference
    repository). Designed for :class:`~pyhealth.tasks.TCGASurvivalPrediction`
    and :class:`~pyhealth.models.BulkRNABert`.

    Data source:

    - **Hugging Face dataset** (cached 256-d BulkRNABert embeddings joined
      with TCGA survival labels):
      https://huggingface.co/datasets/cstein2/TCGABulkBertRNAEmbeddings
    - **Raw RNA-seq and clinical survival data**: NCI Genomic Data Commons,
      open-access tier — https://portal.gdc.cancer.gov/
    - **Embedding model**: Gélard et al., "BulkRNABert: Cancer prognosis
      from bulk RNA-seq based language models" (ML4H 2025),
      https://arxiv.org/abs/2502.06407

    Args:
        root: Directory holding ``tcga_rna_seq.csv``.
        **kwargs: Passed to :class:`~pyhealth.datasets.BaseDataset`
            (e.g. ``cache_dir``, ``dev``).
    Raises:
        FileNotFoundError: If ``tcga_rna_seq.csv`` is missing at ``root``.
            Use :meth:`from_huggingface` to download it automatically.
    Example:
        >>> from pyhealth.datasets import TCGARNASeqDataset
        >>> ds = TCGARNASeqDataset(root="artifacts_dir")  # doctest: +SKIP
        >>> ds.stats()  # doctest: +SKIP
    """

    HF_REPO_ID = HF_TCGA_RNA_SEQ_REPO
    HF_REVISION = HF_TCGA_RNA_SEQ_REVISION

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        csv_path = Path(root) / _CSV_NAME
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. Call "
                "TCGARNASeqDataset.from_huggingface() to download it, or place "
                "tcga_rna_seq.csv at this path yourself. See the class docstring "
                "for the expected schema."
            )
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "tcga_rna_seq.yaml")
        super().__init__(
            root=root,
            tables=tables or ["tcga_rna_seq"],
            dataset_name=dataset_name or "tcga_rna_seq",
            config_path=config_path,
            **kwargs,
        )

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str = HF_TCGA_RNA_SEQ_REPO,
        revision: str = HF_TCGA_RNA_SEQ_REVISION,
        cache_dir: str | Path | None = None,
        **kwargs,
    ) -> TCGARNASeqDataset:
        """Download ``tcga_rna_seq.csv`` from Hugging Face and load it.

        Args:
            repo_id: Hugging Face Dataset repo ID.
            revision: Commit hash or branch to pin.
            cache_dir: PyHealth processed-data cache directory. Separate
                from the Hugging Face download cache.
            **kwargs: Passed to :class:`TCGARNASeqDataset`.

        Returns:
            A loaded :class:`TCGARNASeqDataset`.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as err:
            raise RuntimeError(
                "from_huggingface requires huggingface_hub. Install it with "
                "`pip install huggingface_hub`."
            ) from err

        csv_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=_CSV_NAME,
                repo_type="dataset",
                revision=revision,
            )
        )
        return cls(root=str(csv_path.parent), cache_dir=cache_dir, **kwargs)
