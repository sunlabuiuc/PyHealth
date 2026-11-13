"""TCGA bulk RNA-seq survival dataset.

Thin adapter over the artifacts produced by the BulkRNABert reference
repository (pull_data.py + train.py): per-patient 256-d embeddings from the
frozen encoder, plus survival labels joined from the TCGA clinical tables.
The CSV is small (~20 MB for 11k patients); production use points ``root``
at the directory holding ``artifacts/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

_CSV_NAME = "tcga_rna_seq.csv"
_EMBED_DIM = 256

def prepare_tcga_rna_seq(
    root: str,
    artifacts_dir: str | None = None,
    embedding: str = "repo",
) -> Path:
    """Builds ``tcga_rna_seq.csv`` from the reference repo's artifacts.

    Args:
        root: Directory to write the CSV to.
        artifacts_dir: Directory holding ``samples.tsv``,
            ``survival_labels.tsv``, and ``embeddings_{embedding}.npy``.
            Defaults to ``root/artifacts``.
        embedding: Which cached embedding to use: ``"repo"`` (Haiku
            checkpoint) or ``"hf"`` (Hugging Face port).

    Returns:
        Path to the written CSV.

    Raises:
        FileNotFoundError: If a required artifact file is missing.
        ValueError: If the joined table has duplicate patients.
    """
    root_path = Path(root)
    artifacts = Path(artifacts_dir) if artifacts_dir else root_path / "artifacts"

    samples_path = artifacts / "samples.tsv"
    labels_path = artifacts / "survival_labels.tsv"
    emb_path = artifacts / f"embeddings_{embedding}.npy"
    for path in (samples_path, labels_path, emb_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run the reference repo's pull_data.py and "
                f"train.py first, or pass artifacts_dir=..."
            )

    samples = pd.read_csv(samples_path, sep="\t")
    labels = pd.read_csv(
        labels_path,
        sep="\t",
        keep_default_na=False,
        dtype={"exclusion_reason": str},
    )
    labels = labels[labels.exclusion_reason.eq("")]
    embeddings = np.load(emb_path)
    if embeddings.ndim != 2 or embeddings.shape[1] != _EMBED_DIM:
        raise ValueError(
            f"embeddings_{embedding}.npy must be (N, {_EMBED_DIM}), "
            f"got {embeddings.shape}"
        )
    if embeddings.shape[0] != len(samples):
        raise ValueError(
            f"Embedding row count ({embeddings.shape[0]}) does not match "
            f"samples.tsv row count ({len(samples)})"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")
    
    samples = samples.assign(row=np.arange(len(samples)))

    frame = samples.merge(labels, on="patient_id", how="inner")
    if frame.empty:
        raise ValueError("No overlap between samples and survival labels")

    # Drop Solid Tissue Normal samples (matches the reference repo's training).
    if "sample_type" in frame.columns:
        before = len(frame)
        frame = frame[frame.sample_type != "Solid Tissue Normal"]
        if len(frame) < before:
            logger.info("Dropped %d Solid Tissue Normal samples", before - len(frame))

    # After dropping Solid Tissue Normal, some patients may still have
    # multiple non-normal samples (Primary + Metastatic, Primary + Recurrent,
    # etc.). The paper treats each patient as one input, so we keep the first
    # sample (in source-file order, which matches the reference repo's rule).
    n_before = len(frame)
    frame = frame.drop_duplicates(subset="patient_id", keep="first")
    n_deduped = n_before - len(frame)
    if n_deduped > 0:
        logger.info(
            "Dropped %d duplicate samples from patients with multiple "
            "non-normal specimens (kept first)",
            n_deduped,
        )

    # Validate survival records.
    if frame.patient_id.isna().any() or (frame.patient_id == "").any():
        raise ValueError("Missing patient_id in survival records")
    if frame.cohort.isna().any() or (frame.cohort == "").any():
        raise ValueError("Missing cohort in survival records")
    if not np.isfinite(frame.duration_days.astype(float)).all():
        raise ValueError("Survival times contain non-finite values")
    if not frame.event.isin([0, 1, True, False]).all():
        raise ValueError("Event must be 0 or 1")

    # Negative durations are data errors; the reference repo already
    # excludes these, but be defensive.
    neg_mask = frame.duration_days.astype(float) < 0
    if neg_mask.any():
        logger.warning(
            "Excluding %d patients with negative survival times",
            int(neg_mask.sum()),
        )
        frame = frame[~neg_mask]

    # Zero durations occur when death or last contact is on the diagnosis
    # date. They break the Cox loss and contribute no comparable pairs to
    # the C-index. Clamp to a small positive floor (1 day), matching the
    # METABRIC task's min_duration_months=0.01 convention.
    zero_mask = frame.duration_days.astype(float) == 0
    if zero_mask.any():
        logger.info(
            "Clamping %d patients with zero survival time to 1 day",
            int(zero_mask.sum()),
        )
        frame = frame.assign(
            duration_days=np.where(
                zero_mask, 1.0, frame.duration_days.astype(float),
            )
        )

    # Final sanity check.
    if (frame.duration_days.astype(float) <= 0).any():
        raise ValueError(
            "Survival times must be positive after clamping"
        )

    vecs = embeddings[frame.row.to_numpy()].astype(np.float32)

    out = pd.DataFrame({
        "patient_id": frame.patient_id.to_numpy(),
        "cohort": frame.cohort.to_numpy(),
        "duration": frame.duration_days.astype(float).to_numpy(),
        "event": frame.event.astype(int).to_numpy(),
        "embedding": [json.dumps(vec.tolist()) for vec in vecs],
    })

    root_path.mkdir(parents=True, exist_ok=True)
    csv_path = root_path / _CSV_NAME
    out.to_csv(csv_path, index=False)
    logger.info("Wrote %d patients to %s", len(out), csv_path)
    return csv_path


class TCGARNASeqDataset(BaseDataset):
    """TCGA bulk RNA-seq survival dataset, from frozen BulkRNABert embeddings.

    Each patient is one row: TCGA cohort, survival time in days, event
    indicator, and a 256-d embedding from the BulkRNABert encoder (the mean
    over genes of the last transformer layer, cached by the reference
    repository). Designed for :class:`~pyhealth.tasks.TCGASurvivalPrediction`
    and :class:`~pyhealth.models.BulkRNABert`, which accepts cached
    embeddings via ``use_cached_embeddings=True``.

    Args:
        root: Directory holding ``tcga_rna_seq.csv``. If absent, it is
            built from ``root/artifacts/`` (or ``artifacts_dir``) via
            :func:`prepare_tcga_rna_seq`.
        embedding: ``"repo"`` or ``"hf"``, used only when building the CSV.
        artifacts_dir: Override for where the reference artifacts live.
        **kwargs: Passed to :class:`~pyhealth.datasets.BaseDataset`
            (e.g. ``cache_dir``, ``dev``).

    Example:
        >>> from pyhealth.datasets import TCGARNASeqDataset
        >>> ds = TCGARNASeqDataset(root="artifacts_dir")  # doctest: +SKIP
        >>> ds.stats()  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        embedding: str = "repo",
        artifacts_dir: str | None = None,
        **kwargs,
    ) -> None:
        csv_path = Path(root) / _CSV_NAME
        if not csv_path.exists():
            logger.info("%s not found, building it from artifacts", csv_path)
            prepare_tcga_rna_seq(root, artifacts_dir=artifacts_dir, embedding=embedding)
        if config_path is None:
            config_path = str(
                Path(__file__).parent / "configs" / "tcga_rna_seq.yaml"
            )
        super().__init__(
            root=root,
            tables=tables or ["tcga_rna_seq"],
            dataset_name=dataset_name or "tcga_rna_seq",
            config_path=config_path,
            **kwargs,
        )