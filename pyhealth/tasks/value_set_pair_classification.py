"""Text-pair value set completion task, with hybrid title retrieval.

Author: Wenchang Qi (NetID: qi14)
Paper: Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value
    Set Authoring (Mukherjee et al., 2026)
Paper link: https://arxiv.org/abs/2604.14616
Description: The cross-encoder formulation of RASC (paper section 4.5). Each
    retrieved candidate code becomes one (value set title, candidate display
    name) text pair with a binary label, the input of
    :class:`~pyhealth.models.RASCCrossEncoder`. The module also builds a
    hybrid retrieval index (SapBERT title search fused with BM25 over
    titles; not in the paper).

    :class:`~pyhealth.tasks.ValueSetCompletion` emits precomputed embeddings
    for the paper's MLP. This task reads the same index layout and emits
    text, so it loads no embedding matrix.
"""

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import torch

from ..data import Patient
from .base_task import BaseTask
from .value_set_completion import CODE_SYSTEM_VOCAB, Encoder, normalize_code_system

if TYPE_CHECKING:
    from ..datasets import BaseDataset

logger = logging.getLogger(__name__)

TEXT_FEATURES = ("title_text", "candidate_text")
# Optional numeric inputs, for late fusion in the model.
FUSION_FEATURES = ("code_system", "score")
# The layout written by pyhealth.tasks.build_retrieval_index; the last file
# holds the fingerprint of the others.
_INDEX_FILES = (
    "titles.parquet",
    "title_embeddings.npy",
    "neighbors.parquet",
    "source_codes.parquet",
    "displays.parquet",
    "display_embeddings.npy",
    "description_embeddings.npy",
    "meta.json",
)


_HYBRID_NOTE = (
    "Neighbors are in fused-rank order, not score order. Read this index "
    "with ValueSetPairClassification; ValueSetCompletion assumes score order."
)


def _bm25_scores(
    queries: Sequence[str], sources: Sequence[str], k1: float = 1.5, b: float = 0.75
) -> Any:
    """Okapi BM25 scores of every source title for every query title.

    Titles are lowercased and split into word tokens. A query term counts
    once however often the query repeats it.

    Args:
        queries: Query titles.
        sources: Source titles; document statistics come from these.
        k1: Term-frequency saturation.
        b: Length normalization strength.

    Returns:
        A ``scipy.sparse`` CSR matrix of shape ``(len(queries),
        len(sources))``; 0 where a pair shares no term.
    """
    from scipy import sparse
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    counts = vectorizer.fit_transform(sources).tocsr().astype(np.float32)
    n_sources = counts.shape[0]
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    norm = k1 * (1 - b + b * lengths / max(float(lengths.mean()), 1e-12))
    document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log1p((n_sources - document_frequency + 0.5) / (document_frequency + 0.5))
    weights = counts.tocoo()
    weights.data = (
        idf[weights.col] * weights.data * (k1 + 1) / (weights.data + norm[weights.row])
    )
    present = vectorizer.transform(queries)
    present.data[:] = 1.0
    return sparse.csr_matrix(present @ weights.tocsr().T, dtype=np.float32)


def _ranks(scores: torch.Tensor) -> torch.Tensor:
    """0-based rank of each column within its row, highest score first."""
    order = scores.argsort(dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(scores.shape[1]).expand_as(order))
    return ranks


def _hybrid_neighbors(
    queries: np.ndarray,
    sources: np.ndarray,
    query_ids: list[str],
    source_ids: list[str],
    query_titles: Sequence[str],
    source_titles: Sequence[str],
    max_k: int,
    rrf_k: int = 60,
    chunk: int = 1024,
) -> pl.DataFrame:
    """Reciprocal rank fusion of dense and BM25 title retrieval.

    A source's fused score is ``1 / (rrf_k + dense_rank) + 1 / (rrf_k +
    bm25_rank)``, ranks starting at 1; a source sharing no term with the
    query gets no BM25 part. The stored ``score`` stays the dense inner
    product, so the score input means the same under both retrievers; only
    which neighbors are kept, and their order, changes.

    Args:
        queries: Query vectors, one row per ``query_ids`` entry.
        sources: Source vectors, one row per ``source_ids`` entry.
        query_ids: Query value set ids.
        source_ids: Source value set ids.
        query_titles: Query titles.
        source_titles: Source titles.
        max_k: Neighbors kept per query, excluding the query itself.
        rrf_k: Rank-fusion constant; larger flattens rank differences.
        chunk: Queries scored per matrix product.

    Returns:
        Columns ``oid``, ``neighbor``, ``score`` (dense), and ``position``
        (0 for the best fused rank).
    """
    source_t = torch.from_numpy(sources)
    source_row = {oid: j for j, oid in enumerate(source_ids)}
    lexical = _bm25_scores(query_titles, source_titles)
    k = min(max_k, len(source_ids))
    rows: dict[str, list] = {"oid": [], "neighbor": [], "score": [], "position": []}
    for start in range(0, len(query_ids), chunk):
        ids = query_ids[start : start + chunk]
        dense = torch.from_numpy(queries[start : start + chunk]) @ source_t.T
        bm25 = torch.from_numpy(lexical[start : start + chunk].toarray())
        fused = 1.0 / (rrf_k + 1 + _ranks(dense))
        fused += (bm25 > 0) / (rrf_k + 1 + _ranks(bm25))
        own = [(i, source_row[oid]) for i, oid in enumerate(ids) if oid in source_row]
        if own:
            fused[tuple(zip(*own))] = -1.0
        fused_scores, indices = torch.topk(fused, k, dim=1)
        for i, oid in enumerate(ids):
            kept = [
                j
                for f, j in zip(fused_scores[i].tolist(), indices[i].tolist())
                if f > 0
            ]
            for position, j in enumerate(kept):
                rows["oid"].append(oid)
                rows["neighbor"].append(source_ids[j])
                rows["score"].append(float(dense[i, j]))
                rows["position"].append(position)
    return pl.DataFrame(
        rows,
        schema={
            "oid": pl.String,
            "neighbor": pl.String,
            "score": pl.Float64,
            "position": pl.Int64,
        },
    )


def build_hybrid_retrieval_index(
    dataset: "BaseDataset",
    encoder: Encoder,
    output_dir: str,
    max_k: int = 40,
    source_splits: Sequence[str] | None = None,
    rrf_k: int = 60,
) -> str:
    """Builds a retrieval index from dense and BM25 title search combined.

    Not in the paper, which retrieves by title embedding alone and names
    retrieval coverage as its binding constraint. Dense retrieval can miss a
    value set that shares an exact term with the query, such as a drug name
    or an abbreviation; BM25 ranks those first. The two rankings are merged
    by reciprocal rank fusion at the same ``max_k`` budget.

    Writes the files of :func:`~pyhealth.tasks.build_retrieval_index`, which
    builds the dense-only index, so :class:`ValueSetPairClassification` reads
    either. Run once per corpus, not per value set.

    A hybrid index is for :class:`ValueSetPairClassification` only. Its
    neighbors are stored in fused-rank order while ``score`` stays the dense
    similarity, so scores do not fall with position.
    :class:`~pyhealth.tasks.ValueSetCompletion` assumes they do and keeps
    the first neighbor's score for a code several neighbors share, which on
    a hybrid index is not always the highest. ``meta.json`` records
    ``"retriever": "hybrid"`` and this caveat under ``"note"``.

    Args:
        dataset: A :class:`~pyhealth.datasets.VSACDataset`.
        encoder: Maps texts to unit-norm vectors, e.g.
            :func:`~pyhealth.tasks.value_set_completion.sapbert_encoder`.
        output_dir: Directory for the index files; created if missing.
        max_k: Largest number of neighbors a task may use.
        source_splits: Splits that may be retrieved; ``None`` for all.
        rrf_k: Rank-fusion constant; larger flattens rank differences.

    Returns:
        ``output_dir``, to pass to :class:`ValueSetPairClassification`.

    Raises:
        ValueError: If ``max_k`` or ``rrf_k`` is not positive, no value set
            has a title, or no source remains.

    Examples:
        >>> from pyhealth.tasks import build_hybrid_retrieval_index
        >>> from pyhealth.tasks.value_set_completion import sapbert_encoder
        >>> index_dir = build_hybrid_retrieval_index(  # doctest: +SKIP
        ...     dataset, sapbert_encoder(), "vsac_hybrid_index", max_k=40
        ... )
    """
    if max_k < 1:
        raise ValueError(f"max_k must be positive, got {max_k}.")
    if rrf_k < 1:
        raise ValueError(f"rrf_k must be positive, got {rrf_k}.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    events = dataset.global_event_df
    valuesets = (
        events.filter(pl.col("event_type") == "valuesets")
        .select(
            pl.col("patient_id").alias("oid"),
            pl.col("valuesets/title").fill_null("").str.strip_chars().alias("title"),
            pl.col("valuesets/split").fill_null("").alias("split"),
            pl.col("valuesets/description")
            .fill_null("")
            .str.strip_chars()
            .alias("description"),
        )
        .filter(pl.col("title") != "")
        .sort("oid")
        .collect()
    )
    if valuesets.is_empty():
        raise ValueError("No value set has a title to index.")
    is_source = (
        pl.lit(True)
        if source_splits is None
        else pl.col("split").is_in(list(source_splits))
    )
    sources = valuesets.filter(is_source)
    if sources.is_empty():
        raise ValueError(f"No value set has a split in {source_splits}.")

    title_vecs = np.asarray(encoder(valuesets["title"].to_list()), dtype=np.float32)
    descriptions = valuesets["description"].to_list()
    described = [i for i, text in enumerate(descriptions) if text]
    description_vecs = np.zeros_like(title_vecs)
    if described:
        description_vecs[described] = encoder([descriptions[i] for i in described])
    row = {oid: i for i, oid in enumerate(valuesets["oid"].to_list())}
    source_ids = sources["oid"].to_list()
    neighbors = _hybrid_neighbors(
        title_vecs,
        title_vecs[[row[oid] for oid in source_ids]],
        valuesets["oid"].to_list(),
        source_ids,
        valuesets["title"].to_list(),
        sources["title"].to_list(),
        max_k,
        rrf_k=rrf_k,
    )

    source_codes = (
        events.filter(pl.col("event_type") == "codes")
        .select(
            pl.col("patient_id").alias("oid"),
            pl.col("codes/code").alias("code"),
            pl.col("codes/system").fill_null("").alias("system"),
            pl.col("codes/display").fill_null("").alias("display"),
            pl.col("codes/rank").cast(pl.Int64).alias("rank"),
        )
        # Only sources some query can retrieve supply candidates.
        .filter(pl.col("oid").is_in(neighbors["neighbor"].unique().implode()))
        .sort("oid", "rank")
        .collect()
    )
    displays = source_codes["display"].unique(maintain_order=True)
    display_vecs = np.asarray(encoder(displays.to_list()), dtype=np.float32)
    display_table = pl.DataFrame({"display": displays}).with_row_index("row")
    source_codes = source_codes.join(display_table, on="display").select(
        "oid", "code", "system", pl.col("row").alias("display_row"), "rank"
    )

    valuesets.drop("description").with_row_index("row").write_parquet(
        out / "titles.parquet"
    )
    np.save(out / "title_embeddings.npy", title_vecs)
    neighbors.write_parquet(out / "neighbors.parquet")
    source_codes.write_parquet(out / "source_codes.parquet")
    display_table.write_parquet(out / "displays.parquet")
    np.save(out / "display_embeddings.npy", display_vecs)
    np.save(out / "description_embeddings.npy", description_vecs)

    digest = hashlib.sha256()
    for name in _INDEX_FILES[:-1]:
        digest.update((out / name).read_bytes())
    meta = {
        "max_k": max_k,
        "retriever": "hybrid",
        "note": _HYBRID_NOTE,
        "rrf_k": rrf_k,
        "source_splits": None if source_splits is None else list(source_splits),
        "n_queries": valuesets.height,
        "n_sources": len(source_ids),
        "n_displays": len(displays),
        "n_descriptions": len(described),
        "fingerprint": digest.hexdigest(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Built hybrid retrieval index at {out}: {meta}")
    return str(out)


@dataclass
class _PairIndex:
    """Index tables held in memory by each worker process; no embeddings."""

    neighbors: dict[str, list[tuple[str, float]]]
    source_codes: dict[str, list[tuple[str, str, int, int]]]
    title_row: dict[str, int]
    titles: list[str]
    displays: list[str]


# Loaded once per process; keeping it off the task keeps vars(task), the task
# cache key, small and stable.
_LOADED: dict[str, _PairIndex] = {}


def _load_index(index_dir: str, fingerprint: str) -> _PairIndex:
    """Loads an index's neighbor, code and text tables once per process.

    Args:
        index_dir: An index directory.
        fingerprint: The index's content fingerprint, part of the cache key.

    Returns:
        The loaded tables.
    """
    key = f"{index_dir}:{fingerprint}"
    if key not in _LOADED:
        path = Path(index_dir)
        neighbors: dict[str, list[tuple[str, float]]] = {}
        for oid, neighbor, score in (
            pl.read_parquet(path / "neighbors.parquet")
            .sort("oid", "position")
            .select("oid", "neighbor", "score")
            .iter_rows()
        ):
            neighbors.setdefault(oid, []).append((neighbor, score))
        source_codes: dict[str, list[tuple[str, str, int, int]]] = {}
        for oid, code, system, display_row, rank in (
            pl.read_parquet(path / "source_codes.parquet")
            .sort("oid", "rank")
            .iter_rows()
        ):
            source_codes.setdefault(oid, []).append((code, system, display_row, rank))
        titles = pl.read_parquet(path / "titles.parquet").sort("row")
        displays = pl.read_parquet(path / "displays.parquet").sort("row")
        _LOADED[key] = _PairIndex(
            neighbors=neighbors,
            source_codes=source_codes,
            title_row=dict(zip(titles["oid"], titles["row"])),
            titles=titles["title"].to_list(),
            displays=displays["display"].to_list(),
        )
    return _LOADED[key]


def _one_hot(system: str) -> np.ndarray:
    """One-hot encodes a code system URI over ``CODE_SYSTEM_VOCAB``."""
    vec = np.zeros(len(CODE_SYSTEM_VOCAB), dtype=np.float32)
    vec[CODE_SYSTEM_VOCAB.index(normalize_code_system(system))] = 1.0
    return vec


class ValueSetPairClassification(BaseTask):
    """Labels (value set title, candidate code name) text pairs as in or out.

    Works with :class:`~pyhealth.datasets.VSACDataset` and
    :class:`~pyhealth.models.RASCCrossEncoder`.

    For each target value set, the ``k`` nearest value sets by title are
    read from a retrieval index, and the union of their codes, deduplicated
    on ``(code, system)`` with the highest retrieval score kept, is the
    candidate pool. Each candidate becomes one sample labeled 1 when its
    code is in the target's code set. Positives are rare, so the label uses
    ``pos_weight="balanced"`` (the paper's class-balanced BCE); fit it on
    the training split.

    The index comes from :func:`~pyhealth.tasks.build_retrieval_index`
    (title embeddings, as in the paper) or
    :func:`build_hybrid_retrieval_index` (title embeddings and BM25).

    ``max_codes_per_set`` keeps only each value set's first N codes in VSAC
    expansion order, for both labels and candidate pools. The default,
    1,000, reproduces an unpaged ``$expand`` call (the RASC benchmark);
    ``None`` uses complete expansions. Value sets with fewer than
    ``min_codes`` codes after the cap, or with no candidates, yield no
    samples.

    Inputs:

    - ``title_text``: the target value set's title.
    - ``candidate_text``: the candidate code's display name (the code itself
      when VSAC gives none).
    - ``code_system`` (opt-in): one-hot over ``CODE_SYSTEM_VOCAB``.
    - ``score`` (opt-in): title-embedding similarity of the best neighbor
      that supplied the candidate.

    Samples also carry ``code``, ``system``, and ``n_true_codes`` (the
    target's code count after the cap) for value-set-level metrics.

    Args:
        index_dir: A retrieval index directory.
        k: Number of retrieved value sets. At most the index's ``max_k``.
        max_codes_per_set: Codes kept per value set, or ``None`` for all.
        min_codes: Minimum codes for a target value set.
        fusion_features: Numeric inputs to emit besides the two texts, a
            subset of ``FUSION_FEATURES``. Default is none, the paper's
            cross-encoder input.
        target_splits: Splits whose value sets become targets.

    Raises:
        ValueError: If ``k`` is outside ``[1, max_k]`` of the index,
            ``fusion_features`` has unknown names, ``max_codes_per_set`` or
            ``min_codes`` is not positive, or ``target_splits`` is empty.
        FileNotFoundError: If ``index_dir`` has no ``meta.json``.

    Examples:
        >>> from pyhealth.datasets import VSACDataset
        >>> from pyhealth.models import RASCCrossEncoder
        >>> from pyhealth.tasks import ValueSetPairClassification
        >>> dataset = VSACDataset(root="/path/to/vsac_data")  # doctest: +SKIP
        >>> task = ValueSetPairClassification("vsac_index", k=10)  # doctest: +SKIP
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=dataset.manifest_split()
        ... )
        >>> model = RASCCrossEncoder(train)  # doctest: +SKIP
    """

    task_name: str = "ValueSetPairClassification"

    def __init__(
        self,
        index_dir: str,
        k: int = 10,
        max_codes_per_set: int | None = 1000,
        min_codes: int = 3,
        fusion_features: Sequence[str] = (),
        target_splits: Sequence[str] = ("train", "val", "test"),
    ) -> None:
        meta_path = Path(index_dir) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found; build a retrieval index first."
            )
        meta = json.loads(meta_path.read_text())
        if not 1 <= k <= meta["max_k"]:
            raise ValueError(f"k must be in [1, {meta['max_k']}], got {k}.")
        if set(fusion_features) - set(FUSION_FEATURES):
            raise ValueError(
                f"fusion_features must be a subset of {FUSION_FEATURES}, "
                f"got {tuple(fusion_features)}."
            )
        if max_codes_per_set is not None and max_codes_per_set < 1:
            raise ValueError("max_codes_per_set must be positive or None.")
        if min_codes < 1:
            raise ValueError(f"min_codes must be positive, got {min_codes}.")
        if not target_splits:
            raise ValueError("target_splits must not be empty.")
        self.index_dir = str(index_dir)
        self.fingerprint = meta["fingerprint"]
        self.k = k
        self.max_codes_per_set = max_codes_per_set
        self.min_codes = min_codes
        self.fusion_features = tuple(f for f in FUSION_FEATURES if f in fusion_features)
        self.target_splits = tuple(target_splits)
        self.input_schema = {
            **{name: "tensor" for name in self.fusion_features},
            **{name: "text" for name in TEXT_FEATURES},
        }
        self.output_schema = {"label": ("binary", {"pos_weight": "balanced"})}
        super().__init__()

    def _within_cap(self, rank: int) -> bool:
        """Whether a code at expansion position ``rank`` survives the cap."""
        return self.max_codes_per_set is None or rank < self.max_codes_per_set

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Builds one text-pair sample per candidate code of a value set.

        Args:
            patient: A value set from :class:`~pyhealth.datasets.VSACDataset`.

        Returns:
            Samples with the inputs, ``label``, ``code``, ``system``, and
            ``n_true_codes``; empty when the value set is not a target.
        """
        header = patient.get_events(event_type="valuesets")
        if not header or header[0]["split"] not in self.target_splits:
            return []
        index = _load_index(self.index_dir, self.fingerprint)
        oid = patient.patient_id
        if oid not in index.title_row:
            return []

        true_codes = [
            e["code"]
            for e in patient.get_events(event_type="codes")
            if self._within_cap(int(e["rank"]))
        ]
        if len(true_codes) < self.min_codes:
            return []
        # Labels match on the code alone, ignoring the system, as the
        # authors' build_dataset.py does, so the benchmark labels are kept.
        true_set = set(true_codes)

        # A code keeps its highest-scoring source. Hybrid neighbors arrive
        # in fused-rank order, not score order, so compare.
        pool: dict[tuple[str, str], tuple[float, int]] = {}
        for neighbor, score in index.neighbors.get(oid, [])[: self.k]:
            for code, system, display_row, rank in index.source_codes.get(neighbor, []):
                if not self._within_cap(rank):
                    break
                key = (code, system)
                if key not in pool or score > pool[key][0]:
                    pool[key] = (score, display_row)

        title = index.titles[index.title_row[oid]]
        samples = []
        for (code, system), (score, display_row) in pool.items():
            inputs: dict[str, Any] = {}
            if "code_system" in self.fusion_features:
                inputs["code_system"] = _one_hot(system)
            if "score" in self.fusion_features:
                inputs["score"] = np.array([score], dtype=np.float32)
            samples.append(
                {
                    "patient_id": oid,
                    "code": code,
                    "system": system,
                    "n_true_codes": len(true_set),
                    **inputs,
                    "title_text": title,
                    "candidate_text": index.displays[display_row],
                    "label": int(code in true_set),
                }
            )
        return samples
