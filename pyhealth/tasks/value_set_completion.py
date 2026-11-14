"""Retrieval-augmented value set completion task.

Author: Bryan Chance (NetID: bchance2)
Paper: Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value
    Set Authoring (Mukherjee et al., 2026)
Paper link: https://arxiv.org/abs/2604.14616
Description: RASC frames value set authoring as set completion. A title
    encoder retrieves the K most similar value sets, the union of their codes
    becomes the candidate pool, and a binary classifier decides which
    candidates belong to the target value set. This module builds the
    retrieval index once (:func:`build_retrieval_index`) and emits one sample
    per (value set, candidate) pair (:class:`ValueSetCompletion`), with the
    paper's 1,545-dim features split into four inputs. Retrieval and
    candidate pooling follow create_dataset/build_dataset.py in
    https://github.com/mukhes3/RASC.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import torch

from pyhealth.data import Patient

from .base_task import BaseTask

if TYPE_CHECKING:
    from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

SAPBERT = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
FEATURES = ("title_embedding", "candidate_embedding", "code_system", "score")
# Not in the paper; ablation input from VSAC's purpose text.
ALL_FEATURES = (*FEATURES, "description_embedding")
Encoder = Callable[[Sequence[str]], np.ndarray]

# From create_dataset/build_dataset.py in the RASC repository.
_SYSTEM_URIS = {
    "http://snomed.info/sct": "SNOMED-CT",
    "http://hl7.org/fhir/sid/icd-10-cm": "ICD-10-CM",
    "http://www.nlm.nih.gov/research/umls/rxnorm": "RxNorm",
    "http://loinc.org": "LOINC",
    "http://www.ama-assn.org/go/cpt": "CPT",
    "http://hl7.org/fhir/sid/icd-10-pcs": "ICD-10-PCS",
    "http://www.cms.gov/Medicare/Coding/ICD10": "ICD-10-PCS",
    "http://www.nlm.nih.gov/research/umls/hcpcs": "HCPCS",
}
CODE_SYSTEM_VOCAB = [
    "SNOMED-CT",
    "ICD-10-CM",
    "RxNorm",
    "LOINC",
    "CPT",
    "ICD-10-PCS",
    "HCPCS",
    "OTHER",
]


def normalize_code_system(uri: str) -> str:
    """Maps a FHIR code system URI to one of ``CODE_SYSTEM_VOCAB``.

    Args:
        uri: Code system URI from a VSAC expansion.

    Returns:
        The vocabulary name, or ``"OTHER"`` when the URI is not recognized.

    Examples:
        >>> normalize_code_system("http://loinc.org")
        'LOINC'
        >>> normalize_code_system("urn:oid:2.16.840.1.113883.6.238")
        'OTHER'
    """
    if uri in _SYSTEM_URIS:
        return _SYSTEM_URIS[uri]
    lowered = uri.lower()
    return next(
        (name for name in CODE_SYSTEM_VOCAB[:-1] if name.lower() in lowered),
        "OTHER",
    )


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


def sapbert_encoder(
    model_name: str = SAPBERT,
    device: str | None = None,
    batch_size: int = 256,
    max_length: int = 512,
) -> Encoder:
    """Returns a text encoder with SapBERT [CLS] pooling and L2 normalization.

    Matches the encoder the RASC authors load through sentence-transformers.

    Args:
        model_name: Hugging Face model name.
        device: Torch device. Defaults to CUDA when available.
        batch_size: Texts per forward pass.
        max_length: Token limit per text.

    Returns:
        A function mapping a list of texts to a float32 array of shape
        ``(len(texts), hidden_size)`` with unit-norm rows.

    Examples:
        >>> encode = sapbert_encoder()  # doctest: +SKIP
        >>> encode(["Type 2 diabetes"]).shape  # doctest: +SKIP
        (1, 768)
    """
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def encode(texts: Sequence[str]) -> np.ndarray:
        parts = []
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            cls = model(**batch).last_hidden_state[:, 0]
            parts.append(torch.nn.functional.normalize(cls, dim=-1).float().cpu())
        if not parts:
            return np.zeros((0, model.config.hidden_size), dtype=np.float32)
        return torch.cat(parts).numpy()

    return encode


def _top_neighbors(
    queries: np.ndarray,
    sources: np.ndarray,
    query_ids: list[str],
    source_ids: list[str],
    max_k: int,
    chunk: int = 1024,
) -> pl.DataFrame:
    """Exact inner-product search, as ``faiss.IndexFlatIP``; skips self-matches.

    Args:
        queries: Query vectors, one row per ``query_ids`` entry.
        sources: Source vectors, one row per ``source_ids`` entry.
        query_ids: Query value set ids.
        source_ids: Source value set ids.
        max_k: Neighbors kept per query, excluding the query itself.
        chunk: Queries scored per matrix product.

    Returns:
        Columns ``oid``, ``neighbor``, ``score``, and ``position`` (0 for the
        nearest), sorted by score within each query.
    """
    source_t = torch.from_numpy(sources)
    k = min(max_k + 1, len(source_ids))
    rows: dict[str, list] = {"oid": [], "neighbor": [], "score": [], "position": []}
    for start in range(0, len(query_ids), chunk):
        scores, indices = torch.topk(
            torch.from_numpy(queries[start : start + chunk]) @ source_t.T, k, dim=1
        )
        for offset, (row_scores, row_indices) in enumerate(
            zip(scores.tolist(), indices.tolist())
        ):
            oid = query_ids[start + offset]
            hits = [
                (source_ids[j], s)
                for s, j in zip(row_scores, row_indices)
                if source_ids[j] != oid
            ][:max_k]
            for position, (neighbor, score) in enumerate(hits):
                rows["oid"].append(oid)
                rows["neighbor"].append(neighbor)
                rows["score"].append(score)
                rows["position"].append(position)
    return pl.DataFrame(rows)


def build_retrieval_index(
    dataset: "BaseDataset",
    encoder: Encoder,
    output_dir: str,
    max_k: int = 40,
    source_splits: Sequence[str] | None = None,
) -> str:
    """Builds the title retrieval index and candidate tables for the task.

    Every value set with a title is a query. Value sets whose split is in
    ``source_splits`` (all, when ``None``) form the retrieval corpus, and
    their codes form the candidate pools. Titles and candidate display names
    are encoded once, so one index serves every ``k`` up to ``max_k`` and
    every ``max_codes_per_set``. Only codes of value sets that appear among
    some query's ``max_k`` neighbors are stored and encoded. Non-empty
    descriptions are encoded too, for the ``description_embedding`` input.

    Args:
        dataset: A :class:`~pyhealth.datasets.VSACDataset`.
        encoder: Maps texts to unit-norm vectors, e.g.
            :func:`~pyhealth.tasks.value_set_completion.sapbert_encoder`.
        output_dir: Directory for the index files; created if missing.
        max_k: Largest number of neighbors a task may use.
        source_splits: Splits that may be retrieved; ``None`` for all.

    Returns:
        ``output_dir``, to pass to :class:`ValueSetCompletion`.

    Raises:
        ValueError: If ``max_k`` is not positive, or no value set has a title,
            or no source remains.

    Examples:
        >>> from pyhealth.tasks import build_retrieval_index
        >>> index_dir = build_retrieval_index(  # doctest: +SKIP
        ...     dataset, sapbert_encoder(), "vsac_index", max_k=40
        ... )
    """
    if max_k < 1:
        raise ValueError(f"max_k must be positive, got {max_k}.")
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
    neighbors = _top_neighbors(
        title_vecs,
        title_vecs[[row[oid] for oid in source_ids]],
        valuesets["oid"].to_list(),
        source_ids,
        max_k,
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
    # Rows, not strings, keep the per-worker candidate table small.
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
        "source_splits": None if source_splits is None else list(source_splits),
        "n_queries": valuesets.height,
        "n_sources": len(source_ids),
        "n_displays": len(displays),
        "n_descriptions": len(described),
        "fingerprint": digest.hexdigest(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Built retrieval index at {out}: {meta}")
    return str(out)


@dataclass
class _RetrievalIndex:
    """Index tables held in memory by each worker process."""

    neighbors: dict[str, list[tuple[str, float]]]
    source_codes: dict[str, list[tuple[str, str, int, int]]]
    title_row: dict[str, int]
    title_vecs: np.ndarray
    display_vecs: np.ndarray
    description_vecs: np.ndarray | None


# Loaded once per process; keeping it off the task keeps vars(task), the task
# cache key, small and stable.
_LOADED: dict[str, _RetrievalIndex] = {}


def _load_index(index_dir: str, fingerprint: str) -> _RetrievalIndex:
    """Loads an index once per process; embeddings stay memory-mapped.

    Args:
        index_dir: Output of :func:`build_retrieval_index`.
        fingerprint: The index's content fingerprint, part of the cache key.

    Returns:
        The loaded index.
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
        titles = pl.read_parquet(path / "titles.parquet")
        description_path = path / "description_embeddings.npy"
        _LOADED[key] = _RetrievalIndex(
            neighbors=neighbors,
            source_codes=source_codes,
            title_row=dict(zip(titles["oid"], titles["row"])),
            title_vecs=np.load(path / "title_embeddings.npy", mmap_mode="r"),
            display_vecs=np.load(path / "display_embeddings.npy", mmap_mode="r"),
            description_vecs=(
                np.load(description_path, mmap_mode="r")
                if description_path.exists()
                else None
            ),
        )
    return _LOADED[key]


def _one_hot(system: str) -> np.ndarray:
    """One-hot encodes a code system URI over ``CODE_SYSTEM_VOCAB``.

    Args:
        system: Code system URI.

    Returns:
        A float32 vector with a single 1.
    """
    vec = np.zeros(len(CODE_SYSTEM_VOCAB), dtype=np.float32)
    vec[CODE_SYSTEM_VOCAB.index(normalize_code_system(system))] = 1.0
    return vec


class ValueSetCompletion(BaseTask):
    """Classifies retrieved candidate codes as in or out of a value set.

    Works with :class:`~pyhealth.datasets.VSACDataset`.

    For each target value set, the ``k`` nearest value sets by title are
    retrieved from the index built by :func:`build_retrieval_index`, and the
    union of their codes, deduplicated on ``(code, system)`` with the highest
    retrieval score kept, is the candidate pool. Each candidate becomes one
    sample labeled 1 when its code is in the target's code set. Positives
    are rare, so the label uses ``pos_weight="balanced"`` (the paper's
    class-balanced BCE); fit it on the training split.

    ``max_codes_per_set`` keeps only each value set's first N codes in VSAC
    expansion order, for both labels and candidate pools. The default, 1,000,
    reproduces an unpaged ``$expand`` call (the RASC benchmark); ``None`` uses
    complete expansions. Value sets with fewer than ``min_codes`` codes after
    the cap, or with no candidates, yield no samples.

    Inputs, which together form the paper's 1,545-dim vector with SapBERT:

    - ``title_embedding``: embedding of the target's title.
    - ``candidate_embedding``: embedding of the candidate's display name.
    - ``code_system``: one-hot over ``CODE_SYSTEM_VOCAB``.
    - ``score``: title similarity of the neighbor that supplied the
      candidate.
    - ``description_embedding`` (opt-in, not in the paper): embedding of the
      target's description; zeros when it has none.

    Samples also carry ``code``, ``system``, and ``n_true_codes`` (the
    target's code count after the cap) for value-set-level metrics.

    Args:
        index_dir: Output of :func:`build_retrieval_index`.
        k: Number of retrieved value sets. At most the index's ``max_k``.
        max_codes_per_set: Codes kept per value set, or ``None`` for all.
        min_codes: Minimum codes for a target value set.
        features: Input features to emit, a subset of ``ALL_FEATURES``.
            Defaults to the paper's four, ``FEATURES``.
        target_splits: Splits whose value sets become targets.

    Raises:
        ValueError: If ``k`` is outside ``[1, max_k]`` of the index,
            ``features`` is empty or has unknown names,
            ``max_codes_per_set`` or ``min_codes`` is not positive,
            ``target_splits`` is empty, or ``description_embedding`` is
            requested from an index built without descriptions.
        FileNotFoundError: If ``index_dir`` has no ``meta.json``.

    Examples:
        >>> from pyhealth.datasets import VSACDataset
        >>> from pyhealth.tasks import ValueSetCompletion
        >>> dataset = VSACDataset(root="/path/to/vsac_data")  # doctest: +SKIP
        >>> task = ValueSetCompletion("vsac_index", k=10)  # doctest: +SKIP
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=dataset.manifest_split()
        ... )
    """

    task_name: str = "ValueSetCompletion"

    def __init__(
        self,
        index_dir: str,
        k: int = 10,
        max_codes_per_set: int | None = 1000,
        min_codes: int = 3,
        features: Sequence[str] = FEATURES,
        target_splits: Sequence[str] = ("train", "val", "test"),
    ) -> None:
        meta_path = Path(index_dir) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found; run build_retrieval_index first."
            )
        meta = json.loads(meta_path.read_text())
        if not 1 <= k <= meta["max_k"]:
            raise ValueError(f"k must be in [1, {meta['max_k']}], got {k}.")
        unknown = set(features) - set(ALL_FEATURES)
        if not features or unknown:
            raise ValueError(f"features must be a non-empty subset of {ALL_FEATURES}.")
        if "description_embedding" in features and "n_descriptions" not in meta:
            raise ValueError("This index has no description embeddings; rebuild it.")
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
        self.features = tuple(f for f in ALL_FEATURES if f in features)
        self.target_splits = tuple(target_splits)
        self.input_schema = {name: "tensor" for name in self.features}
        self.output_schema = {"label": ("binary", {"pos_weight": "balanced"})}
        super().__init__()

    def _within_cap(self, rank: int) -> bool:
        """Whether a code at expansion position ``rank`` survives the cap."""
        return self.max_codes_per_set is None or rank < self.max_codes_per_set

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Builds one sample per candidate code of a value set.

        Args:
            patient: A value set from :class:`~pyhealth.datasets.VSACDataset`.

        Returns:
            Samples with the selected features, ``label``, ``code``,
            ``system``, and ``n_true_codes``; empty when the value set is not
            a target.
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

        # Neighbors are sorted by score, so the first sighting is the maximum.
        pool: dict[tuple[str, str], tuple[float, int]] = {}
        for neighbor, score in index.neighbors.get(oid, [])[: self.k]:
            for code, system, display_row, rank in index.source_codes.get(neighbor, []):
                if not self._within_cap(rank):
                    break
                pool.setdefault((code, system), (score, display_row))

        row = index.title_row[oid]
        title_vec = np.array(index.title_vecs[row])
        description_vec = (
            np.array(index.description_vecs[row])
            if "description_embedding" in self.features
            else None
        )
        samples = []
        for (code, system), (score, display_row) in pool.items():
            inputs = {
                "title_embedding": title_vec,
                "candidate_embedding": np.array(index.display_vecs[display_row]),
                "code_system": _one_hot(system),
                "score": np.array([score], dtype=np.float32),
                "description_embedding": description_vec,
            }
            samples.append(
                {
                    "patient_id": oid,
                    "code": code,
                    "system": system,
                    "n_true_codes": len(true_set),
                    **{name: inputs[name] for name in self.features},
                    "label": int(code in true_set),
                }
            )
        return samples
