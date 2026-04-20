"""CBERT-lite neural IR baseline for EHR evidence retrieval.

This baseline is a lightweight reproduction of the "CBERT" retrieval
style used as a comparison point in:

    M. Ahsan et al. "Retrieving Evidence from EHRs with LLMs:
    Possibilities and Challenges." Proceedings of Machine Learning
    Research, 2024.

Given a fixed risk-factor sentence per condition, the baseline encodes
both the risk-factor sentence and each candidate sentence from a note,
then returns the top-K candidates ranked by cosine similarity. The
result is *extractive* — no text is generated.

Author:
    Arnab Karmakar (arnabk3@illinois.edu)
"""
import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

from pyhealth.models import BaseModel
from pyhealth.tasks.evidence_retrieval_mimic3 import split_sentences

logger = logging.getLogger(__name__)


EncoderCallable = Callable[[List[str]], List[List[float]]]


_STOPWORDS: frozenset = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "will", "with", "after", "before", "not",
    "no", "any", "all", "more", "most", "some", "such", "than", "then",
    "there", "these", "they", "also", "into", "over", "under", "about",
    "above", "below", "between", "during", "each", "few", "other", "same",
    "so", "only", "own", "very", "per", "up", "down", "out", "off", "if",
})


DEFAULT_RISK_FACTOR_SENTENCES: Dict[str, str] = {
    "intracranial hemorrhage": (
        "Anticoagulation after recent neurosurgery or trauma is a risk "
        "factor for intracranial hemorrhage."
    ),
    "stroke": (
        "Atrial fibrillation, carotid stenosis, and hypertension are risk "
        "factors for ischemic stroke."
    ),
    "pneumonia": (
        "Fever, productive cough, and focal consolidation on chest imaging "
        "are signs of pneumonia."
    ),
}


@dataclass
class RankedSentence:
    """One scored candidate sentence returned by the baseline.

    Attributes:
        note_id (str): Source note identifier.
        condition (str): Diagnosis condition being queried.
        sentence (str): The extracted candidate sentence.
        score (float): Cosine similarity to the risk-factor sentence.
        rank (int): 1-indexed rank within the note for this condition.
    """

    note_id: str
    condition: str
    sentence: str
    score: float
    rank: int


class HashingEncoder:
    """Token-hash encoder used as the default, dependency-free backend.

    The encoder maps each whitespace-separated alphanumeric token to a
    deterministic bucket in a fixed-size vector. It intentionally
    approximates a bag-of-words Bio/Clinical-BERT sentence embedding so
    unit tests can run without downloading any weights. Swap in a real
    sentence encoder in production via the ``encoder`` argument to
    :class:`CBERTLiteRetriever`.
    """

    def __init__(self, dim: int = 2048) -> None:
        """Initialize the encoder.

        Args:
            dim (int): Output embedding dimension. Defaults to 2048.

        Raises:
            ValueError: If ``dim`` is not a positive integer.
        """
        if dim <= 0:
            raise ValueError("dim must be a positive integer.")
        self.dim = dim

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Embed ``texts`` into ``dim``-dimensional float vectors.

        Args:
            texts (List[str]): Sentences to encode.

        Returns:
            List[List[float]]: One embedding per input sentence.
        """
        embeddings: List[List[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            tokens = [
                t.lower()
                for t in re.findall(r"[a-zA-Z0-9]+", text or "")
                if len(t) >= 4 and t.lower() not in _STOPWORDS
            ]
            if not tokens:
                embeddings.append(vector)
                continue
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                bucket = int(digest, 16) % self.dim
                vector[bucket] += 1.0
            # L2 normalize so cosine similarity is just a dot product.
            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]
            embeddings.append(vector)
        return embeddings


class CBERTLiteRetriever(BaseModel):
    """Neural IR baseline that ranks note sentences by cosine similarity.

    Args:
        dataset: The :class:`~pyhealth.datasets.SampleDataset` returned
            by :class:`~pyhealth.tasks.EvidenceRetrievalMIMIC3`.
        encoder (EncoderCallable): Callable that accepts a list of
            sentences and returns a list of embeddings. Defaults to
            :class:`HashingEncoder`, which needs no external weights.
            Replace with a call to a public clinical/biomedical
            encoder (e.g. Bio_ClinicalBERT) for production use.
        risk_factor_sentences (Dict[str, str]): Per-condition reference
            sentence used as the query. Keys should be lower-case
            conditions.
        top_k (int): Number of candidate sentences to return per note.
        decision_threshold (float): Cosine similarity threshold above
            which the top-1 match is treated as a positive yes/no
            decision. Defaults to ``0.2``.

    Expected inputs (forward):
        - ``note_text``: iterable of raw note strings, one per batch
          element.
        - ``condition``: iterable of condition queries, one per batch
          element.
        - ``is_positive`` (optional): iterable of binary labels for loss
          computation.

    Examples:
        >>> from pyhealth.datasets import SyntheticEHRNotesDataset
        >>> from pyhealth.tasks import EvidenceRetrievalMIMIC3
        >>> from pyhealth.models import CBERTLiteRetriever
        >>> dataset = SyntheticEHRNotesDataset(root="./synthetic_notes")
        >>> task = EvidenceRetrievalMIMIC3()
        >>> samples = dataset.set_task(task)
        >>> model = CBERTLiteRetriever(dataset=samples, top_k=2)
        >>> ranked = model.retrieve_evidence(
        ...     note_text="fever, productive cough, RLL consolidation",
        ...     condition="pneumonia",
        ...     note_id="n0007",
        ... )
    """

    def __init__(
        self,
        dataset: Any = None,
        encoder: Optional[EncoderCallable] = None,
        risk_factor_sentences: Optional[Dict[str, str]] = None,
        top_k: int = 3,
        decision_threshold: float = 0.2,
    ) -> None:
        """Initialize the CBERT-lite retriever.

        Raises:
            ValueError: If ``top_k`` is not a positive integer or
                ``decision_threshold`` is outside ``[0, 1]``.
        """
        super().__init__(dataset=dataset)
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not 0.0 <= decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be in [0, 1].")
        self.encoder: EncoderCallable = encoder or HashingEncoder()
        self.risk_factor_sentences: Dict[str, str] = {
            k.lower(): v
            for k, v in (risk_factor_sentences or DEFAULT_RISK_FACTOR_SENTENCES).items()
        }
        self.top_k = top_k
        self.decision_threshold = decision_threshold
        self.mode = "binary"

    # ------------------------------------------------------------------
    # Core retrieval API
    # ------------------------------------------------------------------
    def _query_sentence(self, condition: str) -> str:
        """Return the reference sentence for ``condition`` (fallback to itself)."""
        return self.risk_factor_sentences.get(
            condition.lower(),
            f"Signs and risk factors of {condition}.",
        )

    def retrieve_evidence(
        self,
        note_text: str,
        condition: str,
        note_id: str = "",
    ) -> List[RankedSentence]:
        """Return the top-K candidate sentences for one note.

        Args:
            note_text (str): Raw note text.
            condition (str): Target diagnosis condition.
            note_id (str): Source note identifier to thread through the
                returned snippets.

        Returns:
            List[RankedSentence]: Scored and ranked candidate sentences
                in descending-score order (at most ``top_k`` entries).
        """
        candidates = split_sentences(note_text)
        if not candidates:
            return []
        query = self._query_sentence(condition)
        embeddings = self.encoder([query] + candidates)
        if not embeddings:
            return []
        query_vec = embeddings[0]
        scored = []
        for sentence, vec in zip(candidates, embeddings[1:]):
            scored.append((sentence, _cosine(query_vec, vec)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top = scored[: self.top_k]
        return [
            RankedSentence(
                note_id=note_id,
                condition=condition,
                sentence=sentence,
                score=float(score),
                rank=rank + 1,
            )
            for rank, (sentence, score) in enumerate(top)
        ]

    def retrieve_evidence_batch(
        self,
        notes: List[str],
        conditions: List[str],
        note_ids: Optional[List[str]] = None,
    ) -> List[List[RankedSentence]]:
        """Vectorized wrapper around :meth:`retrieve_evidence`.

        Args:
            notes (List[str]): One note per element.
            conditions (List[str]): One condition per element.
            note_ids (Optional[List[str]]): Optional note identifiers,
                defaulting to empty strings.

        Returns:
            List[List[RankedSentence]]: Per-note ranked snippets.
        """
        if len(notes) != len(conditions):
            raise ValueError("notes and conditions must have equal length.")
        ids = note_ids or [""] * len(notes)
        return [
            self.retrieve_evidence(n, c, i)
            for n, c, i in zip(notes, conditions, ids)
        ]

    # ------------------------------------------------------------------
    # PyHealth forward API
    # ------------------------------------------------------------------
    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Run the baseline and return PyHealth-style outputs.

        Args:
            **kwargs: Must contain ``note_text`` and ``condition`` as
                sequences of strings with equal length. When
                ``is_positive`` is provided, a binary cross-entropy
                loss is computed against it.

        Returns:
            Dict[str, torch.Tensor]: Keys ``logit``, ``y_prob``, and
            optionally ``loss`` and ``y_true``.
        """
        note_text = _flatten_text(kwargs.get("note_text", []))
        condition = _flatten_text(kwargs.get("condition", []))
        if len(note_text) != len(condition):
            raise ValueError(
                "note_text and condition must have the same batch size."
            )
        ids = _flatten_text(kwargs.get("note_id", [""] * len(note_text)))
        if len(ids) != len(note_text):
            ids = [""] * len(note_text)

        per_note = self.retrieve_evidence_batch(note_text, condition, ids)
        probs: List[float] = []
        for ranked in per_note:
            if not ranked:
                probs.append(0.0)
                continue
            top_score = max(r.score for r in ranked)
            if top_score >= self.decision_threshold:
                probs.append(min(1.0, 0.5 + (top_score - self.decision_threshold)))
            else:
                probs.append(max(0.0, top_score))
        prob_tensor = torch.tensor(
            probs, dtype=torch.float, device=self.device
        ).unsqueeze(-1)
        clamped = prob_tensor.clamp(min=1e-6, max=1 - 1e-6)
        logits = torch.log(clamped / (1.0 - clamped))

        outputs: Dict[str, torch.Tensor] = {"logit": logits, "y_prob": prob_tensor}
        outputs["snippets"] = per_note  # type: ignore[assignment]

        y_true_raw = kwargs.get("is_positive")
        if y_true_raw is not None:
            y_true = _coerce_binary_tensor(y_true_raw, self.device).unsqueeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y_true
            )
            outputs["y_true"] = y_true
            outputs["loss"] = loss
        return outputs


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    # Vectors from HashingEncoder are already unit-normalized, but we
    # guard against callers that pass unnormalized custom embeddings.
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return num / (norm_a * norm_b)


def _flatten_text(value: Any) -> List[str]:
    """Normalize ``value`` into a list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, torch.Tensor):
        return [str(v) for v in value.tolist()]
    return [str(value)]


def _coerce_binary_tensor(value: Any, device: torch.device) -> torch.Tensor:
    """Convert ``value`` (list/tensor/sequence) to a float binary tensor."""
    if isinstance(value, torch.Tensor):
        return value.float().to(device)
    if isinstance(value, (list, tuple)):
        return torch.tensor([float(v) for v in value], dtype=torch.float, device=device)
    return torch.tensor([float(value)], dtype=torch.float, device=device)
