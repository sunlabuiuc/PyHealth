# Author: Amin Torabi
# Paper title: Classifying Unstructured Clinical Notes via Automatic Weak Supervision
# Paper link: https://arxiv.org/abs/2206.12088v2
# Description: Note cleaning, automatic phrase mining, labeling-function generation,
# categorical weak-label aggregation, and confident-sample selection.

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_DEID_PLACEHOLDER = re.compile(r"\[\*\*.*?\*\*\]", re.DOTALL)
_DIGITS = re.compile(r"\d+")
# CountVectorizer's default tokens are 2+ word characters. Stopwords that can never
# equal such a token ("a", "don't") have no effect, so dropping them changes nothing
# except silencing sklearn's inconsistent-stopword warning.
_TOKEN = re.compile(r"\w\w+")


class TextEncoder(Protocol):
    """Any object that embeds texts, such as :class:`ClinicalTextEncoder`."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embeds texts.

        Args:
            texts: Texts to embed.

        Returns:
            A finite float array of shape ``[len(texts), dimension]``.
        """
        ...


@lru_cache(maxsize=1)
def nltk_english_stopwords() -> tuple[str, ...]:
    """Returns NLTK's English stopwords, as the KeyClass authors use them.

    The list is NLTK corpus data, installed separately from the package with
    ``python -m nltk.downloader stopwords``, and it differs across NLTK releases.
    Entries that can never match a ``CountVectorizer`` token (single letters,
    contractions such as "don't") are dropped. The result is cached.

    Returns:
        Sorted, unique stopwords.

    Raises:
        ImportError: If NLTK is not installed.
        LookupError: If the NLTK stopwords corpus has not been downloaded.
    """
    try:
        from nltk.corpus import stopwords
    except ImportError as exc:
        raise ImportError(
            'KeywordLabeler needs NLTK; pip install "nltk~=3.9.1"'
        ) from exc
    try:
        words = stopwords.words("english")
    except LookupError as exc:
        raise LookupError(
            "NLTK stopwords corpus not found; run: python -m nltk.downloader stopwords"
        ) from exc
    return tuple(sorted({w for w in words if _TOKEN.fullmatch(w)}))


def clean_note(text: str) -> str:
    """Removes MIMIC de-identification placeholders and digits, then lowercases.

    Placeholders such as ``[**Hospital1 18**]`` otherwise become phrases like
    "hospital1" or "namepattern1 md" that fire on nearly every note. The
    authors' MIMIC loader also lowercases and strips digits before mining and
    encoding.

    Args:
        text: A raw clinical note.

    Returns:
        The cleaned note with whitespace collapsed to single spaces.

    Example:
        >>> clean_note("Seen at [**Hospital1 18**]. ASA 81 mg")
        'seen at . asa mg'
    """
    text = _DEID_PLACEHOLDER.sub(" ", text)
    text = _DIGITS.sub(" ", text.lower())
    return " ".join(text.split())


@dataclass(frozen=True)
class LabelingFunction:
    """A keyword rule: vote for one category whenever a phrase occurs.

    Attributes:
        phrase: The mined n-gram that triggers the vote.
        category_index: Column of the category it votes for, in
            ``KeywordLabeler.category_ids_`` order.
        similarity: Cosine similarity between the phrase and the category
            description embeddings.
    """

    phrase: str
    category_index: int
    similarity: float


class KeywordLabeler:
    """Builds KeyClass labeling functions from category descriptions alone.

    Mines frequent 1-3 grams from training notes, embeds them together with the
    category descriptions, assigns each phrase to its most similar description,
    and keeps the top ``top_k`` phrases per category as keyword rules. No gold
    labels are used.

    Mining follows the authors' code: NLTK English stopwords and unicode accent
    stripping inside ``CountVectorizer``, so stopwords are dropped before
    n-grams are formed and negations ("no", "not") are not modeled. By default
    each category keeps its own top ``top_k``, as the paper describes.

    Args:
        encoder: Object with ``encode(texts) -> [len(texts), dimension]`` array.
        top_k: Maximum rules kept per category.
        ngram_range: Smallest and largest n-gram length to mine.
        min_df: A float in (0, 1) keeps phrases in strictly more than
            ``min_df * notes`` notes (the paper's rule); an int is an inclusive
            minimum note count.
        equalize_top_k: If True, caps ``top_k`` at the smallest nonempty
            category pool (the authors' ``min_topk`` option). With raw BlueBERT
            vectors one category can win only a few phrases, which then shrinks
            every category's rule set.
        max_candidates: If set, ``fit`` raises before encoding when mining finds
            more candidate phrases than this.

    Attributes:
        category_ids_: Category IDs in output column order, set by ``fit``.
        lfs_: The fitted :class:`LabelingFunction` rules.
        min_count_: Minimum note count implied by ``min_df``.
        effective_top_k_: Rules kept per category after any equalization.
        vectorizer_: The fitted ``CountVectorizer``.

    Raises:
        ValueError: If ``top_k`` is not a positive integer.

    Example:
        >>> labeler = KeywordLabeler(encoder, top_k=300, min_df=0.001)
        >>> labeler.fit(train_notes, {"circulatory": "heart cardiac ..."})
        >>> votes = labeler.transform(train_notes)  # [notes, rules], -1 abstains
    """

    def __init__(
        self,
        encoder: TextEncoder,
        top_k: int = 300,
        ngram_range: tuple[int, int] = (1, 3),
        min_df: float | int = 0.001,
        equalize_top_k: bool = False,
        max_candidates: int | None = None,
    ) -> None:
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.encoder = encoder
        self.top_k = top_k
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.equalize_top_k = equalize_top_k
        self.max_candidates = max_candidates

    def _min_count(self, n_texts: int) -> int:
        """Converts ``min_df`` into an inclusive minimum note count.

        Args:
            n_texts: Number of training notes.

        Returns:
            The smallest note count a phrase needs to be kept.

        Raises:
            ValueError: If ``min_df`` is neither a float in (0, 1) nor a
                positive int.
        """
        if isinstance(self.min_df, float):
            if not 0 < self.min_df < 1:
                raise ValueError("A float min_df must lie in (0, 1)")
            return math.floor(self.min_df * n_texts) + 1
        if isinstance(self.min_df, int) and self.min_df >= 1:
            return self.min_df
        raise ValueError("min_df must be a float in (0, 1) or a positive int")

    def fit(
        self, texts: Iterable[str], descriptions: Mapping[Any, str]
    ) -> "KeywordLabeler":
        """Mines phrases from training notes and builds the keyword rules.

        Args:
            texts: Training notes, already cleaned (see :func:`clean_note`).
            descriptions: Ordered mapping from category ID to description text.
                Its order fixes the output columns.

        Returns:
            The fitted labeler.

        Raises:
            ValueError: If there are no notes, fewer than two categories, an
                empty description, too many candidate phrases, or invalid
                encoder output.
        """
        texts = list(texts)
        if not texts or len(descriptions) < 2:
            raise ValueError("Need training notes and at least two categories")
        if any(not isinstance(x, str) or not x.strip() for x in descriptions.values()):
            raise ValueError("Category descriptions must be nonempty strings")
        self.category_ids_ = tuple(descriptions)
        self.min_count_ = self._min_count(len(texts))
        self.vectorizer_ = CountVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_count_,
            stop_words=list(nltk_english_stopwords()),
            strip_accents="unicode",
        )
        self.vectorizer_.fit(texts)
        phrases = self.vectorizer_.get_feature_names_out()
        if self.max_candidates is not None and len(phrases) > self.max_candidates:
            raise ValueError(
                f"{len(phrases)} candidate phrases exceed max_candidates="
                f"{self.max_candidates}; raise min_df for small samples"
            )
        vectors = np.asarray(
            self.encoder.encode(list(phrases) + list(descriptions.values()))
        )
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(phrases) + len(descriptions)
            or not np.isfinite(vectors).all()
            or (np.linalg.norm(vectors, axis=1) == 0).any()
        ):
            raise ValueError(
                "Encoder must return finite, nonzero vectors, one per text"
            )
        scores = cosine_similarity(vectors[: len(phrases)], vectors[len(phrases) :])
        owners = scores.argmax(axis=1)
        pools = np.bincount(owners, minlength=len(descriptions))
        k = self.top_k
        if self.equalize_top_k and (pools > 0).any():
            k = min(k, int(pools[pools > 0].min()))
        self.effective_top_k_ = k
        self.lfs_ = []
        for category in range(len(descriptions)):
            indices = np.flatnonzero(owners == category)
            ranked = sorted(indices, key=lambda i: (-scores[i, category], phrases[i]))
            self.lfs_.extend(
                LabelingFunction(str(phrases[i]), category, float(scores[i, category]))
                for i in ranked[:k]
            )
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        """Applies the fitted rules to notes.

        Args:
            texts: Notes, cleaned the same way as the training notes.

        Returns:
            An int64 array of shape ``[notes, rules]``: -1 where a rule abstains,
            otherwise the column index of the category it votes for.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not hasattr(self, "lfs_"):
            raise RuntimeError("Fit KeywordLabeler on training notes first")
        texts = list(texts)
        votes = np.full((len(texts), len(self.lfs_)), -1, dtype=np.int64)
        if not texts or not self.lfs_:
            return votes
        columns = [self.vectorizer_.vocabulary_[lf.phrase] for lf in self.lfs_]
        categories = np.array([lf.category_index for lf in self.lfs_], dtype=np.int64)
        hits = self.vectorizer_.transform(texts)[:, columns].tocoo()
        present = hits.data > 0
        rows, lf_columns = hits.row[present], hits.col[present]
        votes[rows, lf_columns] = categories[lf_columns]
        return votes


def _validate_votes(votes: np.ndarray, n_classes: int) -> np.ndarray:
    """Checks a vote matrix and returns it as an array.

    Args:
        votes: Integer array of shape ``[notes, rules]`` with -1 or a category.
        n_classes: Number of categories.

    Returns:
        The votes as a NumPy array.

    Raises:
        ValueError: If the shape, dtype or values are invalid.
    """
    votes = np.asarray(votes)
    if n_classes < 2 or votes.ndim != 2 or not np.issubdtype(votes.dtype, np.integer):
        raise ValueError("Need an integer [notes, LFs] matrix and >= 2 classes")
    if ((votes < -1) | (votes >= n_classes)).any():
        raise ValueError("Votes must be -1 or category indices")
    return votes


def snorkel_memory_gb(n_rows: int, n_lfs: int, n_classes: int) -> float:
    """Estimates peak host memory for Snorkel's ``LabelModel.fit``.

    Snorkel one-hot encodes the votes into a dense float64
    ``[rows, rules * classes]`` matrix, then forms its float64 Gram matrix and a
    float32 copy on the host before moving it to the device. Memory therefore
    grows with the square of ``rules * classes``, and a GPU does not avoid it.

    Args:
        n_rows: Number of notes passed to Snorkel.
        n_lfs: Number of labeling functions.
        n_classes: Number of categories.

    Returns:
        Estimated memory in gigabytes.

    Example:
        >>> round(snorkel_memory_gb(945, 5700, 19))  # 300 rules x 19 categories
        142
    """
    width = n_lfs * n_classes
    return (n_rows * width * 8 + width * width * (8 + 4)) / 1e9


def aggregate_votes(
    votes: np.ndarray,
    n_classes: int,
    method: str = "majority",
    seed: int = 0,
    max_memory_gb: float | None = None,
    device: str = "cpu",
    **fit_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Turns rule votes into one categorical weak label per note.

    The majority baseline normalizes vote counts, preserving ties. Snorkel fits
    a categorical label model on covered notes only; it is never silently
    replaced by the baseline. Notes no rule covers get a uniform distribution
    and must be excluded from initial training through the returned mask.

    Args:
        votes: Integer array of shape ``[notes, rules]`` from
            :meth:`KeywordLabeler.transform`.
        n_classes: Number of categories.
        method: ``"majority"`` or ``"snorkel"``.
        seed: Random seed for Snorkel.
        max_memory_gb: If set, refuses Snorkel when :func:`snorkel_memory_gb`
            exceeds it.
        device: Snorkel ``LabelModel`` device, for example ``"cuda:0"``.
        **fit_kwargs: Passed to Snorkel's ``LabelModel.fit``.

    Returns:
        A tuple ``(probabilities, covered)``: float32 ``[notes, n_classes]``
        rows summing to one, and a bool ``[notes]`` mask of notes with at least
        one vote.

    Raises:
        ValueError: If the votes or method are invalid, or Snorkel lacks rules
            or covered notes.
        MemoryError: If Snorkel would exceed ``max_memory_gb``.
        ImportError: If Snorkel is requested but not installed.

    Example:
        >>> p, covered = aggregate_votes(np.array([[0, 1, 1], [-1, -1, -1]]), 2)
        >>> p.astype(float).round(2).tolist(), covered.tolist()
        ([[0.33, 0.67], [0.5, 0.5]], [True, False])
    """
    votes = _validate_votes(votes, n_classes)
    covered = (votes != -1).any(axis=1)
    probabilities = np.full((len(votes), n_classes), 1 / n_classes)
    if method not in ("majority", "snorkel"):
        raise ValueError("method must be majority or snorkel")
    if method == "snorkel" and max_memory_gb is not None:
        needed = snorkel_memory_gb(int(covered.sum()), votes.shape[1], n_classes)
        if needed > max_memory_gb:
            raise MemoryError(
                f"Snorkel needs about {needed:.1f} GB for {votes.shape[1]} LFs x "
                f"{n_classes} classes (limit {max_memory_gb} GB); use fewer LFs "
                "or majority voting"
            )
    if method == "majority":
        counts = np.stack([(votes == c).sum(axis=1) for c in range(n_classes)], axis=1)
        probabilities[covered] = counts[covered] / counts[covered].sum(
            axis=1, keepdims=True
        )
    else:
        if votes.shape[1] < 3 or not covered.any():
            raise ValueError("Snorkel needs >= 3 LFs and covered training rows")
        try:
            from snorkel.labeling.model import LabelModel
        except ImportError as exc:
            raise ImportError(
                'Snorkel is not installed; pip install "snorkel>=0.10,<0.11"'
            ) from exc
        model = LabelModel(cardinality=n_classes, device=device, verbose=False)
        model.fit(L_train=votes[covered], seed=seed, **fit_kwargs)
        probabilities[covered] = model.predict_proba(L=votes[covered])
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Categorical weak labels must sum to one per note")
    return probabilities.astype(np.float32), covered


def confident_balanced_mask(
    probabilities: np.ndarray, covered: np.ndarray, max_num: int
) -> np.ndarray:
    """Selects the most confident covered notes for initial training.

    Mirrors the authors' ``get_balanced_data_mask`` with uniform class balance:
    for each category keep the ``max_num // categories`` covered notes with the
    highest weak probability for it, then take the union. Uncovered notes are
    never selected because their weak label is only the uniform prior.

    Args:
        probabilities: Weak labels of shape ``[notes, categories]``.
        covered: Bool mask of shape ``[notes]`` from :func:`aggregate_votes`.
        max_num: Total note budget, split evenly across categories.

    Returns:
        A bool mask of shape ``[notes]`` marking the selected notes.

    Raises:
        ValueError: If the shapes do not match or ``max_num`` is smaller than
            the number of categories.
    """
    probabilities = np.asarray(probabilities)
    covered = np.asarray(covered, dtype=bool)
    if probabilities.ndim != 2 or covered.shape != (len(probabilities),):
        raise ValueError("Need [notes, categories] probabilities and a covered mask")
    if max_num < probabilities.shape[1]:
        raise ValueError("max_num must allow at least one note per category")
    per_class = max_num // probabilities.shape[1]
    candidates = np.flatnonzero(covered)
    mask = np.zeros(len(probabilities), dtype=bool)
    for category in range(probabilities.shape[1]):
        order = np.argsort(-probabilities[candidates, category], kind="stable")
        mask[candidates[order[:per_class]]] = True
    return mask


def vote_diagnostics(votes: np.ndarray, n_classes: int) -> dict[str, Any]:
    """Summarizes how much of the corpus the rules cover.

    Notes voted for by several categories may reflect true comorbidity, not
    rule conflict.

    Args:
        votes: Integer array of shape ``[notes, rules]``.
        n_classes: Number of categories.

    Returns:
        A dict with ``"coverage"`` (share of notes with any vote),
        ``"cross_category_fraction"`` (share voted for by more than one
        category) and ``"lf_coverage"`` (per-rule share of notes it fires on).
    """
    votes = _validate_votes(votes, n_classes)
    present = np.stack([(votes == c).any(axis=1) for c in range(n_classes)], axis=1)
    return {
        "coverage": float(present.any(axis=1).mean()) if len(votes) else 0.0,
        "cross_category_fraction": float((present.sum(axis=1) > 1).mean())
        if len(votes)
        else 0.0,
        "lf_coverage": (votes != -1).mean(axis=0)
        if len(votes)
        else np.zeros(votes.shape[1]),
    }
