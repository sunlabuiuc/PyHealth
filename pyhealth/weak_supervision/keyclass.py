# Author: Amin Torabi
# Paper title: Classifying Unstructured Clinical Notes via Automatic Weak Supervision
# Paper link: https://arxiv.org/abs/2206.12088v2
# Description: Note cleaning, automatic phrase mining, labeling-function generation,
# categorical weak-label aggregation, and confident-sample selection.

from dataclasses import dataclass
from functools import lru_cache
import math
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_DEID_PLACEHOLDER = re.compile(r"\[\*\*.*?\*\*\]", re.DOTALL)
_DIGITS = re.compile(r"\d+")
# CountVectorizer's default tokens are 2+ word characters. Stopwords that can never
# equal such a token ("a", "don't") have no effect, so dropping them changes nothing
# except silencing sklearn's inconsistent-stopword warning.
_TOKEN = re.compile(r"\w\w+")


@lru_cache(maxsize=1)
def nltk_english_stopwords():
    """Return NLTK's English stopwords, as the authors pass to CountVectorizer.

    The word list is NLTK corpus data, installed separately from the package:
    ``python -m nltk.downloader stopwords``. It differs across NLTK releases.
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


def clean_note(text):
    """Remove MIMIC de-identification placeholders and digits, then lowercase.

    Placeholders such as [**Hospital1 18**] otherwise become phrases like
    "hospital1" or "namepattern1 md" that fire on nearly every note. The authors'
    MIMIC loader also lowercases and strips digits before mining and encoding.
    """
    text = _DEID_PLACEHOLDER.sub(" ", text)
    text = _DIGITS.sub(" ", text.lower())
    return " ".join(text.split())


@dataclass(frozen=True)
class LabelingFunction:
    phrase: str
    category_index: int
    similarity: float


class KeywordLabeler:
    """Assign each training phrase to its nearest description; keep top K/class.

    descriptions is an insertion-ordered mapping of stable category IDs to text.
    category_ids_ fixes all output columns; IDs need not be consecutive integers.

    Mining follows the authors' code: NLTK English stopwords and unicode accent
    stripping inside CountVectorizer, so stopwords are dropped before n-grams are
    formed and negations ("no", "not") are not modeled. A float min_df is the
    paper's strict rule (document frequency > min_df * notes); an int is an
    inclusive document count. By default each category keeps its own top K, as the
    paper describes. equalize_top_k caps K at the smallest nonempty category pool,
    as in the authors' min_topk option; with raw BlueBERT vectors one category can
    win only a few phrases, which then shrinks every category's rule set.
    """

    def __init__(
        self,
        encoder,
        top_k=300,
        ngram_range=(1, 3),
        min_df=0.001,
        equalize_top_k=False,
        max_candidates=None,
    ):
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.encoder = encoder
        self.top_k = top_k
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.equalize_top_k = equalize_top_k
        self.max_candidates = max_candidates

    def _min_count(self, n_texts):
        if isinstance(self.min_df, float):
            if not 0 < self.min_df < 1:
                raise ValueError("A float min_df must lie in (0, 1)")
            return math.floor(self.min_df * n_texts) + 1
        if isinstance(self.min_df, int) and self.min_df >= 1:
            return self.min_df
        raise ValueError("min_df must be a float in (0, 1) or a positive int")

    def fit(self, texts, descriptions):
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

    def transform(self, texts):
        """Return int64 [notes, LFs], with -1 abstain or a category column index."""
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


def _validate_votes(votes, n_classes):
    votes = np.asarray(votes)
    if n_classes < 2 or votes.ndim != 2 or not np.issubdtype(votes.dtype, np.integer):
        raise ValueError("Need an integer [notes, LFs] matrix and >= 2 classes")
    if ((votes < -1) | (votes >= n_classes)).any():
        raise ValueError("Votes must be -1 or category indices")
    return votes


def snorkel_memory_gb(n_rows, n_lfs, n_classes):
    """Estimate peak host memory (GB) for Snorkel's LabelModel.fit.

    Snorkel one-hot encodes the votes into a dense float64 [rows, LFs * classes]
    matrix, then forms its float64 Gram matrix and a float32 copy on the host before
    moving it to the device. Memory therefore grows with the square of
    LFs * classes and a GPU does not avoid it; the device also needs the float32 copy.
    """
    width = n_lfs * n_classes
    return (n_rows * width * 8 + width * width * (8 + 4)) / 1e9


def aggregate_votes(
    votes,
    n_classes,
    method="majority",
    seed=0,
    max_memory_gb=None,
    device="cpu",
    **fit_kwargs,
):
    """Return (categorical probabilities, covered mask).

    Majority baseline is normalized vote counts, preserving ties. Uncovered
    rows receive a uniform distribution and MUST be excluded from initial
    training via the returned mask. Snorkel is optional and never silently
    replaced by the baseline. Fits only on covered training rows, and refuses
    to start when snorkel_memory_gb exceeds max_memory_gb.
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
                "Install the experiments extra (uv sync --extra experiments) "
                "to use Snorkel"
            ) from exc
        model = LabelModel(cardinality=n_classes, device=device, verbose=False)
        model.fit(L_train=votes[covered], seed=seed, **fit_kwargs)
        probabilities[covered] = model.predict_proba(L=votes[covered])
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Categorical weak labels must sum to one per note")
    return probabilities.astype(np.float32), covered


def confident_balanced_mask(probabilities, covered, max_num):
    """Select the most confident covered notes for initial training.

    Mirrors the authors' get_balanced_data_mask with uniform class balance: for
    each category keep the int(max_num / categories) covered notes with the
    highest weak probability for it, then take the union. Uncovered notes are
    never selected because their weak label is only the uniform prior.
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


def vote_diagnostics(votes, n_classes):
    """Coverage and cross-category disagreement (may reflect true comorbidity)."""
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
