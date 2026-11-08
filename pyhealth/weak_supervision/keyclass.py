# Author: Amin Torabi
# Paper title: Classifying Unstructured Clinical Notes via Automatic Weak Supervision
# Paper link: https://arxiv.org/abs/2206.12088v2
# Description: Automatic phrase mining, labeling-function generation, and categorical weak-label aggregation.

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class LabelingFunction:
    phrase: str
    category_index: int
    similarity: float


class KeywordLabeler:
    """Assign each training phrase to its nearest description; keep top K/class.

    descriptions is an insertion-ordered mapping of stable category IDs to text.
    category_ids_ fixes all output columns; IDs need not be consecutive integers.
    Matching uses the same lowercasing, tokenization, and stopword filtering as
    mining. These are lexical rules without negation or temporality handling.
    min_df is an inclusive sklearn document-frequency threshold.
    """

    def __init__(self, encoder, top_k=10, ngram_range=(1, 3), min_df=0.001):
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.encoder = encoder
        self.top_k = top_k
        self.ngram_range = ngram_range
        self.min_df = min_df

    def fit(self, texts, descriptions):
        texts = list(texts)
        if not texts or len(descriptions) < 2:
            raise ValueError("Need training notes and at least two categories")
        if any(not isinstance(x, str) or not x.strip() for x in descriptions.values()):
            raise ValueError("Category descriptions must be nonempty strings")
        self.category_ids_ = tuple(descriptions)
        self.vectorizer_ = CountVectorizer(
            ngram_range=self.ngram_range, min_df=self.min_df, stop_words="english"
        )
        self.vectorizer_.fit(texts)
        phrases = self.vectorizer_.get_feature_names_out()
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
        self.lfs_ = []
        for category in range(len(descriptions)):
            indices = np.flatnonzero(owners == category)
            ranked = sorted(indices, key=lambda i: (-scores[i, category], phrases[i]))
            self.lfs_.extend(
                LabelingFunction(str(phrases[i]), category, float(scores[i, category]))
                for i in ranked[: self.top_k]
            )
        return self

    def transform(self, texts):
        """Return int64 [notes, LFs], with -1 abstain or a category column index."""
        if not hasattr(self, "lfs_"):
            raise RuntimeError("Fit KeywordLabeler on training notes first")
        texts = list(texts)
        votes = np.full((len(texts), len(self.lfs_)), -1, dtype=np.int64)
        if not texts:
            return votes
        counts = self.vectorizer_.transform(texts)
        for j, lf in enumerate(self.lfs_):
            matched = (
                counts[:, self.vectorizer_.vocabulary_[lf.phrase]].toarray().ravel() > 0
            )
            votes[matched, j] = lf.category_index
        return votes


def _validate_votes(votes, n_classes):
    votes = np.asarray(votes)
    if n_classes < 2 or votes.ndim != 2 or not np.issubdtype(votes.dtype, np.integer):
        raise ValueError("Need an integer [notes, LFs] matrix and >= 2 classes")
    if ((votes < -1) | (votes >= n_classes)).any():
        raise ValueError("Votes must be -1 or category indices")
    return votes


def aggregate_votes(votes, n_classes, method="majority", seed=0, **fit_kwargs):
    """Return (categorical probabilities, covered mask).

    Majority baseline is normalized vote counts, preserving ties. Uncovered
    rows receive a uniform distribution and MUST be excluded from initial
    training via the returned mask. Snorkel is optional and never silently
    replaced by the baseline. Fits only on covered training rows.
    """
    votes = _validate_votes(votes, n_classes)
    covered = (votes != -1).any(axis=1)
    probabilities = np.full((len(votes), n_classes), 1 / n_classes)
    if method not in ("majority", "snorkel"):
        raise ValueError("method must be majority or snorkel")
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
            raise ImportError("Install pyhealth[keyclass] to use Snorkel") from exc
        model = LabelModel(cardinality=n_classes, verbose=False)
        model.fit(L_train=votes[covered], seed=seed, **fit_kwargs)
        probabilities[covered] = model.predict_proba(L=votes[covered])
    return probabilities.astype(np.float32), covered


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
