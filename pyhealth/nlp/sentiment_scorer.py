"""
Sentence-level sentiment scoring for clinical text
====================================================
Provides a VADER-based ``SentimentScorer`` that avoids the full-text
saturation problem specific to clinical discharge notes, and a utility for
Z-score normalising a column of raw scores across a sample dataset.

Background
----------
Standard full-text VADER compound scoring saturates at −1.0 for >94 % of
clinical discharge summaries because clinical language is lexically negative
("pain", "failure", "respiratory distress").  After Z-scoring a saturated
distribution every patient receives the same score, making the metric useless
for discrimination.

Sentence-level averaging (score each sentence independently, take the mean)
avoids saturation and closely approximates the word-averaged approach used by
the ``pattern.en`` library in the original Boag et al. 2018 implementation.

Reference
---------
Boag et al. "Racial Disparities and Mistrust in End-of-Life Care."
MLHC 2018. https://arxiv.org/abs/1808.03827
"""

from typing import Dict, List, Optional

import numpy as np


class SentimentScorer:
    """Sentence-level VADER sentiment scorer for clinical text.

    Scores a document by:
      1. Tokenising into sentences with NLTK's ``sent_tokenize``.
      2. Computing the VADER compound score for each sentence.
      3. Returning the mean compound score across all sentences.

    This avoids the full-text VADER saturation problem that affects clinical
    discharge notes (>94 % saturate at −1.0).

    The scorer is intentionally stateless and thread-safe after initialisation.

    Args:
        language: Reserved for future multilingual support. Only ``"english"``
            is currently supported (NLTK sentence tokeniser).

    Examples:
        >>> scorer = SentimentScorer()
        >>> scorer.score("The patient is calm and alert. No acute distress.")
        0.412
        >>> scorer.score("Patient unresponsive, severe respiratory failure.")
        -0.613
    """

    def __init__(self, language: str = "english") -> None:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            from nltk.tokenize import sent_tokenize
        except ImportError as e:
            raise ImportError(
                "nltk is required for SentimentScorer. "
                "Install with: pip install nltk && python -c "
                "\"import nltk; nltk.download('vader_lexicon'); "
                "nltk.download('punkt_tab')\""
            ) from e

        self._sid = SentimentIntensityAnalyzer()
        self._sent_tokenize = sent_tokenize
        self.language = language

    def score(self, text: str) -> float:
        """Compute the mean sentence-level VADER compound score for a document.

        Args:
            text: Raw document text. Empty or whitespace-only text returns 0.0.

        Returns:
            Mean VADER compound score in [−1.0, +1.0].
            Higher values indicate more positive sentiment.
        """
        if not text or not text.strip():
            return 0.0
        sentences = self._sent_tokenize(text)
        if not sentences:
            return 0.0
        scores = [
            self._sid.polarity_scores(s)["compound"]
            for s in sentences
        ]
        return float(np.mean(scores))

    def score_batch(self, texts: List[str]) -> List[float]:
        """Score a list of documents.

        Args:
            texts: List of raw document strings.

        Returns:
            List of mean sentence-level compound scores.
        """
        return [self.score(t) for t in texts]

    def negate_and_zscore(self, raw_scores: Dict) -> Dict:
        """Negate and Z-score a dict of {key: raw_score} values.

        Applies the normalisation from Boag et al. 2018:

            neg_score[key] = -(raw_score[key] - μ) / σ

        Higher output value → more negative sentiment → more mistrust signal.

        Args:
            raw_scores: Dict mapping any key to a raw compound score.

        Returns:
            Dict with the same keys mapped to negated Z-scores.
        """
        vals = np.array(list(raw_scores.values()), dtype=np.float64)
        mu, sigma = vals.mean(), vals.std()
        if sigma == 0.0:
            return {k: 0.0 for k in raw_scores}
        return {k: float(-(v - mu) / sigma) for k, v in raw_scores.items()}


def normalize_sentiment_scores(sample_dataset, feature_key: str = "neg_sentiment") -> None:
    """Z-score normalise a raw sentiment feature column in-place across all samples.

    The ``MistrustSentimentMIMIC3`` task stores the *raw* negated sentiment
    score (``-mean_sentence_compound``) per sample.  Because Z-scoring requires
    the global mean and standard deviation across all patients, it cannot be
    done inside ``__call__`` (which processes one patient at a time).  Call
    this utility **after** ``dataset.set_task()`` to complete the normalisation.

    The transformation applied is identical to Boag et al. 2018:

        neg_score = -(raw_score - μ_all) / σ_all

    Args:
        sample_dataset: A PyHealth ``SampleDataset`` produced by
            ``base_dataset.set_task(MistrustSentimentMIMIC3(...))``.
        feature_key: Name of the sentiment feature in each sample.
            Defaults to ``"neg_sentiment"``.

    Example:
        >>> sample_dataset = base_dataset.set_task(MistrustSentimentMIMIC3(...))
        >>> normalize_sentiment_scores(sample_dataset)
        >>> # neg_sentiment values are now Z-scored across the full dataset
    """
    raw_vals = np.array(
        [s[feature_key][0] for s in sample_dataset.samples],
        dtype=np.float64,
    )
    mu, sigma = raw_vals.mean(), raw_vals.std()
    if sigma == 0.0:
        return
    for sample in sample_dataset.samples:
        raw = sample[feature_key][0]
        sample[feature_key] = [float((raw - mu) / sigma)]
