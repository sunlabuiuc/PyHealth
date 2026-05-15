"""
Negative Sentiment Mistrust Task for MIMIC-III
===============================================
Implements the sentiment-based mistrust proxy from:

    Boag et al. "Racial Disparities and Mistrust in End-of-Life Care."
    MLHC 2018. https://arxiv.org/abs/1808.03827

The original paper used ``pattern.en.sentiment(text.split())`` — a
word-averaged polarity approach — on concatenated discharge notes.  Full-text
VADER compound scoring saturates at −1.0 for >94 % of clinical notes, making
it unsuitable as a direct replacement.  Sentence-level VADER averaging avoids
saturation and closely reproduces the original's discriminative power.

Method
------
For each hospital admission:
  1. Collect all discharge summary notes (``CATEGORY = 'Discharge summary'``).
  2. Score each note by averaging VADER compound scores across its sentences.
  3. Average across multiple notes for the same admission.
  4. Negate: ``raw_neg_score = -mean_sentence_polarity``
     (higher → more negative → more mistrust signal).

Z-score normalisation (``neg_score = -(raw - μ) / σ``) requires global
statistics and must be applied **after** ``set_task()`` using the provided
``normalize_sentiment_scores`` utility from ``pyhealth.nlp``.

Output feature
--------------
``neg_sentiment`` — a single-element list ``[float]`` stored as a ``"tensor"``
feature.  The one-element list satisfies PyHealth's TensorProcessor which
expects an iterable of numerics.

Usage
-----
    >>> from pyhealth.datasets import MIMIC3Dataset
    >>> from pyhealth.tasks import MistrustSentimentMIMIC3
    >>> from pyhealth.nlp import normalize_sentiment_scores
    >>> from pyhealth.models import LogisticRegression
    >>>
    >>> base_dataset = MIMIC3Dataset(
    ...     root="/path/to/mimic-iii/1.4",
    ...     tables=["NOTEEVENTS"],
    ... )
    >>> task = MistrustSentimentMIMIC3()
    >>> sample_dataset = base_dataset.set_task(task)
    >>> normalize_sentiment_scores(sample_dataset)   # Z-score in-place
    >>> model = LogisticRegression(dataset=sample_dataset)
"""

from typing import Any, Dict, List, Optional

from pyhealth.tasks.base_task import BaseTask


class MistrustSentimentMIMIC3(BaseTask):
    """Compute negative-sentiment mistrust proxy from MIMIC-III discharge notes.

    For each hospital admission the task produces one sample:

    - ``neg_sentiment``: a one-element list ``[float]`` — the raw negated
      mean sentence-level VADER compound score across all discharge summaries
      for this admission (schema: ``"tensor"``).  Values are negated so that
      higher = more negative sentiment = more mistrust signal.  Call
      ``pyhealth.nlp.normalize_sentiment_scores(sample_dataset)`` after
      ``set_task()`` to complete the Z-score normalisation step.

    - ``noncompliance``: ``1`` if any note for this admission contains
      ``"noncompliant"``, else ``0`` (schema: ``"binary"``).  This matches
      the output label of ``MistrustNoncomplianceMIMIC3``, enabling direct
      comparison of the three mistrust proxies on the same task.

    Args:
        min_notes: Minimum number of discharge summary notes required for a
            sample to be included.  Defaults to 1.
        output_label: Column name and key of the binary output label.
            Change to ``"autopsy_consent"`` to align with
            ``MistrustAutopsyMIMIC3``. Defaults to ``"noncompliance"``.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import MistrustSentimentMIMIC3
        >>> from pyhealth.nlp import normalize_sentiment_scores
        >>>
        >>> base_dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["NOTEEVENTS"],
        ... )
        >>> task = MistrustSentimentMIMIC3()
        >>> sample_dataset = base_dataset.set_task(task)
        >>> normalize_sentiment_scores(sample_dataset)
        >>> len(sample_dataset)
        52726
    """

    task_name: str = "MistrustSentimentMIMIC3"
    input_schema: Dict[str, str] = {"neg_sentiment": "tensor"}
    output_schema: Dict[str, str] = {"noncompliance": "binary"}

    def __init__(
        self,
        min_notes: int = 1,
        output_label: str = "noncompliance",
    ) -> None:
        self.min_notes = min_notes
        self.output_label = output_label
        # output_schema is a class attribute; update it to reflect output_label
        self.output_schema = {output_label: "binary"}

        # Lazy-initialise scorer to avoid importing nltk at module load time
        self._scorer: Optional[Any] = None

    def _get_scorer(self):
        """Lazily initialise SentimentScorer on first use."""
        if self._scorer is None:
            from pyhealth.nlp import SentimentScorer
            self._scorer = SentimentScorer()
        return self._scorer

    @staticmethod
    def _noncompliance_label(noteevents: List[Any]) -> int:
        """Return 1 if any note contains 'noncompliant', else 0."""
        for ev in noteevents:
            if "noncompliant" in str(getattr(ev, "text", "") or "").lower():
                return 1
        return 0

    @staticmethod
    def _autopsy_label(noteevents: List[Any]) -> Optional[int]:
        """Return autopsy consent label (1/0) or None if absent/ambiguous."""
        consented = declined = False
        for ev in noteevents:
            text = str(getattr(ev, "text", "") or "").lower()
            if "autopsy" not in text:
                continue
            for line in text.split("\n"):
                if "autopsy" not in line:
                    continue
                if any(w in line for w in ("decline", "not consent", "refuse", "denied")):
                    declined = True
                if any(w in line for w in ("consent", "agree", "request")):
                    consented = True
        if consented and declined:
            return None
        if consented:
            return 1
        if declined:
            return 0
        return None

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient into negative-sentiment classification samples.

        Args:
            patient: a PyHealth Patient object with ``noteevents`` loaded.

        Returns:
            List of dicts, one per admission that has ≥ ``min_notes`` discharge
            summaries, each containing:
                - ``patient_id``
                - ``visit_id`` (hadm_id)
                - ``neg_sentiment`` (list of one float — raw negated score)
                - output label (``noncompliance`` or ``autopsy_consent``)
        """
        scorer = self._get_scorer()
        samples = []
        admissions = patient.get_events(event_type="admissions")

        for admission in admissions:
            hadm_id = admission.hadm_id

            noteevents = patient.get_events(
                event_type="noteevents",
                filters=[("hadm_id", "==", hadm_id)],
            )

            # Extract discharge summaries only
            discharge_notes = [
                ev for ev in noteevents
                if str(getattr(ev, "category", "") or "").strip().lower()
                == "discharge summary"
            ]

            if len(discharge_notes) < self.min_notes:
                continue

            # Score each note; average across notes for this admission
            note_scores = [
                scorer.score(str(getattr(ev, "text", "") or ""))
                for ev in discharge_notes
            ]
            raw_mean = float(sum(note_scores) / len(note_scores))

            # Negate: higher value = more negative sentiment = more mistrust
            raw_neg = -raw_mean

            # Derive output label
            if self.output_label == "noncompliance":
                label = self._noncompliance_label(noteevents)
            elif self.output_label == "autopsy_consent":
                label = self._autopsy_label(noteevents)
                if label is None:
                    continue   # exclude ambiguous/absent autopsy signal
            else:
                raise ValueError(
                    f"output_label must be 'noncompliance' or 'autopsy_consent', "
                    f"got '{self.output_label}'"
                )

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": hadm_id,
                    "neg_sentiment": [raw_neg],   # 1-element list for TensorProcessor
                    self.output_label: label,
                }
            )

        return samples
