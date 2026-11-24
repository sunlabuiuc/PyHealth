"""Automatic keyword supervision, independent of gold diagnosis labels."""

from .encoders import ClinicalTextEncoder
from .keyclass import (
    KeywordLabeler,
    LabelingFunction,
    aggregate_votes,
    clean_note,
    confident_balanced_mask,
    nltk_english_stopwords,
    snorkel_memory_gb,
    vote_diagnostics,
)
