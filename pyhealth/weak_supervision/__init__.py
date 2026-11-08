"""Automatic keyword supervision, independent of gold diagnosis labels."""

from .encoders import ClinicalTextEncoder
from .keyclass import (
    KeywordLabeler,
    LabelingFunction,
    aggregate_votes,
    vote_diagnostics,
)
