"""Automatic keyword supervision, independent of gold diagnosis labels."""

from .encoders import ClinicalTextEncoder as ClinicalTextEncoder
from .keyclass import KeywordLabeler as KeywordLabeler
from .keyclass import LabelingFunction as LabelingFunction
from .keyclass import aggregate_votes as aggregate_votes
from .keyclass import clean_note as clean_note
from .keyclass import confident_balanced_mask as confident_balanced_mask
from .keyclass import nltk_english_stopwords as nltk_english_stopwords
from .keyclass import snorkel_memory_gb as snorkel_memory_gb
from .keyclass import vote_diagnostics as vote_diagnostics
