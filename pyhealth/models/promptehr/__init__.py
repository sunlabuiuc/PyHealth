"""PromptEHR: Prompt-based BART model for synthetic EHR generation.

This module provides a demographic-conditioned sequence-to-sequence model
for generating realistic synthetic electronic health records.

Main components:
    - PromptEHR: Main model class (inherits from BaseModel)
    - ConditionalPromptEncoder: Demographic conditioning with reparameterization
    - PromptBartEncoder: Modified BART encoder with prompt injection
    - PromptBartDecoder: Modified BART decoder with prompt injection
    - VisitStructureSampler: Utility for structure-constrained generation
    - Generation functions: sample_demographics, parse_sequence_to_visits, etc.
"""

from .model import PromptEHR
from .conditional_prompt import ConditionalPromptEncoder
from .bart_encoder import PromptBartEncoder
from .bart_decoder import PromptBartDecoder
from .visit_sampler import VisitStructureSampler
from .generation import (
    DemographicSampler,
    sample_demographics,
    decode_patient_demographics,
    parse_sequence_to_visits,
    generate_patient_sequence_conditional,
    generate_patient_with_structure_constraints
)

__all__ = [
    "PromptEHR",
    "ConditionalPromptEncoder",
    "PromptBartEncoder",
    "PromptBartDecoder",
    "VisitStructureSampler",
    "DemographicSampler",
    "sample_demographics",
    "decode_patient_demographics",
    "parse_sequence_to_visits",
    "generate_patient_sequence_conditional",
    "generate_patient_with_structure_constraints",
]
