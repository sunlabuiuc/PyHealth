"""PromptEHR: Prompt-based BART model for synthetic EHR generation.

This module provides a demographic-conditioned sequence-to-sequence model
for generating realistic synthetic electronic health records.

Main components:
    - PromptEHR: Main model class (inherits from BaseModel)
    - ConditionalPromptEncoder: Demographic conditioning with reparameterization
    - PromptBartEncoder: Modified BART encoder with prompt injection
    - PromptBartDecoder: Modified BART decoder with prompt injection
    - VisitStructureSampler: Utility for structure-constrained generation
"""

from .model import PromptEHR
from .conditional_prompt import ConditionalPromptEncoder
from .bart_encoder import PromptBartEncoder
from .bart_decoder import PromptBartDecoder

__all__ = [
    "PromptEHR",
    "ConditionalPromptEncoder",
    "PromptBartEncoder",
    "PromptBartDecoder",
]
