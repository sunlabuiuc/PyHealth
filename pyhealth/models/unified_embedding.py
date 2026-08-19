"""Deprecated import path for the unified multimodal encoder.

Use ``pyhealth.models.embedding`` instead. This module re-exports the live
classes so older ``from pyhealth.models.unified_embedding import ...`` call
sites keep working against one implementation.
"""

from pyhealth.models.embedding.unified import (
    SinusoidalTimeEmbedding,
    UnifiedMultimodalEmbeddingModel,
)

__all__ = [
    "SinusoidalTimeEmbedding",
    "UnifiedMultimodalEmbeddingModel",
]
