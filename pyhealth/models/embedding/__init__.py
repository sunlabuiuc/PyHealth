"""Embedding models for PyHealth multimodal pipelines.

All embedding models share the :class:`BaseEmbeddingModel` interface:
they expose an ``embedding_dim`` property and a ``forward`` method that
transforms processor output tensors into dense vector embeddings.

Available models:

- :class:`EmbeddingModel` — generic encoder for codes, sequences, timeseries
- :class:`VisionEmbeddingModel` — ViT-style patch encoder for medical images (Josh)
- :class:`TextEmbeddingModel` — BERT-based encoder for clinical text (Rian)
- :class:`UnifiedMultimodalEmbeddingModel` — temporally-aligned multi-modal encoder

Helper utilities:

- :class:`SinusoidalTimeEmbedding` — continuous time positional encoding
- :func:`init_embedding_with_pretrained` — load GloVe-style pretrained vectors
"""

from .base import BaseEmbeddingModel
from .vanilla import EmbeddingModel, init_embedding_with_pretrained
from .vision import VisionEmbeddingModel, PatchEmbedding, Permute
from .text import TextEmbeddingModel, TextEmbedding
from .unified import UnifiedMultimodalEmbeddingModel, SinusoidalTimeEmbedding

__all__ = [
    "BaseEmbeddingModel",
    "EmbeddingModel",
    "VisionEmbeddingModel",
    "PatchEmbedding",
    "Permute",
    "TextEmbeddingModel",
    "TextEmbedding",
    "UnifiedMultimodalEmbeddingModel",
    "SinusoidalTimeEmbedding",
    "init_embedding_with_pretrained",
]
