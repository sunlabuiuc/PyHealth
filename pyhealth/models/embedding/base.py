from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """Abstract base class for all embedding models in PyHealth.

    All embedding models share a common contract:

    - They expose an ``embedding_dim`` property indicating the output vector dimension.
    - Their ``forward`` method accepts processor output tensors and returns
      vector embeddings.

    Concrete subclasses:

    - :class:`EmbeddingModel` – generic encoder for codes, sequences, timeseries
    - :class:`VisionEmbeddingModel` – patch-based encoder for medical images (Josh)
    - :class:`TextEmbeddingModel` – BERT-based encoder for clinical text (Rian)
    - :class:`UnifiedMultimodalEmbeddingModel` – temporally-aligned multi-modal encoder
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output embedding dimension shared across all modalities."""
        ...

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Transform processor outputs into embeddings."""
        ...
