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
        """Transform processor outputs into embeddings.

        Subclass return types
        ---------------------
        EmbeddingModel
            ``Dict[str, Tensor]`` mapping each field name to its embedded
            tensor.  When ``output_mask=True`` is passed, returns a
            ``(Dict[str, Tensor], Dict[str, Tensor])`` tuple of
            (embeddings, masks).

        VisionEmbeddingModel
            ``Tensor`` of shape ``[batch, embedding_dim]``.

        TextEmbeddingModel
            ``(Tensor, BoolTensor)`` of shapes ``([B, T, E], [B, T])``
            when ``return_mask=True`` (default), or a plain
            ``Tensor [B, T, E]`` when ``return_mask=False``.

        UnifiedMultimodalEmbeddingModel
            ``Dict[str, Tensor]`` with keys ``"sequence"`` ``[B, S, E]``,
            ``"mask"`` ``[B, S]``, ``"time"`` ``[B, S]``, and
            ``"type_ids"`` ``[B, S]``.
        """
        ...
