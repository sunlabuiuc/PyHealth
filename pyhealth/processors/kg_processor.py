from collections.abc import Iterable
from typing import Any

import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("kg_entity_list")
class KGProcessor(FeatureProcessor):
    """Pads a variable-length list of knowledge-graph entity ids to a fixed length.

    Intended for fields holding lists of entity ids, such as the known-true
    ``ground_truth_head`` / ``ground_truth_tail`` filter sets used for
    filtered link-prediction ranking evaluation (e.g. TransE, RotatE style
    KG embedding models). The target length is the maximum list length
    observed for that field during ``fit``, following the same per-field
    convention as :class:`~pyhealth.processors.nested_sequence_processor.NestedSequenceProcessor`.

    Note:
        The padded ``pad_token_id`` is not a valid entity id on its own: any
        code consuming ``ground_truth_head``/``ground_truth_tail`` must use
        the accompanying ``mask`` to recover the true (unpadded) entity list
        before doing membership filtering, since ``pad_token_id`` may collide
        with a real entity id (e.g. 0). See
        :meth:`~pyhealth.medcode.pretrained_embeddings.kg_emb.models.kg_base.KGEBaseModel.train_neg_sample_gen`.

    Args:
        pad_token_id: Entity id used to pad lists shorter than ``max_length``.
            Default is 0.

    Example:
        >>> processor = KGProcessor(pad_token_id=0)
        >>> samples = [
        ...     {"ground_truth_tail": [6, 7, 8]},
        ...     {"ground_truth_tail": [16]},
        ... ]
        >>> processor.fit(samples, "ground_truth_tail")
        >>> processor.process([16])
        {'value': tensor([16, 0, 0]), 'mask': tensor([1, 0, 0])}
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        self.max_length = 1

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Determine the maximum list length observed for ``field``.

        Args:
            samples: Iterable of sample dictionaries.
            field: Name of the field holding a list of entity ids.
        """
        max_len = 0
        for sample in samples:
            value = sample.get(field)
            if value is not None:
                max_len = max(max_len, len(value))
        self.max_length = max(1, max_len)

    def process(self, value: list[int]) -> dict[str, torch.Tensor]:
        """Pad a list of entity ids to ``max_length`` and build its attention mask.

        Lists longer than ``max_length`` (e.g. seen only at inference time,
        after ``fit`` was called on a different split) are truncated.

        Args:
            value: List of entity ids.

        Returns:
            Dict with:
                - ``"value"``: LongTensor of shape ``(max_length,)``, padded with
                  ``pad_token_id``.
                - ``"mask"``: LongTensor of shape ``(max_length,)``, 1 for real
                  entities and 0 for padding.
        """
        entities = list(value) if value is not None else []
        seq_len = len(entities)

        if seq_len >= self.max_length:
            padded = entities[: self.max_length]
            mask = [1] * self.max_length
        else:
            padded = entities + [self.pad_token_id] * (self.max_length - seq_len)
            mask = [1] * seq_len + [0] * (self.max_length - seq_len)

        return {
            "value": torch.tensor(padded, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
        }

    def size(self) -> int:
        """Return the fitted padding length."""
        return self.max_length

    def is_token(self) -> bool:
        """Entity ids are discrete token indices."""
        return True

    def schema(self) -> tuple:
        return ("value", "mask")

    def dim(self) -> tuple:
        return (1, 1)

    def spatial(self) -> tuple:
        return (False, False)

    def __repr__(self) -> str:
        return f"KGProcessor(max_length={self.max_length}, pad_token_id={self.pad_token_id})"
