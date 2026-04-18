"""Native BaseModel entrypoint for EOL mistrust downstream tasks."""

from __future__ import annotations

import hashlib
from typing import Dict, Sequence

import torch
import torch.nn as nn

from pyhealth.datasets.sample_dataset import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.processors import SequenceProcessor, TensorProcessor, TextProcessor


def _stable_bucket_index(text: str, num_buckets: int) -> int:
    """Map text to a stable non-zero embedding bucket."""

    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return (int(digest, 16) % max(num_buckets - 1, 1)) + 1


class EOLMistrustClassifier(BaseModel):
    """Simple multimodal classifier for EOL mistrust task samples.

    The model is designed for the task schema used by
    ``pyhealth.tasks.eol_mistrust``:

    - coded EHR history fields use learned sequence embeddings with mean pooling
    - scalar numeric fields use linear projections
    - text and categorical string fields use stable hashed token embeddings

    Args:
        dataset: SampleDataset returned by ``dataset.set_task(...)``.
        embedding_dim: Shared feature embedding dimension.
        hidden_dim: Hidden layer width before the output head.
        dropout: Dropout applied to the pooled patient representation.
        text_hash_buckets: Number of buckets for hashed text embeddings.

    Attributes:
        label_key: The single label field name consumed from the task schema.
        embedding_dim: Dimension of every per-modality pooled representation.
        hidden_dim: Width of the hidden layer before the classification head.
        text_hash_buckets: Vocabulary size (excluding pad) for text embeddings.
        sequence_embeddings: ``nn.ModuleDict`` of learned sequence embeddings.
        tensor_projections: ``nn.ModuleDict`` of linear tensor projections.
        text_embeddings: ``nn.ModuleDict`` of hashed-token text embeddings.
        hidden_layer: Linear layer that mixes concatenated modality features.
        output_layer: Final linear head producing task logits.

    Raises:
        ValueError: If the task has anything other than exactly one label key.
        TypeError: If the task schema contains a processor type this model does
            not support (only sequence, tensor, and text are handled).

    Example:
        >>> from pyhealth.datasets import EOLMistrustDataset
        >>> from pyhealth.tasks import EOLMistrustMortalityPredictionMIMIC3
        >>> from pyhealth.models import EOLMistrustClassifier
        >>> dataset = EOLMistrustDataset(root="/data/eol_mistrust")
        >>> samples = dataset.set_task(EOLMistrustMortalityPredictionMIMIC3())
        >>> model = EOLMistrustClassifier(
        ...     dataset=samples,
        ...     embedding_dim=32,
        ...     hidden_dim=64,
        ... )
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        text_hash_buckets: int = 2048,
    ) -> None:
        """Build the multimodal classification head over the task schema.

        Args:
            dataset: Fitted :class:`~pyhealth.datasets.SampleDataset`.
            embedding_dim: Shared per-modality embedding dimension.
            hidden_dim: Width of the hidden layer before the output head.
            dropout: Dropout probability applied to the patient representation.
            text_hash_buckets: Number of hashed text token buckets (the
                embedding table has ``text_hash_buckets + 1`` rows, with row 0
                reserved for padding).

        Raises:
            ValueError: If the task does not have exactly one label key.
            TypeError: If a processor in the schema is not a sequence, tensor,
                or text processor.
        """
        super().__init__(dataset)

        if len(self.label_keys) != 1:
            raise ValueError("EOLMistrustClassifier supports exactly one label key.")

        self.label_key = self.label_keys[0]
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.text_hash_buckets = int(text_hash_buckets)

        self.sequence_embeddings = nn.ModuleDict()
        self.tensor_projections = nn.ModuleDict()
        self.text_embeddings = nn.ModuleDict()

        for feature_key, processor in self.dataset.input_processors.items():
            if isinstance(processor, SequenceProcessor):
                self.sequence_embeddings[feature_key] = nn.Embedding(
                    num_embeddings=len(processor.code_vocab),
                    embedding_dim=self.embedding_dim,
                    padding_idx=0,
                )
            elif isinstance(processor, TensorProcessor):
                self.tensor_projections[feature_key] = nn.Linear(
                    self._infer_tensor_input_size(feature_key),
                    self.embedding_dim,
                )
            elif isinstance(processor, TextProcessor):
                self.text_embeddings[feature_key] = nn.Embedding(
                    num_embeddings=self.text_hash_buckets + 1,
                    embedding_dim=self.embedding_dim,
                    padding_idx=0,
                )
            else:
                raise TypeError(
                    f"Unsupported processor for EOLMistrustClassifier: "
                    f"{feature_key} -> {processor.__class__.__name__}"
                )

        total_modalities = (
            len(self.sequence_embeddings)
            + len(self.tensor_projections)
            + len(self.text_embeddings)
        )
        representation_dim = total_modalities * self.embedding_dim
        self.hidden_layer = nn.Linear(representation_dim, self.hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(self.hidden_dim, self.get_output_size())

    def _infer_tensor_input_size(self, feature_key: str) -> int:
        for index in range(len(self.dataset)):
            if feature_key not in self.dataset[index]:
                continue
            value = self.dataset[index][feature_key]
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    return 1
                return int(value.shape[-1])
            return 1
        return 1

    def _mean_pool_sequence(
        self, values: torch.Tensor, feature_key: str
    ) -> torch.Tensor:
        if values.dim() == 1:
            values = values.unsqueeze(0)
        values = values.long().to(self.device)
        embeddings = self.sequence_embeddings[feature_key](values)
        mask = (values != 0).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        return (embeddings * mask).sum(dim=1) / denom

    def _project_tensor(self, values: torch.Tensor, feature_key: str) -> torch.Tensor:
        values = values.to(self.device).float()
        if values.dim() == 0:
            values = values.view(1, 1)
        elif values.dim() == 1:
            values = values.unsqueeze(-1)
        return self.tensor_projections[feature_key](values)

    def _embed_text_field(
        self, values: Sequence[str], feature_key: str
    ) -> torch.Tensor:
        token_lists = []
        max_len = 1
        for raw_value in values:
            normalized = str(raw_value or "").strip().lower()
            tokens = normalized.split()
            if not tokens:
                tokens = ["<empty>"]
            token_lists.append(tokens)
            max_len = max(max_len, len(tokens))

        index_rows = []
        for tokens in token_lists:
            row = [
                _stable_bucket_index(token, self.text_hash_buckets) for token in tokens
            ]
            if len(row) < max_len:
                row.extend([0] * (max_len - len(row)))
            index_rows.append(row)

        indices = torch.tensor(index_rows, dtype=torch.long, device=self.device)
        embeddings = self.text_embeddings[feature_key](indices)
        mask = (indices != 0).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        return (embeddings * mask).sum(dim=1) / denom

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Run a forward pass over one batch of task samples.

        Each feature in ``self.feature_keys`` is routed to the appropriate
        per-modality handler (sequence mean pool, tensor projection, or hashed
        text embedding), the results are concatenated, and a two-layer MLP
        produces logits for the single task label.

        Args:
            **kwargs: Batch dictionary with one entry per task feature key and
                one entry for the task label key. Values may be tensors
                (sequence / tensor features) or strings / sequences of strings
                (text features).

        Returns:
            Dict with keys ``loss``, ``y_prob``, ``y_true``, and ``logit``, as
            expected by :class:`~pyhealth.trainer.Trainer`.

        Raises:
            KeyError: If an unexpected feature key is encountered at runtime.
        """
        pooled_features = []

        for feature_key in self.feature_keys:
            value = kwargs[feature_key]
            if feature_key in self.sequence_embeddings:
                pooled_features.append(self._mean_pool_sequence(value, feature_key))
            elif feature_key in self.tensor_projections:
                pooled_features.append(self._project_tensor(value, feature_key))
            elif feature_key in self.text_embeddings:
                if isinstance(value, str):
                    text_values = [value]
                else:
                    text_values = list(value)
                pooled_features.append(self._embed_text_field(text_values, feature_key))
            else:
                raise KeyError(
                    "Unexpected feature key for EOLMistrustClassifier: "
                    f"{feature_key}"
                )

        patient_representation = torch.cat(pooled_features, dim=1)
        hidden = self.activation(self.hidden_layer(patient_representation))
        hidden = self.dropout(hidden)
        logits = self.output_layer(hidden)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }


__all__ = ["EOLMistrustClassifier"]
