"""PyHealth EHRMamba supervised clinical prediction implementation.

This is a small, unfused selective-SSM adaptation for sequence features.
It does not reproduce foundation-model pretraining from the original paper.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.processors import SequenceProcessor

from .cehr_embeddings import CEHREmbeddings
from .selective_ssm import SelectiveSSMBlock


class EHRMamba(BaseModel):
    """Causal selective-SSM model for categorical clinical event sequences.

    Each input feature has its own code embedding and Mamba backbone. The
    last valid representation of each feature is concatenated for prediction.

    Args:
        dataset: Fitted PyHealth SampleDataset with ``sequence`` features.
        embedding_dim: Embedding and backbone width.
        num_layers: Number of selective SSM blocks per feature.
        state_dim: State size per expanded SSM channel.
        expansion: Expansion ratio inside the SSM.
        conv_kernel: Width of causal depthwise convolution.
        dropout: Dropout rate.
        use_cehr_metadata: Accept optional ``{feature}_ages``,
            ``{feature}_times``, and ``{feature}_segments`` tensors in
            ``forward``; metadata must align with code tokens.

    Raises:
        ValueError: If the dataset has unsupported features or labels.
        
        
    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"patient_id": "p0", "codes": ["A", "B"], "label": 1},
        ...     {"patient_id": "p1", "codes": ["C"], "label": 0},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"codes": "sequence"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="ehrmamba_example",
        ... )
        >>> model = EHRMamba(dataset, embedding_dim=8, num_layers=1, state_dim=4)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> model(**batch)["y_prob"].shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        num_layers: int = 2,
        state_dim: int = 16,
        expansion: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.1,
        use_cehr_metadata: bool = False,
    ) -> None:
        super().__init__(dataset)
        if not self.feature_keys or len(self.label_keys) != 1:
            raise ValueError("Expected at least one feature and one label")
        if embedding_dim < 1 or num_layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid embedding_dim, num_layers, or dropout")
        self.label_key = self.label_keys[0]
        self.use_cehr_metadata = use_cehr_metadata
        self.embeddings = nn.ModuleDict()
        self.blocks = nn.ModuleDict()
        for key in self.feature_keys:
            processor = dataset.input_processors[key]
            if not isinstance(processor, SequenceProcessor):
                raise TypeError(f"{key}: only 'sequence' inputs are supported")
            self.embeddings[key] = CEHREmbeddings(
                vocab_size=len(processor.code_vocab),
                hidden_dim=embedding_dim,
                dropout=dropout,
            )
            self.blocks[key] = nn.ModuleList(
                SelectiveSSMBlock(
                    hidden_dim=embedding_dim,
                    state_dim=state_dim,
                    expansion=expansion,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embedding_dim * len(self.feature_keys),
            self.label_processor.output_size(),
        )

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Predict clinical outcomes from processed patient code sequences.

        Args:
            **kwargs: One integer tensor per input feature, each of shape
                (batch, sequence). Optional labels and ``embed=True``.
                Optional CEHR metadata is accepted when enabled.

        Returns:
            A dictionary containing ``logit`` and ``y_prob``; when labels
            are provided, also ``loss`` and ``y_true``. Optionally ``embed``.

        Raises:
            ValueError: If a patient has no valid codes or shapes are wrong.
        """
        patient_vectors = []
        for key in self.feature_keys:
            value = kwargs[key]
            if isinstance(value, (tuple, list)):
                if len(value) != 1:
                    raise ValueError(f"{key} must contain one value tensor")
                value = value[0]
            codes = torch.as_tensor(value, device=self.device).long()
            if codes.ndim != 2:
                raise ValueError(f"{key} must be a 2D code tensor")
            mask = codes.ne(0)
            if not mask.any(dim=1).all():
                raise ValueError(f"{key} contains an empty patient sequence")
            metadata = {}
            if self.use_cehr_metadata:
                for suffix, arg in (
                    ("ages", "ages"),
                    ("times", "times"),
                    ("segments", "segments"),
                ):
                    raw = kwargs.get(f"{key}_{suffix}")
                    if raw is not None:
                        metadata[arg] = torch.as_tensor(raw, device=self.device).long()
            x = self.embeddings[key](codes, mask, **metadata)
            for block in self.blocks[key]:
                x = block(x, mask)
            indices = (
                torch.arange(codes.shape[1], device=self.device)[None, :]
                .expand_as(mask)
                .masked_fill(~mask, -1)
                .max(dim=1)
                .values
            )
            rows = torch.arange(codes.shape[0], device=self.device)
            patient_vectors.append(x[rows, indices])
        representation = torch.cat(patient_vectors, dim=-1)
        logits = self.classifier(self.dropout(representation))
        output = {
            "logit": logits,
            "y_prob": self.label_processor.activation(logits),
        }
        if self.label_key in kwargs:
            labels = kwargs[self.label_key]
            if isinstance(labels, (tuple, list)):
                labels = labels[0]
            y_true = torch.as_tensor(labels, device=self.device)
            if self.mode == "binary":
                y_true = y_true.float().reshape_as(logits)
            output["loss"] = self.label_processor.loss(logits, y_true)
            output["y_true"] = y_true
        if kwargs.get("embed", False):
            output["embed"] = representation
        return output
