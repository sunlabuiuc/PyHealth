"""Temporal Pointwise Convolution model for PyHealth.

Contributor: Hasham Ul Haq (huhaq2)
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
    in the Intensive Care Unit
Paper link: https://arxiv.org/abs/2007.09483
Description: PyHealth-adapted Temporal Pointwise Convolution (TPC) model for
    sequential EHR prediction tasks. The implementation follows the
    ``BaseModel`` interface, supports standard sequence processors together
    with StageNet-style temporal inputs, and is intended for reproducible
    length-of-stay experiments on existing PyHealth MIMIC-IV pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type, Union

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.models.utils import get_last_visit
from pyhealth.processors import (
    MultiHotProcessor,
    SequenceProcessor,
    StageNetProcessor,
    StageNetTensorProcessor,
    TensorProcessor,
    TimeseriesProcessor,
)
from pyhealth.processors.base_processor import FeatureProcessor


class CausalTrim1d(nn.Module):
    """Trim the right side of a padded 1D convolution output."""

    def __init__(self, trim_size: int):
        super().__init__()
        self.trim_size = trim_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.trim_size == 0:
            return x
        return x[:, :, :-self.trim_size]


class TPCBlock(nn.Module):
    """Single temporal-pointwise convolution residual block.

    The block first applies a causal temporal convolution over the sequence axis
    and then a pointwise ``1x1`` convolution to mix channels. A residual
    connection keeps the feature dimension fixed across blocks.

    Args:
        feature_dim: Input and output feature size.
        hidden_dim: Hidden channel count used inside the block.
        kernel_size: Temporal convolution kernel size.
        dropout: Dropout probability applied before the residual addition.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = max(kernel_size - 1, 0)
        self.norm = nn.LayerNorm(feature_dim)
        self.temporal_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=hidden_dim * 2,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.trim = CausalTrim1d(padding)
        self.activation = nn.GLU(dim=1)
        self.pointwise_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=feature_dim,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the residual TPC block.

        Args:
            x: Input tensor of shape ``[batch, seq_len, feature_dim]``.
            mask: Optional boolean padding mask ``[batch, seq_len]``.

        Returns:
            Tensor of shape ``[batch, seq_len, feature_dim]``.
        """

        residual = x
        y = self.norm(x).transpose(1, 2)
        y = self.temporal_conv(y)
        y = self.trim(y)
        y = self.activation(y)
        y = self.pointwise_conv(y).transpose(1, 2)
        y = self.dropout(y)
        x = residual + y
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x


class TPCLayer(nn.Module):
    """Stack of temporal-pointwise convolution blocks.

    Args:
        feature_dim: Input and output embedding dimension.
        hidden_dim: Hidden channel count in each TPC block.
        num_layers: Number of residual TPC blocks.
        kernel_size: Temporal convolution kernel size.
        dropout: Dropout probability.

    Examples:
        >>> x = torch.randn(2, 5, 32)
        >>> layer = TPCLayer(feature_dim=32, hidden_dim=64, num_layers=2)
        >>> outputs, pooled = layer(x)
        >>> outputs.shape
        torch.Size([2, 5, 32])
        >>> pooled.shape
        torch.Size([2, 32])
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout

        self.blocks = nn.ModuleList(
            [
                TPCBlock(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a feature sequence and return sequence + pooled outputs.

        Args:
            x: Tensor of shape ``[batch, seq_len, feature_dim]``.
            mask: Optional boolean mask ``[batch, seq_len]``.

        Returns:
            Tuple containing:
                - Encoded sequence ``[batch, seq_len, feature_dim]``
                - Last valid embedding ``[batch, feature_dim]``
        """

        if mask is None:
            mask = torch.ones(
                x.size(0),
                x.size(1),
                dtype=torch.bool,
                device=x.device,
            )
        else:
            mask = mask.bool().to(x.device)

        for block in self.blocks:
            x = block(x, mask)

        x = self.output_norm(x)
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        pooled = get_last_visit(x, mask)
        return x, pooled


class TPC(BaseModel):
    """Temporal Pointwise Convolution model for sequential EHR prediction.

    This implementation adapts the high-level TPC idea to PyHealth's processor
    abstractions. Each feature stream is embedded independently, optionally
    enriched with temporal information when available, encoded by a dedicated
    :class:`TPCLayer`, and finally concatenated for prediction.

    Supported input processor families include:
    - ``SequenceProcessor``
    - ``StageNetProcessor``
    - ``StageNetTensorProcessor``
    - ``TimeseriesProcessor``
    - ``TensorProcessor``
    - ``MultiHotProcessor``

    Args:
        dataset: SampleDataset used to configure feature and label processors.
        embedding_dim: Shared embedding dimension for all feature streams.
        hidden_dim: Hidden channel size inside each TPC block.
        num_layers: Number of TPC blocks per feature stream.
        kernel_size: Temporal convolution kernel size.
        dropout: Dropout probability.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "conditions": ["A", "B"],
        ...         "procedures": ["X"],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v0",
        ...         "conditions": ["C"],
        ...         "procedures": ["Y", "Z"],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "sequence", "procedures": "sequence"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="test_tpc",
        ... )
        >>> model = TPC(dataset=dataset, embedding_dim=32, hidden_dim=64)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))
        >>> output = model(**batch)
        >>> sorted(output.keys())
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout

        assert len(self.label_keys) == 1, "TPC supports single-label tasks only"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.feature_processors = {
            key: self.dataset.input_processors[key] for key in self.feature_keys
        }

        self.time_projections = nn.ModuleDict()
        self.tpc_layers = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.tpc_layers[feature_key] = TPCLayer(
                feature_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            self.time_projections[feature_key] = nn.Linear(1, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            len(self.feature_keys) * embedding_dim,
            self.get_output_size(),
        )

    @staticmethod
    def _split_temporal(feature: Any) -> Tuple[Optional[torch.Tensor], Any]:
        """Split a feature tuple into ``(time, value)`` when applicable."""
        if isinstance(feature, tuple) and len(feature) == 2:
            return feature
        return None, feature

    def _ensure_tensor(self, feature_key: str, value: Any) -> torch.Tensor:
        """Convert a raw feature value into a tensor when needed."""
        if isinstance(value, torch.Tensor):
            return value
        processor = self.feature_processors[feature_key]
        if isinstance(processor, (SequenceProcessor, StageNetProcessor)):
            return torch.tensor(value, dtype=torch.long)
        return torch.tensor(value, dtype=torch.float)

    @staticmethod
    def _ensure_time_tensor(time_value: Optional[Any]) -> Optional[torch.Tensor]:
        """Convert optional temporal metadata into a float tensor."""
        if time_value is None:
            return None
        if isinstance(time_value, torch.Tensor):
            return time_value.float()
        return torch.tensor(time_value, dtype=torch.float)

    def _create_mask(self, feature_key: str, value: torch.Tensor) -> torch.Tensor:
        """Create a sequence mask tailored to the feature processor."""
        processor = self.feature_processors[feature_key]
        if isinstance(processor, SequenceProcessor):
            mask = value != 0
        elif isinstance(processor, StageNetProcessor):
            mask = torch.any(value != 0, dim=-1) if value.dim() >= 3 else value != 0
        elif isinstance(processor, (TimeseriesProcessor, StageNetTensorProcessor)):
            if value.dim() >= 3:
                mask = torch.any(torch.abs(value) > 0, dim=-1)
            elif value.dim() == 2:
                mask = torch.any(torch.abs(value) > 0, dim=-1, keepdim=True)
            else:
                mask = torch.ones(
                    value.size(0),
                    1,
                    dtype=torch.bool,
                    device=value.device,
                )
        elif isinstance(processor, (TensorProcessor, MultiHotProcessor)):
            mask = torch.ones(
                value.size(0),
                1,
                dtype=torch.bool,
                device=value.device,
            )
        else:
            mask = (
                torch.any(value != 0, dim=-1)
                if value.dim() >= 2
                else torch.ones(
                    value.size(0),
                    1,
                    dtype=torch.bool,
                    device=value.device,
                )
            )
        if mask.dim() == 1:
            mask = mask.unsqueeze(1)
        mask = mask.bool()
        invalid = ~mask.any(dim=1)
        if invalid.any():
            mask[invalid, 0] = True
        return mask

    @staticmethod
    def _pool_embedding(x: torch.Tensor) -> torch.Tensor:
        """Collapse unordered dimensions so every feature becomes sequential."""
        if x.dim() == 4:
            x = x.sum(dim=2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    @staticmethod
    def _align_time(
        time_tensor: Optional[torch.Tensor],
        target_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Pad or trim a time tensor to match the encoded sequence length."""
        if time_tensor is None:
            return None
        time_tensor = time_tensor.to(device).float()
        if time_tensor.dim() == 3 and time_tensor.size(-1) == 1:
            time_tensor = time_tensor.squeeze(-1)
        if time_tensor.dim() == 1:
            time_tensor = time_tensor.unsqueeze(0)
        if time_tensor.size(1) > target_length:
            time_tensor = time_tensor[:, :target_length]
        elif time_tensor.size(1) < target_length:
            pad = torch.zeros(
                time_tensor.size(0),
                target_length - time_tensor.size(1),
                device=device,
                dtype=time_tensor.dtype,
            )
            time_tensor = torch.cat([time_tensor, pad], dim=1)
        return time_tensor

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Run the TPC model on a batch of PyHealth samples."""
        patient_embeddings = []
        embedding_inputs: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}
        time_tensors: Dict[str, Optional[torch.Tensor]] = {}

        for feature_key in self.feature_keys:
            time_value, value = self._split_temporal(kwargs[feature_key])
            value_tensor = self._ensure_tensor(feature_key, value)
            embedding_inputs[feature_key] = value_tensor
            masks[feature_key] = self._create_mask(feature_key, value_tensor)
            time_tensors[feature_key] = self._ensure_time_tensor(time_value)

        embedded = self.embedding_model(embedding_inputs)

        for feature_key in self.feature_keys:
            x = embedded[feature_key].to(self.device)
            x = self._pool_embedding(x)

            mask = masks[feature_key].to(self.device)
            time_tensor = self._align_time(
                time_tensors[feature_key],
                target_length=x.size(1),
                device=self.device,
            )
            if time_tensor is not None:
                x = x + self.time_projections[feature_key](time_tensor.unsqueeze(-1))

            _, pooled = self.tpc_layers[feature_key](x, mask)
            patient_embeddings.append(pooled)

        patient_embedding = torch.cat(patient_embeddings, dim=1)
        logits = self.fc(self.dropout(patient_embedding))
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        output = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            output["embed"] = patient_embedding
        return output


if __name__ == "__main__":
    from pyhealth.datasets import create_sample_dataset, get_dataloader

    samples = [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "conditions": ["A", "B"],
            "procedures": ["X"],
            "label": 1,
        },
        {
            "patient_id": "p1",
            "visit_id": "v0",
            "conditions": ["C"],
            "procedures": ["Y", "Z"],
            "label": 0,
        },
    ]
    input_schema: Dict[str, Union[str, Type[FeatureProcessor]]] = {
        "conditions": "sequence",
        "procedures": "sequence",
    }
    output_schema: Dict[str, Union[str, Type[FeatureProcessor]]] = {"label": "binary"}

    dataset = create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="test_tpc_dataset",
    )
    model = TPC(dataset=dataset, embedding_dim=32, hidden_dim=64)
    batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))
    out = model(**batch)
    print("keys:", sorted(out.keys()))
    out["loss"].backward()
