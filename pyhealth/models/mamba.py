"""Mamba model for PyHealth 2.0 datasets.

This implementation mirrors the Transformer model structure but swaps the
transformer blocks for Mamba blocks.

Currently, it relies on the mambapy package to build the Mamba blocks,
and only supports SequenceProcessor embeddings (as this is only what has 
been tested)
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.processors import SequenceProcessor
from pyhealth.processors.base_processor import FeatureProcessor

try:
    from mambapy.mamba import MambaConfig, ResidualBlock
except Exception as exc:
    ResidualBlock = None
    MambaConfig = None
    _mambapy_import_error = exc
else:
    _mambapy_import_error = None


class MambaLayers(nn.Module):
    """Stacked mambapy residual blocks returning sequence and CLS embeddings."""

    def __init__(
        self,
        feature_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if ResidualBlock is None or MambaConfig is None:
            raise ImportError(
                "mambapy is required for the Mamba model; install mambapy first."
            ) from _mambapy_import_error

        config = MambaConfig(
            d_model=feature_size,
            n_layers=num_layers,
            d_state=d_state,
            expand_factor=expand,
            d_conv=d_conv,
            use_cuda=False, # If this is enabled, then mamba_ssm must also be installed.
        )

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _apply_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Applies a mask to a tensor.
        """
        if mask is None:
            return x
        mask = mask.to(x.device)
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)
        return x * mask.float()

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Mamba layers.

        Args:
            x:  a tensor of shape ``[batch, sequence len, feature_size]``.
            mask: an optional tensor of shape ``[batch, sequence len]``, where
                1 indicates valid and 0 indicates invalid.

        Returns:
            emb: a tensor of shape ``[batch, sequence len, feature_size]``,
                containing the output features for each time step.
            cls_emb: a tensor of shape ``[batch, feature_size]``, containing
                the output features for the last valid time step.
        """

        for layer in self.layers:
            x = self._apply_mask(x, mask)
            x = layer(x)
            x = self.dropout(x)
            x = self._apply_mask(x, mask)
        emb = x
        if mask is None:
            cls_emb = x[:, -1, :]
        else: # Carefully find the last valid hidden state
            seq_mask = mask
            if seq_mask.dim() == 1:
                seq_mask = seq_mask.unsqueeze(1)
            if seq_mask.dim() > 2:
                seq_mask = seq_mask.view(seq_mask.size(0), seq_mask.size(1), -1).any(
                    dim=-1
                )
            lengths = torch.count_nonzero(seq_mask, dim=1).clamp(min=1)
            last_idx = (lengths - 1).view(-1, 1, 1)
            cls_emb = torch.take_along_dim(
                x, last_idx.expand(-1, 1, x.size(-1)), dim=1
            ).squeeze(1)
        return emb, cls_emb


class Mamba(BaseModel):
    """Mamba model for PyHealth datasets.

    Each feature stream is embedded with :class:`EmbeddingModel` and encoded by
    a seperate stack of Mamba blocks (:class:`MambaLayers`). The resulting embeddings 
    from each stream are concatenated and passed through a linear classification head.

    Args:
        dataset: a :class:`pyhealth.datasets.SampleDataset` object.
        embedding_dim: shared embedding dimension
        num_layers: the number of Mamba blocks to stack.
        dropout: the dropout rate to apply to the output of the Mamba blocks.
        d_state: shape of the state space latents
        d_conv: size of the convolution kernel
        expand: expansion factor determining the intermediate size
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__(dataset=dataset)
        if ResidualBlock is None or MambaConfig is None:
            raise ImportError(
                "mambapy is required for the Mamba model; install mambapy first."
            ) from _mambapy_import_error

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if Mamba is initialized"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.feature_processors = {
            feature_key: self.dataset.input_processors[feature_key]
            for feature_key in self.feature_keys
        }
        invalid_processors = {
            feature_key: type(processor).__name__
            for feature_key, processor in self.feature_processors.items()
            if not isinstance(processor, SequenceProcessor)
        }
        if invalid_processors:
            raise TypeError(
                "Mamba supports only SequenceProcessor inputs; "
                f"found unsupported processors: {invalid_processors}"
            )

        self.mamba_layers = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.mamba_layers[feature_key] = MambaLayers(
                feature_size=embedding_dim,
                num_layers=num_layers,
                dropout=dropout,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * embedding_dim, output_size)

    @staticmethod
    def _split_temporal(feature: Any) -> Tuple[Optional[torch.Tensor], Any]:
        """Seperate temporal metadata from a feature payload. Possibly
        not needed for Mamba?
        """
        if isinstance(feature, tuple) and len(feature) == 2:
            return feature
        return None, feature

    def _ensure_tensor(self, feature_key: str, value: Any) -> torch.Tensor:
        """Convert raw feature payloads into tensors on demand.
        Convert raw feature payloads into tensors on demand.

        Args:
            feature_key: Name of the feature in the dataset schema.
            value: Raw payload from the dataloader (tensor, list of codes, etc.).

        Returns:
            torch.Tensor: Tensor representation suitable for the embedding model.
        """
        if isinstance(value, torch.Tensor):
            return value
        processor = self.feature_processors[feature_key]
        if not isinstance(processor, SequenceProcessor):
            raise TypeError(
                "Mamba supports only SequenceProcessor inputs; "
                f"got {type(processor).__name__} for feature {feature_key}"
            )
        return torch.tensor(value, dtype=torch.long)

    def _create_mask(self, feature_key: str, value: torch.Tensor) -> torch.Tensor:
        """Create a boolean mask indicating valid sequence positions.
        """
        processor = self.feature_processors[feature_key]
        if isinstance(processor, SequenceProcessor):
            mask = value != 0
        else:
            raise TypeError(
                "Mamba supports only SequenceProcessor inputs; "
                f"got {type(processor).__name__} for feature {feature_key}"
            )


        if mask.dim() == 1:
            mask = mask.unsqueeze(1)
        mask = mask.bool()
        if mask.dim() == 2:
            invalid_rows = ~mask.any(dim=1)
            if invalid_rows.any():
                mask[invalid_rows, 0] = True
        return mask

    @staticmethod
    def _pool_embedding(x: torch.Tensor) -> torch.Tensor:
        """Pool nested embeddings to ``[batch, seq_len, hidden]`` format.
        """
        if x.dim() == 4:
            x = x.sum(dim=2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    @staticmethod
    def _mask_from_embeddings(x: torch.Tensor) -> torch.Tensor:
        """Infer a boolean mask directly from embedded representations."""

        mask = torch.any(torch.abs(x) > 0, dim=-1)
        if mask.dim() == 1:
            mask = mask.unsqueeze(1)
        invalid_rows = ~mask.any(dim=1)
        if invalid_rows.any():
            mask[invalid_rows, 0] = True
        return mask.bool()

    def forward_from_embedding(
        self,
        feature_embeddings: Dict[str, torch.Tensor],
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass that consumes pre-computed embeddings."""
        patient_emb = []

        for feature_key in self.feature_keys:
            x = feature_embeddings[feature_key].to(self.device)
            x = self._pool_embedding(x)
            mask = self._mask_from_embeddings(x).to(self.device)
            _, cls_emb = self.mamba_layers[feature_key](x, mask)
            patient_emb.append(cls_emb)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with PyHealth 2.0 inputs.

        Args:
            **kwargs: Keyword arguments that include every feature key defined in
                the dataset schema plus the label key.

        Returns:
            Dict[str, torch.Tensor]: Prediction dictionary containing the loss,
            probabilities, logits, labels, and (optionally) embeddings.

        Example:
            This method is invoked by :class:`pyhealth.trainer.Trainer`, which
            supplies batches from :func:`pyhealth.datasets.get_dataloader`.
        """

        patient_emb = []
        embedding_inputs: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}

        for feature_key in self.feature_keys:
            _, value = self._split_temporal(kwargs[feature_key])
            value_tensor = self._ensure_tensor(feature_key, value)
            embedding_inputs[feature_key] = value_tensor
            masks[feature_key] = self._create_mask(feature_key, value_tensor)

        embedded = self.embedding_model(embedding_inputs)

        for feature_key in self.feature_keys:
            x = embedded[feature_key].to(self.device)
            mask = masks[feature_key].to(self.device)
            x = self._pool_embedding(x)
            _, cls_emb = self.mamba_layers[feature_key](x, mask)
            patient_emb.append(cls_emb)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results

if __name__ == "__main__":
    from pyhealth.datasets import SampleDataset, get_dataloader

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "diagnoses": ["A", "B", "C"],
            "procedures": ["X", "Y"],
            "label": 1,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-0",
            "diagnoses": ["D", "E"],
            "procedures": ["Z"],
            "label": 0,
        },
    ]

    input_schema: Dict[str, Union[str, type[FeatureProcessor]]] = {
        "diagnoses": "sequence",
        "procedures": "sequence",
    }
    output_schema: Dict[str, Union[str, type[FeatureProcessor]]] = {"label": "binary"}

    dataset = SampleDataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="test",
    )

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    model = Mamba(dataset=dataset, embedding_dim=64, num_layers=2)

    data_batch = next(iter(train_loader))

    result = model(**data_batch)
    print(result)

    result["loss"].backward()