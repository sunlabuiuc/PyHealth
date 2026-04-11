# Author: Joshua Steier
# Paper title: Jamba: A Hybrid Transformer-Mamba Language Model (AI21 Labs)
# Paper link: https://arxiv.org/abs/2403.19887
# Description: Hybrid Transformer-Mamba model for EHR sequential clinical
#     prediction tasks. Interleaves PyHealth's TransformerBlock (self-attention)
#     and MambaBlock (selective SSM) in a configurable ratio following the Jamba
#     architecture, combining attention's long-range dependency modeling with
#     SSM's linear-time efficiency for long patient histories.

from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.models.transformer import TransformerBlock
from pyhealth.models.ehrmamba import MambaBlock
from pyhealth.models.utils import get_last_visit


def build_layer_schedule(
    num_transformer_layers: int,
    num_mamba_layers: int,
) -> List[str]:
    """Build an interleaved layer schedule distributing Transformer layers evenly.

    Follows the Jamba design principle of spacing attention layers throughout
    the stack rather than clustering them. For example, with 2 Transformer
    and 6 Mamba layers the schedule becomes:
        ``['mamba', 'mamba', 'mamba', 'transformer', 'mamba', 'mamba',
          'mamba', 'transformer']``

    Args:
        num_transformer_layers (int): Number of Transformer (attention) layers.
        num_mamba_layers (int): Number of Mamba (SSM) layers.

    Returns:
        List[str]: Ordered list of ``"transformer"`` or ``"mamba"`` strings.

    Example:
        >>> build_layer_schedule(2, 6)
        ['mamba', 'mamba', 'mamba', 'transformer', 'mamba', 'mamba', 'mamba', 'transformer']
    """
    total = num_transformer_layers + num_mamba_layers
    if total == 0:
        return []
    if num_transformer_layers == 0:
        return ["mamba"] * num_mamba_layers
    if num_mamba_layers == 0:
        return ["transformer"] * num_transformer_layers

    schedule = ["mamba"] * total
    stride = total / num_transformer_layers
    for i in range(num_transformer_layers):
        idx = int((i + 1) * stride) - 1
        idx = min(idx, total - 1)
        schedule[idx] = "transformer"

    return schedule


class JambaLayer(nn.Module):
    """Hybrid Transformer-Mamba encoder stack.

    Interleaves :class:`TransformerBlock` and :class:`MambaBlock` layers
    following a configurable schedule derived from the Jamba architecture
    (AI21 Labs, 2024). Both layer types operate on ``(batch, seq_len, hidden)``
    tensors, making them composable within a single sequential stack.

    Args:
        feature_size (int): Hidden dimension shared by all layers.
        num_transformer_layers (int): Number of attention layers. Default 2.
        num_mamba_layers (int): Number of SSM layers. Default 6.
        heads (int): Attention heads for Transformer layers. Default 4.
        dropout (float): Dropout rate for Transformer layers. Default 0.3.
        state_size (int): SSM state size for Mamba layers. Default 16.
        conv_kernel (int): Causal conv kernel for Mamba layers. Default 4.

    Examples:
        >>> from pyhealth.models.jamba_ehr import JambaLayer
        >>> x = torch.randn(3, 128, 64)
        >>> layer = JambaLayer(64, num_transformer_layers=1, num_mamba_layers=3)
        >>> emb, cls_emb = layer(x)
        >>> emb.shape
        torch.Size([3, 128, 64])
        >>> cls_emb.shape
        torch.Size([3, 64])
    """

    def __init__(
        self,
        feature_size: int,
        num_transformer_layers: int = 2,
        num_mamba_layers: int = 6,
        heads: int = 4,
        dropout: float = 0.3,
        state_size: int = 16,
        conv_kernel: int = 4,
    ):
        super(JambaLayer, self).__init__()
        self.feature_size = feature_size
        self.num_transformer_layers = num_transformer_layers
        self.num_mamba_layers = num_mamba_layers

        self.schedule = build_layer_schedule(
            num_transformer_layers, num_mamba_layers
        )

        self.layers = nn.ModuleList()
        for layer_type in self.schedule:
            if layer_type == "transformer":
                self.layers.append(
                    TransformerBlock(
                        hidden=feature_size,
                        attn_heads=heads,
                        dropout=dropout,
                    )
                )
            else:
                self.layers.append(
                    MambaBlock(
                        d_model=feature_size,
                        state_size=state_size,
                        conv_kernel=conv_kernel,
                    )
                )

        self.final_norm = nn.LayerNorm(feature_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation through the hybrid layer stack.

        Args:
            x (torch.Tensor): Input of shape ``[batch, seq_len, feature_size]``.
            mask (Optional[torch.Tensor]): Padding mask ``[batch, seq_len]``
                where 1 = valid, 0 = pad.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ``emb``: ``[batch, seq_len, feature_size]`` per-step features.
                - ``cls_emb``: ``[batch, feature_size]`` from last valid step.
        """
        attn_mask = None
        if mask is not None:
            attn_mask = torch.einsum("ab,ac->abc", mask, mask)

        for i, layer in enumerate(self.layers):
            if self.schedule[i] == "transformer":
                x = layer(x, attn_mask)
            else:
                x = layer(x)

        x = self.final_norm(x)
        emb = x
        cls_emb = get_last_visit(x, mask)
        return emb, cls_emb


class JambaEHR(BaseModel):
    """JambaEHR: Hybrid Transformer-Mamba model for clinical EHR prediction.

    Paper: Lieber et al. Jamba: A Hybrid Transformer-Mamba Language Model.
    arXiv 2403.19887, 2024.

    This model interleaves Transformer self-attention and Mamba selective-SSM
    layers for each feature stream. Configurable layer counts control the
    attention-to-SSM ratio, trading off between global dependency modeling
    and linear-time sequence processing for long EHR histories.

    Each feature stream is embedded with :class:`EmbeddingModel` and encoded
    by an independent :class:`JambaLayer`. The resulting patient embeddings
    are concatenated and projected through a classification head.

    Args:
        dataset (SampleDataset): Dataset providing processed inputs.
        embedding_dim (int): Embedding and hidden dimension. Default 128.
        num_transformer_layers (int): Transformer layers per stream. Default 2.
        num_mamba_layers (int): Mamba layers per stream. Default 6.
        heads (int): Attention heads per Transformer block. Default 4.
        dropout (float): Dropout rate. Default 0.3.
        state_size (int): SSM state size in Mamba blocks. Default 16.
        conv_kernel (int): Causal conv kernel in Mamba blocks. Default 4.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "diagnoses": ["A", "B", "C"],
        ...         "procedures": ["X", "Y"],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-0",
        ...         "diagnoses": ["D"],
        ...         "procedures": ["Z", "Y"],
        ...         "label": 0,
        ...     },
        ... ]
        >>> input_schema = {
        ...     "diagnoses": "sequence",
        ...     "procedures": "sequence",
        ... }
        >>> output_schema = {"label": "binary"}
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     input_schema,
        ...     output_schema,
        ...     dataset_name="demo",
        ... )
        >>> model = JambaEHR(dataset=dataset, embedding_dim=64)
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> batch = next(iter(loader))
        >>> output = model(**batch)
        >>> sorted(output.keys())
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        num_transformer_layers: int = 2,
        num_mamba_layers: int = 6,
        heads: int = 4,
        dropout: float = 0.3,
        state_size: int = 16,
        conv_kernel: int = 4,
    ):
        super(JambaEHR, self).__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_mamba_layers = num_mamba_layers
        self.heads = heads
        self.dropout_rate = dropout
        self.state_size = state_size
        self.conv_kernel = conv_kernel

        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if JambaEHR is initialized"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        self.jamba: nn.ModuleDict = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.jamba[feature_key] = JambaLayer(
                feature_size=embedding_dim,
                num_transformer_layers=num_transformer_layers,
                num_mamba_layers=num_mamba_layers,
                heads=heads,
                dropout=dropout,
                state_size=state_size,
                conv_kernel=conv_kernel,
            )

        output_size = self.get_output_size()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            len(self.feature_keys) * embedding_dim, output_size
        )

    @staticmethod
    def _pool_embedding(x: torch.Tensor) -> torch.Tensor:
        """Collapse nested embeddings to ``[batch, seq_len, hidden]``.

        Handles 4-D inputs from categorical processors by summing over the
        inner token dimension, and 2-D inputs by adding a length-1 sequence
        dimension.

        Args:
            x (torch.Tensor): Embedded tensor from EmbeddingModel.

        Returns:
            torch.Tensor: 3-D tensor ``[batch, seq_len, hidden]``.
        """
        if x.dim() == 4:
            x = x.sum(dim=2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    @staticmethod
    def _mask_from_embeddings(x: torch.Tensor) -> torch.Tensor:
        """Derive a padding mask from embedded representations.

        Marks positions where all hidden features are zero as invalid. Ensures
        at least one position per sample is valid (sets index 0 if needed).

        Args:
            x (torch.Tensor): Embedded tensor ``[batch, seq_len, hidden]``.

        Returns:
            torch.Tensor: Boolean mask ``[batch, seq_len]``.
        """
        mask = torch.any(torch.abs(x) > 0, dim=-1)
        if mask.dim() == 1:
            mask = mask.unsqueeze(1)
        invalid_rows = ~mask.any(dim=1)
        if invalid_rows.any():
            mask[invalid_rows, 0] = True
        return mask.bool()

    def forward(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Embeds each feature stream, encodes through the hybrid
        Transformer-Mamba stack, concatenates per-stream patient
        representations, and projects to label space.

        Args:
            **kwargs: Must include all feature keys (tensors or tuples
                following processor schema) and the label key.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with keys ``loss``,
                ``y_prob``, ``y_true``, ``logit``, and optionally
                ``embed`` if ``kwargs["embed"] is True``.
        """
        patient_emb = []

        for feature_key in self.feature_keys:
            feature = kwargs[feature_key]

            if isinstance(feature, torch.Tensor):
                feature = (feature,)

            schema = self.dataset.input_processors[feature_key].schema()

            value = (
                feature[schema.index("value")]
                if "value" in schema
                else None
            )
            mask = (
                feature[schema.index("mask")]
                if "mask" in schema
                else None
            )

            if len(feature) == len(schema) + 1 and mask is None:
                mask = feature[-1]

            if value is None:
                raise ValueError(
                    f"Feature '{feature_key}' must contain 'value' "
                    f"in the schema."
                )
            else:
                value = value.to(self.device)

            value = self.embedding_model(
                {feature_key: value}
            )[feature_key]
            value = self._pool_embedding(value)

            if mask is not None:
                mask = mask.to(self.device)
                if mask.dim() == value.dim():
                    mask = mask.any(dim=-1)
                mask = mask.float()
            else:
                mask = self._mask_from_embeddings(value).float()

            _, cls_emb = self.jamba[feature_key](value, mask)
            patient_emb.append(cls_emb)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(self.dropout(patient_emb))
        y_prob = self.prepare_y_prob(logits)

        results = {
            "logit": logits,
            "y_prob": y_prob,
        }

        if self.label_key in kwargs:
            y_true = cast(
                torch.Tensor, kwargs[self.label_key]
            ).to(self.device)
            loss = self.get_loss_function()(logits, y_true)
            results["loss"] = loss
            results["y_true"] = y_true

        if kwargs.get("embed", False):
            results["embed"] = patient_emb

        return results


if __name__ == "__main__":
    from pyhealth.datasets import create_sample_dataset, get_dataloader

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

    input_schema = {
        "diagnoses": "sequence",
        "procedures": "sequence",
    }
    output_schema = {"label": "binary"}

    dataset = create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="test",
    )

    # Default Jamba ratio: 2 Transformer + 6 Mamba
    print("=== JambaEHR (2T + 6M) ===")
    model = JambaEHR(
        dataset=dataset,
        embedding_dim=64,
        num_transformer_layers=2,
        num_mamba_layers=6,
        heads=2,
    )
    loader = get_dataloader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    out = model(**batch)
    print("keys:", sorted(out.keys()))
    print("logit shape:", out["logit"].shape)
    out["loss"].backward()
    print("backward OK\n")

    # Pure Transformer fallback
    print("=== JambaEHR pure Transformer (4T + 0M) ===")
    model_t = JambaEHR(
        dataset=dataset,
        embedding_dim=64,
        num_transformer_layers=4,
        num_mamba_layers=0,
        heads=2,
    )
    batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=True)))
    out = model_t(**batch)
    print("keys:", sorted(out.keys()))
    out["loss"].backward()
    print("backward OK\n")

    # Pure Mamba fallback
    print("=== JambaEHR pure Mamba (0T + 4M) ===")
    model_m = JambaEHR(
        dataset=dataset,
        embedding_dim=64,
        num_transformer_layers=0,
        num_mamba_layers=4,
    )
    batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=True)))
    out = model_m(**batch)
    print("keys:", sorted(out.keys()))
    out["loss"].backward()
    print("backward OK")