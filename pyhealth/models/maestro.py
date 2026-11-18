# Author: Chinmaykatpatal
# Description: MAESTRO model implementation for PyHealth.

"""MAESTRO: adaptive sparse attention for multimodal dynamic time series.

Paper:
    Mohapatra, P., Sui, Y., Pandey, A., Xia, S., & Zhu, Q. (2025).
    MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal
    Dynamic Time Series. NeurIPS 2025. https://arxiv.org/abs/2509.25278

Official code: https://github.com/payalmohapatra/MAESTRO

Each input feature is one modality: a tensor of shape
``[batch, time, variates]`` sharing the same time axis. Every modality is
projected and encoded separately with ProbSparse attention (Informer-style
distilling), the encoded sequences are concatenated along time, and a fusion
stack with a sparse mixture-of-experts feed-forward layer combines them. During
training, whole modalities are randomly dropped; the model's per-modality
factor gate then adapts the sparsity of each modality's attention.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class ProbSparseAttention(nn.Module):
    """Multi-head ProbSparse self-attention.

    Only the ``u`` queries with the largest sampled sparsity score attend to
    all keys. The remaining queries receive the mean of the values, so the
    cost grows with ``u * L`` rather than ``L * L``.

    Args:
        d_model: Model width. Must be divisible by ``n_heads``.
        n_heads: Number of attention heads.
        dropout: Dropout applied to the attention output.
        factor: Sparsity factor; ``u = factor * ceil(log L)`` queries are kept.

    Examples:
        >>> import torch
        >>> from pyhealth.models.maestro import ProbSparseAttention
        >>> attn = ProbSparseAttention(d_model=8, n_heads=2)
        >>> attn(torch.randn(3, 6, 8), factor=2).shape
        torch.Size([3, 6, 8])
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        factor: int = 5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.default_factor = factor

    def forward(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """Attend over ``x`` of shape ``[batch, length, d_model]``."""
        batch, length, _ = x.shape
        heads, d_head = self.n_heads, self.d_head

        q = self.q_linear(x).view(batch, length, heads, d_head).transpose(1, 2)
        k, v = self.kv_linear(x).chunk(2, dim=-1)
        k = k.view(batch, length, heads, d_head).transpose(1, 2)
        v = v.view(batch, length, heads, d_head).transpose(1, 2)

        # At least one query is always kept, including for length-1 inputs.
        top_k = max(1, min(factor * math.ceil(math.log(length)), length))

        # Estimate query sparsity from a random subset of keys per query.
        sample_idx = torch.randint(length, (length, top_k), device=x.device)
        k_sample = k[:, :, sample_idx]  # [B, H, L, u, D]
        qk_sample = torch.matmul(q.unsqueeze(3), k_sample.transpose(-2, -1))
        qk_sample = qk_sample.squeeze(3)  # [B, H, L, u]
        sparsity = qk_sample.max(dim=-1)[0] - qk_sample.mean(dim=-1)
        top_idx = torch.topk(sparsity, top_k, dim=-1)[1]  # [B, H, u]

        top_q = q.gather(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, d_head))
        scores = torch.matmul(top_q, k.transpose(-2, -1)) * self.scale
        attn_update = torch.matmul(F.softmax(scores, dim=-1), v)  # [B, H, u, D]

        # Lazy queries take the mean value; selected queries are scattered back.
        context = v.mean(dim=2, keepdim=True).expand(-1, -1, length, -1)
        context = context.scatter(
            2, top_idx.unsqueeze(-1).expand(-1, -1, -1, d_head), attn_update
        )
        context = context.transpose(1, 2).reshape(batch, length, self.d_model)
        return self.dropout(self.out(context))


class SparseMoEFeedForward(nn.Module):
    """Feed-forward layer with top-``k`` routing over ``num_experts`` experts.

    Args:
        d_model: Input and output width.
        expert_dim: Hidden width of each expert.
        num_experts: Number of expert MLPs.
        k: Number of experts each token is routed to.

    Examples:
        >>> import torch
        >>> from pyhealth.models.maestro import SparseMoEFeedForward
        >>> moe = SparseMoEFeedForward(d_model=8, expert_dim=16, num_experts=3)
        >>> moe(torch.randn(2, 5, 8)).shape
        torch.Size([2, 5, 8])
    """

    def __init__(
        self,
        d_model: int,
        expert_dim: int = 256,
        num_experts: int = 4,
        k: int = 1,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, d_model),
            )
            for _ in range(num_experts)
        )
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route each time step of ``x`` ``[batch, length, d_model]``."""
        batch, length, width = x.shape
        flat = x.reshape(batch * length, width)
        scores, indices = torch.topk(self.gate(flat), self.k, dim=-1)

        output = torch.zeros_like(flat)
        for slot in range(self.k):
            expert_ids = indices[:, slot]
            for expert_idx in range(self.num_experts):
                mask = expert_ids == expert_idx
                if not bool(mask.any()):
                    continue
                result = self.experts[expert_idx](flat[mask])
                output[mask] = output[mask] + scores[mask, slot].unsqueeze(1) * result
        return output.reshape(batch, length, width)


class ModalityPositionalEncoder(nn.Module):
    """Adds a learned temporal embedding and a per-modality embedding.

    Args:
        d_model: Embedding width.
        max_len: Longest supported sequence length.
        num_modalities: Number of modalities.

    Examples:
        >>> import torch
        >>> from pyhealth.models.maestro import ModalityPositionalEncoder
        >>> enc = ModalityPositionalEncoder(d_model=8, max_len=10, num_modalities=2)
        >>> enc(torch.zeros(1, 4, 8), modality_id=1).shape
        torch.Size([1, 4, 8])
    """

    def __init__(self, d_model: int, max_len: int, num_modalities: int) -> None:
        super().__init__()
        self.temporal_pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.modality_pe = nn.Embedding(num_modalities, d_model)
        nn.init.normal_(self.temporal_pe, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, modality_id: int) -> torch.Tensor:
        """Add positional information to ``x`` ``[batch, length, d_model]``."""
        length = x.shape[1]
        modality = self.modality_pe(
            torch.tensor(modality_id, device=x.device)
        ).view(1, 1, -1)
        return x + self.temporal_pe[:, :length, :] + modality


class TemporalPositionalEncoder(nn.Module):
    """Adds fixed sinusoidal position encodings.

    Args:
        d_model: Embedding width.
        max_len: Longest supported sequence length.

    Examples:
        >>> import torch
        >>> from pyhealth.models.maestro import TemporalPositionalEncoder
        >>> enc = TemporalPositionalEncoder(d_model=8, max_len=10)
        >>> enc(torch.zeros(1, 4, 8)).shape
        torch.Size([1, 4, 8])
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position encodings to ``x`` ``[batch, length, d_model]``."""
        return x + self.pe[:, : x.shape[1], :]


class InformerEncoderLayer(nn.Module):
    """ProbSparse attention block with distilling convolution and a feed-forward.

    The distilling step halves the sequence length (stride 2), so the output has
    length ``ceil(L / 2)``.

    Args:
        d_model: Model width.
        n_heads: Number of attention heads.
        d_ff: Hidden width of the dense feed-forward. Ignored when ``moe`` is
            true.
        dropout: Dropout rate.
        moe: Use a :class:`SparseMoEFeedForward` instead of a dense MLP.
        num_experts: Number of experts when ``moe`` is true.
        k: Experts per token when ``moe`` is true.

    Examples:
        >>> import torch
        >>> from pyhealth.models.maestro import InformerEncoderLayer
        >>> layer = InformerEncoderLayer(d_model=8, n_heads=2, d_ff=16)
        >>> layer(torch.randn(2, 9, 8), factor=2).shape
        torch.Size([2, 5, 8])
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        moe: bool = False,
        num_experts: int = 4,
        k: int = 1,
    ) -> None:
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        if moe:
            self.ff: nn.Module = SparseMoEFeedForward(
                d_model, d_model, num_experts=num_experts, k=k
            )
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """Encode ``x`` of shape ``[batch, length, d_model]``.

        Returns a tensor of shape ``[batch, ceil(length / 2), d_model]``.
        """
        x = self.norm1(x + self.attention(x, factor))
        distilled = x.permute(0, 2, 1)
        x = (self.conv(distilled) + self.pool(distilled)).permute(0, 2, 1)
        return self.norm2(x + self.ff(x))


def _modality_dropout(
    inputs: dict[str, torch.Tensor],
    keys: list[str],
    drop_prob: float,
    training: bool,
) -> list[float]:
    """Zero out whole modalities at random for the batch; return presence flags.

    The draw is per modality and shared by the whole batch, as in the official
    implementation.
    """
    present: list[float] = []
    for key in keys:
        dropped = training and drop_prob > 0 and bool(torch.rand(1).item() < drop_prob)
        if dropped:
            inputs[key] = torch.zeros_like(inputs[key])
        present.append(0.0 if dropped else 1.0)
    return present


class MAESTRO(BaseModel):
    """MAESTRO: multimodal time-series classifier with sparse attention.

    Every dataset input feature is treated as one modality. Each feature must be
    a tensor of shape ``(time, variates)`` per sample, and all modalities must
    share the same ``time`` length. Missing modalities can be expressed by
    passing zeros for that feature.

    Paper:
        Mohapatra et al. (2025). MAESTRO: Adaptive Sparse Attention and Robust
        Learning for Multimodal Dynamic Time Series. NeurIPS 2025.

    Args:
        dataset: Fitted ``SampleDataset`` with one label key and one or more
            tensor input features.
        d_model: Width of the model. Must be divisible by ``n_heads``.
            Defaults to 64.
        n_heads: Attention heads per layer. Defaults to 8.
        num_layers_per_modal: ProbSparse layers per modality. Defaults to 2.
        num_layers: Fusion layers with mixture-of-experts. Defaults to 2.
        num_experts: Experts per fusion layer. Defaults to 4.
        dropout: Dropout rate. Defaults to 0.1.
        modality_dropout: Probability of dropping each modality during training.
            Defaults to 0.2.
        base_factor: Maximum ProbSparse factor. Defaults to 5.
        input_length: Longest supported time length. Inputs longer than this
            raise an error. Defaults to 256.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import MAESTRO
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "ecg": [[0.1, 0.2]] * 8,
        ...         "eda": [[0.3]] * 8,
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "ecg": [[0.4, 0.5]] * 8,
        ...         "eda": [[0.6]] * 8,
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"ecg": "tensor", "eda": "tensor"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="maestro_example",
        ... )
        >>> model = MAESTRO(dataset=dataset, d_model=16, n_heads=2)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> output = model(**batch)
        >>> output["y_prob"].shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        d_model: int = 64,
        n_heads: int = 8,
        num_layers_per_modal: int = 2,
        num_layers: int = 2,
        num_experts: int = 4,
        dropout: float = 0.1,
        modality_dropout: float = 0.2,
        base_factor: int = 5,
        input_length: int = 256,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.label_keys) != 1:
            raise ValueError("MAESTRO supports exactly one label key.")
        if not self.feature_keys:
            raise ValueError("MAESTRO needs at least one input feature (modality).")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1).")
        if input_length < 1 or base_factor < 1:
            raise ValueError("input_length and base_factor must be positive.")

        self.label_key = self.label_keys[0]
        self.modalities: list[str] = list(self.feature_keys)
        self.variates: dict[str, int] = {
            key: self._infer_variates(key) for key in self.modalities
        }
        self.modality_dropout_prob = modality_dropout
        self.base_factor = base_factor
        self.input_length = input_length

        self.input_projections = nn.ModuleDict(
            {key: nn.Linear(self.variates[key], d_model) for key in self.modalities}
        )
        self.temporal_pos_encoder = TemporalPositionalEncoder(d_model, input_length)
        self.modality_pos_encoder = ModalityPositionalEncoder(
            d_model, input_length, len(self.modalities)
        )
        self.per_modal_layers = nn.ModuleDict(
            {
                key: nn.ModuleList(
                    InformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout)
                    for _ in range(num_layers_per_modal)
                )
                for key in self.modalities
            }
        )
        self.fusion_layers = nn.ModuleList(
            InformerEncoderLayer(
                d_model,
                n_heads,
                4 * d_model,
                dropout,
                moe=True,
                num_experts=num_experts,
                k=1,
            )
            for _ in range(num_layers)
        )
        self.factor_gate = nn.Sequential(
            nn.Linear(len(self.modalities), len(self.modalities)),
            nn.ReLU(),
            nn.Linear(len(self.modalities), len(self.modalities)),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.label_processor.output_size()),
        )

    def _extract_feature_value(
        self,
        feature_key: str,
        feature: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Return the tensor value of a feature, handling processor tuples."""
        if isinstance(feature, torch.Tensor):
            return feature
        schema = self.dataset.input_processors[feature_key].schema()
        if "value" in schema:
            value = feature[schema.index("value")]
            if isinstance(value, torch.Tensor):
                return value
        raise ValueError(
            f"Feature '{feature_key}' must provide a tensor value in its schema."
        )

    def _infer_variates(self, feature_key: str) -> int:
        """Infer the number of variates of a modality from its first sample."""
        value = self._extract_feature_value(feature_key, self.dataset[0][feature_key])
        if value.dim() != 2:
            raise ValueError(
                f"Modality '{feature_key}' must have shape (time, variates); "
                f"got {tuple(value.shape)}."
            )
        return int(value.shape[-1])

    def forward(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Classify a batch of multimodal time series.

        Args:
            **kwargs: One tensor per modality with shape
                ``[batch, time, variates]``, plus the label tensor when a loss
                is wanted.

        Returns:
            A dictionary with ``logit`` and ``y_prob``. When the label key is
            present, it also contains ``loss`` and ``y_true``.
        """
        inputs: dict[str, torch.Tensor] = {}
        for key in self.modalities:
            if key not in kwargs:
                raise ValueError(f"Missing modality input '{key}'.")
            value = self._extract_feature_value(
                key, cast(torch.Tensor | tuple[torch.Tensor, ...], kwargs[key])
            ).to(self.device).float()
            if value.dim() != 3 or value.shape[-1] != self.variates[key]:
                raise ValueError(
                    f"Modality '{key}' must have shape "
                    f"(batch, time, {self.variates[key]})."
                )
            inputs[key] = value

        lengths = {value.shape[1] for value in inputs.values()}
        batch_sizes = {value.shape[0] for value in inputs.values()}
        if len(lengths) != 1 or len(batch_sizes) != 1:
            raise ValueError("All modalities must share batch size and time length.")
        if lengths.pop() > self.input_length:
            raise ValueError(
                f"Time length exceeds input_length={self.input_length}."
            )

        present = _modality_dropout(
            inputs, self.modalities, self.modality_dropout_prob, self.training
        )
        mask = torch.tensor(present, device=self.device)
        gate = self.factor_gate(mask) * self.base_factor
        # The ProbSparse factor is an integer, so it is set from the gate with
        # ceil(). Gradients do not flow through this step.
        factors = torch.ceil(gate.detach() * mask + 1e-3 * (1 - mask))
        factors = factors.clamp(min=1).long().tolist()

        pieces = []
        for idx, key in enumerate(self.modalities):
            h = self.input_projections[key](inputs[key])
            h = self.temporal_pos_encoder(h)
            for layer in self.per_modal_layers[key]:
                h = layer(h, factor=factors[idx])
            pieces.append(self.modality_pos_encoder(h, modality_id=idx))

        fused = torch.cat(pieces, dim=1)
        for layer in self.fusion_layers:
            fused = layer(fused, self.base_factor)

        logits = self.classifier(fused.mean(dim=1))
        y_prob = self.label_processor.activation(logits)
        results: dict[str, torch.Tensor] = {"logit": logits, "y_prob": y_prob}
        if self.label_key in kwargs:
            y_true = cast(torch.Tensor, kwargs[self.label_key]).to(self.device)
            results["loss"] = self.label_processor.loss(logits, y_true)
            results["y_true"] = y_true
        return results
