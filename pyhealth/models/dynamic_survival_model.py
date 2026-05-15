"""Dynamic Survival Analysis model for PyHealth.

GRU / LSTM / causal-Transformer based model implementing the DSA pipeline
from Yèche et al. (CHIL 2024).  Compatible with
:class:`~pyhealth.datasets.SampleDataset` and the PyHealth
:class:`~pyhealth.trainer.Trainer`.

References:
    Yèche H. et al., "Dynamic Survival Analysis for Early Event Prediction",
    Proceedings of Machine Learning for Health (CHIL), 2024.
    https://proceedings.mlr.press/v248/yeche24a.html

Example::

    from pyhealth.datasets import create_sample_dataset, get_dataloader
    from pyhealth.models import DynamicSurvivalModel
    from pyhealth.tasks import DecompensationDSA
    from pyhealth.tasks.decompensation_dsa import make_synthetic_dsa_samples

    samples = make_synthetic_dsa_samples(n_patients=100, n_features=8, horizon=24)
    dataset = create_sample_dataset(
        samples=samples,
        input_schema=DecompensationDSA.input_schema,
        output_schema=DecompensationDSA.output_schema,
        dataset_name="dsa_synthetic",
    )
    model = DynamicSurvivalModel(dataset=dataset, input_dim=8, horizon=24)
    loader = get_dataloader(dataset, batch_size=16, shuffle=True)
    batch  = next(iter(loader))
    out    = model(**batch)
    # out keys: "loss", "y_prob", "y_true", "logit"
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


# ---------------------------------------------------------------------------
# Internal sub-modules
# ---------------------------------------------------------------------------


class _LinearEmbedding(nn.Module):
    """Linear time-step embedding with L1 regularisation on weights.

    Args:
        input_dim: Input feature dimension D.
        embedding_dim: Output embedding dimension E.
        l1_weight: Scale factor for the L1 regularisation term.
    """

    def __init__(self, input_dim: int, embedding_dim: int, l1_weight: float = 1.0) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.linear = nn.Linear(input_dim, embedding_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input features.

        Args:
            x: FloatTensor of shape ``(B, T, D)``.

        Returns:
            FloatTensor of shape ``(B, T, E)``.
        """
        return self.linear(x)

    def l1_loss(self) -> torch.Tensor:
        """Return the L1 regularisation term (scalar)."""
        return self.l1_weight * self.linear.weight.abs().sum()


class _CausalTransformerEncoder(nn.Module):
    """Causal self-attention encoder with learned positional embeddings.

    A drop-in alternative to the GRU / LSTM backbone.  Each time step
    attends only to itself and earlier steps so the encoder preserves real-
    time semantics for the DSA task.

    Args:
        embedding_dim: Dimensionality of the input (post-embedding) tensor.
        hidden_dim: Transformer model dimension (d_model); also the output
            dimension.  Must be divisible by ``num_heads``.
        num_layers: Number of stacked transformer blocks.
        num_heads: Number of attention heads.
        ffn_dim: Dimensionality of the feed-forward network.
        dropout: Dropout probability for attention and FFN.
        max_len: Maximum supported sequence length.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        if embedding_dim != hidden_dim:
            self.input_proj: nn.Module = nn.Linear(embedding_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Run causal self-attention.

        Args:
            emb: FloatTensor of shape ``(B, T, embedding_dim)``.

        Returns:
            FloatTensor of shape ``(B, T, hidden_dim)``.
        """
        _, seq_len, _ = emb.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len={self.max_len}."
            )
        x = self.input_proj(emb)
        positions = torch.arange(seq_len, device=emb.device)
        x = x + self.position_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=emb.device),
            diagonal=1,
        )
        return self.transformer(x, mask=causal_mask)


class _HazardHead(nn.Module):
    """Projects hidden states to per-horizon hazard probabilities λ̂(k|X_t).

    Args:
        hidden_dim: Recurrent hidden state dimension H.
        horizon: Number of prediction horizons h.
    """

    def __init__(self, hidden_dim: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.fc = nn.Linear(hidden_dim, horizon)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.sigmoid = nn.Sigmoid()

    def init_bias_from_rates(self, rates: np.ndarray) -> None:
        """Initialise output bias from empirical mean hazard rates.

        Args:
            rates: Float array of shape ``(horizon,)`` with values in (0, 1).
        """
        rates = np.clip(rates, 1e-6, 1.0 - 1e-6)
        logits = np.log(rates / (1.0 - rates))
        with torch.no_grad():
            self.fc.bias.copy_(torch.tensor(logits, dtype=torch.float32))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute hazard probabilities.

        Args:
            h: FloatTensor of shape ``(B, T, H)``.

        Returns:
            FloatTensor of shape ``(B, T, horizon)`` in (0, 1).
        """
        return self.sigmoid(self.fc(h))


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------


class DynamicSurvivalModel(BaseModel):
    """GRU / LSTM / Transformer-based Dynamic Survival Analysis model.

    Implements the DSA pipeline from Yèche et al. (CHIL 2024):

    1. **Linear embedding** – projects each time-step feature vector to a
       dense embedding, with L1 regularisation to encourage sparsity.
    2. **Temporal backbone** – a stacked GRU, LSTM, or causal Transformer
       captures temporal dependencies and produces per-step hidden states.
    3. **Hazard head** – a linear layer maps each hidden state to
       λ̂(k | X_t) for k = 1…horizon.
    4. **Risk score** – cumulative failure probability F(h | X_t) at the
       last observed timestep, used as the scalar alarm score.
    5. **Loss** – binary cross-entropy on the final risk score plus an L1
       regularisation term on the embedding weights.

    The model is compatible with the PyHealth :class:`~pyhealth.trainer.Trainer`
    and expects a :class:`~pyhealth.datasets.SampleDataset` built with
    :class:`~pyhealth.tasks.DecompensationDSA` (or any task whose
    ``input_schema`` contains ``"timeseries": "tensor"`` and
    ``output_schema`` contains ``"label": "binary"``).

    Args:
        dataset: A fitted :class:`~pyhealth.datasets.SampleDataset`.
        input_dim: Dimensionality D of the feature vector at each time step.
        hidden_dim: Backbone hidden / model dimension. Default: ``256``.
        embedding_dim: Linear embedding output size. Default: ``128``.
        num_layers: Number of stacked backbone layers. Default: ``2``.
        encoder_type: Backbone choice – ``"gru"``, ``"lstm"``, or
            ``"transformer"``. Default: ``"gru"``.
        dropout: Dropout probability inside the backbone (only applied
            between recurrent layers when ``num_layers == 1``). Default: ``0.0``.
        l1_reg: L1 regularisation coefficient on the embedding weights.
            Default: ``1.0``.
        horizon: Prediction horizon h in time steps. Default: ``24``.
        num_heads: Number of attention heads (transformer backbone only).
            Default: ``4``.
        ffn_dim: Feed-forward dimension (transformer backbone only).
            Default: ``512``.
        max_seq_len: Maximum supported sequence length for the transformer
            positional table. Default: ``4096``.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import DynamicSurvivalModel
        >>> from pyhealth.tasks import DecompensationDSA
        >>> from pyhealth.tasks.decompensation_dsa import make_synthetic_dsa_samples
        >>> samples = make_synthetic_dsa_samples(n_patients=50, n_features=8)
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema=DecompensationDSA.input_schema,
        ...     output_schema=DecompensationDSA.output_schema,
        ...     dataset_name="dsa_synthetic",
        ... )
        >>> model = DynamicSurvivalModel(dataset=dataset, input_dim=8, horizon=24)
        >>> loader = get_dataloader(dataset, batch_size=4, shuffle=False)
        >>> out = model(**next(iter(loader)))
        >>> out["y_prob"].shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 2,
        encoder_type: Literal["gru", "lstm", "transformer"] = "gru",
        dropout: float = 0.0,
        l1_reg: float = 1.0,
        horizon: int = 24,
        num_heads: int = 4,
        ffn_dim: int = 512,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__(dataset=dataset)

        if encoder_type not in ("gru", "lstm", "transformer"):
            raise ValueError(
                "encoder_type must be 'gru', 'lstm', or 'transformer', "
                f"got '{encoder_type}'"
            )

        self.horizon = horizon
        self.encoder_type = encoder_type
        self._label_key: str = self.label_keys[0] if self.label_keys else "label"

        self.embedding = _LinearEmbedding(input_dim, embedding_dim, l1_weight=l1_reg)
        self.drop = nn.Dropout(p=dropout)

        if encoder_type in ("gru", "lstm"):
            rnn_drop = dropout if num_layers > 1 else 0.0
            rnn_cls = nn.GRU if encoder_type == "gru" else nn.LSTM
            self.rnn: Optional[nn.Module] = rnn_cls(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_drop,
            )
            self.transformer: Optional[_CausalTransformerEncoder] = None
        else:
            self.rnn = None
            self.transformer = _CausalTransformerEncoder(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                max_len=max_seq_len,
            )
        self.head = _HazardHead(hidden_dim, horizon)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def initialise_bias(self, mean_hazard_rates: np.ndarray) -> None:
        """Initialise the hazard head bias from empirical mean hazard rates.

        Call this before training to speed up convergence, as described in
        Yèche et al. (2024).

        Args:
            mean_hazard_rates: Float array of shape ``(horizon,)`` with
                per-step mean hazard rates estimated from the training set.
        """
        self.head.init_bias_from_rates(mean_hazard_rates)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Embed and run the temporal backbone.

        Args:
            x: FloatTensor of shape ``(B, T, D)``.

        Returns:
            FloatTensor of shape ``(B, T, hidden_dim)``.
        """
        emb = self.drop(self.embedding(x))
        if self.transformer is not None:
            return self.transformer(emb)
        assert self.rnn is not None
        out, _ = self.rnn(emb)
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Forward pass compatible with the PyHealth Trainer.

        Expects keyword arguments matching the dataset's input/output schema:

        * ``"timeseries"`` – FloatTensor of shape ``(B, T, input_dim)``
        * ``"label"``      – FloatTensor of shape ``(B,)`` or ``(B, 1)``
          (binary 0/1; only required during training)

        Args:
            **kwargs: Batch dictionary unpacked from the DataLoader.

        Returns:
            Dictionary with keys:

            * ``"loss"``   – scalar BCE + L1 training loss *(training only)*
            * ``"y_prob"`` – risk score FloatTensor of shape ``(B, 1)``
            * ``"y_true"`` – ground-truth labels ``(B, 1)`` *(training only)*
            * ``"logit"``  – raw risk score FloatTensor of shape ``(B, 1)``
        """
        x: torch.Tensor = kwargs["timeseries"]
        if x.dim() == 2:
            x = x.unsqueeze(0)

        hidden_seq = self._encode(x)            # (B, T, hidden_dim)
        hazard = self.head(hidden_seq)          # (B, T, horizon)

        # Cumulative failure: F(k | X_t) = 1 − ∏_{j=1}^{k} (1 − λ(j | X_t))
        survival = torch.cumprod(1.0 - hazard, dim=-1)
        cum_failure = 1.0 - survival            # (B, T, horizon)

        # Scalar alarm score: cumulative failure at last timestep, full horizon
        risk_score = cum_failure[:, -1, -1].unsqueeze(-1)   # (B, 1)

        result: Dict[str, torch.Tensor] = {
            "y_prob": risk_score,
            "logit":  risk_score,
        }

        if self._label_key in kwargs:
            y_true = kwargs[self._label_key].float().to(x.device)
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(-1)   # (B, 1)

            l1 = self.embedding.l1_loss()
            bce = F.binary_cross_entropy(risk_score, y_true)
            result["loss"]   = bce + l1
            result["y_true"] = y_true

        return result
