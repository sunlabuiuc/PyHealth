"""MIRA: a medical time-series foundation model for irregular data.

Paper:
    Hao Li, Bowen Deng, Chang Xu, Zhiyuan Feng, Viktor Schlegel, Yu-Hao
    Huang, Yizheng Sun, Jingyuan Sun, Kailai Yang, Yiyao Yu and Jiang Bian.
    "MIRA: Medical Time Series Foundation Model for Real-World Health Data."
    NeurIPS 2025. https://arxiv.org/abs/2506.07584

MIRA is a decoder-only transformer over univariate series of irregularly
timed values. Each value is embedded by a gated linear layer; the attention
uses continuous-time rotary positions (CT-RoPE) computed from the
observation times; every feed-forward layer is a sparse mixture of experts
(top-2 of 8, plus a gated shared expert); and an optional neural-ODE block
moves the last hidden state to the target time before the output layer
predicts the next value.

This module ports the model from the authors' code
(https://github.com/microsoft/MIRA, ``MIRA/models/modeling_mira.py``) as plain
PyTorch, without the Hugging Face ``PreTrainedModel`` machinery. The parameter
names are the original ones, so the published checkpoint
(https://huggingface.co/MIRA-Mode/MIRA, "MIRA-Large", MIT License) loads with
every weight matched; see :func:`load_mira_checkpoint`. The differences from
the authors' code are listed in :class:`MIRABackbone`.

The original code is released under the MIT License:

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions: The above copyright notice and this permission
    notice shall be included in all copies or substantial portions of the
    Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

Parts of the authors' code come from Time-MoE (Apache License 2.0,
https://github.com/Time-MoE/Time-MoE; license text at
https://www.apache.org/licenses/LICENSE-2.0) and Mistral/Llama.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import SampleDataset
from ..processors import LabelProcessor, RegressionLabelProcessor
from .base_model import BaseModel

# MIRA keeps its own copies of these small pieces, so a later change to APN
# or EHRMamba cannot silently change its outputs or its loss.


def masked_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the finite entries of ``target``.

    The same loss as APN's, so the two models are scored identically on
    ``PhysioNet2012Forecasting``.

    Args:
        prediction: Forecasts, any shape.
        target: Targets of the same shape, NaN where not observed.

    Returns:
        A scalar; 0 (with gradient) when nothing is observed.

    Examples:
        >>> import torch
        >>> masked_mse(torch.tensor([1.0, 2.0]), torch.tensor([0.0, float("nan")]))
        tensor(1.)
    """
    mask = torch.isfinite(target)
    if not mask.any():
        return prediction.sum() * 0.0
    return ((prediction[mask] - target[mask]) ** 2).mean()


class _ForecastLabel(RegressionLabelProcessor):
    """A regression over the observed (finite) targets of a forecast."""

    def loss(self, logits: torch.Tensor, y_true: Any, **kwargs: Any) -> torch.Tensor:
        return masked_mse(logits, y_true)


class _RMSNorm(nn.Module):
    """Root mean square layer normalization, as in the MIRA checkpoint."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return x * rms * self.weight


HF_REPO_ID = "MIRA-Mode/MIRA"
"""Hugging Face repository with the published MIRA-Large checkpoint."""

HF_REVISION = "932caadcee65242e439921ff37d99fb1906f7aad"
"""The commit of :data:`HF_REPO_ID` that the checkpoint is read from."""


@dataclass
class MIRAConfig:
    """Sizes and options of a MIRA model.

    The defaults are those of the published MIRA-Large checkpoint
    (``config.json`` in :data:`HF_REPO_ID`): 12 layers, hidden size 768,
    8 experts with 2 active per token, about 455M parameters. Field names
    follow the authors' ``MIRAConfig``; :meth:`from_dict` ignores the keys
    that only the Hugging Face classes use.

    Args:
        input_size: Values per time step (1: MIRA is univariate).
        hidden_size: Model width.
        intermediate_size: Width of the shared expert; each routed expert is
            ``intermediate_size // num_experts_per_tok`` wide.
        num_hidden_layers: Number of decoder layers.
        num_attention_heads: Attention heads.
        num_key_value_heads: Key/value heads (grouped-query attention);
            ``None`` uses ``num_attention_heads``.
        num_experts: Routed experts per layer.
        num_experts_per_tok: Experts each token is routed to.
        use_dense: Use one dense feed-forward layer instead of experts.
        rms_norm_eps: Epsilon of the RMS norms.
        rope_theta: Base of the rotary frequencies.
        time_scale: CT-RoPE scale ``alpha``: the times of a series are
            rescaled to ``[0, alpha * (n - 1)]`` for ``n`` observations.
        time_snap: Grid the rescaled times are rounded to (0.1 in the
            authors' code); ``None`` disables rounding.
        use_terminal_ode: Build the neural-ODE extrapolation block.
        ode_func_hidden_dims: Hidden sizes of the ODE function.
        ode_func_use_time: Feed the elapsed time to the ODE function.
        ode_solver_method: ``torchdiffeq`` method used when ``torchdiffeq`` is
            installed (``"dopri5"`` in the checkpoint).
        ode_solver_atol: Absolute tolerance of the adaptive solver.
        ode_solver_rtol: Relative tolerance of the adaptive solver.
        ode_rk4_steps: Steps of the built-in fixed-step RK4 solver used when
            ``torchdiffeq`` is not installed.

    Examples:
        >>> MIRAConfig().hidden_size
        768
        >>> MIRAConfig.from_dict({"hidden_size": 48, "model_type": "mira"}).hidden_size
        48
    """

    input_size: int = 1
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int | None = 12
    num_experts: int = 8
    num_experts_per_tok: int = 2
    use_dense: bool = False
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    time_scale: float = 1.0
    time_snap: float | None = 0.1
    use_terminal_ode: bool = True
    ode_func_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    ode_func_use_time: bool = True
    ode_solver_method: str = "dopri5"
    ode_solver_atol: float = 1e-6
    ode_solver_rtol: float = 1e-6
    ode_rk4_steps: int = 8

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> MIRAConfig:
        """Builds a config from a dict such as the checkpoint's ``config.json``.

        Unknown keys (``architectures``, ``torch_dtype``, ...) are ignored.

        Args:
            values: Config values.

        Returns:
            The config.

        Examples:
            >>> MIRAConfig.from_dict({"num_experts": 4}).num_experts
            4
        """
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})

    def to_dict(self) -> dict[str, Any]:
        """The config as a plain dict.

        Returns:
            A dict of the fields.

        Examples:
            >>> MIRAConfig().to_dict()["num_hidden_layers"]
            12
        """
        return asdict(self)


def normalize_time_for_ctrope(
    times: torch.Tensor,
    mask: torch.Tensor | None = None,
    alpha: float = 1.0,
    snap: float | None = 0.1,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Rescales each series' times to the CT-RoPE scale.

    The observed times of a series of ``n`` points are mapped linearly onto
    ``[0, alpha * (n - 1)]``, rounded to a grid of ``snap``, and made strictly
    increasing by moving ties forward by ``eps``, as
    ``normalize_time_for_ctrope`` in the authors' code does. Padding (``mask``
    0) is ignored and set to 0; ``n`` counts each row's observed points, so a
    left-padded row gets the same times as the unpadded series.

    Args:
        times: Times ``[B, L]``.
        mask: 1 for observed points, ``[B, L]``; ``None`` if all are.
        alpha: Scale (``time_scale`` in :class:`MIRAConfig`).
        snap: Grid to round to; ``None`` keeps the rescaled times.
        eps: Gap that separates tied times.

    Returns:
        The rescaled times ``[B, L]`` (float32).

    Examples:
        >>> import torch
        >>> normalize_time_for_ctrope(torch.tensor([[0.0, 2.0, 3.0, 6.0]]))
        tensor([[0.0000, 1.0000, 1.5000, 3.0000]])
    """
    t = times.float()
    valid = torch.ones_like(t, dtype=torch.bool) if mask is None else mask.bool()
    t_min = torch.where(valid, t, torch.inf).amin(dim=1, keepdim=True)
    t_max = torch.where(valid, t, -torch.inf).amax(dim=1, keepdim=True)
    count = valid.sum(dim=1, keepdim=True)
    span = alpha * (count - 1).clamp(min=0).float()
    scaled = (t - t_min) / (t_max - t_min).clamp(min=1e-8) * span
    scaled = torch.where(valid, scaled, 0.0)
    if snap is not None:
        scaled = torch.round(scaled / snap) * snap
    # out[i] = max(t[i], out[i-1] + eps) over the observed points, as a
    # running maximum of t[i] - rank * eps.
    rank = valid.long().cumsum(dim=1).float() - 1.0
    shifted = torch.where(valid, scaled - rank * eps, -torch.inf)
    if shifted.device.type == "mps":  # cummax has no MPS kernel
        running = torch.cummax(shifted.cpu(), dim=1).values.to(shifted.device)
    else:
        running = torch.cummax(shifted, dim=1).values
    return torch.where(valid, running + rank * eps, 0.0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _InputEmbedding(nn.Module):
    """Gated linear embedding of each value: ``silu(W_g x) * W_e x``."""

    def __init__(self, config: MIRAConfig):
        super().__init__()
        self.emb_layer = nn.Linear(config.input_size, config.hidden_size, bias=False)
        self.gate_layer = nn.Linear(config.input_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate_layer(x)) * self.emb_layer(x)


class _GatedMLP(nn.Module):
    """SwiGLU feed-forward block (an expert)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _SparseExperts(nn.Module):
    """Top-k mixture of experts plus a sigmoid-gated shared expert."""

    def __init__(self, config: MIRAConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            _GatedMLP(config.hidden_size, config.intermediate_size // self.top_k)
            for _ in range(config.num_experts)
        )
        self.shared_expert = _GatedMLP(config.hidden_size, config.intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        routing = F.softmax(self.gate(x), dim=-1, dtype=torch.float)
        # Top-k weights are not renormalized, as in the authors' code.
        weights, chosen = torch.topk(routing, self.top_k, dim=-1)
        weights = weights.to(x.dtype)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            token, slot = torch.where(chosen == e)
            if token.numel():
                out.index_add_(0, token, expert(x[token]) * weights[token, slot, None])
        out = out + torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return out.reshape(shape)


class _Attention(nn.Module):
    """Causal self-attention with continuous-time rotary positions."""

    def __init__(self, config: MIRAConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        size, kv_size = config.hidden_size, self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(size, kv_size, bias=True)
        self.v_proj = nn.Linear(size, kv_size, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, size, bias=False)
        inv_freq = 1.0 / config.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, times: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, -1).transpose(1, 2)
        angles = torch.einsum("bl,d->bld", times.float(), self.inv_freq)
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, -1))


class _DecoderLayer(nn.Module):
    def __init__(self, config: MIRAConfig):
        super().__init__()
        self.self_attn = _Attention(config)
        self.ffn_layer = (
            _GatedMLP(config.hidden_size, config.intermediate_size)
            if config.use_dense
            else _SparseExperts(config)
        )
        self.input_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, times: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), times, mask)
        return x + self.ffn_layer(self.post_attention_layernorm(x))


class _Decoder(nn.Module):
    def __init__(self, config: MIRAConfig):
        super().__init__()
        self.embed_layer = _InputEmbedding(config)
        self.layers = nn.ModuleList(
            _DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = _RMSNorm(config.hidden_size, config.rms_norm_eps)


class _ODEFunc(nn.Module):
    def __init__(self, config: MIRAConfig):
        super().__init__()
        self.use_time = config.ode_func_use_time
        layers: list[nn.Module] = []
        width = config.hidden_size + int(self.use_time)
        for hidden in config.ode_func_hidden_dims:
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, config.hidden_size))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self.use_time:
            s = torch.as_tensor(s, device=h.device, dtype=h.dtype)
            h = torch.cat([s.reshape(-1, 1).expand(h.shape[0], 1), h], dim=-1)
        return self.net(h)


class _TerminalODE(nn.Module):
    """Integrates ``dh/ds = f(s, h)`` from the last time to the target time.

    As in the authors' code (after their per-sample fix), each series is
    integrated over its own interval ``dt``: ``dh/dtau = dt * f(tau * dt, h)``
    for ``tau`` in ``[0, 1]``; ``dt <= 0`` keeps the state.
    """

    def __init__(self, config: MIRAConfig):
        super().__init__()
        self.ode_func = _ODEFunc(config)
        self.method = config.ode_solver_method
        self.atol, self.rtol = config.ode_solver_atol, config.ode_solver_rtol
        self.rk4_steps = config.ode_rk4_steps

    def forward(self, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        dt = dt.to(h.dtype).clamp(min=0.0)

        def dynamics(tau: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            return dt.unsqueeze(-1) * self.ode_func(tau * dt, state)

        try:
            from torchdiffeq import odeint
        except ImportError:
            odeint = None
        if odeint is not None:
            span = torch.tensor([0.0, 1.0], device=h.device, dtype=h.dtype)
            return odeint(
                dynamics, h, span, method=self.method, atol=self.atol, rtol=self.rtol
            )[-1]
        step = 1.0 / self.rk4_steps
        for i in range(self.rk4_steps):
            tau = torch.tensor(i * step, device=h.device, dtype=h.dtype)
            k1 = dynamics(tau, h)
            k2 = dynamics(tau + step / 2, h + step / 2 * k1)
            k3 = dynamics(tau + step / 2, h + step / 2 * k2)
            k4 = dynamics(tau + step, h + step * k3)
            h = h + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return h


class _OutputLayer(nn.Module):
    def __init__(self, config: MIRAConfig):
        super().__init__()
        self.out_layer = nn.Linear(config.hidden_size, config.input_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(x)


class MIRABackbone(nn.Module):
    """The MIRA network: next-value prediction for irregular univariate series.

    Submodule and parameter names match the authors' ``MIRAForPrediction``,
    so its state dict loads unchanged (see :func:`load_mira_checkpoint`).
    Differences from the authors' code, none of which changes the outputs of
    an unpadded series:

    - attention is computed with ``scaled_dot_product_attention`` and no KV
      cache (series are re-encoded at each forecast step);
    - left padding is supported: the CT-RoPE time rescaling counts each
      row's observed points instead of the padded length, and padded
      positions are masked out of the attention;
    - the neural-ODE block uses ``torchdiffeq`` when it is installed and a
      fixed-step RK4 solver otherwise (the checkpoint was trained with
      ``dopri5``); it is off unless target times are given, as in the
      authors' inference example;
    - ``horizon_lengths`` is fixed to ``[1]`` (the authors' class forces it
      too) and the router's load-balancing loss is not computed.

    Args:
        config: The model sizes; MIRA-Large by default.

    Examples:
        >>> import torch
        >>> config = MIRAConfig(hidden_size=16, intermediate_size=32,
        ...                     num_hidden_layers=2, num_attention_heads=2,
        ...                     num_key_value_heads=2, num_experts=4,
        ...                     ode_func_hidden_dims=[8])
        >>> backbone = MIRABackbone(config)
        >>> values, times = torch.randn(3, 10), torch.arange(10.0).repeat(3, 1)
        >>> backbone(values, times).shape
        torch.Size([3])
    """

    def __init__(self, config: MIRAConfig | None = None):
        super().__init__()
        self.config = config or MIRAConfig()
        self.model = _Decoder(self.config)
        self.ode_extrapolation_block = (
            _TerminalODE(self.config) if self.config.use_terminal_ode else None
        )
        self.lm_heads = nn.ModuleList([_OutputLayer(self.config)])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hidden states of every position, ``[B, L, hidden_size]``.

        Args:
            values: Values ``[B, L]``, already normalized per series.
            times: Their times ``[B, L]``, in any unit (they are rescaled).
            mask: 1 for observed points, 0 for (left) padding; ``None`` if
                all are observed.

        Returns:
            The final-norm hidden states.
        """
        valid = torch.ones_like(values, dtype=torch.bool) if mask is None else mask.bool()
        scaled = normalize_time_for_ctrope(
            times, valid, self.config.time_scale, self.config.time_snap
        )
        length = values.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=values.device).tril()
        eye = torch.eye(length, dtype=torch.bool, device=values.device)
        # Padded queries see themselves so their (unused) rows are not NaN.
        attn_mask = (causal & valid[:, None, :]) | eye
        x = values.unsqueeze(-1).to(self.model.embed_layer.emb_layer.weight.dtype)
        h = self.model.embed_layer(x)
        for layer in self.model.layers:
            h = layer(h, scaled, attn_mask)
        return self.model.norm(h)

    def forward(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
        target_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predicts the next value of each series, ``[B]``.

        Args:
            values: Values ``[B, L]``, normalized per series; left-padded.
            times: Their times ``[B, L]``.
            mask: 1 for observed points, 0 for padding.
            target_times: Times of the values to predict ``[B]``, in the units
                of ``times``. When given and the ODE block exists, the last
                hidden state is integrated over ``target_times - times[:, -1]``
                (raw units, as in the authors' code) before the output layer.

        Returns:
            The predicted next values (in the normalized scale).
        """
        h = self.encode(values, times, mask)[:, -1]
        if target_times is not None and self.ode_extrapolation_block is not None:
            dt = target_times.to(h.device).float() - times[:, -1].float()
            h = self.ode_extrapolation_block(h, dt)
        return self.lm_heads[0](h).squeeze(-1)

    def forecast(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        query_times: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_ode: bool = False,
    ) -> torch.Tensor:
        """Forecasts autoregressively at the query times.

        Each step predicts the next value from the series so far and appends
        it at its query time, as the authors' ``mira_predict_autoregressive``
        does.

        Args:
            values: Values ``[B, L]``, normalized per series; left-padded.
            times: Their times ``[B, L]``.
            query_times: Times to forecast ``[B, K]``, in increasing order.
            mask: 1 for observed points, 0 for padding.
            use_ode: Pass each query time to the ODE block.

        Returns:
            Forecasts ``[B, K]`` (in the normalized scale).
        """
        valid = torch.ones_like(values, dtype=torch.bool) if mask is None else mask.bool()
        preds = []
        for step in range(query_times.shape[1]):
            target = query_times[:, step]
            nxt = self(values, times, valid, target if use_ode else None)
            preds.append(nxt)
            values = torch.cat([values, nxt.unsqueeze(1).to(values.dtype)], dim=1)
            times = torch.cat([times, target.unsqueeze(1).to(times.dtype)], dim=1)
            valid = torch.cat([valid, valid.new_ones(valid.shape[0], 1)], dim=1)
        return torch.stack(preds, dim=1)


def load_mira_checkpoint(
    path: str | None = None, revision: str = HF_REVISION
) -> tuple[MIRAConfig, dict[str, torch.Tensor]]:
    """Loads the published MIRA weights and config.

    Args:
        path: A local directory with ``config.json`` and
            ``model.safetensors`` (a Hugging Face snapshot). Defaults to
            downloading them from :data:`HF_REPO_ID` at ``revision`` with
            ``huggingface_hub`` (about 0.9 GB, cached).
        revision: Commit of :data:`HF_REPO_ID` to download.

    Returns:
        ``(config, state_dict)`` for :class:`MIRABackbone`; the weights are
        stored in bfloat16 and cast when loaded into a float32 model.

    Examples:
        >>> config, state = load_mira_checkpoint()  # doctest: +SKIP
        >>> backbone = MIRABackbone(config)  # doctest: +SKIP
        >>> backbone.load_state_dict(state)  # doctest: +SKIP
        <All keys matched successfully>
    """
    import os

    from safetensors.torch import load_file

    if path is None:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(HF_REPO_ID, "config.json", revision=revision)
        weights_path = hf_hub_download(
            HF_REPO_ID, "model.safetensors", revision=revision
        )
    else:
        config_path = os.path.join(path, "config.json")
        weights_path = os.path.join(path, "model.safetensors")
    with open(config_path) as f:
        config = MIRAConfig.from_dict(json.load(f))
    return config, load_file(weights_path)


class MIRA(BaseModel):
    """MIRA forecaster for irregular multivariate time series (NeurIPS 2025).

    MIRA is a univariate foundation model, so each variable of a sample is
    forecast on its own (channel-independent). A variable's history is its
    observed values and their times; it is standardized with its own mean
    and standard deviation, :class:`MIRABackbone` forecasts the query times
    autoregressively, and the forecasts are mapped back to the input scale.
    A variable with no observed value is forecast as 0 (the mean, for
    z-scored inputs).

    The model reads the same fields as :class:`~pyhealth.models.APN`, so it
    runs on :class:`~pyhealth.tasks.PhysioNet2012Forecasting` unchanged:

    - ``x``: values ``[B, L, N]``, NaN where not observed;
    - ``x_time``: their times ``[B, L]`` (shared by all variables);
    - ``y_time``: the times to forecast ``[B, K]``;
    - ``y``: the targets ``[B, K, N]``, NaN where not observed.

    With ``pretrained=True`` the published MIRA-Large checkpoint is loaded
    (every weight must match) and the model forecasts zero-shot; it can also
    be fine-tuned with the masked MSE that :meth:`forward` returns. As with
    APN, ``y_true`` and ``y_prob`` are the observed targets and their
    forecasts, flattened, so the :class:`~pyhealth.trainer.Trainer` scores it
    in ``"regression"`` mode (``metrics=["mse", "mae"]``).

    Args:
        dataset: The samples; used for the field names.
        pretrained: ``True`` to load the checkpoint from Hugging Face, a
            local snapshot directory to load it from there, or ``False`` for
            random weights with ``config``.
        config: Model sizes when not pretrained; MIRA-Large by default.
            With ``pretrained`` the checkpoint's config is used, and passing
            ``config`` too raises ``ValueError``.
        use_ode: Integrate the last hidden state to each query time with the
            neural-ODE block. Off by default, as in the authors' inference
            example. ``torchdiffeq`` is optional and not a PyHealth
            dependency: with it the block uses the checkpoint's adaptive
            solver (as the authors' code does); without it, a fixed-step RK4,
            whose forecasts differ slightly.
        ode_time_scale: Factor applied to the times before they are passed
            to the ODE block (e.g. 48 to turn PhysioNet 2012's ``hour / 48``
            back into hours).
        min_std: Lower bound of each series' standard deviation; a series
            with one observation uses 1.
        chunk_size: Most series encoded at once; series are grouped by
            history length, and each group is cut to its longest history.
        value_key: Input field with the observed values.
        time_key: Input field with their times.
        query_key: Input field with the times to forecast.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> rng = np.random.default_rng(0)
        >>> samples = [
        ...     {"patient_id": f"p{i}", "x": rng.normal(size=(6, 2)).astype("float32"),
        ...      "x_time": np.arange(6, dtype="float32") / 8,
        ...      "y_time": np.array([6, 7], dtype="float32") / 8,
        ...      "y": rng.normal(size=(2, 2)).astype("float32")}
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     {"x": "tensor", "x_time": "tensor", "y_time": "tensor"},
        ...     {"y": "tensor"},
        ... )
        >>> config = MIRAConfig(hidden_size=16, intermediate_size=32,
        ...                     num_hidden_layers=1, num_attention_heads=2,
        ...                     num_key_value_heads=2, num_experts=2,
        ...                     ode_func_hidden_dims=[8])
        >>> model = MIRA(dataset, config=config)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> model(**batch)["forecast"].shape
        torch.Size([4, 2, 2])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        pretrained: bool | str = False,
        config: MIRAConfig | None = None,
        use_ode: bool = False,
        ode_time_scale: float = 1.0,
        min_std: float = 1e-6,
        chunk_size: int = 1024,
        value_key: str = "x",
        time_key: str = "x_time",
        query_key: str = "y_time",
    ) -> None:
        super().__init__(dataset)
        if len(self.label_keys) > 1:
            raise ValueError("MIRA expects at most one label field (the targets).")
        self.label_key = self.label_keys[0] if self.label_keys else None
        self.value_key, self.time_key, self.query_key = value_key, time_key, query_key
        self.use_ode = use_ode
        self.ode_time_scale = ode_time_scale
        self.min_std = min_std
        self.chunk_size = chunk_size
        state = None
        if pretrained and config is not None:
            raise ValueError(
                "config is read from the checkpoint when pretrained is set; "
                "pass one or the other."
            )
        if pretrained:
            config, state = load_mira_checkpoint(
                pretrained if isinstance(pretrained, str) else None
            )
        self.backbone = MIRABackbone(config)
        if state is not None:
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise ValueError(
                    "checkpoint does not match the backbone: missing "
                    f"{missing}, unexpected {unexpected}"
                )
        if use_ode and self.backbone.ode_extrapolation_block is None:
            raise ValueError("use_ode needs config.use_terminal_ode=True.")

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """The forecasts, scored as a regression with :func:`masked_mse`.

        The label is a ``"tensor"`` field, which is not a label processor, so
        MIRA supplies its own, as APN does: the Trainer scores the observed
        targets as a regression. Without a label field there is none.

        Examples:
            >>> model.mode  # doctest: +SKIP
            'regression'
        """
        if self.label_key is None:
            return {}
        return {self.label_key: _ForecastLabel()}

    def predict(
        self, x: torch.Tensor, x_time: torch.Tensor, y_time: torch.Tensor
    ) -> torch.Tensor:
        """Forecasts every variable at the query times.

        Args:
            x: Values ``[B, L, N]``, NaN where not observed.
            x_time: Times ``[B, L]``.
            y_time: Query times ``[B, K]``.

        Returns:
            Forecasts ``[B, K, N]`` in the scale of ``x``.

        Examples:
            >>> model.predict(x, x_time, y_time).shape  # doctest: +SKIP
            torch.Size([4, 3, 36])
        """
        batch, length, n_vars = x.shape
        steps = y_time.shape[1]
        observed = torch.isfinite(x).transpose(1, 2).reshape(-1, length)  # [B*N, L]
        values = torch.nan_to_num(x, nan=0.0).transpose(1, 2).reshape(-1, length)
        times = x_time.unsqueeze(1).expand(batch, n_vars, length).reshape(-1, length)
        queries = y_time.unsqueeze(1).expand(batch, n_vars, steps).reshape(-1, steps)
        out = values.new_zeros(batch * n_vars, steps)
        rows = observed.any(dim=1).nonzero().squeeze(1)
        if rows.numel():
            observed, values = observed[rows], values[rows]
            times, queries = times[rows], queries[rows]
            # Left-align padding: observed points to the right, in time order.
            order = torch.argsort(observed.int(), dim=1, stable=True)
            observed = observed.gather(1, order)
            values = values.gather(1, order)
            times = times.gather(1, order)
            count = observed.sum(dim=1, keepdim=True)
            mean = values.sum(dim=1, keepdim=True) / count
            var = ((values - mean) ** 2 * observed).sum(dim=1, keepdim=True)
            std = (var / (count - 1).clamp(min=1)).sqrt().clamp(min=self.min_std)
            std = torch.where(count > 1, std, torch.ones_like(std))
            normed = torch.where(observed, (values - mean) / std, 0.0)
            scale = self.ode_time_scale
            times, queries = times * scale, queries * scale
            # Series of similar length are encoded together, cut to their own
            # longest history, so short series do not pay for long padding.
            pred = torch.empty_like(out[rows])
            by_length = torch.argsort(count.squeeze(1))
            for chunk in by_length.split(self.chunk_size):
                keep = int(count[chunk].max())
                pred[chunk] = self.backbone.forecast(
                    normed[chunk, -keep:], times[chunk, -keep:],
                    queries[chunk], observed[chunk, -keep:], self.use_ode,
                )
            out[rows] = pred * std + mean
        return out.reshape(batch, n_vars, steps).transpose(1, 2)

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forecasts the query times and, with targets, the masked MSE.

        Returns:
            A dict with ``forecast`` (``[B, K, N]``), ``logit`` (the same),
            and, when the targets are given, ``loss``, ``y_true`` and
            ``y_prob`` (the observed targets and their forecasts, flattened)
            and ``mask`` (``[B, K, N]``, 1 where a target was observed).
        """
        x = kwargs[self.value_key].to(self.device).float()
        x_time = kwargs[self.time_key].to(self.device).float()
        y_time = kwargs[self.query_key].to(self.device).float()
        forecast = self.predict(x, x_time, y_time)
        out = {"forecast": forecast, "logit": forecast}
        if self.label_key is not None and self.label_key in kwargs:
            y = kwargs[self.label_key].to(self.device).float()
            mask = torch.isfinite(y)
            out["loss"] = masked_mse(forecast, y)
            out["y_true"] = y[mask]
            out["y_prob"] = forecast[mask]
            out["mask"] = mask.float()
        else:
            out["y_prob"] = forecast
        return out


__all__ = [
    "MIRA",
    "MIRABackbone",
    "MIRAConfig",
    "load_mira_checkpoint",
    "normalize_time_for_ctrope",
]
