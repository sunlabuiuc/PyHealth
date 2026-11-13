"""A selective state-space (Mamba) block in pure PyTorch.

Shared by :class:`~pyhealth.models.ARMR` (over visits) and
:class:`~pyhealth.models.APRICOTM` (over clinical events).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["SelectiveSSM", "selective_scan"]


def selective_scan(log_decay: Tensor, drive: Tensor, chunk_size: int = 32) -> Tensor:
    """Parallel scan of ``h_t = exp(log_decay_t) * h_{t-1} + drive_t``, ``h_0 = 0``.

    Computed along dimension 1 in log space: within a chunk, with ``S_t``
    the cumulative sum of ``log_decay`` since the chunk start,
    ``h_t = exp(S_t) h_start + sum_{s <= t} exp(S_t - S_s) drive_s``. The
    positive and negative parts of ``drive`` are summed separately with
    ``logcumsumexp``, so no ``exp(-S)`` can overflow; chunking keeps ``S``
    small enough for float32 precision.

    Args:
        log_decay: Non-positive log decay factors, shape ``[B, L, ...]``.
        drive: Inputs, same shape.
        chunk_size: Steps per chunk.

    Returns:
        The states ``h``, same shape.

    Examples:
        >>> import torch
        >>> log_decay = torch.log(torch.tensor([[0.5, 0.5, 0.5]]))
        >>> drive = torch.tensor([[1.0, -2.0, 4.0]])
        >>> selective_scan(log_decay, drive)  # h = 1, 0.5 - 2, -0.75 + 4
        tensor([[ 1.0000, -1.5000,  3.2500]])
    """
    very_negative = -1e30

    def part(values: Tensor, cumulative: Tensor) -> Tensor:
        positive = values > 0
        log_values = torch.where(
            positive, torch.log(torch.where(positive, values, 1.0)), very_negative
        )
        return torch.exp(
            torch.logcumsumexp(log_values - cumulative, dim=1) + cumulative
        )

    chunks = []
    carry = None
    for begin in range(0, log_decay.shape[1], chunk_size):
        decay = log_decay[:, begin : begin + chunk_size]
        inputs = drive[:, begin : begin + chunk_size]
        cumulative = torch.cumsum(decay, dim=1)
        states = part(inputs, cumulative) - part(-inputs, cumulative)
        if carry is not None:
            states = states + torch.exp(cumulative) * carry.unsqueeze(1)
        carry = states[:, -1]
        chunks.append(states)
    if not chunks:
        return drive.clone()
    return torch.cat(chunks, dim=1)


class SelectiveSSM(nn.Module):
    """A Mamba block (selective state-space model) in pure PyTorch.

    Paper: Albert Gu and Tri Dao. "Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces." 2023. An input projection into a gated branch
    and an SSM branch, a causal depthwise convolution, input-dependent step
    size ``dt`` and matrices ``B`` and ``C``, a diagonal state matrix ``A``,
    and the scan ``h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t``,
    ``y_t = C_t h_t + D x_t``.

    By default (``scan="loop"``) the scan is a plain loop over time, which
    is fast enough for short sequences such as visits (tens of steps).
    ``scan="parallel"`` computes it with :func:`selective_scan` instead,
    which is much faster on long sequences (hundreds of events) and gives
    the same result up to floating-point rounding.

    This differs from :class:`pyhealth.models.ehrmamba.MambaBlock`: there the
    step size and the ``B`` and ``C`` matrices are fixed per channel (a
    time-invariant SSM computed as a convolution), while here they are
    computed from each input step (Mamba's selection mechanism), which ARMR's
    Eq. 2 and APRICOT-M rely on. Hence the separate block.

    Args:
        dim: input and output size.
        state_size: SSM state size per channel. Default is 16.
        expand: inner width as a multiple of ``dim``. Default is 2.
        conv_size: width of the causal depthwise convolution. Default is 4.
        scan: ``"loop"`` (default) or ``"parallel"``.
        chunk_size: steps per chunk of the parallel scan. Default is 32.

    Examples:
        >>> import torch
        >>> block = SelectiveSSM(dim=8)
        >>> block(torch.randn(2, 5, 8)).shape
        torch.Size([2, 5, 8])
        >>> block = SelectiveSSM(dim=8, scan="parallel")
        >>> x = torch.randn(2, 5, 8)
        >>> mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]).bool()
        >>> y = block(x, mask)
        >>> torch.allclose(y[0, :3], block(x[:1, :3])[0], atol=1e-6)
        True
    """

    def __init__(
        self,
        dim: int,
        state_size: int = 16,
        expand: int = 2,
        conv_size: int = 4,
        scan: str = "loop",
        chunk_size: int = 32,
    ):
        super().__init__()
        if scan not in ("loop", "parallel"):
            raise ValueError(f"scan must be 'loop' or 'parallel', got {scan!r}")
        inner = expand * dim
        self.dt_rank = math.ceil(dim / 16)
        self.state_size = state_size
        self.scan = scan
        self.chunk_size = chunk_size
        self.in_proj = nn.Linear(dim, 2 * inner)
        self.conv = nn.Conv1d(
            inner, inner, conv_size, groups=inner, padding=conv_size - 1
        )
        self.x_proj = nn.Linear(inner, self.dt_rank + 2 * state_size, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, inner)
        a = torch.arange(1, state_size + 1, dtype=torch.float).repeat(inner, 1)
        self.a_log = nn.Parameter(torch.log(a))
        self.d = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Runs the block over ``x`` of shape [batch, steps, dim].

        Args:
            x: input sequences, shape [batch, steps, dim].
            mask: optional bool tensor [batch, steps], True for real steps.
                Padded steps do not change the state and their outputs are
                zero; use right padding (the convolution is causal, so real
                steps never see later padding).

        Returns:
            Tensor of shape [batch, steps, dim].
        """
        steps = x.size(1)
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :steps].transpose(1, 2)
        u = F.silu(u)
        dt, b, c = self.x_proj(u).split(
            [self.dt_rank, self.state_size, self.state_size], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))  # [batch, steps, inner]
        if mask is not None:
            dt = dt * mask.unsqueeze(-1)  # dt = 0: state kept, no input
        a = -torch.exp(self.a_log)  # [inner, state]
        if self.scan == "parallel":
            log_decay = dt.unsqueeze(-1) * a  # [batch, steps, inner, state]
            drive = (dt * u).unsqueeze(-1) * b.unsqueeze(2)
            states = selective_scan(log_decay, drive, self.chunk_size)
            y = torch.einsum("bldn,bln->bld", states, c)
        else:
            da = torch.exp(dt.unsqueeze(-1) * a)  # [batch, steps, inner, state]
            dbu = dt.unsqueeze(-1) * b.unsqueeze(2) * u.unsqueeze(-1)
            h = torch.zeros_like(da[:, 0])
            ys = []
            for t in range(steps):
                h = da[:, t] * h + dbu[:, t]
                ys.append((h * c[:, t].unsqueeze(1)).sum(-1))
            y = torch.stack(ys, dim=1)
        y = self.out_proj((y + u * self.d) * F.silu(gate))
        if mask is not None:
            y = y * mask.unsqueeze(-1)
        return y
