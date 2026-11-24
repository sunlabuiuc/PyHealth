"""Pure-PyTorch selective state-space Mamba-style block.

Uses an explicit sequential scan, avoiding CUDA-only dependencies. This is a
reference implementation, not a fused high-performance Mamba kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root mean square layer normalization.

    Examples:
        >>> import torch
        >>> norm = RMSNorm(8)
        >>> output = norm(torch.randn(2, 4, 8))
        >>> output.shape
        torch.Size([2, 4, 8])
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension."""
        return (
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
            * (self.weight)
        )


class SelectiveSSMBlock(nn.Module):
    """Causal depthwise convolution and input-dependent diagonal SSM.

    Args:
        hidden_dim: Input and output feature width.
        state_dim: State size per expanded channel.
        expansion: Inner width multiplier.
        conv_kernel: Causal convolution width.
        dt_rank: Rank of the input-dependent time-step projection.
        dropout: Residual-branch dropout.
        
    Examples:
        >>> import torch
        >>> block = SelectiveSSMBlock(hidden_dim=8, state_dim=4)
        >>> x = torch.randn(2, 3, 8)
        >>> block(x).shape
        torch.Size([2, 3, 8])
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 16,
        expansion: int = 2,
        conv_kernel: int = 4,
        dt_rank: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(hidden_dim, state_dim, expansion, conv_kernel) < 1:
            raise ValueError("SSM dimensions must be positive")
        inner = hidden_dim * expansion
        rank = dt_rank or max(1, (hidden_dim + 15) // 16)
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.inner = inner
        self.conv_kernel = conv_kernel
        self.norm = RMSNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, inner * 2)
        self.conv = nn.Conv1d(inner, inner, conv_kernel, groups=inner, padding=0)
        self.x_proj = nn.Linear(inner, rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(rank, inner)
        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1).float())
            .unsqueeze(0)
            .repeat(inner, 1)
        )
        self.d = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _scan(self, u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply a selective diagonal state-space recurrence."""
        batch, length, _ = u.shape
        state = u.new_zeros(batch, self.inner, self.state_dim)
        a = -self.a_log.exp().to(u.dtype)
        outputs = []
        for t in range(length):
            dt_bc = self.x_proj(u[:, t])
            dt_raw, b, c = torch.split(
                dt_bc,
                [self.dt_proj.in_features, self.state_dim, self.state_dim],
                dim=-1,
            )
            dt = F.softplus(self.dt_proj(dt_raw))
            decay = torch.exp(dt.unsqueeze(-1) * a)
            proposed = decay * state + (
                dt.unsqueeze(-1) * b.unsqueeze(1) * u[:, t].unsqueeze(-1)
            )
            valid = mask[:, t, None, None]
            state = torch.where(valid, proposed, state)
            y = (state * c.unsqueeze(1)).sum(-1) + self.d * u[:, t]
            outputs.append(y * mask[:, t, None].to(y.dtype))
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transform (batch, sequence, hidden_dim) with causal masking.

        Padding is compacted before convolution so left and right padding
        cannot contaminate valid positions. Outputs preserve original positions.
        """
        if x.ndim != 3 or x.shape[-1] != self.hidden_dim:
            raise ValueError("Expected (batch, sequence, hidden_dim)")
        batch, length, _ = x.shape
        mask = (
            torch.ones(batch, length, dtype=torch.bool, device=x.device)
            if attention_mask is None
            else attention_mask.bool()
        )
        if mask.shape != x.shape[:2] or not mask.any(dim=1).all():
            raise ValueError("Each sequence must contain at least one valid event")
        # Compact valid tokens; padding never advances convolution or SSM state.
        lengths = mask.sum(1)
        indices = torch.arange(length, device=x.device)[None, :]
        ranks = mask.long().cumsum(1) - 1
        compact = x.new_zeros(x.shape)
        compact = compact.scatter_add(
            1,
            ranks.clamp_min(0).unsqueeze(-1).expand_as(x),
            x * mask.unsqueeze(-1),
        )
        compact_mask = indices < lengths[:, None]
        u, gate = self.in_proj(self.norm(compact)).chunk(2, dim=-1)
        u = u * compact_mask.unsqueeze(-1)
        u = self.conv(F.pad(u.transpose(1, 2), (self.conv_kernel - 1, 0)))
        u = F.silu(u.transpose(1, 2)) * compact_mask.unsqueeze(-1)
        y = self._scan(u, compact_mask)
        y = self.out_proj(y * F.silu(gate))
        y = self.dropout(y) * compact_mask.unsqueeze(-1)
        # Gather valid results back into their original locations.
        y = y.gather(1, ranks.clamp_min(0).unsqueeze(-1).expand_as(x))
        return (x + y * mask.unsqueeze(-1)) * mask.unsqueeze(-1)
