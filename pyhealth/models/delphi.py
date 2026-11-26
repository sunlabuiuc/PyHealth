"""Delphi-2M: a generative time-to-event transformer for disease trajectories.

Contributor: Evgeniy Tikhonov (GitHub: etikhonov-dev)

Paper: Shmatko et al., Learning the natural history of human disease with
generative transformers, Nature 2025.
https://doi.org/10.1038/s41586-025-09529-3

The continuous age encoding, tie-masked causal attention, weight-tied token
head and exponential competing-risks time loss follow the Gerstung Lab
reference model (https://github.com/gerstung-lab/Delphi). This module is
self-contained: it subclasses :class:`~pyhealth.models.BaseModel` and does
not import other concrete models or tasks.

Two changes adapt that model to PyHealth's :class:`~pyhealth.trainer.Trainer`:

* ``codes`` and ``next_code`` can use different index spaces. Embeddings and
  ``lm_head`` share one vocabulary that starts with the code vocabulary and
  appends any label that is not already in it. Class logits are gathered
  with ``label_to_token_idx`` so they line up with ``MultiClassLabelProcessor``.
* Padding is ``codes == 0``, never ``age == 0``. A birth event at age 0
  therefore does not look like a padded step, including when ``mask_ties``
  compares timestamps. The attention mask is squeezed on the head axis only,
  so a one-token history (``T == 1``) stays a row vector.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset

from .base_model import BaseModel


def _new_gelu(x: torch.Tensor) -> torch.Tensor:
    """GPT-2 GELU, matching the Delphi reference."""
    return 0.5 * x * (
        1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
    )


def _attention_mask(
    codes: torch.Tensor, ages: torch.Tensor, mask_ties: bool
) -> torch.Tensor:
    """Causal mask of shape ``[B, 1, T, T]`` (query, key).

    Query ``i`` attends to key ``j <= i`` when both tokens are real
    (``codes != 0``). With ``mask_ties``, an earlier token at the same age
    is hidden. The diagonal stays on, so a tied or padded row still has a
    finite softmax.
    """
    batch, length = codes.shape
    valid = codes != 0
    causal = torch.tril(torch.ones(length, length, device=codes.device, dtype=torch.bool))
    allow = valid.view(batch, length, 1) & valid.view(batch, 1, length) & causal
    if mask_ties:
        same_age = ages.unsqueeze(2) == ages.unsqueeze(1)
        earlier = torch.tril(
            torch.ones(length, length, device=codes.device, dtype=torch.bool),
            diagonal=-1,
        )
        tied = same_age & earlier & valid.view(batch, length, 1) & valid.view(batch, 1, length)
        allow = allow & ~tied
    eye = torch.eye(length, device=codes.device, dtype=torch.bool)
    return (allow | eye).view(batch, 1, length, length)


def _competing_nll(
    logit: torch.Tensor, delta_t: torch.Tensor, t_min: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-example exponential competing-risks NLL and its log-rate.

    ``lse = -log(exp(-logsumexp(logit)) + t_min)`` and
    ``nll = -(lse - exp(lse - ldt))`` with ``ldt = -log(clamp(dt, min=1) + t_min)``.
    """
    lse = torch.logsumexp(logit, dim=-1)
    lse = -torch.log(torch.exp(-lse) + t_min)
    ldt = -torch.log(delta_t.clamp(min=1.0) + t_min)
    return -(lse - torch.exp(lse - ldt)), lse


class DelphiAgeEncoding(nn.Module):
    """Sinusoidal encoding of age in days, then a bias-free linear map.

    Even channels are ``sin(age / 365.25 * omega)`` and odd channels are
    ``cos``. ``omega`` is the same geometric schedule as a transformer
    positional encoding. ``n_embd`` must be even.

    Args:
        n_embd: Embedding width. Must be even.

    Examples:
        >>> import torch
        >>> from pyhealth.models.delphi import DelphiAgeEncoding
        >>> tuple(DelphiAgeEncoding(4)(torch.zeros(1, 2)).shape)
        (1, 2, 4)
    """

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        if n_embd % 2 != 0:
            raise ValueError(f"n_embd must be even, got {n_embd}.")
        div_term = torch.exp(
            torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd)
        )
        self.register_buffer("div_term", div_term)
        self.n_embd = n_embd
        self.linear = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ages of shape ``[B, T]`` or ``[B, T, 1]`` as ``[B, T, n_embd]``."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        y = x.new_zeros(*x.shape[:-1], self.n_embd)
        scaled = x / 365.25 * self.div_term
        y[..., 0::2] = torch.sin(scaled)
        y[..., 1::2] = torch.cos(scaled)
        return self.linear(y)


class _LayerNorm(nn.Module):
    def __init__(self, n_embd: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class _CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, width = x.shape
        query, key, value = (
            part.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
            for part in self.c_attn(x).split(width, dim=2)
        )
        scale = 1.0 / math.sqrt(key.size(-1))
        scores = (query @ key.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~attn_mask, float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        mixed = weights @ value
        mixed = mixed.transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.c_proj(mixed)), weights


class _MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(_new_gelu(self.c_fc(x))))


class _Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.ln_1 = _LayerNorm(n_embd, bias=bias)
        self.attn = _CausalSelfAttention(n_embd, n_head, dropout, bias)
        self.ln_2 = _LayerNorm(n_embd, bias=bias)
        self.mlp = _MLP(n_embd, dropout, bias)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update, weights = self.attn(self.ln_1(x), attn_mask)
        x = x + update
        x = x + self.mlp(self.ln_2(x))
        return x, weights


class Delphi(BaseModel):
    """Generative multi-disease time-to-event transformer (Delphi-2M).

    The input schema is ``codes`` (a token sequence) and ``ages`` (days,
    a float tensor). The output schema is ``next_code`` (one multiclass
    label, so :class:`~pyhealth.trainer.Trainer` computes multiclass
    metrics) and ``time_to_next`` (days until that event). The loss is
    next-event cross-entropy plus the exponential competing-risks
    log-likelihood of the wait. Cross-entropy uses the label vocabulary.
    The time loss and ``dt_pred`` use every token except padding, which is
    the rate ``generate_trajectory`` samples before ``no_repeat`` or
    ``top_k``.

    Args:
        dataset: Sample dataset whose processors supply the code and label
            vocabularies.
        n_embd: Embedding width. Even, and divisible by ``n_head``.
            Default 64.
        n_layer: Transformer blocks. Default 2.
        n_head: Attention heads. Default 2.
        dropout: Residual and attention dropout. Default 0.
        token_dropout: Dropout on token embeddings before age is added.
            Default 0.
        t_min: Positive floor inside the time log-rate. Default 1, as in
            the reference configuration.
        bias: Bias on linear layers and layer norms. Default False. The
            age projection is always bias-free.
        mask_ties: Hide earlier tokens that share the query's age.
            Default True.
        causal_sequence_loss: Add teacher-forced next-token losses on
            prefix steps, as well as the labeled next event. Default True.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import Delphi
        >>> samples = [
        ...     {"patient_id": "p", "codes": ["A"], "ages": [0.0],
        ...      "next_code": "B", "time_to_next": 10.0},
        ...     {"patient_id": "p", "codes": ["B"], "ages": [10.0],
        ...      "next_code": "A", "time_to_next": 5.0},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     {"codes": "sequence", "ages": "tensor"},
        ...     {"next_code": "multiclass", "time_to_next": "tensor"},
        ... )
        >>> Delphi(dataset, n_embd=8, n_layer=1, n_head=2).mode
        'multiclass'
    """

    def __init__(
        self,
        dataset: SampleDataset,
        n_embd: int = 64,
        n_layer: int = 2,
        n_head: int = 2,
        dropout: float = 0.0,
        token_dropout: float = 0.0,
        t_min: float = 1.0,
        bias: bool = False,
        mask_ties: bool = True,
        causal_sequence_loss: bool = True,
    ) -> None:
        super().__init__(dataset)
        if n_embd % 2 != 0:
            raise ValueError(f"n_embd must be even, got {n_embd}.")
        if n_head < 1 or n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head}).")
        if n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {n_layer}.")
        if t_min <= 0:
            raise ValueError(f"t_min must be positive, got {t_min}.")
        code_vocab = dict(dataset.input_processors["codes"].code_vocab)
        label_vocab = dataset.output_processors["next_code"].label_vocab
        unified = dict(code_vocab)
        next_index = len(unified)
        label_to_token: list[int] = []
        inverse_labels = [""] * len(label_vocab)
        for label, class_index in label_vocab.items():
            inverse_labels[class_index] = label
        for label in inverse_labels:
            if label in unified:
                label_to_token.append(unified[label])
            else:
                unified[label] = next_index
                label_to_token.append(next_index)
                next_index += 1

        self.n_embd = n_embd
        self.mask_ties = mask_ties
        self.causal_sequence_loss = causal_sequence_loss
        self.t_min = float(t_min)
        self.vocab_size = len(unified)
        age = DelphiAgeEncoding(n_embd)
        blocks = nn.ModuleList(
            [_Block(n_embd, n_head, dropout, bias) for _ in range(n_layer)]
        )
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(self.vocab_size, n_embd, padding_idx=0),
                "wae": age,
                "token_drop": nn.Dropout(token_dropout),
                "drop": nn.Dropout(dropout),
                "h": blocks,
                "ln_f": _LayerNorm(n_embd, bias=bias),
            }
        )
        self.lm_head = nn.Linear(n_embd, self.vocab_size, bias=False)
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        self.transformer.wte.weight = self.lm_head.weight
        with torch.no_grad():
            self.lm_head.weight[0].zero_()
        self.register_buffer(
            "label_to_token_idx",
            torch.tensor(label_to_token, dtype=torch.long),
            persistent=True,
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def _codes_and_ages(
        self, codes: torch.Tensor, ages: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes = codes.to(self.device).long()
        ages = ages.to(self.device).float()
        if ages.dim() == 3 and ages.size(-1) == 1:
            ages = ages.squeeze(-1)
        return codes, ages

    def _hidden(
        self, codes: torch.Tensor, ages: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.transformer.token_drop(self.transformer.wte(codes))
        hidden = self.transformer.drop(token + self.transformer.wae(ages))
        mask = _attention_mask(codes, ages, self.mask_ties)
        weights = []
        for block in self.transformer.h:
            hidden, layer_weights = block(hidden, mask)
            weights.append(layer_weights)
        hidden = self.transformer.ln_f(hidden)
        # [layer, batch, head, query, key] -> [batch, query, key]
        attn = torch.stack(weights).mean(dim=(0, 2))
        return hidden, attn

    def _last_logits(self, full_logit: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        last = (codes != 0).sum(dim=1).clamp(min=1) - 1
        batch = torch.arange(codes.size(0), device=codes.device)
        return full_logit[batch, last]

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Score the next disease and the waiting time.

        Returns:
            ``logit`` and ``y_prob`` over the label vocabulary at the last
            real token, plus ``loss``, ``loss_ce``, ``loss_dt``, ``y_true``
            and ``dt_pred`` when labels are present. ``dt_pred`` is the
            expected waiting time in days. It is ``exp(-logsumexp)`` over
            the full vocabulary at the last real token, with padding
            removed, the same rate as the time loss. ``t_min`` shifts only
            that loss, so it is not included. Class logits stay on the
            label vocabulary for cross-entropy. ``attn`` is the attention
            averaged over layers and heads.
        """
        codes, ages = self._codes_and_ages(kwargs["codes"], kwargs["ages"])
        hidden, attn = self._hidden(codes, ages)
        full_logit = self.lm_head(hidden)
        last_logit = self._last_logits(full_logit, codes)
        class_logit = last_logit.index_select(-1, self.label_to_token_idx)
        output: dict[str, torch.Tensor] = {
            "logit": class_logit,
            "y_prob": F.softmax(class_logit, dim=-1),
            "attn": attn,
        }
        if "next_code" not in kwargs or "time_to_next" not in kwargs:
            return output
        y_true = kwargs["next_code"].to(self.device).long().view(-1)
        waits = kwargs["time_to_next"].to(self.device).float().view(-1)
        target_hazard = last_logit.clone()
        target_hazard[..., 0] = float("-inf")
        target_nll, _ = _competing_nll(target_hazard, waits, self.t_min)
        loss_ce = F.cross_entropy(class_logit, y_true)
        loss_dt = target_nll.mean()
        n_target = class_logit.size(0)
        if (
            self.causal_sequence_loss
            and codes.size(1) >= 2
        ):
            step = (codes[:, :-1] != 0) & (codes[:, 1:] != 0)
            if bool(step.any()):
                prefix_logit = full_logit[:, :-1][step]
                prefix_ce = F.cross_entropy(prefix_logit, codes[:, 1:][step])
                prefix_hazard = full_logit[:, :-1].clone()
                prefix_hazard[..., 0] = float("-inf")
                prefix_dt = (ages[:, 1:] - ages[:, :-1]).clamp(min=0)
                prefix_nll, _ = _competing_nll(
                    prefix_hazard[step], prefix_dt[step], self.t_min
                )
                n_prefix = int(step.sum())
                loss_ce = (loss_ce * n_target + prefix_ce * n_prefix) / (
                    n_target + n_prefix
                )
                loss_dt = (loss_dt * n_target + prefix_nll.mean() * n_prefix) / (
                    n_target + n_prefix
                )
        output.update(
            {
                "y_true": y_true,
                "loss_ce": loss_ce,
                "loss_dt": loss_dt,
                "loss": loss_ce + loss_dt,
                "dt_pred": torch.exp(-torch.logsumexp(target_hazard, dim=-1)),
            }
        )
        return output

    @torch.no_grad()
    def generate_trajectory(
        self,
        codes: torch.Tensor,
        ages: torch.Tensor,
        max_new_tokens: int = 32,
        max_age: float = 85 * 365.25,
        no_repeat: bool = True,
        termination_tokens: list[int] | None = None,
        top_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a continuation of each trajectory.

        Waiting time is the minimum of independent exponential races, one
        per token except padding. That total rate is the same one the time
        loss and ``dt_pred`` use. The sampled disease is which race wins.
        Padding (index 0) is never sampled. Tokens already in the history
        are blocked when ``no_repeat`` is set, except ``<unk>`` (index 1),
        and ``top_k`` can drop further tokens before the draw. Generation
        of a row stops at ``max_new_tokens``, ``max_age``, or any
        ``termination_tokens``.

        Args:
            codes: Token ids ``[B, T]``. Padding is 0 and is stripped.
            ages: Ages in days, ``[B, T]`` or ``[B, T, 1]``.
            max_new_tokens: Maximum events to append. Default 32.
            max_age: Stop once the latest age exceeds this many days.
                Default 85 years.
            no_repeat: Do not sample a disease token twice. Default True.
            termination_tokens: Token ids that end a trajectory.
            top_k: Sample only from the ``top_k`` finite logits.

        Returns:
            Padded token ids and ages, each ``[B, T']``.
        """
        was_training = self.training
        self.eval()
        try:
            codes, ages = self._codes_and_ages(codes, ages)
            device = codes.device
            rows_codes: list[torch.Tensor] = []
            rows_ages: list[torch.Tensor] = []
            for index in range(codes.size(0)):
                valid = codes[index] != 0
                row_codes = codes[index][valid].view(1, -1)
                row_ages = ages[index][valid].view(1, -1)
                if row_codes.numel() == 0:
                    row_codes = codes[index, :1].view(1, -1)
                    row_ages = ages[index, :1].view(1, -1)
                row_codes, row_ages = self._rollout(
                    row_codes,
                    row_ages,
                    max_new_tokens=max_new_tokens,
                    max_age=max_age,
                    no_repeat=no_repeat,
                    termination_tokens=termination_tokens or [],
                    top_k=top_k,
                )
                rows_codes.append(row_codes.view(-1))
                rows_ages.append(row_ages.view(-1))
            return (
                nn.utils.rnn.pad_sequence(
                    rows_codes, batch_first=True, padding_value=0
                ).to(device),
                nn.utils.rnn.pad_sequence(
                    rows_ages, batch_first=True, padding_value=0.0
                ).to(device),
            )
        finally:
            self.train(was_training)

    def _rollout(
        self,
        codes: torch.Tensor,
        ages: torch.Tensor,
        max_new_tokens: int,
        max_age: float,
        no_repeat: bool,
        termination_tokens: list[int],
        top_k: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blocked = set(termination_tokens)
        for _ in range(max_new_tokens):
            if float(ages[0, -1]) > max_age:
                break
            present = {int(token) for token in codes.view(-1).tolist()}
            if blocked and present & blocked:
                break
            hidden, _ = self._hidden(codes, ages)
            logits = self.lm_head(hidden)[0, -1].clone()
            logits[0] = float("-inf")
            if no_repeat:
                seen = codes.clone()
                seen[seen == 1] = 0
                logits[torch.unique(seen)] = float("-inf")
                logits[0] = float("-inf")
            if top_k is not None:
                finite = int(torch.isfinite(logits).sum())
                keep = min(top_k, finite)
                if keep <= 0:
                    break
                values, indices = torch.topk(logits, keep)
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(0, indices, values)
                logits = filtered
            if not torch.isfinite(logits).any():
                break
            unit = torch.rand(logits.shape, device=logits.device).clamp(1e-6, 1 - 1e-6)
            waits = -torch.exp(-logits) * unit.log()
            waits = torch.where(
                torch.isfinite(logits), waits, torch.full_like(waits, float("inf"))
            )
            wait, token = torch.min(waits, dim=0)
            if not torch.isfinite(wait):
                break
            wait = wait.clamp(min=1e-3)
            codes = torch.cat([codes, token.view(1, 1)], dim=1)
            ages = torch.cat([ages, (ages[0, -1] + wait).view(1, 1)], dim=1)
        return codes, ages
