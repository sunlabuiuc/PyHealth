"""BulkRNABert survival model for PyHealth.

Reproduces the downstream survival prediction head of Gélard et al. (2025),
"BulkRNABert: Cancer prognosis from bulk RNA-seq based language models"
(ML4H 2025): a small MLP over 256-d frozen-encoder embeddings, trained
with Cox partial likelihood.

The shipped pipeline consumes cached 256-d embeddings produced offline by
the frozen BulkRNABert encoder. The encoder architecture, tokenizer, weight
loader, and parity validation live in the accompanying research repository
as preprocessing/validation tools, not in this PyHealth model.
"""

from __future__ import annotations

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.trajsurv import cox_partial_likelihood_loss

EMBED_DIM = 256


def _unpack(v):
    """Unwrap ``(value, mask)`` tuples that some processors return."""
    if isinstance(v, (tuple, list)):
        return v[0]
    return v


class SurvivalHead(nn.Module):
    """MLP head: 256 -> 512 (SELU) -> 256 (SELU) -> LayerNorm -> 1.
    Example:
        >>> import torch
        >>> head = SurvivalHead()
        >>> head(torch.zeros(2, 256)).shape
        torch.Size([2])
    """

    def __init__(
        self, in_dim: int = EMBED_DIM, hidden: tuple[int, ...] = (512, 256)
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = in_dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.SELU()]
            width = size
        self.hidden = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(width, eps=1e-5)
        self.out = nn.Linear(width, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = module.in_features**-0.5
                nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(self.hidden(x))).squeeze(-1)


def cox_loss(
    risk: torch.Tensor,
    duration: torch.Tensor,
    event: torch.Tensor,
    group: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean negative Cox partial log-likelihood (Breslow ties).

    Delegates to :func:`pyhealth.models.trajsurv.cox_partial_likelihood_loss`
    for the global case. With ``group``, one risk set is built per cohort
    and the per-cohort losses are combined by an event-count-weighted
    average, so the result is still a mean over all observed events rather
    than a mean of per-cohort means (which would weight small cohorts
    equally with large ones).

    Args:
        risk: Risk scores, shape ``(batch,)``.
        duration: Survival times, shape ``(batch,)``.
        event: Event indicators, shape ``(batch,)``.
        group: Optional cohort ids, shape ``(batch,)``.


    Returns:
        Scalar loss. If no events are present, returns a zero-valued tensor
        that still propagates gradients (safe for minibatches).
    Example:
        >>> import torch
        >>> risk = torch.tensor([0.9, 0.5, 0.1])
        >>> duration = torch.tensor([1.0, 2.0, 3.0])
        >>> event = torch.tensor([1.0, 1.0, 1.0])
        >>> bool(cox_loss(risk, duration, event) > 0)
        True
    """
    if group is None:
        return cox_partial_likelihood_loss(risk, duration, event)

    total_weighted_loss = 0.0
    total_events = 0.0
    for g in group.unique():
        mask = group == g
        n_events = float(event[mask].sum().item())
        if n_events == 0:
            continue
        per_group_loss = cox_partial_likelihood_loss(
            risk[mask], duration[mask], event[mask]
        )
        total_weighted_loss = total_weighted_loss + per_group_loss * n_events
        total_events += n_events
    if total_events == 0:
        return risk.sum() * 0.0
    return total_weighted_loss / total_events


class BulkRNABert(BaseModel):
    """BulkRNABert survival head over cached 256-d embeddings.

    Small MLP (256 -> 512 -> 256 -> 1) with SELU and LayerNorm, trained
    with Cox partial likelihood. Consumes the ``embedding`` field emitted
    by :class:`~pyhealth.tasks.TCGASurvivalPrediction`.

    Args:
        dataset: PyHealth ``SampleDataset``.
        feature_key: Fallback input field name when ``dataset`` is None.
        cox_loss_mode: ``"global"`` (paper default) or ``"cohort"``
            (risk sets restricted to the same cancer cohort).
    Note:
        This model computes its own Cox partial likelihood and returns it as
        ``out["loss"]``, matching the ``MetabricSurvivalPrediction`` and
        ``InHospitalSurvivalMIMIC4`` task conventions in this repo. Because
        the task's outputs are all ``"tensor"`` fields (no ``LabelProcessor``),
        ``model.mode`` is ``None`` and PyHealth's stock ``Trainer.evaluate()``
        reports only average loss, not C-index. Survival evaluation is done
        outside the Trainer using ``pyhealth.metrics.survival.concordance_index``
        — see ``examples/tcga_rna_seq_tcga_survival_prediction_bulk_rna_bert.py``
        for the pattern.

        ``y_prob`` is returned for PyHealth API compatibility but is not a
        calibrated probability: it is the raw Cox risk score. Higher means
        higher predicted risk.
    Example:
        >>> import torch
        >>> from pyhealth.models import BulkRNABert
        >>> m = BulkRNABert()
        >>> out = m(embedding=torch.randn(4, 256))
        >>> out["logit"].shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset | None = None,
        feature_key: str = "embedding",
        cox_loss_mode: str = "global",
    ) -> None:
        super().__init__(dataset=dataset)
        self._feature_key = feature_key
        if cox_loss_mode not in {"global", "cohort"}:
            raise ValueError(
                f"cox_loss_mode must be 'global' or 'cohort', got {cox_loss_mode!r}"
            )
        self.cox_loss_mode = cox_loss_mode
        self.head = SurvivalHead()

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Compute risk scores (and loss if survival labels are present).

        Returns:
            logit: (B, 1) unbounded Cox risk scores.
            y_prob: same as logit (Cox scores are not calibrated probabilities).
            loss: scalar Cox loss (only when ``duration`` and ``event``
                are in ``kwargs``).
            y_true: (B, 1) durations (only when labels are present).
        """
        key = self.feature_keys[0] if self.feature_keys else self._feature_key
        emb = _unpack(kwargs[key])
        if emb.shape[-1] != EMBED_DIM:
            raise ValueError(
                f"embedding must have last dim {EMBED_DIM}, got {emb.shape[-1]}"
            )
        logit = self.head(emb).unsqueeze(-1)
        out: dict[str, torch.Tensor] = {
            "logit": logit,
            "y_prob": logit,
            "risk_score": logit,
        }

        if "duration" in kwargs and "event" in kwargs:
            duration = _unpack(kwargs["duration"]).reshape(-1).float()
            event = _unpack(kwargs["event"]).reshape(-1).float()
            group = None
            if self.cox_loss_mode == "cohort":
                if "cohort" not in kwargs:
                    raise ValueError(
                        "cox_loss_mode='cohort' requires cohort labels in the batch"
                    )
                group = _unpack(kwargs["cohort"]).reshape(-1).long()
            out["loss"] = cox_loss(logit.squeeze(-1), duration, event, group)
            out["y_true"] = _unpack(kwargs["duration"])

        return out
