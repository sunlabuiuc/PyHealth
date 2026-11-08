"""Exact TreeSHAP attributions for gradient-boosted tree models."""

from __future__ import annotations

from typing import Any

import torch

from pyhealth.interpret.methods.base_interpreter import BaseInterpreter
from pyhealth.models.gradient_boosted_trees import (
    _RESERVED_CODES,
    GradientBoostedTreeModel,
)


class TreeSHAP(BaseInterpreter):
    """Exact TreeSHAP for :class:`~pyhealth.models.GradientBoostedTreeModel`.

    Uses the tree library's own exact Shapley values (``pred_contribs`` in
    XGBoost) rather than sampling, so it is fast and deterministic.
    Attributions are on the margin (log-odds for classification) and are
    returned in the shape of each input field, like the other interpreters:

    - tensor / multi_hot fields: one value per input element.
    - code fields encoded with ``bag_of_codes`` and ``nested_multihot``: each
      code's contribution is split evenly over the positions holding it, and
      ``<pad>`` / ``<unk>`` positions get 0. A code that is absent can still
      contribute (its absence moved the prediction), but has no input position;
      :meth:`GradientBoostedTreeModel.explain` keeps those in column form.

    Args:
        model: A fitted gradient-boosted tree model.

    Examples:
        >>> from pyhealth.interpret.methods import TreeSHAP
        >>> attributions = TreeSHAP(model).attribute(**batch)  # doctest: +SKIP
        >>> attributions["labs"].shape == batch["labs"].shape  # doctest: +SKIP
        True
    """

    def __init__(self, model: GradientBoostedTreeModel):
        if not isinstance(model, GradientBoostedTreeModel):
            raise TypeError(
                "TreeSHAP needs a gradient-boosted tree model such as "
                f"XGBoostModel, not {type(model).__name__}."
            )
        super().__init__(model)

    def attribute(
        self, target_class_idx: int | None = None, **data: Any
    ) -> dict[str, torch.Tensor]:
        """Computes TreeSHAP attributions in the shape of each input field.

        Args:
            target_class_idx: Class (multiclass) or label (multilabel) to
                explain; defaults to the highest-scoring one per sample.
                Ignored for binary and regression.
            **data: A batch from the dataloader.

        Returns:
            ``{field: attribution tensor}`` with each tensor shaped like the
            field's input.
        """
        out = self.model.explain(**data)
        logit = out["logit"]
        targets = self._resolve_target_indices(logit, target_class_idx)
        rows = torch.arange(logit.shape[0], device=logit.device)
        result = {}
        for field in self.model.feature_layout:
            key = field["key"]
            a = out["attributions"][key]
            if a.ndim == 3:
                a = a[rows, :, targets]
            x = data[key].to(a.device)
            if field["kind"] in ("dense", "multi_hot"):
                result[key] = a.reshape(x.shape)
            elif field["kind"] == "visit_counts":
                counts = x.sum(dim=1)[:, _RESERVED_CODES:]
                per = torch.where(counts > 0, a / counts.clamp(min=1), 0.0)
                full = torch.zeros(x.shape[0], x.shape[-1], device=a.device)
                full[:, _RESERVED_CODES:] = per
                result[key] = x.float() * full[:, None, :]
            else:  # "counts": token indices
                n = x.shape[0]
                idx = x.reshape(n, -1).long()
                counts = torch.zeros(n, field["width"] + _RESERVED_CODES, device=a.device)
                counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=counts.dtype))
                per = torch.zeros_like(counts)
                body = counts[:, _RESERVED_CODES:]
                per[:, _RESERVED_CODES:] = torch.where(body > 0, a / body.clamp(min=1), 0.0)
                result[key] = per.gather(1, idx).reshape(x.shape)
        return result
