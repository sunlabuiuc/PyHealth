"""Average ensemble interpreter.

This module implements the AGGMean ensemble strategy, which aggregates
attributions from multiple interpretability experts by taking the uniform
average of their competitively-ranked importance scores.
"""

from __future__ import annotations

import torch

from pyhealth.models import BaseModel
from .base_ensemble import BaseInterpreterEnsemble
from .base_interpreter import BaseInterpreter


class AvgEnsemble(BaseInterpreterEnsemble):
    """Ensemble interpreter using uniform averaging (AGGMean / Borda).

    Computes the consensus attribution as the simple arithmetic mean
    of the competitively-ranked attributions from all expert interpreters.
    This is the simplest ensemble strategy — every expert contributes
    equally regardless of its agreement with the others.

    Because the inputs are already competitively ranked, averaging is
    equivalent (up to a constant factor) to the Borda count, which sums
    the ranks instead.  The two methods therefore produce identical
    feature orderings.

    Implements the AGGMean method from:

        Rieger, L. and Hansen, L. K. "Aggregating Explanation Methods
        for Stable and Robust Explainability." arXiv preprint
        arXiv:1903.00519, 2019.

    See also the Borda aggregation in:

        Chen, Y., Mancini, M., Zhu, X., and Akata, Z. "Ensemble
        Interpretation: A Unified Method for Interpretable Machine
        Learning." arXiv preprint arXiv:2312.06255, 2023.

    Args:
        model: The PyHealth model to interpret.
        experts: A list of at least three :class:`BaseInterpreter` instances
            whose ``attribute`` methods will be called to produce individual
            attribution maps.

    Example:
        >>> from pyhealth.interpret.methods import IntegratedGradients, DeepLift, LimeExplainer
        >>> experts = [IntegratedGradients(model), DeepLift(model), LimeExplainer(model)]
        >>> ensemble = AvgEnsemble(model, experts)
        >>> attrs = ensemble.attribute(**batch)
    """

    def __init__(
        self,
        model: BaseModel,
        experts: list[BaseInterpreter],
    ):
        super().__init__(model, experts)

    # ------------------------------------------------------------------
    # Ensemble implementation
    # ------------------------------------------------------------------
    def _ensemble(self, attributions: torch.Tensor) -> torch.Tensor:
        """Aggregate expert attributions by uniform averaging.

        Args:
            attributions: Normalized attribution tensor of shape
                ``(B, I, M)`` with values in [0, 1], where *B* is the
                batch size, *I* is the number of experts, and *M* is the
                total number of flattened features.

        Returns:
            Consensus tensor of shape ``(B, M)`` with values in [0, 1].
        """
        return torch.mean(attributions, dim=1)