"""Variance-weighted ensemble interpreter.

This module implements the AGGVar ensemble strategy, which aggregates
attributions from multiple interpretability experts by dividing the mean
attribution by the standard deviation, penalising features where experts
disagree.
"""

from __future__ import annotations

import torch

from pyhealth.models import BaseModel
from .base_ensemble import BaseInterpreterEnsemble
from .base_interpreter import BaseInterpreter


class VarEnsemble(BaseInterpreterEnsemble):
    """Ensemble interpreter using variance-weighted averaging (AGGVar).

    Computes the consensus attribution by dividing the mean of the
    competitively-ranked expert attributions by their standard deviation
    (plus a small constant ε for numerical stability).  Features that
    all experts agree are important receive high scores, while features
    with high inter-expert disagreement are suppressed.

    Implements the AGGVar method from:

        Rieger, L. and Hansen, L. K. "Aggregating Explanation Methods
        for Stable and Robust Explainability." arXiv preprint
        arXiv:1903.00519, 2019.

    Args:
        model: The PyHealth model to interpret.
        experts: A list of at least three :class:`BaseInterpreter` instances
            whose ``attribute`` methods will be called to produce individual
            attribution maps.

    Example:
        >>> from pyhealth.interpret.methods import IntegratedGradients, DeepLift, LimeExplainer
        >>> experts = [IntegratedGradients(model), DeepLift(model), LimeExplainer(model)]
        >>> ensemble = VarEnsemble(model, experts)
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
        """Aggregate expert attributions via variance-weighted averaging.

        Computes ``mean / (std + ε)`` across experts for each feature,
        rewarding consensus and penalising disagreement.

        Args:
            attributions: Normalized attribution tensor of shape
                ``(B, I, M)`` with values in [0, 1], where *B* is the
                batch size, *I* is the number of experts, and *M* is the
                total number of flattened features.

        Returns:
            Consensus tensor of shape ``(B, M)``.
        """
        mean = torch.mean(attributions, dim=1)   # (B, M)
        std = torch.std(attributions, dim=1)      # (B, M)
        return mean / (std + 1e-6)