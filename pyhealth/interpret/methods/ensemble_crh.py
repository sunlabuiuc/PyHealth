"""CRH ensemble interpreter.

This module implements the Conflict Resolution on Heterogeneous data (CRH)
ensemble strategy for aggregating attributions from multiple interpretability
experts into a single consensus attribution.
"""

from __future__ import annotations

import torch

from pyhealth.models import BaseModel
from .base_ensemble import BaseInterpreterEnsemble
from .base_interpreter import BaseInterpreter


class CrhEnsemble(BaseInterpreterEnsemble):
    """Ensemble interpreter using Conflict Resolution on Heterogeneous data (CRH).

    Iteratively estimates a consensus attribution by reweighting experts
    according to their agreement with the current consensus estimate.
    Experts whose attributions are closer to the consensus receive higher
    weights, which in turn pulls the consensus toward more reliable
    experts.

    This implements the truth-discovery algorithm from:

        Li, Q., Li, Y., Gao, J., Zhao, B., Fan, W., and Han, J.
        "Resolving Conflicts in Heterogeneous Data by Truth Discovery
        and Source Reliability Estimation."  In *Proceedings of the 2014
        ACM SIGMOD International Conference on Management of Data*
        (SIGMOD'14), pp. 1187–1198, 2014.

    Args:
        model: The PyHealth model to interpret.
        experts: A list of at least three :class:`BaseInterpreter` instances
            whose ``attribute`` methods will be called to produce individual
            attribution maps.
        n_iter: Maximum number of CRH refinement iterations.  Higher values
            allow more precise convergence at the cost of computation.
        low_confidence_threshold: If set, batches where the standard
            deviation of the final expert weights falls below this value
            are considered low-confidence, and their consensus is replaced
            by a simple uniform average of all experts.
        early_stopping_threshold: If set, the CRH loop terminates early
            when the maximum absolute change in the consensus vector
            between successive iterations is below this value.

    Example:
        >>> from pyhealth.interpret.methods import GradientShap, IntegratedGradients, Saliency
        >>> experts = [GradientShap(model), IntegratedGradients(model), Saliency(model)]
        >>> ensemble = CrhEnsemble(model, experts, n_iter=30)
        >>> attrs = ensemble.attribute(**batch)
    """

    def __init__(
        self,
        model: BaseModel,
        experts: list[BaseInterpreter],
        n_iter: int = 20,
        low_confidence_threshold: float | None = None,
        early_stopping_threshold: float | None = None,
    ):
        super().__init__(model, experts)
        self.n_iter = n_iter
        self.low_confidence_threshold = low_confidence_threshold
        self.early_stopping_threshold = early_stopping_threshold

    # ------------------------------------------------------------------
    # Ensemble implementation
    # ------------------------------------------------------------------
    def _ensemble(self, attributions: torch.Tensor) -> torch.Tensor:
        """Aggregate expert attributions via the CRH truth-discovery algorithm.

        The consensus is initialised as the median across experts and then
        iteratively refined: on each iteration, expert weights are set
        inversely proportional to their mean squared error against the
        current consensus, and the consensus is updated as the weighted
        average of all experts.

        Args:
            attributions: Normalized attribution tensor of shape
                ``(B, I, M)`` with values in [0, 1], where *B* is the
                batch size, *I* is the number of experts, and *M* is the
                total number of flattened features.

        Returns:
            Consensus tensor of shape ``(B, M)`` with values in [0, 1].
        """
        # Step 1: Initialize truth as median across experts (B, M)
        t = torch.median(attributions, dim=1).values  # (B, M)
        
        # Iterative refinement
        eps = 1e-6
        
        for _ in range(self.n_iter):
            t_old = t.clone()
            
            # Step 2: Compute expert reliability per batch
            # errors: (B, I) - mean squared error per expert per batch
            errors = torch.mean((attributions - t.unsqueeze(1)) ** 2, dim=2)  # (B, I)
            
            # weights: (B, I)
            w = 1.0 / (eps + errors)  # (B, I)
            w = w / w.sum(dim=1, keepdim=True)  # normalize per batch
            
            # Step 3: Update truth as weighted average
            # t: (B, M) = sum over experts of w * attributions
            t = torch.sum(w.unsqueeze(2) * attributions, dim=1)  # (B, M)
            
            # Early stopping: check convergence per batch
            if self.early_stopping_threshold is not None:
                if torch.allclose(t, t_old, atol=self.early_stopping_threshold):
                    break
        
        if self.low_confidence_threshold is None:
            # If no low confidence threshold is set, just return the CRH result
            return t
        
        # Detect low-confidence batches where all experts are equally weighted
        # If std(w) is very low, it means no expert is clearly better
        w_std = torch.std(w, dim=1)  # type: ignore[assignment] (B,)
        low_confidence = w_std < self.low_confidence_threshold  # (B,)
        
        # For low-confidence batches, fall back to uniform weighting (mean)
        if low_confidence.any():
            uniform_consensus = torch.mean(attributions, dim=1)  # (B, M)
            t = torch.where(
                low_confidence.unsqueeze(1),  # (B, 1)
                uniform_consensus,
                t
            )
        
        return t