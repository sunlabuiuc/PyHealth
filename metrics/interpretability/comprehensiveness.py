"""Comprehensiveness metric for interpretability evaluation.

This module implements the Comprehensiveness metric which measures how much
the predicted probability drops when important features are removed.
"""

from typing import Dict

import torch

from .base import RemovalBasedMetric


class ComprehensivenessMetric(RemovalBasedMetric):
    """Comprehensiveness metric for interpretability evaluation.

    Measures the drop in predicted class probability when important features
    are REMOVED (ablated). Higher scores indicate more faithful
    interpretations.

    The metric is computed as:
        COMP = (1/|B|) × Σ[p_c(x)(x) - p_c(x)(x \\ x:q%)]
                        q∈B

    Where:
        - x is the original input
        - x:q% are the top q% most important features
        - x \\ x:q% is input with top q% features removed (ablated)
        - p_c(x)(·) is predicted probability for original predicted class
        - B is the set of percentages (default: {1, 5, 10, 20, 50})

    Examples:
        >>> import torch
        >>> from pyhealth.models import MLP
        >>> from pyhealth.metrics.interpretability import (
        ...     ComprehensivenessMetric
        ... )
        >>>
        >>> # Assume we have a trained model
        >>> model = MLP(dataset=dataset)
        >>>
        >>> # Initialize metric
        >>> comp = ComprehensivenessMetric(model)
        >>>
        >>> # Prepare inputs and attributions
        >>> inputs = {'conditions': torch.randn(32, 50)}
        >>> attributions = {'conditions': torch.randn(32, 50)}
        >>>
        >>> # Compute metric
        >>> scores, valid_mask = comp.compute(inputs, attributions)
        >>> print(f"Mean comprehensiveness: {scores[valid_mask].mean():.3f}")
        Mean comprehensiveness: 0.234
        >>>
        >>> # Get detailed scores per percentage
        >>> detailed = comp.compute(
        ...     inputs, attributions, return_per_percentage=True
        ... )
        >>> for pct, scores in detailed.items():
        ...     print(f"  {pct}%: {scores.mean():.3f}")
          1%: 0.045
          5%: 0.123
          10%: 0.234
          20%: 0.345
          50%: 0.456
    """

    def _create_ablated_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Create ablated inputs by REMOVING (ablating) important features.

        For comprehensiveness, mask==1 indicates important features to remove.

        Args:
            inputs: Original model inputs
            masks: Binary masks (1=remove, 0=keep)

        Returns:
            Ablated inputs with important features removed
        """
        return self._apply_ablation(inputs, masks)
