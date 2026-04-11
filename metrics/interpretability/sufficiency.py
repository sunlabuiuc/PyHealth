"""Sufficiency metric for interpretability evaluation.

This module implements the Sufficiency metric which measures how much
the predicted probability drops when only important features are kept.
"""

from typing import Dict

import torch

from .base import RemovalBasedMetric


class SufficiencyMetric(RemovalBasedMetric):
    """Sufficiency metric for interpretability evaluation.

    Measures the drop in predicted class probability when ONLY important
    features are KEPT (all others removed). Lower scores indicate more
    faithful interpretations.

    The metric is computed as:
        SUFF = (1/|B|) × Σ[p_c(x)(x) - p_c(x)(x:q%)]
                        q∈B

    Where:
        - x is the original input
        - x:q% are the top q% most important features (all others removed)
        - p_c(x)(·) is predicted probability for original predicted class
        - B is the set of percentages (default: {1, 5, 10, 20, 50})

    Examples:
        >>> import torch
        >>> from pyhealth.models import MLP
        >>> from pyhealth.metrics.interpretability import SufficiencyMetric
        >>>
        >>> # Assume we have a trained model
        >>> model = MLP(dataset=dataset)
        >>>
        >>> # Initialize metric
        >>> suff = SufficiencyMetric(model)
        >>>
        >>> # Prepare inputs and attributions
        >>> inputs = {'conditions': torch.randn(32, 50)}
        >>> attributions = {'conditions': torch.randn(32, 50)}
        >>>
        >>> # Compute metric
        >>> scores, valid_mask = suff.compute(inputs, attributions)
        >>> print(f"Mean sufficiency: {scores[valid_mask].mean():.3f}")
        Mean sufficiency: 0.089
        >>>
        >>> # Get detailed scores per percentage
        >>> detailed = suff.compute(
        ...     inputs, attributions, return_per_percentage=True
        ... )
        >>> for pct, scores in detailed.items():
        ...     print(f"  {pct}%: {scores.mean():.3f}")
          1%: 0.234
          5%: 0.178
          10%: 0.089
          20%: 0.045
          50%: 0.012
    """

    def _create_ablated_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Create ablated inputs by KEEPING only important features.

        For sufficiency, mask==1 indicates important features to keep.
        We invert the mask to ablate everything else.

        Args:
            inputs: Original model inputs
            masks: Binary masks (1=keep, 0=remove)

        Returns:
            Ablated inputs with only important features kept
        """
        # Invert masks: ablate where mask==0 (keep only where mask==1)
        inverted_masks = {key: 1 - mask for key, mask in masks.items()}
        return self._apply_ablation(inputs, inverted_masks)
