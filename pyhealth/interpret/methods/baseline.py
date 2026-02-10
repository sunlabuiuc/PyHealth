"""Random baseline attribution method.

This module implements a simple random attribution method that assigns
uniformly random importance scores to each input feature. It serves as a
baseline for evaluating the quality of more sophisticated interpretability
methods — any useful attribution technique should outperform random
assignments.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from pyhealth.models import BaseModel
from .base_interpreter import BaseInterpreter


class Random(BaseInterpreter):
    """Random feature attribution baseline for PyHealth models.

    This interpreter assigns uniformly random importance scores to input
    features. It is intended as a **sanity-check baseline**: any meaningful
    interpretability method should produce attributions that are
    significantly more faithful to the model than random scores.

    The method works with all PyHealth models and requires no gradients,
    embeddings, or special model capabilities.

    Args:
        model: A trained PyHealth model to interpret. Can be any model
            that inherits from :class:`~pyhealth.models.BaseModel`.
        random_seed: Optional random seed for reproducibility. When set,
            the same inputs will always receive the same random
            attributions. Default is ``None`` (non-deterministic).
        distribution: Distribution used to sample importances. One of
            ``"uniform"`` (values in ``[0, 1]``), ``"normal"`` (standard
            normal), or ``"signed_uniform"`` (values in ``[-1, 1]``).
            Default is ``"uniform"``.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.interpret.methods.baseline import Random
        >>> from pyhealth.models import MLP
        >>>
        >>> samples = [
        ...     {"patient_id": "p0", "visit_id": "v0",
        ...      "conditions": ["cond-33", "cond-86", "cond-80"],
        ...      "procedures": [1.0, 2.0, 3.5, 4.0], "label": 1},
        ...     {"patient_id": "p1", "visit_id": "v1",
        ...      "conditions": ["cond-55", "cond-12"],
        ...      "procedures": [5.0, 2.0, 3.5, 4.0], "label": 0},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "sequence", "procedures": "tensor"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = MLP(dataset=dataset, embedding_dim=32, hidden_dim=32)
        >>> model.eval()
        >>> test_loader = get_dataloader(dataset, batch_size=1, shuffle=False)
        >>>
        >>> rand_interp = Random(model, random_seed=42)
        >>> batch = next(iter(test_loader))
        >>> attributions = rand_interp.attribute(**batch)
        >>> print({k: v.shape for k, v in attributions.items()})
    """

    _DISTRIBUTIONS = {"uniform", "normal", "signed_uniform"}

    def __init__(
        self,
        model: BaseModel,
        random_seed: Optional[int] = None,
        distribution: str = "uniform",
    ):
        super().__init__(model)
        if distribution not in self._DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {self._DISTRIBUTIONS}, "
                f"got '{distribution}'"
            )
        self.random_seed = random_seed
        self.distribution = distribution

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attribute(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """Compute random attributions for input features.

        Generates random importance scores with the same shape as each
        input feature tensor. No gradients or forward passes are needed.

        Args:
            **kwargs: Input data dictionary from a dataloader batch.
                Should contain feature tensors (or tuples of tensors)
                keyed by the model's feature keys, plus optional label
                or metadata tensors (which are ignored).

        Returns:
            Dictionary mapping each feature key to a random attribution
            tensor whose shape matches the raw input values.
        """
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        device = next(self.model.parameters()).device

        # Filter kwargs to only include model feature keys
        inputs = {
            k: (v,) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
            if k in self.model.feature_keys
        }

        # Extract raw value tensors via processor schema
        values: dict[str, torch.Tensor] = {}
        for k, v in inputs.items():
            schema = self.model.dataset.input_processors[k].schema()
            values[k] = v[schema.index("value")]

        # Generate random attributions matching each value shape
        attributions: dict[str, torch.Tensor] = {}
        for k, v in values.items():
            if self.distribution == "uniform":
                attributions[k] = torch.rand_like(v.float()).to(device)
            elif self.distribution == "normal":
                attributions[k] = torch.randn_like(v.float()).to(device)
            elif self.distribution == "signed_uniform":
                attributions[k] = (torch.rand_like(v.float()) * 2 - 1).to(device)

        return attributions
