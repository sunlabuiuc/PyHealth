from __future__ import annotations

from typing import Dict, Optional

import torch

from pyhealth.models import BaseModel
from .base_interpreter import BaseInterpreter


class BaseInterpreterEnsemble(BaseInterpreter):
    """Abstract base class for ensemble interpreters.

    Provides the shared workflow for ensemble-based attribution:

    1. Each expert interpreter independently computes attributions.
    2. The per-expert attribution maps are flattened, then normalized to
       a common [0, 1] scale via competitive ranking.
    3. The normalized attributions are passed to :meth:`_ensemble`, which
       concrete subclasses must override to implement a specific
       aggregation strategy (e.g., CRH truth discovery, simple averaging,
       majority voting).
    4. The aggregated result is unflattened back to the original tensor
       shapes.

    Subclasses only need to implement :meth:`_ensemble`.

    Args:
        model: The PyHealth model to interpret.
        experts: A list of at least three :class:`BaseInterpreter` instances
            whose ``attribute`` methods will be called to produce individual
            attribution maps.
    """

    def __init__(
        self,
        model: BaseModel,
        experts: list[BaseInterpreter],
    ):
        super().__init__(model)
        assert len(experts) >= 3, "Ensemble must contain at least three interpreters for majority voting"
        self.experts = experts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attribute(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """Compute consensus attributions by ensembling all expert interpreters.

        Each expert's ``attribute`` method is called with the same inputs.
        The resulting attribution maps are flattened, competitively ranked
        to a common [0, 1] scale, and aggregated via the subclass-defined
        :meth:`_ensemble` strategy.

        Args:
            **kwargs: Input data dictionary from a dataloader batch.
                Should contain feature tensors (or tuples of tensors)
                keyed by the model's feature keys, plus optional label
                or metadata tensors (which are forwarded to experts).

        Returns:
            Dictionary mapping each feature key to a consensus attribution
            tensor whose shape matches the corresponding input tensor.
        """
        out_shape: dict[str, torch.Size] | None = None
        attr_lst: list[torch.Tensor] = []
        for expert in self.experts:
            attr = expert.attribute(**kwargs)
            
            # record the output shape from the first interpreter, 
            # since all interpreters should produce the same shape
            if out_shape is None:
                out_shape = {k: v.shape for k, v in attr.items()}
            
            flat_attr = self._flatten_attributions(attr) # shape (B, M)
            attr_lst.append(flat_attr)
        
        # Combine the flattened attributions from all interpreters
        attributions = torch.stack(attr_lst, dim=1)  # shape (B, I, M)
        # Normalize the attributions across items for each interpreter (e.g., by competitive ranking)
        attributions = self._competitive_ranking_normalize(attributions) # shape (B, I, M)
        
        # Resolve conflicts and aggregate across interpreters using CRH
        consensus = self._ensemble(attributions)  # shape (B, M)
        assert out_shape is not None, "Output shape should have been determined from the first interpreter"
        return self._unflatten_attributions(consensus, out_shape)  # dict of tensors with original shapes

    def _ensemble(self, attributions: torch.Tensor) -> torch.Tensor:
        """Aggregate normalized expert attributions into a single consensus.

        Subclasses must override this method to define the aggregation
        strategy (e.g., iterative truth discovery, simple averaging).

        Args:
            attributions: Normalized attribution tensor of shape
                ``(B, I, M)`` with values in [0, 1], where *B* is the
                batch size, *I* is the number of experts, and *M* is the
                total number of flattened features.

        Returns:
            Aggregated tensor of shape ``(B, M)`` with values in [0, 1].
        """
        raise NotImplementedError("Subclasses must implement their ensemble aggregation strategy in this method")

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_attributions(
        values: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Flatten values dictionary to a single tensor.

        Takes a dictionary of tensors with shape (B, *) and flattens each to (B, M_i),
        then concatenates them along the feature dimension to get (B, M).

        Args:
            values: Dictionary mapping feature keys to tensors of shape (B, *).

        Returns:
            Flattened tensor of shape (B, M) where M is the sum of all flattened dimensions.
        """
        flattened_list = []
        for key in sorted(values.keys()):  # Sort for consistency
            tensor = values[key]
            batch_size = tensor.shape[0]
            # Flatten all dimensions except batch
            flattened = tensor.reshape(batch_size, -1)
            flattened_list.append(flattened)

        # Concatenate along feature dimension
        return torch.cat(flattened_list, dim=1)

    @staticmethod
    def _unflatten_attributions(
        flattened: torch.Tensor,
        shapes: dict[str, torch.Size],
    ) -> dict[str, torch.Tensor]:
        """Unflatten tensor back to values dictionary.

        Takes a flattened tensor of shape (B, M) and original shapes,
        and reconstructs the original dictionary of tensors.

        Args:
            flattened: Flattened tensor of shape (B, M).
            shapes: Dictionary mapping feature keys to original tensor shapes.

        Returns:
            Dictionary mapping feature keys to tensors with original shapes.
        """
        values = {}
        offset = 0

        for key in sorted(shapes.keys()):  # Must match the order in _flatten_values
            shape = shapes[key]
            batch_size = shape[0]

            # Calculate the size of the flattened feature
            feature_size = 1
            for s in shape[1:]:
                feature_size *= s

            # Extract the relevant portion and reshape
            values[key] = flattened[:, offset : offset + feature_size].reshape(shape)
            offset += feature_size

        return values


    @staticmethod
    def _competitive_ranking_normalize(x: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor via competitive (standard competition) ranking.

        For each (batch, expert) slice, items are ranked ascendingly from
        0 to ``total_item - 1``.  Tied scores receive the same rank — the
        smallest position index among the tied group (standard competition /
        "1224" ranking).  The ranks are then divided by ``total_item - 1``
        so that the output lies in [0, 1].

        Args:
            x: Tensor of shape ``(B, I, M)``
                containing unbounded floating-point scores.

        Returns:
            Tensor of the same shape with values in [0, 1].
        """
        batch_size, num_experts, num_items = x.shape

        if num_items <= 1:
            # With a single item the rank is 0 and 0/0 is undefined;
            # return zeros as a safe default.
            return torch.zeros_like(x)

        # 1. Sort ascending along the item dimension
        sorted_vals, sort_indices = x.sort(dim=-1)

        # 2. Build a mask that is True at positions where the value changes
        #    from the previous position (i.e. the start of a new rank group).
        change_mask = torch.ones(batch_size, num_experts, num_items, dtype=torch.bool, device=x.device)
        change_mask[..., 1:] = sorted_vals[..., 1:] != sorted_vals[..., :-1]

        # 3. Assign competitive ranks in sorted order.
        #    At change positions the rank equals the position index;
        #    at tie positions we propagate the rank of the first occurrence
        #    via cummax (all non-change positions are set to -1 so cummax
        #    naturally carries forward the last "real" rank).
        positions = torch.arange(num_items, device=x.device, dtype=torch.long).expand(batch_size, num_experts, num_items)
        ranks_sorted = torch.where(
            change_mask,
            positions,
            torch.full_like(positions, -1),
        )
        ranks_sorted, _ = ranks_sorted.cummax(dim=-1)

        # 4. Scatter the ranks back to the original (unsorted) order
        ranks = torch.zeros_like(x)
        ranks.scatter_(-1, sort_indices, ranks_sorted.to(x.dtype))

        # 5. Normalize to [0, 1]
        return ranks / (num_items - 1)
