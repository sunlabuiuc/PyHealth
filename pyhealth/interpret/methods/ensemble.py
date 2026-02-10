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


class Ensemble(BaseInterpreter):
    def __init__(
        self,
        model: BaseModel,
        interpreters: list[BaseInterpreter],
    ):
        super().__init__(model)
        assert len(interpreters) >= 3, "Ensemble must contain at least three interpreters for majority voting"
        self.interpreters = interpreters

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------
    def _flatten_attributions(
        self,
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

    def _unflatten_attributions(
        self,
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
        out_shape: dict[str, torch.Size] | None = None
        attr_lst: list[torch.Tensor] = []
        for interpreter in self.interpreters:
            attr = interpreter.attribute(**kwargs)
            
            # record the output shape from the first interpreter, 
            # since all interpreters should produce the same shape
            if out_shape is None:
                out_shape = {k: v.shape for k, v in attr.items()}
            
            flat_attr = self._flatten_attributions(attr) # shape (B, M)
            attr_lst.append(flat_attr)
        
        # Combine the flattened attributions from all interpreters
        attributions = torch.stack(attr_lst, dim=1)  # shape (B, I, M)
        # Normalize the attributions across items for each interpreter (e.g., by competitive ranking)
        attributions = self.competitive_ranking_noramlize(attributions) # shape (B, I, M)
        
        
        
        # normalize the attributions across interpreters (e.g., by ranking)
        _, rank = attributions.sort
            
            
            
            
            
        
        raise NotImplementedError("Ensemble attribution method is not implemented yet. This is a placeholder for future development.")

    @staticmethod
    def competitive_ranking_noramlize(x: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor via competitive (standard competition) ranking.

        For each (batch, expert) slice, items are ranked ascendingly from
        0 to ``total_item - 1``.  Tied scores receive the same rank — the
        smallest position index among the tied group (standard competition /
        "1224" ranking).  The ranks are then divided by ``total_item - 1``
        so that the output lies in [0, 1].

        Args:
            x: Tensor of shape ``(batch_size, expert_size, total_item)``
                containing unbounded floating-point scores.

        Returns:
            Tensor of the same shape with values in [0, 1].
        """
        B, I, M = x.shape

        if M <= 1:
            # With a single item the rank is 0 and 0/0 is undefined;
            # return zeros as a safe default.
            return torch.zeros_like(x)

        # 1. Sort ascending along the item dimension
        sorted_vals, sort_indices = x.sort(dim=-1)

        # 2. Build a mask that is True at positions where the value changes
        #    from the previous position (i.e. the start of a new rank group).
        change_mask = torch.ones(B, I, M, dtype=torch.bool, device=x.device)
        change_mask[..., 1:] = sorted_vals[..., 1:] != sorted_vals[..., :-1]

        # 3. Assign competitive ranks in sorted order.
        #    At change positions the rank equals the position index;
        #    at tie positions we propagate the rank of the first occurrence
        #    via cummax (all non-change positions are set to -1 so cummax
        #    naturally carries forward the last "real" rank).
        positions = torch.arange(M, device=x.device, dtype=torch.long).expand(B, I, M)
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
        return ranks / (M - 1)
