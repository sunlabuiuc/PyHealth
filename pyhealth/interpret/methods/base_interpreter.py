"""Abstract base class for interpretability methods.

This module defines the interface that all interpretability/attribution
methods must implement. It ensures consistency across different methods
and makes it easy to swap between different attribution techniques.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn

from pyhealth.models import BaseModel


class BaseInterpreter(ABC):
    """Abstract base class for interpretability methods.

    This class defines the interface that all attribution methods must
    implement. Attribution methods compute importance scores for input
    features, explaining which features contributed most to the model's
    prediction.

    All interpretability methods should:
    1. Take a trained model in their constructor
    2. Implement the `attribute()` method
    3. Return attributions as a dictionary matching input shapes
    4. Work with any PyHealth model (or at least clearly document
       compatibility requirements)

    The `attribute()` method is the core interface that:
    - Takes model inputs (as would be passed to model.forward())
    - Computes attribution scores for each input feature
    - Returns a dictionary mapping feature keys to attribution tensors
    - Attribution tensors have the same shape as input tensors
    - Higher absolute values indicate more important features

    Subclasses should implement:
        - __init__(self, model, **kwargs): Initialize with model and
          method-specific parameters
        - attribute(self, **data): Compute attributions for given inputs

    Args:
        model (BaseModel or nn.Module): A trained PyHealth model to
            interpret. Should be in evaluation mode during attribution
            computation.

    Examples:
        >>> # Example of implementing a new attribution method
        >>> class MyAttributionMethod(BaseInterpreter):
        ...     def __init__(self, model, some_param=1.0):
        ...         super().__init__(model)
        ...         self.some_param = some_param
        ...
        ...     def attribute(self, **data):
        ...         # Implement attribution computation
        ...         attributions = {}
        ...         for key in self.model.feature_keys:
        ...             # Compute importance scores
        ...             attributions[key] = compute_scores(data[key])
        ...         return attributions
        >>>
        >>> # Using the attribution method
        >>> model = StageNet(dataset=dataset)
        >>> interpreter = MyAttributionMethod(model)
        >>> batch = next(iter(dataloader))
        >>> attributions = interpreter.attribute(**batch)
    """

    def __init__(self, model: nn.Module):
        """Initialize the base interpreter.

        Args:
            model: A trained PyHealth model to interpret. The model
                should be in evaluation mode during interpretation.

        Note:
            Subclasses should call super().__init__(model) and then
            perform any method-specific initialization.
        """
        self.model = model
        if hasattr(model, "eval"):
            self.model.eval()

    @abstractmethod
    def attribute(
        self,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute attribution scores for input features.

        This is the main method that all interpretability methods must
        implement. It takes model inputs and returns attribution scores
        indicating which features were most important for the model's
        prediction.

        Args:
            **data: Input data dictionary from a dataloader batch. Should
                contain at minimum:
                - Feature keys (e.g., 'conditions', 'procedures', 'icd_codes'):
                  Input tensors or sequences for each modality. The exact
                  keys depend on the model's feature_keys.
                - 'label' (optional): Ground truth labels, may be needed by
                  some methods but not used in attribution computation.
                - Additional method-specific parameters can be passed here
                  (e.g., target_class_idx, baseline, steps).

                The data dictionary should match what would be passed to
                the model's forward() method.

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping each feature key to
                its attribution tensor. Each attribution tensor:
                - Has the same shape as the corresponding input tensor
                - Contains real-valued importance scores
                - Higher absolute values = more important features
                - Can be positive (increases prediction) or negative
                  (decreases prediction) depending on the method
                - Should be on the same device as the input

        Raises:
            NotImplementedError: If subclass doesn't implement this method.

        Note:
            **Attribution Properties:**

            Different attribution methods may produce scores with different
            properties:

            1. **Sign**: Some methods produce only positive scores (e.g.,
               attention weights), while others can produce both positive
               and negative scores (e.g., Integrated Gradients).

            2. **Magnitude**: Scores may be:
               - Normalized to sum to 1 (probability-like)
               - Unnormalized gradients or relevance scores
               - Relative importance within each feature modality

            3. **Interpretation**: Higher absolute values generally mean
               more important, but the exact interpretation depends on the
               method.

            **Common Patterns:**

            - Gradient-based methods (IG, Saliency): Can be positive or
              negative, represent contribution to output change
            - Attention-based methods (Chefer): Usually positive, represent
              relevance or importance
            - Perturbation-based methods (LIME, SHAP): Can be positive or
              negative, represent feature contribution

        Examples:
            >>> # Basic usage
            >>> interpreter = IntegratedGradients(model)
            >>> batch = next(iter(test_loader))
            >>> attributions = interpreter.attribute(**batch)
            >>> print(attributions.keys())  # Feature keys
            >>> print(attributions['conditions'].shape)  # Same as input
            >>>
            >>> # With method-specific parameters
            >>> attributions = interpreter.attribute(
            ...     **batch,
            ...     target_class_idx=1,  # Attribute to specific class
            ...     steps=50  # Method-specific parameter
            ... )
            >>>
            >>> # Analyze most important features
            >>> cond_attr = attributions['conditions'][0]  # First sample
            >>> top_features = torch.topk(torch.abs(cond_attr), k=5)
            >>> print(f"Top 5 features: {top_features.indices}")
            >>> print(f"Importance scores: {top_features.values}")
        """
        pass

    def __call__(self, **data) -> Dict[str, torch.Tensor]:
        """Convenience method to call attribute().

        This allows the interpreter to be used as a callable:
            attributions = interpreter(**batch)

        instead of:
            attributions = interpreter.attribute(**batch)

        Args:
            **data: Same as attribute() method.

        Returns:
            Same as attribute() method.
        """
        return self.attribute(**data)

    def __repr__(self) -> str:
        """String representation of the interpreter."""
        return f"{self.__class__.__name__}(model={self.model.__class__.__name__})"
