"""Abstract base class for interpretability methods.

This module defines the interface that all interpretability/attribution
methods must implement. It ensures consistency across different methods
and makes it easy to swap between different attribution techniques.

The key API contract is that ``attribute()`` returns a dictionary keyed by
the model's feature keys (as defined by the task schema), making it easy to
map attributions back to specific input modalities.
"""

from abc import ABC, abstractmethod
from typing import Dict, cast

import torch
import torch.nn as nn

from pyhealth.models import BaseModel


class BaseInterpreter(ABC):
    """Abstract base class for interpretability methods.

    This class defines the interface that all attribution methods must
    implement. Attribution methods compute importance scores for input
    features, explaining which features contributed most to the model's
    prediction.

    **API Contract:**

    All interpretability methods should:

    1. Take a trained model in their constructor
    2. Implement the ``attribute()`` method
    3. Return attributions as a dictionary **keyed by the model's feature keys**
       (as defined by the task's ``input_schema``)
    4. Work with any PyHealth model (or clearly document compatibility)

    The ``attribute()`` method returns a dictionary that mirrors the task schema:

    - For EHR tasks with ``input_schema={"conditions": "sequence", "procedures": "sequence"}``,
      returns ``{"conditions": attr_tensor, "procedures": attr_tensor}``
    - For image tasks with ``input_schema={"image": "image"}``,
      returns ``{"image": attr_tensor}``

    This design ensures attributions are dynamically tied to dataset feature keys,
    making the API consistent across CXR datasets, EHR datasets, or any custom
    task schema.

    Subclasses should implement:
        - ``__init__(self, model, **kwargs)``: Initialize with model and
          method-specific parameters
        - ``attribute(self, **data)``: Compute attributions for given inputs

    Args:
        model (BaseModel or nn.Module): A trained PyHealth model to interpret.
            The model must have ``feature_keys`` (list of input feature names)
            derived from the dataset's task schema. Should be in evaluation
            mode during attribution computation.

    Example:
        >>> # Example 1: EHR data with multiple feature keys
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import Transformer
        >>>
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "conditions": ["A05B", "A05C", "A06A"],
        ...         "procedures": ["P01", "P02"],
        ...         "label": 1,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "sequence", "procedures": "sequence"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="ehr_example",
        ... )
        >>> model = Transformer(dataset=dataset)
        >>> # model.feature_keys == ["conditions", "procedures"]
        >>>
        >>> interpreter = CheferRelevance(model)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=1)))
        >>> attributions = interpreter.attribute(**batch)
        >>> # Returns: {"conditions": tensor(...), "procedures": tensor(...)}
        >>> print(attributions.keys())  # dict_keys(['conditions', 'procedures'])
        >>>
        >>> # Example 2: Image data (CXR) with single feature key
        >>> # Given task schema: input_schema={"image": "image"}
        >>> # model.feature_keys == ["image"] (or model.feature_key == "image")
        >>>
        >>> interpreter = CheferRelevance(vit_model)
        >>> attributions = interpreter.attribute(**batch)
        >>> # Returns: {"image": tensor(...)} - keyed by the task's feature key
        >>> print(attributions["image"].shape)  # [batch, 1, H, W]
    """

    def __init__(self, model: BaseModel):
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

        **Important:** The returned dictionary must be keyed by the model's
        feature keys (from ``model.feature_keys`` or ``model.feature_key``),
        which are derived from the task's ``input_schema``. This ensures
        attributions map directly to the input modalities defined in the task.

        Args:
            **data: Input data dictionary from a dataloader batch. Should
                contain at minimum:

                - Feature keys (e.g., ``"conditions"``, ``"procedures"``,
                  ``"image"``): Input tensors for each modality as defined
                  by the task's ``input_schema``.
                - Label key (optional): Ground truth labels, may be needed
                  by some methods for loss computation.
                - ``class_index`` (optional): Target class for attribution.
                  If not provided, uses the predicted class.
                - Additional method-specific parameters (e.g., ``baseline``,
                  ``steps``, ``interpolate``).

                The data dictionary should match what would be passed to
                the model's ``forward()`` method.

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping **each feature key**
                to its attribution tensor. The keys must match the model's
                feature keys from the task schema.

                For EHR tasks::

                    {"conditions": tensor, "procedures": tensor, ...}

                For image tasks::

                    {"image": tensor}  # Shape: [batch, 1, H, W] for spatial

                Each attribution tensor:

                - Contains real-valued importance scores
                - Higher absolute values = more important features
                - Can be positive or negative depending on the method
                - Should be on the same device as the input

        Raises:
            NotImplementedError: If subclass doesn't implement this method.

        Note:
            **Attribution Properties:**

            Different attribution methods produce scores with different
            properties:

            1. **Sign**: Some methods produce only positive scores (e.g.,
               attention weights), while others produce both positive and
               negative scores (e.g., Integrated Gradients, DeepLift).

            2. **Magnitude**: Scores may be normalized (sum to 1) or
               unnormalized (raw gradients/relevance).

            3. **Shape**: For sequential data, shape matches input tokens.
               For images, shape is typically ``[batch, 1, H, W]`` for
               spatial attribution maps.

            **Common Patterns:**

            - Gradient-based (IG, Saliency): +/- scores, contribution to output
            - Attention-based (Chefer): Usually positive, relevance/importance
            - Perturbation-based (LIME, SHAP): +/- scores, feature contribution

        Example:
            >>> # EHR model with multiple feature keys
            >>> interpreter = DeepLift(model)
            >>> batch = next(iter(test_loader))
            >>> attributions = interpreter.attribute(**batch)
            >>> print(attributions.keys())  # ['conditions', 'procedures']
            >>>
            >>> # Image model (CXR) with single feature key
            >>> interpreter = CheferRelevance(vit_model)
            >>> attributions = interpreter.attribute(**batch)
            >>> print(attributions.keys())  # ['image']
            >>> print(attributions['image'].shape)  # [1, 1, 224, 224]
        """
        pass
    
    def _prediction_mode(self) -> str:
        """Resolve the prediction mode from the model. 
        
        Returns:
            str: The prediction mode, one of "binary", "multiclass", "multilabel" or "regression".
        """
        
        assert (
            len(self.model.label_keys) == 1
        ), "Only one label key is supported if get_loss_function is called"
        label_key = self.model.label_keys[0]
        return self.model._resolve_mode(self.model.dataset.output_schema[label_key])

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
