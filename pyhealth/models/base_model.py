from abc import ABC
import logging

import torch
import torch.nn as nn

from ..datasets import SampleDataset
from ..processors import LabelProcessor

logger = logging.getLogger(__name__)


class BaseModel(ABC, nn.Module):
    """Abstract class for PyTorch models.

    Args:
        dataset (SampleDataset): The dataset to train the model. It is used to query certain
            information such as the set of all tokens.
            
    Interpretability
    --------
        To use a model with interpretability methods, the model must implement a method
        `forward_from_embedding` that takes in embeddings as input instead of raw features;
        for the models that already take in dense features as input, this method can simply
        call the existing `forward` method. 
        
        For certain gradient-based interpretability methods (e.g., DeepLIFT), the model must also
        ensure all non-linearity (e.g. ReLU, Sigmoid, Softmax) are using nn.Module versions instead of
        functional versions (e.g., F.relu, F.sigmoid, F.softmax) so that hooks can be registered properly.

    Labels
    --------
        The output processors that are
        :class:`~pyhealth.processors.LabelProcessor` instances define the labels:
        :attr:`label_processors` maps each to its key, and
        :attr:`label_processor` is the one of a single-label model. A model
        takes its output size, activation and default loss from it::

            self.fc = nn.Linear(hidden, self.label_processor.output_size())
            ...
            loss = self.label_processor.loss(logits, y_true)
            y_prob = self.label_processor.activation(logits)

        A model that scores something other than its dataset's label (e.g. a
        reconstruction) overrides :attr:`label_processors`. :attr:`mode` is
        the single label processor's kind, e.g. ``"binary"``, or ``None``.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import RNN
        >>> from pyhealth.processors import MultiLabelProcessor
        >>> samples = [
        ...     {"patient_id": "p0", "codes": ["a", "b"], "labels": ["x"]},
        ...     {"patient_id": "p1", "codes": ["b"], "labels": ["x", "y"]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"codes": "sequence"},
        ...     output_schema={"labels": MultiLabelProcessor},
        ... )
        >>> RNN(dataset=dataset).mode
        'multilabel'
    """

    def __init__(self, dataset: SampleDataset):
        """
        Initializes the BaseModel.

        Args:
            dataset (SampleDataset): The dataset to train the model.
        """
        super(BaseModel, self).__init__()
        self.dataset = dataset
        self.feature_keys = []
        self.label_keys = []
        if dataset:
            self.feature_keys = list(dataset.input_schema.keys())
            self.label_keys = list(dataset.output_schema.keys())
        # used to query the device of the model
        self._dummy_param = nn.Parameter(torch.empty(0))

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """The label processors, by output key.

        These are the dataset's output processors that are
        :class:`~pyhealth.processors.LabelProcessor` instances; other output fields
        (e.g. ``"tensor"`` targets of a model with its own loss) are left out.
        """
        dataset = self.__dict__.get("dataset")
        processors = getattr(dataset, "output_processors", None) or {}
        return {
            key: processor
            for key, processor in processors.items()
            if isinstance(processor, LabelProcessor)
        }

    @property
    def label_processor(self) -> LabelProcessor:
        """The label processor of a single-label model.

        Raises:
            ValueError: If the model does not have exactly one label processor.
        """
        processors = self.label_processors
        if len(processors) != 1:
            raise ValueError(
                f"{type(self).__name__} has {len(processors)} label processors "
                f"({', '.join(processors) or 'none'}); label_processor needs "
                "exactly one. Use label_processors."
            )
        return next(iter(processors.values()))

    @property
    def mode(self) -> str | None:
        """The kind of the single label, e.g. ``"binary"``, or ``None``."""
        processors = self.label_processors
        if len(processors) != 1:
            return None
        return next(iter(processors.values())).mode

    def forward(self, 
            **kwargs: torch.Tensor | tuple[torch.Tensor, ...]
        ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            **kwargs: A variable number of keyword arguments representing input features.
                Each keyword argument is a tensor or a tuple of tensors of shape (batch_size, ...).
        
        Returns:
            A dictionary with the following keys:
                logit: a tensor of predicted logits.
                y_prob: a tensor of predicted probabilities.
                loss [optional]: a scalar tensor representing the final loss, if self.label_keys in kwargs.
                y_true [optional]: a tensor representing the true labels, if self.label_keys in kwargs.
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """
        Gets the device of the model.

        Returns:
            torch.device: The device on which the model is located.
        """
        return self._dummy_param.device
