from typing import Any

from . import register_processor
from .label_processor import MultiLabelProcessor


@register_processor("multi_hot")
class MultiHotProcessor(MultiLabelProcessor):
    """Processor for converting categorical variables into multi-hot encoded vectors.

    This is an alias of :class:`MultiLabelProcessor` specifically for input feature schemas.
    The implementation is identical to :class:`MultiLabelProcessor`, but this class provides
    semantic clarity: use ``MultiLabelProcessor`` for output labels in classification tasks,
    and ``MultiHotProcessor`` for input features like patient demographics (e.g., ethnicity,
    race, comorbidities).

    The processor builds a vocabulary during the fit() phase and converts each sample's
    categorical values into a fixed-size binary vector.

    Input:
        - List of categorical tokens (e.g., ["asian", "non_hispanic"])
        - Each token is a hashable value (typically strings or integers)

    Processing:
        1. During fit(): builds vocabulary by collecting all unique categorical values
           across the dataset and mapping each to a unique index
        2. During process(): converts a list of tokens into a 1D binary tensor where
           positions corresponding to present categories are set to 1.0, others to 0.0

    Output:
        - torch.Tensor of shape (num_categories,) with dtype float32
        - Binary encoding: 1.0 at indices where categories are present, 0.0 elsewhere
        - The size() method returns num_categories (vocabulary size)

    Example:
        Given samples with ethnicity field:
            - Sample 1: ["asian", "non_hispanic"]
            - Sample 2: ["white", "hispanic"]
            - Sample 3: ["black"]

        After fit():
            vocabulary = {"asian": 0, "black": 1, "hispanic": 2, "non_hispanic": 3, "white": 4}
            size() returns 5

        Processing Sample 1 produces:
            tensor([1.0, 0.0, 0.0, 1.0, 0.0])  # asian and non_hispanic are present

    Note:
        The processor sorts categories alphabetically during vocabulary construction
        to ensure consistent index assignments across runs.

    See Also:
        :class:`MultiLabelProcessor`: Parent class with identical implementation.
            Use for output labels in multi-label classification tasks.
    """

    def process(self, value: Any):
        return super().process(value)


