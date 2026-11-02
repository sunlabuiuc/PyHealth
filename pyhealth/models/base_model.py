from abc import ABC
from collections.abc import Iterable, Sequence
from typing import Callable, Any, Optional
import functools
import inspect
import logging

import torch
import torch.nn as nn

from ..datasets import SampleDataset
from ..processors import PROCESSOR_REGISTRY, LabelProcessor

logger = logging.getLogger(__name__)

# Label kinds understood by the loss helpers, Trainer and the calibration methods.
# The timeseries modes are per-step labels: a TimeseriesLabelProcessor packs
# them as (values, time, batch offsets), one record per label.
VALID_MODES = (
    "binary",
    "multiclass",
    "multilabel",
    "regression",
    "timeseries_binary",
    "timeseries_regression",
)


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

    Mode
    --------
        ``mode`` is always one of ``"binary"``, ``"multiclass"``, ``"multilabel"``,
        ``"regression"``, ``"timeseries_binary"``, ``"timeseries_regression"``
        or ``None``. It is resolved from the single label's
        ``output_schema`` entry, which may be a string, a processor class, a
        processor instance or a ``(name, kwargs)`` tuple; a processor gives
        its :attr:`~pyhealth.processors.LabelProcessor.mode`. Assigning
        ``self.mode`` in a subclass resolves the value the same way. The
        default loss and ``y_prob`` come from the label's
        :class:`~pyhealth.processors.LabelProcessor`.

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
        # Keep a mode a subclass may have set before calling this __init__.
        self._mode: str | None = self.__dict__.get("_mode")
        if dataset:
            self.feature_keys = list(dataset.input_schema.keys())
            self.label_keys = list(dataset.output_schema.keys())
            # if single label, resolve mode for Trainer and calibration usage
            if len(self.label_keys) == 1:
                try:
                    m = self._resolve_mode(dataset.output_schema[self.label_keys[0]])
                except ValueError:
                    m = None
                if m in VALID_MODES:
                    self._mode = m
        # used to query the device of the model
        self._dummy_param = nn.Parameter(torch.empty(0))

    @property
    def mode(self) -> str | None:
        """The label kind: one of ``VALID_MODES``, or ``None`` if not applicable."""
        return self.__dict__.get("_mode")

    @mode.setter
    def mode(self, value: Any) -> None:
        # Subclasses often assign the raw output_schema entry (e.g. a processor
        # class); resolve it so consumers can always compare against strings.
        if value is None:
            self._mode = None
            return
        try:
            resolved = self._resolve_mode(value)
        except ValueError:
            resolved = None
        if resolved not in VALID_MODES:
            logger.warning(
                "Cannot use %r as a model mode (expected one of %s); "
                "setting mode to None.",
                value,
                ", ".join(VALID_MODES),
            )
            resolved = None
        self._mode = resolved
        
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_mode(self, schema_entry: Any) -> str:
        """Resolve a mode string from an output_schema entry.

        Supports:
          - direct string ("binary", ...)
          - processor class
          - processor instance
          - (string or processor class, kwargs) tuple
        A string is returned lowercased; a processor gives the ``mode`` of its
        :class:`~pyhealth.processors.LabelProcessor`.
        """
        if isinstance(schema_entry, tuple):
            schema_entry = schema_entry[0]
        if isinstance(schema_entry, str):
            return schema_entry.lower()

        cls = schema_entry if inspect.isclass(schema_entry) else schema_entry.__class__
        if issubclass(cls, LabelProcessor) and getattr(cls, "mode", None) in VALID_MODES:
            return cls.mode
        raise ValueError(
            f"Cannot resolve mode from output_schema entry {schema_entry}. Use a supported string"
        )

    def _label_processor(self) -> LabelProcessor:
        """The label processor of the single label key.

        The fitted processor of the dataset when there is one, otherwise a
        new instance of the processor registered under the label's mode
        (enough for the stateless activation and loss).
        """
        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported for the default loss and y_prob"
        label_key = self.label_keys[0]
        try:
            processor = self.dataset.output_processors.get(label_key)
        except (AttributeError, RuntimeError):  # no or unfitted processors
            processor = None
        if isinstance(processor, LabelProcessor):
            return processor
        mode = self._resolve_mode(self.dataset.output_schema[label_key])
        cls = PROCESSOR_REGISTRY.get(mode)
        if not (inspect.isclass(cls) and issubclass(cls, LabelProcessor)):
            raise ValueError(f"Invalid mode: {mode}")
        return cls()

    @property
    def device(self) -> torch.device:
        """
        Gets the device of the model.

        Returns:
            torch.device: The device on which the model is located.
        """
        return self._dummy_param.device

    def get_output_size(self) -> int:
        """
        Gets the default output size using the label tokenizer and `self.mode`.

        If the mode is "binary", the output size is 1. If the mode is "multiclass"
        or "multilabel", the output size is the number of classes or labels.

        Returns:
            int: The output size of the model.
        """
        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if get_output_size is called"
        output_size = self.dataset.output_processors[self.label_keys[0]].size()
        return output_size

    @property
    def pos_weight(self) -> torch.Tensor | None:
        """Weight of positive examples in the binary/multilabel loss, or None."""
        return self.__dict__.get("_pos_weight")

    def set_pos_weight(
        self,
        pos_weight: float | Sequence[float] | torch.Tensor | str | None,
        dataset: Iterable[dict] | None = None,
    ) -> None:
        """Weights positive examples in the default loss, for rare outcomes.

        Applies to binary and multilabel labels, through
        :meth:`get_loss_function` (``pos_weight`` of
        ``F.binary_cross_entropy_with_logits``). Models that compute their own
        loss (e.g. the drug-recommendation models) are not affected. Weighting
        changes the scale of the predicted probabilities, so check calibration
        when you use it.

        Args:
            pos_weight: A number (binary), one number per label (multilabel),
                ``"balanced"`` for negatives / positives in ``dataset``, or
                None to remove the weight.
            dataset: The training samples, required for ``"balanced"``. Pass
                the training split, not the full dataset, so validation and
                test labels do not set the weight.

        Raises:
            ValueError: If the label is not binary or multilabel, or
                ``"balanced"`` is given without a dataset.

        Examples:
            >>> model.set_pos_weight(4.0)  # doctest: +SKIP
            >>> model.set_pos_weight("balanced", train_dataset)  # doctest: +SKIP
        """
        if pos_weight is None:
            self._pos_weight = None
            return
        if self.mode is None or not self._label_processor().supports_pos_weight:
            raise ValueError(
                "pos_weight applies to binary or multilabel labels; this "
                f"model's mode is {self.mode!r}."
            )
        if isinstance(pos_weight, str):
            if pos_weight != "balanced":
                raise ValueError(
                    "pos_weight must be a number, a sequence, a tensor, "
                    f"'balanced' or None, not {pos_weight!r}."
                )
            if dataset is None:
                raise ValueError(
                    "pos_weight='balanced' needs the training dataset: "
                    "set_pos_weight('balanced', train_dataset)."
                )
            label_key = self.label_keys[0]
            positives, count = None, 0
            for sample in dataset:
                y = torch.as_tensor(sample[label_key], dtype=torch.float32).reshape(-1)
                positives = y.clone() if positives is None else positives + y
                count += 1
            if count == 0:
                raise ValueError("pos_weight='balanced' got an empty dataset.")
            if bool((positives == 0).any()):
                logger.warning(
                    "pos_weight='balanced': some labels have no positive "
                    "examples in the dataset; their weight is set to 1."
                )
            weight = torch.where(
                positives > 0, (count - positives) / positives.clamp(min=1), 1.0
            )
        else:
            weight = torch.as_tensor(pos_weight, dtype=torch.float32).reshape(-1)
        self._pos_weight = weight

    def get_loss_function(self) -> Callable:
        """
        Gets the default loss function: the label processor's
        :meth:`~pyhealth.processors.LabelProcessor.loss`.

        The default loss functions are:
            - binary: `F.binary_cross_entropy_with_logits`
            - multiclass: `F.cross_entropy`
            - multilabel: `F.binary_cross_entropy_with_logits`
            - regression: `F.mse_loss`
            - timeseries_binary: `F.binary_cross_entropy_with_logits` over
              the label records
            - timeseries_regression: `F.mse_loss` over the label records

        The timeseries losses take one logit per label record, ``[ΣN, 1]``,
        and the packed batch of a
        :class:`~pyhealth.processors.TimeseriesLabelProcessor`, ``(values
        [ΣN, 1], time [ΣN], batch offsets [B + 1])``.

        For binary and multilabel labels, a weight set with
        :meth:`set_pos_weight` is passed as ``pos_weight``.

        Returns:
            Callable: The default loss function.
        """
        return functools.partial(
            self._label_processor().loss, pos_weight=self.pos_weight
        )

    def prepare_y_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Prepares the predicted probabilities for model evaluation.

        This function converts the predicted logits to predicted probabilities
        with the label processor's
        :meth:`~pyhealth.processors.LabelProcessor.activation`. The default
        formats are:
            - binary: a tensor of shape (batch_size, 1) with values in [0, 1],
                which is obtained with `torch.sigmoid()`
            - multiclass: a tensor of shape (batch_size, num_classes) with
                values in [0, 1] and sum to 1, which is obtained with
                `torch.softmax()`
            - multilabel: a tensor of shape (batch_size, num_labels) with values
                in [0, 1], which is obtained with `torch.sigmoid()`
            - regression: a tensor of shape (batch_size, 1) with raw logits
            - timeseries_binary: a tensor of shape (label records, 1) with
                values in [0, 1], from `torch.sigmoid()`
            - timeseries_regression: a tensor of shape (label records, 1)
                with raw logits

        Args:
            logits (torch.Tensor): The predicted logit tensor.

        Returns:
            torch.Tensor: The predicted probability tensor.
        """
        return self._label_processor().activation(logits)
