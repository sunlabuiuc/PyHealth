"""LSTM classifier for raw multichannel sensor windows.

Authors: Joshua Gundersen (jgundersen3), Zachary Gundersen (zgundersen3)
Paper: Zhang et al. (2024), "Daily Physical Activity Monitoring: Adaptive
    Learning from Multi-source Motion Sensor Data", CHIL 2024.
    https://proceedings.mlr.press/v248/zhang24a.html
Description: The LSTM classifier used as the base network of the paper's
    transfer framework. It reads one sensor stream of shape
    ``(batch, time, channels)`` and predicts the activity. ``input_key``
    selects which stream, so one set of weights can be trained on source
    sensors first and fine-tuned on the target sensor afterwards.
"""

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class SensorLSTM(BaseModel):
    """LSTM over a single raw sensor stream.

    The last hidden state of the (stacked, optionally bidirectional) LSTM is
    passed through a linear layer to produce one logit per class. The loss
    and the output activation come from the dataset's label processor.

    Args:
        dataset: The sample dataset, used to find the label and output size.
        input_key: Which input feature to read, ``"signal"`` for the target
            sensor or e.g. ``"source_torso"`` for a source sensor.
        input_dim: Number of channels per time step (9 for DSA).
        hidden_dim: LSTM hidden size.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout between LSTM layers (used only if ``num_layers > 1``)
            and before the classifier.
        bidirectional: Whether to use a bidirectional LSTM.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import SensorLSTM
        >>> import numpy as np
        >>> samples = [
        ...     {"patient_id": f"p{i}", "signal": np.zeros((10, 9)), "label": i % 2}
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": "multiclass"},
        ...     dataset_name="demo",
        ... )  # doctest: +SKIP
        >>> model = SensorLSTM(dataset=dataset, hidden_dim=8)  # doctest: +SKIP
    """

    def __init__(
        self,
        dataset: SampleDataset,
        input_key: str = "signal",
        input_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(dataset=dataset)
        if input_key not in self.feature_keys:
            raise ValueError(
                f"input_key '{input_key}' is not an input of the dataset: "
                f"{self.feature_keys}"
            )
        if len(self.label_keys) != 1:
            raise ValueError("SensorLSTM supports exactly one label.")
        self.label_key = self.label_keys[0]
        self.input_key = input_key
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        directions = 2 if bidirectional else 1
        self.fc = nn.Linear(
            hidden_dim * directions, self.label_processor.output_size()
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of streams into one vector per sample."""
        _, (hidden, _) = self.lstm(x)
        if self.bidirectional:
            return torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return hidden[-1]

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Predicts the activity of a batch.

        Args:
            **kwargs: A batch from :func:`~pyhealth.datasets.get_dataloader`.
                Must contain ``input_key``; may contain the label key and
                ``embed=True`` to also return the sample embeddings.

        Returns:
            A dict with ``logit`` and ``y_prob``; ``loss`` and ``y_true`` if
            the label is in the batch; and ``embed`` if requested.

        Examples:
            >>> out = model(**next(iter(loader)))  # doctest: +SKIP
            >>> out["logit"].shape  # doctest: +SKIP
            torch.Size([4, 3])
        """
        feature = kwargs[self.input_key]
        if isinstance(feature, (tuple, list)):
            feature = feature[0]
        x = feature.to(self.device).float()

        embed = self.dropout(self._encode(x))
        logits = self.fc(embed)
        results = {
            "logit": logits,
            "y_prob": self.label_processor.activation(logits),
        }
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            results["loss"] = self.label_processor.loss(logits, y_true)
            results["y_true"] = y_true
        if kwargs.get("embed", False):
            results["embed"] = embed
        return results