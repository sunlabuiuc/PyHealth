"""Feature-level attention LSTM model for PyHealth time-series tasks."""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from pyhealth.models import BaseModel
from pyhealth.datasets import SampleDataset

__all__ = ["FeatureAttentionLSTM"]


class FeatureAttentionLSTM(BaseModel):
    """Feature-level Attention LSTM for clinical time-series prediction.

    This model implements a dense attention layer across the feature dimension
    before passing the weighted representation into an LSTM network, as
    detailed in Yadav & Subbian (2025).

    Attributes:
        dataset: The PyHealth dataset object tracker.
        feature_keys: List of string keys representing input features.
        label_key: String key representing the target label.
        mode: Type of task, 'binary', 'multiclass', or 'multilabel'.

    Args:
        dataset: The PyHealth dataset used to determine model dimensions.
        feature_keys: Keys for the input time-series features.
        label_key: Key for the target label.
        mode: Task mode, such as "binary", "multiclass", or "multilabel".
        embedding_dim: Reserved embedding dimension. Default is 128.
        hidden_dim: Hidden size of the LSTM. Default is 128.
        num_layers: Number of LSTM layers. Default is 1.
        dropout: Dropout between LSTM layers. Default is 0.2.
        **kwargs: Additional model keyword arguments.

    Examples:
        >>> import torch
        >>> from pyhealth.models import FeatureAttentionLSTM
        >>> class MockDataset:
        ...     input_info = {"features": {"input_dim": 50}}
        ...     output_info = {"label": {"output_dim": 2}}
        >>> dataset = MockDataset()
        >>> model = FeatureAttentionLSTM(
        ...     dataset=dataset,
        ...     feature_keys=["features"],
        ...     label_key="label",
        ...     mode="binary",
        ... )
        >>> batch = {
        ...     "features": torch.randn(4, 24, 50),
        ...     "label": torch.tensor([0, 1, 0, 1]),
        ... }
        >>> output = model(**batch)
        >>> output["attention_weights"].shape
        torch.Size([4, 24, 50])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super(FeatureAttentionLSTM, self).__init__(dataset)

        if not feature_keys:
            raise ValueError("feature_keys must contain at least one feature key")
        if len(feature_keys) != 1:
            raise ValueError("FeatureAttentionLSTM supports exactly one feature key")
        if not label_key:
            raise ValueError("label_key must be a non-empty string")

        self.feature_keys = list(feature_keys)
        self.label_keys = [label_key]
        self.label_key = label_key
        self.task_mode = mode

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = self._resolve_feature_dim(dataset, feature_keys[0])
        self.output_dim = self._resolve_output_dim(dataset, label_key, mode)

        self.attention_matrix = nn.Linear(
            self.feature_dim, self.feature_dim, bias=False
        )
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, self.output_dim)

    @staticmethod
    def _resolve_feature_dim(dataset: Any, feature_key: str) -> int:
        """Resolve the input feature dimension from a PyHealth dataset."""
        input_info = getattr(dataset, "input_info", None)
        if isinstance(input_info, dict) and feature_key in input_info:
            feature_info = input_info[feature_key]
            if isinstance(feature_info, dict) and "input_dim" in feature_info:
                return int(feature_info["input_dim"])

        input_processor = getattr(dataset, "input_processors", {}).get(feature_key)
        if input_processor is not None:
            size = getattr(input_processor, "size", None)
            if callable(size):
                return int(size())
            n_features = getattr(input_processor, "n_features", None)
            if n_features is not None:
                return int(n_features)

        raise ValueError(f"Could not determine feature dimension for '{feature_key}'")

    @staticmethod
    def _resolve_output_dim(dataset: Any, label_key: str, mode: str) -> int:
        """Resolve the label output dimension from a PyHealth dataset."""
        output_info = getattr(dataset, "output_info", None)
        if isinstance(output_info, dict) and label_key in output_info:
            label_info = output_info[label_key]
            if isinstance(label_info, dict) and "output_dim" in label_info:
                return int(label_info["output_dim"])

        label_processor = getattr(dataset, "output_processors", {}).get(label_key)
        if label_processor is not None:
            return int(label_processor.output_size())
        if mode == "binary":
            return 1
        if mode == "multilabel":
            return 1
        if mode == "multiclass":
            return 2
        if mode == "regression":
            return 1
        raise ValueError(f"Unsupported task mode: {mode}")

    def get_loss_function(self) -> Any:
        """Return a PyHealth-compatible loss function for the configured label."""
        dataset_loss = getattr(
            getattr(self, "dataset", None),
            "get_loss_function",
            None,
        )
        if callable(dataset_loss):
            return dataset_loss

        label_processor = getattr(
            getattr(self, "dataset", None),
            "output_processors",
            {},
        ).get(self.label_key)
        if label_processor is not None:
            return label_processor.loss

        if self.task_mode == "binary":
            if self.output_dim > 1:
                return nn.CrossEntropyLoss()
            return nn.BCEWithLogitsLoss()
        if self.task_mode == "multilabel":
            return nn.BCEWithLogitsLoss()
        if self.task_mode == "multiclass":
            return nn.CrossEntropyLoss()
        if self.task_mode == "regression":
            return nn.MSELoss()
        raise ValueError(f"Unsupported task mode: {self.task_mode}")

    def forward(
        self,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            **kwargs: Batch values keyed by feature and label names.

        Returns:
            Dict[str, torch.Tensor]: The loss, logits, labels, and attention weights.
        """
        x = kwargs[self.feature_keys[0]]
        if not isinstance(x, torch.Tensor):
            raise TypeError("The time-series feature must be a torch.Tensor")
        if x.dim() != 3:
            raise ValueError(
                "The time-series tensor must have shape "
                "[batch_size, time_steps, feature_dim]"
            )
        y_true = kwargs[self.label_key].to(self.device)
        x = x.to(self.device)

        attention_scores = self.attention_matrix(x)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        x_new = x * attention_weights
        lstm_out, _ = self.lstm(x_new)
        last_hidden = lstm_out[:, -1, :]
        y_hat = self.fc(last_hidden)
        loss = self.get_loss_function()(y_hat, y_true)

        return {
            "loss": loss,
            "y_hat": y_hat,
            "y_true": y_true,
            "attention_weights": attention_weights,
        }
