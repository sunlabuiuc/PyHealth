from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from pyhealth.models import BaseModel


class WearableMLP(BaseModel):
    """MLP model for short-window wearable classification.

WearableMLP is a lightweight PyHealth model for binary or multiclass
prediction from dense wearable features. It is suitable for short-window
monitoring tasks where several days of wearable measurements have already
been converted into a fixed-length numeric vector.

Typical input features may include:
    - resting heart rate summaries across multiple days
    - sleep duration summaries across multiple days
    - calendar features such as day-of-week one-hot encodings

The model expects one dense tensor feature in ``kwargs`` with shape:
    - ``[batch_size, input_dim]``, or
    - ``[batch_size, seq_len, feature_dim]``

If the input has more than 2 dimensions, it is flattened from dimension 1
before being passed through the MLP backbone.

Args:
    dataset: A PyHealth dataset object containing ``input_schema`` and
        ``output_schema``.
    feature_key: Name of the dense feature field. If not provided, the model
        requires exactly one feature key in ``dataset.input_schema``.
    hidden_dim: Hidden layer size of the MLP.
    dropout: Dropout probability applied after each hidden layer.
    activation: Activation function name. Supported values are ``"relu"``,
        ``"gelu"``, and ``"tanh"``.

Returns:
    A dictionary containing:
        - ``logit``: raw model outputs
        - ``y_prob``: predicted probabilities
        - ``loss``: training loss when labels are provided
        - ``y_true``: ground-truth labels when labels are provided

Examples:
    >>> model = WearableMLP(dataset=sample_dataset, feature_key="wearable")
    >>> batch = {"wearable": torch.randn(4, 13)}
    >>> output = model(**batch)
    >>> output["logit"].shape
    torch.Size([4, 1])
"""

    def __init__(
        self,
        dataset: Any,
        feature_key: Optional[str] = None,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        activation: str = "relu",
    ) -> None:
        super().__init__(dataset=dataset)

        available_feature_keys = list(getattr(dataset, "input_schema", {}).keys())
        if len(available_feature_keys) == 0:
            raise ValueError("WearableMLP requires dataset.input_schema to be set.")

        if feature_key is None:
            if len(available_feature_keys) != 1:
                raise ValueError(
                    "WearableMLP requires exactly one feature key when "
                    "`feature_key` is not provided."
                )
            feature_key = available_feature_keys[0]

        if feature_key not in available_feature_keys:
            raise ValueError(
                f"feature_key={feature_key!r} not found in dataset input schema: "
                f"{available_feature_keys}"
            )

        self.feature_key = feature_key
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.activation_name = activation

        input_dim = self._infer_input_dim(dataset, feature_key)

        self.loss_fn = self.get_loss_function()

        if getattr(self, "mode", None) == "binary":
            output_size = 1
        else:
            output_size = self.get_output_size()

        act_layer = self._get_activation(activation)

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, output_size)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Returns the activation module."""
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        raise ValueError(
            f"Unsupported activation {name!r}. Choose from: relu, gelu, tanh."
        )

    @staticmethod
    def _infer_input_dim(dataset: Any, feature_key: str) -> int:
        """Infers the dense input dimension from dataset schema."""
        schema = getattr(dataset, "input_schema", None)
        if schema is None:
            raise ValueError(
                "dataset.input_schema is required to infer input_dim for "
                "WearableMLP."
            )

        spec = schema.get(feature_key)
        if spec is None:
            raise ValueError(
                f"Feature key {feature_key!r} not found in dataset.input_schema."
            )

        if isinstance(spec, dict):
            for key in ("dim", "input_dim", "feature_dim"):
                if key in spec:
                    return int(spec[key])

            shape = spec.get("shape")
            if shape is not None:
                if isinstance(shape, int):
                    return int(shape)
                if len(shape) == 0:
                    raise ValueError("Feature shape cannot be empty.")
                prod = 1
                for s in shape:
                    prod *= int(s)
                return prod

        if isinstance(spec, int):
            return int(spec)

        if isinstance(spec, (tuple, list)):
            prod = 1
            for s in spec:
                prod *= int(s)
            return prod

        raise ValueError(
            "Could not infer input_dim from dataset.input_schema. "
            "Please encode the feature spec using one of: "
            "{'dim': ...}, {'input_dim': ...}, {'shape': (...)}, int, or tuple."
        )

    @staticmethod
    def _prepare_x(x: torch.Tensor) -> torch.Tensor:
        """Flattens input to [batch_size, input_dim]."""
        if not torch.is_tensor(x):
            raise TypeError(
                "WearableMLP expects a dense torch.Tensor input for the selected "
                "feature key."
            )
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        return x.float()

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        x = kwargs[self.feature_key]
        if isinstance(x, (tuple, list)):
            if len(x) != 1:
                raise ValueError(
                    "WearableMLP expects a single tensor for the feature input."
                )
            x = x[0]

        x = self._prepare_x(x)
        hidden = self.backbone(x)
        logit = self.classifier(hidden)

        if logit.shape[-1] == 1:
            y_prob = torch.sigmoid(logit)
        else:
            y_prob = torch.softmax(logit, dim=-1)

        results: Dict[str, torch.Tensor] = {
            "logit": logit,
            "y_prob": y_prob,
        }

        label_key = self.label_keys[0]
        if label_key in kwargs:
            y_true = kwargs[label_key]

            if logit.shape[-1] == 1:
                y_true = y_true.float()
            else:
                y_true = y_true.long()

            results["y_true"] = y_true
            results["loss"] = self.loss_fn(logit, y_true)

        return results

    def forward_from_embedding(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Compatibility hook for interpretability methods."""
        return self.forward(**kwargs)