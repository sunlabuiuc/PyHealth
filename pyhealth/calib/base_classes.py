from abc import ABC
from typing import Dict

import torch


class PostHocCalibrator(ABC, torch.nn.Module):
    def __init__(self, model, **kwargs) -> None:
        super().__init__()
        self.model = model

    def calibrate(self, cal_dataset):
        ...

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        ...


    def to(self, device):
        super().to(device)
        self.device = device
        return self


class _BinaryClassView(torch.nn.Module):
    """Present a binary model as two classes to prediction-set algorithms."""

    mode = "multiclass"

    def __init__(self, model):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("model"), name)

    def forward(self, **kwargs):
        from pyhealth.calib.utils import binary_to_two_columns

        output = dict(self.model(**kwargs))
        native = {"y_prob": output["y_prob"]}
        output["y_prob"] = binary_to_two_columns(output["y_prob"])
        if "y_true" in output:
            native["y_true"] = output["y_true"]
            output["y_true"] = output["y_true"].reshape(-1).long()

        output["_binary_outputs"] = native

        return output


class SetPredictor(ABC, torch.nn.Module):
    """Base for methods implementing calibrate() and _predict().

    Binary models are exposed internally as two-class models. Public forward()
    restores their original probabilities and labels after set construction.
    The predictor's mode remains the original model's mode.

    Examples:
        >>> model = torch.nn.Linear(2, 1)
        >>> model.mode = "binary"
        >>> predictor = SetPredictor(model)
        >>> predictor.mode, predictor.model.mode
        ('binary', 'multiclass')
    """

    def __init__(self, model, **kwargs) -> None:
        super().__init__()
        self.mode = model.mode
        self.model = _BinaryClassView(model) if self.mode == "binary" else model

    def calibrate(self, cal_dataset):
        ...

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Construct prediction sets and return the model's native outputs."""
        output = self._predict(**kwargs)
        native = output.pop("_binary_outputs", None)
        if native is not None:
            output.update(native)

        return output

    def _predict(self, **kwargs) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def to(self, device):
        super().to(device)
        self.device = device
        return self
