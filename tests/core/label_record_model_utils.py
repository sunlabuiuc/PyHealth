"""A minimal per-step model for tests of the timeseries label plumbing."""

from typing import Any

from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class LabelRecordModel(BaseModel):
    """Predicts each label record from its time with one linear layer.

    It ignores the input features, so tests of the label processors, the loss,
    the Trainer and the metrics do not depend on a particular architecture.
    """

    def __init__(self, dataset: SampleDataset):
        super().__init__(dataset)
        self.label_key = self.label_keys[0]
        self.fc = nn.Linear(1, 1)

    def forward(self, **batch: Any) -> dict[str, Any]:
        # Packed label records: (values [ΣN, 1], time [ΣN], offsets [B + 1]).
        label = tuple(t.to(self.device) for t in batch[self.label_key])
        logits = self.fc(label[1].unsqueeze(1))
        return {
            "loss": self.get_loss_function()(logits, label),
            "y_prob": self.prepare_y_prob(logits),
            "y_true": label,
            "logit": logits,
        }
