"""Models predict without a label in the batch.

At inference time there is often no label. ``forward`` should still return
``logit`` and ``y_prob``, and add ``loss`` and ``y_true`` only when the label
is present.
"""

import random
import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import (
    CNN,
    GRASP,
    RETAIN,
    RNN,
    TCN,
    AdaCare,
    ConCare,
    EHRMamba,
    LogisticRegression,
    MultimodalAdaCare,
    MultimodalRETAIN,
    MultimodalRNN,
)

SEQUENCE_MODELS = {
    "RNN": lambda ds: RNN(dataset=ds, embedding_dim=8, hidden_dim=8),
    "RETAIN": lambda ds: RETAIN(dataset=ds, embedding_dim=8),
    "TCN": lambda ds: TCN(dataset=ds, embedding_dim=8, num_channels=8),
    "CNN": lambda ds: CNN(dataset=ds, embedding_dim=8, hidden_dim=8),
    "LogisticRegression": lambda ds: LogisticRegression(
        dataset=ds, embedding_dim=8
    ),
    "GRASP": lambda ds: GRASP(
        dataset=ds, embedding_dim=8, hidden_dim=8, cluster_num=2
    ),
    "AdaCare": lambda ds: AdaCare(dataset=ds, embedding_dim=8, hidden_dim=8),
    "ConCare": lambda ds: ConCare(dataset=ds, embedding_dim=8, hidden_dim=8),
    "EHRMamba": lambda ds: EHRMamba(dataset=ds, embedding_dim=8, num_layers=1),
}

MULTIMODAL_MODELS = {
    "MultimodalRNN": lambda ds: MultimodalRNN(
        dataset=ds, embedding_dim=8, hidden_dim=8
    ),
    "MultimodalRETAIN": lambda ds: MultimodalRETAIN(dataset=ds, embedding_dim=8),
    "MultimodalAdaCare": lambda ds: MultimodalAdaCare(
        dataset=ds, embedding_dim=8, hidden_dim=8
    ),
}


def _samples(with_tensor):
    samples = []
    for i in range(4):
        sample = {
            "patient_id": f"p{i}",
            "codes": ["a", "b", "c", "d"][: 2 + i % 3],
            "label": i % 2,
        }
        if with_tensor:
            sample["vitals"] = [float(i), 1.0]
        samples.append(sample)
    return samples


class TestForwardWithoutLabels(unittest.TestCase):
    def _check(self, models, samples, schema):
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=schema,
            output_schema={"label": "binary"},
            dataset_name="test_no_labels",
        )
        batch = next(iter(get_dataloader(dataset, batch_size=len(samples))))
        unlabeled = {k: v for k, v in batch.items() if k != "label"}
        for name, build in models.items():
            with self.subTest(model=name):
                torch.manual_seed(0)
                model = build(dataset).eval()
                with torch.no_grad():
                    # GRASP draws cluster centers at random in forward.
                    random.seed(0)
                    torch.manual_seed(0)
                    labeled = model(**batch)
                    random.seed(0)
                    torch.manual_seed(0)
                    out = model(**unlabeled)
                self.assertNotIn("loss", out)
                self.assertNotIn("y_true", out)
                self.assertIn("loss", labeled)
                self.assertIn("y_true", labeled)
                torch.testing.assert_close(out["y_prob"], labeled["y_prob"])
                torch.testing.assert_close(out["logit"], labeled["logit"])

    def test_sequence_models(self):
        self._check(SEQUENCE_MODELS, _samples(False), {"codes": "sequence"})

    def test_multimodal_models(self):
        self._check(
            MULTIMODAL_MODELS,
            _samples(True),
            {"codes": "sequence", "vitals": "tensor"},
        )


if __name__ == "__main__":
    unittest.main()
