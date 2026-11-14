"""Agent, StageNet and Transformer on 2-D and deeply nested embeddings.

``tensor`` and ``multi_hot`` fields embed to one vector per sample
(``[batch, dim]``), and ``deep_nested_sequence`` fields embed to
``[batch, visits, groups, codes, dim]``. These models used to accept only
3-D or 4-D embeddings.
"""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import Agent, StageNet, Transformer

MODELS = {
    "Agent": lambda ds: Agent(dataset=ds, embedding_dim=8, hidden_dim=8),
    "StageNet": lambda ds: StageNet(
        dataset=ds, embedding_dim=8, chunk_size=4, levels=2
    ),
    "Transformer": lambda ds: Transformer(
        dataset=ds, embedding_dim=8, heads=2, num_layers=1
    ),
}


def _dataset(samples, schema):
    return create_sample_dataset(
        samples=samples,
        input_schema=schema,
        output_schema={"label": "binary"},
        dataset_name="test_input_shapes",
    )


class TestModelInputShapes(unittest.TestCase):
    def _train_step(self, name, samples, schema):
        dataset = _dataset(samples, schema)
        torch.manual_seed(0)
        model = MODELS[name](dataset)
        batch = next(iter(get_dataloader(dataset, batch_size=len(samples))))
        ret = model(**batch)
        ret["loss"].backward()
        self.assertEqual(tuple(ret["y_prob"].shape), (len(samples), 1))
        self.assertTrue(torch.isfinite(ret["loss"]))

    def test_one_vector_per_sample_inputs(self):
        """tensor and multi_hot fields give 2-D embeddings (one step)."""
        samples = [
            {
                "patient_id": f"p{i}",
                "vitals": [float(i), 1.0, 2.0],
                "flags": ["a", "b"][: 1 + i % 2],
                "label": i % 2,
            }
            for i in range(4)
        ]
        for name in MODELS:
            for schema in ({"vitals": "tensor"}, {"flags": "multi_hot"}):
                with self.subTest(model=name, schema=schema):
                    self._train_step(name, samples, schema)

    def test_deep_nested_codes(self):
        """deep_nested_sequence fields give 5-D embeddings."""
        samples = [
            {
                "patient_id": f"p{i}",
                "label": i % 2,
                "visits": [[["a", "b"], ["c"]], [["d"]], [["a"], ["e", "b"]]][
                    : 1 + i % 3
                ],
            }
            for i in range(4)
        ]
        for name in MODELS:
            with self.subTest(model=name):
                self._train_step(
                    name, samples, {"visits": "deep_nested_sequence"}
                )

    def test_deep_nested_last_visit_is_used(self):
        """Patients that differ only in their last visit get different outputs."""
        samples = [
            {
                "patient_id": "p0",
                "label": 0,
                "visits": [[["a"], ["b"]], [["c"]]],
            },
            {
                "patient_id": "p1",
                "label": 1,
                "visits": [[["a"], ["b"]], [["d"]]],
            },
        ]
        dataset = _dataset(samples, {"visits": "deep_nested_sequence"})
        for name, build in MODELS.items():
            with self.subTest(model=name):
                torch.manual_seed(0)
                model = build(dataset).eval()
                batch = next(iter(get_dataloader(dataset, batch_size=2)))
                with torch.no_grad():
                    prob = model(**batch)["y_prob"]
                self.assertFalse(torch.allclose(prob[0], prob[1]))


if __name__ == "__main__":
    unittest.main()
