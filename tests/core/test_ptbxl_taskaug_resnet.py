import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import PTBXLResNet, TaskAugResNet


def _make_dataset(length=5000, n=4):
    rng = np.random.RandomState(0)
    samples = [
        {
            "patient_id": f"p{i}",
            "signal": rng.randn(12, length).astype(np.float32),
            "label": i % 2,
        }
        for i in range(n)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": "binary"},
        dataset_name="ptbxl_test",
    )


class TestPTBXLModels(unittest.TestCase):
    """Tests for PTBXLResNet and TaskAugResNet."""

    def setUp(self):
        self.dataset = _make_dataset()
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        self.batch = next(iter(loader))

    def _models(self):
        return (
            PTBXLResNet(dataset=self.dataset),
            TaskAugResNet(dataset=self.dataset, k=2),
        )

    def test_initialization(self):
        for model in self._models():
            self.assertEqual(model.feature_keys, ["signal"])
            self.assertEqual(model.label_keys, ["label"])

    def test_forward_shapes(self):
        for model in self._models():
            model.train()
            ret = model(**self.batch)
            self.assertEqual(tuple(ret["y_prob"].shape), (4, 1))
            self.assertEqual(tuple(ret["logit"].shape), (4, 1))
            self.assertEqual(ret["loss"].dim(), 0)

    def test_backward(self):
        for model in self._models():
            model.train()
            model(**self.batch)["loss"].backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_k_sets_number_of_stages(self):
        for k in (1, 2, 3):
            model = TaskAugResNet(dataset=self.dataset, k=k)
            self.assertEqual(model.k, k)
            self.assertEqual(len(model.taskaug.sub_policies[0].stages), k)

    def test_eval_mode_is_not_augmented(self):
        model = TaskAugResNet(dataset=self.dataset, k=2).eval()
        with torch.no_grad():
            first = model(**self.batch)["logit"]
            second = model(**self.batch)["logit"]
        self.assertTrue(torch.allclose(first, second))

    def test_already_trimmed_input(self):
        dataset = _make_dataset(length=2496)
        batch = next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))
        for model in (PTBXLResNet(dataset=dataset), TaskAugResNet(dataset=dataset)):
            model.train()
            self.assertEqual(tuple(model(**batch)["y_prob"].shape), (4, 1))
    
    def test_training_rejects_unsupported_length(self):
        dataset = _make_dataset(length=1000)
        batch = next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))
        model = TaskAugResNet(dataset=dataset)
        model.train()
        with self.assertRaises(ValueError):
            model(**batch)
        model.eval()
        self.assertEqual(tuple(model(**batch)["y_prob"].shape), (4, 1))

    def test_training_mode_augments_input(self):
        model = TaskAugResNet(dataset=self.dataset, k=2)
        seen = []
        model.encoder.register_forward_hook(
            lambda module, args, out: seen.append(args[0].detach().clone())
        )
        torch.manual_seed(0)
        model.train()
        model(**self.batch)
        model.eval()
        model(**self.batch)
        self.assertFalse(torch.allclose(seen[0], seen[1]))
    
    def test_state_dict_round_trip(self):
        for k in (1, 2):
            source = TaskAugResNet(dataset=self.dataset, k=k).eval()
            target = TaskAugResNet(dataset=self.dataset, k=k).eval()
            target.load_state_dict(source.state_dict())
            with torch.no_grad():
                expected = source(**self.batch)["logit"]
                actual = target(**self.batch)["logit"]
            self.assertTrue(torch.allclose(expected, actual))


if __name__ == "__main__":
    unittest.main()
