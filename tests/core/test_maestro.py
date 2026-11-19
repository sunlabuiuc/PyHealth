import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MAESTRO
from pyhealth.models.maestro import (
    InformerEncoderLayer,
    ProbSparseAttention,
    SparseMoEFeedForward,
)


def _make_samples(labels, seed=0, time=8, ecg_dim=3, eda_dim=2):
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for index, label in enumerate(labels):
        samples.append(
            {
                "patient_id": f"patient-{index}",
                "visit_id": f"visit-{index}",
                "ecg": torch.randn(time, ecg_dim, generator=generator).tolist(),
                "eda": torch.randn(time, eda_dim, generator=generator).tolist(),
                "label": label,
            }
        )
    return samples


class TestMAESTRO(unittest.TestCase):
    """Test cases for the MAESTRO model."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.binary_dataset = create_sample_dataset(
            samples=_make_samples([0, 1, 1, 0]),
            input_schema={"ecg": "tensor", "eda": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="maestro_binary_test",
        )
        cls.multiclass_dataset = create_sample_dataset(
            samples=_make_samples([0, 1, 2, 1]),
            input_schema={"ecg": "tensor", "eda": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="maestro_multiclass_test",
        )

    @staticmethod
    def _build(dataset, **overrides):
        kwargs = {
            "d_model": 16,
            "n_heads": 2,
            "num_layers_per_modal": 1,
            "num_layers": 1,
            "num_experts": 2,
            "dropout": 0.0,
            "modality_dropout": 0.0,
            "input_length": 16,
        }
        kwargs.update(overrides)
        return MAESTRO(dataset=dataset, **kwargs)

    def _batch(self, dataset, batch_size=2):
        return next(iter(get_dataloader(dataset, batch_size=batch_size)))

    def test_binary_forward_shapes_and_loss(self):
        model = self._build(self.binary_dataset)
        output = model(**self._batch(self.binary_dataset))
        self.assertEqual(output["logit"].shape, (2, 1))
        self.assertEqual(output["y_prob"].shape, (2, 1))
        self.assertTrue(torch.all((output["y_prob"] >= 0) & (output["y_prob"] <= 1)))
        self.assertIn("loss", output)
        self.assertIn("y_true", output)
        self.assertEqual(output["loss"].dim(), 0)

    def test_multiclass_forward_shapes(self):
        model = self._build(self.multiclass_dataset)
        output = model(**self._batch(self.multiclass_dataset))
        self.assertEqual(output["logit"].shape, (2, 3))
        self.assertTrue(torch.allclose(output["y_prob"].sum(dim=-1), torch.ones(2)))

    def test_loss_backward_updates_parameters(self):
        model = self._build(self.binary_dataset)
        output = model(**self._batch(self.binary_dataset))
        output["loss"].backward()
        grads = [
            p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        self.assertTrue(len(grads) > 0)
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in grads))

    def test_modality_inputs_are_recorded_from_dataset(self):
        model = self._build(self.binary_dataset)
        self.assertEqual(model.modalities, ["ecg", "eda"])
        self.assertEqual(model.variates, {"ecg": 3, "eda": 2})

    def test_missing_modality_raises(self):
        model = self._build(self.binary_dataset)
        batch = self._batch(self.binary_dataset)
        batch.pop("eda")
        with self.assertRaises(ValueError):
            model(**batch)

    def test_mismatched_time_length_raises(self):
        model = self._build(self.binary_dataset)
        batch = self._batch(self.binary_dataset)
        batch["eda"] = batch["eda"][:, :4, :]
        with self.assertRaises(ValueError):
            model(**batch)

    def test_too_long_input_raises(self):
        model = self._build(self.binary_dataset, input_length=4)
        with self.assertRaises(ValueError):
            model(**self._batch(self.binary_dataset))

    def test_wrong_variate_count_raises(self):
        model = self._build(self.binary_dataset)
        batch = self._batch(self.binary_dataset)
        batch["ecg"] = batch["ecg"][:, :, :2]
        with self.assertRaises(ValueError):
            model(**batch)

    def test_invalid_hyperparameters_raise(self):
        with self.assertRaises(ValueError):
            self._build(self.binary_dataset, d_model=15, n_heads=2)
        with self.assertRaises(ValueError):
            self._build(self.binary_dataset, modality_dropout=1.0)
        with self.assertRaises(ValueError):
            self._build(self.binary_dataset, input_length=0)

    def test_modality_dropout_only_in_training(self):
        torch.manual_seed(0)
        model = self._build(self.binary_dataset, modality_dropout=0.99)
        batch = self._batch(self.binary_dataset)
        model.eval()
        torch.manual_seed(1)
        first = model(**batch)["logit"]
        torch.manual_seed(1)
        second = model(**batch)["logit"]
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_training_mode_runs_with_dropped_modalities(self):
        model = self._build(self.binary_dataset, modality_dropout=0.99)
        model.train()
        output = model(**self._batch(self.binary_dataset))
        self.assertEqual(output["logit"].shape, (2, 1))
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_state_dict_round_trip(self):
        torch.manual_seed(0)
        model = self._build(self.binary_dataset)
        clone = self._build(self.binary_dataset)
        clone.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        clone.eval()
        torch.manual_seed(3)
        a = model(**self._batch(self.binary_dataset))["logit"]
        torch.manual_seed(3)
        b = clone(**self._batch(self.binary_dataset))["logit"]
        self.assertTrue(torch.allclose(a, b, atol=1e-6))


class TestMAESTROBlocks(unittest.TestCase):
    """Test the building blocks of MAESTRO."""

    def test_probsparse_attention_shape(self):
        attn = ProbSparseAttention(d_model=8, n_heads=2)
        out = attn(torch.randn(3, 6, 8), factor=2)
        self.assertEqual(out.shape, (3, 6, 8))

    def test_probsparse_attention_handles_length_one(self):
        attn = ProbSparseAttention(d_model=8, n_heads=2)
        out = attn(torch.randn(2, 1, 8), factor=5)
        self.assertEqual(out.shape, (2, 1, 8))

    def test_informer_layer_halves_length(self):
        layer = InformerEncoderLayer(d_model=8, n_heads=2, d_ff=16, dropout=0.0)
        out = layer(torch.randn(2, 9, 8), factor=2)
        self.assertEqual(out.shape, (2, 5, 8))

    def test_moe_layer_shape_and_routing(self):
        moe = SparseMoEFeedForward(d_model=8, expert_dim=16, num_experts=3, k=1)
        out = moe(torch.randn(2, 5, 8))
        self.assertEqual(out.shape, (2, 5, 8))

    def test_factor_one_still_attends(self):
        # Factors are integers clamped to at least 1, so attention always runs.
        attn = ProbSparseAttention(d_model=4, n_heads=1)
        out = attn(torch.randn(1, 2, 4), factor=1)
        self.assertEqual(out.shape, (1, 2, 4))


if __name__ == "__main__":
    unittest.main()
