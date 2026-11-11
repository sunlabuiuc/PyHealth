"""Tests for the IPD-weighted transfer trainer (tiny synthetic data)."""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.sensor_lstm import SensorLSTM
from pyhealth.models.sensor_transfer import (
    compute_source_distances,
    evaluate_rcc,
    finetune,
    pretrain_on_sources,
    train_epochs,
    transfer_learn,
)

STEPS, CHANNELS = 8, 9
LABELS = [0, 1, 2, 0, 1, 2]


def _make_loader():
    rng = np.random.default_rng(0)
    samples = []
    for i, label in enumerate(LABELS):
        signal = rng.normal(size=(STEPS, CHANNELS))
        samples.append(
            {
                "patient_id": f"p{i}",
                "signal": signal,
                "source_near": signal + 0.01 * rng.normal(size=signal.shape),
                "source_far": rng.normal(size=(STEPS, CHANNELS)),
                "label": label,
            }
        )
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={
            "signal": "tensor",
            "source_near": "tensor",
            "source_far": "tensor",
        },
        output_schema={"label": "multiclass"},
        dataset_name="transfer-test",
    )
    return dataset, get_dataloader(dataset, batch_size=len(LABELS))


class TestSensorTransfer(unittest.TestCase):
    """Tests distances, pretraining, fine-tuning and evaluation."""

    def setUp(self):
        torch.manual_seed(0)
        self.dataset, self.loader = _make_loader()
        self.sources = ["source_near", "source_far"]

    def _model(self):
        return SensorLSTM(self.dataset, hidden_dim=4)

    def test_distances_rank_the_near_source_closer(self):
        d = compute_source_distances(
            self.loader, "signal", self.sources, max_pairs=4, window=3
        )
        self.assertEqual(set(d), set(self.sources))
        self.assertLess(d["source_near"], d["source_far"])

    def test_train_epochs_returns_one_loss_per_epoch(self):
        losses = train_epochs(self._model(), self.loader, lr=1e-2, epochs=2)
        self.assertEqual(len(losses), 2)
        self.assertTrue(all(np.isfinite(losses)))

    def test_pretraining_updates_weights_and_restores_input_key(self):
        model = self._model()
        before = [p.detach().clone() for p in model.parameters()]
        history = pretrain_on_sources(
            model, self.loader, {"source_near": 0.8, "source_far": 0.2}
        )
        self.assertEqual(set(history), set(self.sources))
        self.assertEqual(model.input_key, "signal")
        changed = any(
            not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters())
        )
        self.assertTrue(changed)

    def test_source_order(self):
        weights = {"source_near": 0.8, "source_far": 0.2}
        ascending = pretrain_on_sources(
            self._model(), self.loader, weights, order="ascending_weight"
        )
        self.assertEqual(list(ascending), ["source_far", "source_near"])
        descending = pretrain_on_sources(
            self._model(), self.loader, weights, order="descending_weight"
        )
        self.assertEqual(list(descending), ["source_near", "source_far"])
        with self.assertRaises(ValueError):
            pretrain_on_sources(self._model(), self.loader, weights, order="random")

    def test_finetune_switches_to_the_target(self):
        model = self._model()
        model.input_key = "source_near"
        finetune(model, self.loader, "signal", epochs=1)
        self.assertEqual(model.input_key, "signal")

    def test_rcc_is_a_fraction(self):
        rcc = evaluate_rcc(self._model(), self.loader)
        self.assertGreaterEqual(rcc, 0.0)
        self.assertLessEqual(rcc, 1.0)

    def test_transfer_learn_runs_for_every_weighting(self):
        for weighting in ("ipd", "equal", "inverse"):
            with self.subTest(weighting=weighting):
                result = transfer_learn(
                    self._model(),
                    self.loader,
                    self.loader,
                    "signal",
                    self.sources,
                    weighting=weighting,
                    max_pairs=4,
                )
                self.assertAlmostEqual(sum(result["weights"].values()), 1.0)
                self.assertEqual(set(result["pretrain_losses"]), set(self.sources))
                self.assertGreaterEqual(result["rcc"], 0.0)

    def test_weighting_direction_in_results(self):
        ipd = transfer_learn(
            self._model(), self.loader, self.loader, "signal", self.sources,
            weighting="ipd", max_pairs=4,
        )
        inverse = transfer_learn(
            self._model(), self.loader, self.loader, "signal", self.sources,
            weighting="inverse", distances=ipd["distances"],
        )
        self.assertGreater(ipd["weights"]["source_near"], ipd["weights"]["source_far"])
        self.assertLess(
            inverse["weights"]["source_near"], inverse["weights"]["source_far"]
        )

    def test_no_transfer_baseline_skips_pretraining(self):
        result = transfer_learn(
            self._model(), self.loader, self.loader, "signal", self.sources,
            weighting=None,
        )
        self.assertEqual(result["pretrain_losses"], {})
        self.assertEqual(result["weights"], {})
        self.assertEqual(len(result["finetune_losses"]), 1)

    def test_invalid_weighting_raises(self):
        with self.assertRaises(ValueError):
            transfer_learn(
                self._model(), self.loader, self.loader, "signal", self.sources,
                weighting="random", max_pairs=4,
            )


if __name__ == "__main__":
    unittest.main()
