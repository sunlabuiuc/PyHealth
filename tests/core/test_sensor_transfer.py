"""Tests for the IPD-weighted transfer trainer (tiny synthetic data)."""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.ipd import decay_rates
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
                "source_far": rng.normal(size=(STEPS, CHANNELS)) + 3.0,
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
    return dataset, get_dataloader(dataset, batch_size=len(LABELS), shuffle=False)


class TestSensorTransfer(unittest.TestCase):
    """Tests distances, pretraining, fine-tuning and evaluation."""

    def setUp(self):
        torch.manual_seed(0)
        self.dataset, self.loader = _make_loader()
        self.sources = ["source_near", "source_far"]
        self.distances = {"source_near": 1.0, "source_far": 9.0}
        self.alphas = decay_rates(self.distances, "ipd")

    def _model(self):
        torch.manual_seed(0)
        return SensorLSTM(self.dataset, hidden_dim=4)

    def test_distances_rank_the_near_source_closer(self):
        d = compute_source_distances(
            self.loader, "signal", self.sources, max_pairs=6, window=3
        )
        self.assertEqual(set(d), set(self.sources))
        self.assertLess(d["source_near"], d["source_far"])

    def test_distances_are_reproducible_for_a_seed(self):
        first = compute_source_distances(
            self.loader, "signal", self.sources, max_pairs=4, window=3, seed=5
        )
        again = compute_source_distances(
            self.loader, "signal", self.sources, max_pairs=4, window=3, seed=5
        )
        self.assertEqual(first, again)

    def test_more_pairs_than_samples_uses_all_samples(self):
        d = compute_source_distances(
            self.loader, "signal", self.sources, max_pairs=1000, window=3
        )
        self.assertEqual(set(d), set(self.sources))

    def test_train_epochs_returns_one_loss_per_epoch(self):
        losses = train_epochs(self._model(), self.loader, lr=1e-2, epochs=2)
        self.assertEqual(len(losses), 2)
        self.assertTrue(all(np.isfinite(losses)))

    def test_zero_decay_freezes_the_model_after_the_first_epoch(self):
        once = self._model()
        train_epochs(once, self.loader, lr=1e-2, epochs=1)
        twice = self._model()
        train_epochs(twice, self.loader, lr=1e-2, epochs=2, lr_decay=0.0)
        for a, b in zip(once.parameters(), twice.parameters()):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_pretraining_updates_weights_and_restores_input_key(self):
        model = self._model()
        before = [p.detach().clone() for p in model.parameters()]
        history = pretrain_on_sources(model, self.loader, self.distances, self.alphas)
        self.assertEqual(set(history), set(self.sources))
        self.assertEqual(model.input_key, "signal")
        changed = any(
            not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters())
        )
        self.assertTrue(changed)

    def test_source_order(self):
        descending = pretrain_on_sources(
            self._model(), self.loader, self.distances, self.alphas
        )
        self.assertEqual(list(descending), ["source_far", "source_near"])
        ascending = pretrain_on_sources(
            self._model(),
            self.loader,
            self.distances,
            self.alphas,
            order="ascending_distance",
        )
        self.assertEqual(list(ascending), ["source_near", "source_far"])
        with self.assertRaises(ValueError):
            pretrain_on_sources(
                self._model(), self.loader, self.distances, self.alphas, order="x"
            )

    def test_mismatched_keys_raise(self):
        with self.assertRaises(ValueError):
            pretrain_on_sources(
                self._model(), self.loader, self.distances, {"source_near": 1.0}
            )

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
                    max_pairs=6,
                )
                self.assertAlmostEqual(sum(result["alphas"].values()), 1.0)
                self.assertEqual(set(result["pretrain_losses"]), set(self.sources))
                self.assertGreaterEqual(result["rcc"], 0.0)

    def test_weighting_direction_in_results(self):
        ipd = transfer_learn(
            self._model(), self.loader, self.loader, "signal", self.sources,
            weighting="ipd", max_pairs=6,
        )
        inverse = transfer_learn(
            self._model(), self.loader, self.loader, "signal", self.sources,
            weighting="inverse", distances=ipd["distances"],
        )
        # IPD: the closer source decays more slowly (smaller alpha).
        self.assertLess(ipd["alphas"]["source_near"], ipd["alphas"]["source_far"])
        self.assertGreater(
            inverse["alphas"]["source_near"], inverse["alphas"]["source_far"]
        )

    def test_no_transfer_baseline_skips_pretraining(self):
        result = transfer_learn(
            self._model(), self.loader, self.loader, "signal", self.sources,
            weighting=None,
        )
        self.assertEqual(result["pretrain_losses"], {})
        self.assertEqual(result["alphas"], {})
        self.assertEqual(len(result["finetune_losses"]), 1)

    def test_invalid_weighting_raises(self):
        with self.assertRaises(ValueError):
            transfer_learn(
                self._model(), self.loader, self.loader, "signal", self.sources,
                weighting="random", max_pairs=6,
            )


if __name__ == "__main__":
    unittest.main()
