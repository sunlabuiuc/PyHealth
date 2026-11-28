# Authors: Hanyang Chen (hc81), Siqi Song (siqis3)
# Paper: Frost, Li & Harris. Robust Real-Time Mortality Prediction in the Intensive
#   Care Unit using Temporal Difference Learning. ML4H 2024, PMLR 259:350-363.
# Paper link: https://proceedings.mlr.press/v259/frost25a.html
# Description: Tests for the RealTimeTDCNNLSTM model with small synthetic tensors only.

import math
import tempfile
import unittest

import torch
import torch.nn.functional as F

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RealTimeTDCNNLSTM
from pyhealth.trainer import Trainer

NAN = float("nan")
INPUT_SCHEMA = {"measurements": "tensor", "next_measurements": "tensor"}
OUTPUT_SCHEMA = {
    "mortality": "binary",
    "terminal": "tensor",
    "terminal_reward": "tensor",
    "mortality_horizons": "tensor",
}


def _observation(n, offset=0.0):
    """n clinical tuples (features 1..3) plus 3 demographics (features 4..6)."""
    rows = []
    for i in range(n):
        delta = [NAN, NAN] if i < 3 else [30.0, 0.1 * i]
        rows.append([float(i % 3 + 1), 60.0 * (n - i), 1.0 + i + offset, *delta])
    rows += [[4.0, 0.0, 65.0, NAN, NAN], [5.0, 0.0, 1.0, NAN, NAN]]
    rows.append([6.0, 0.0, 80.0, NAN, NAN])
    return rows


def _samples():
    samples = []
    for i in range(6):
        died = i % 2
        samples.append(
            {
                "patient_id": f"p{i}",
                "measurements": _observation(2 + i, offset=died),
                "next_measurements": _observation(3 + i, offset=died),
                "terminal": float(i >= 4),
                "terminal_reward": float(died),
                "mortality": died,
                "mortality_horizons": [0.0, 0.0, float(died), float(died), float(died)],
            }
        )
    return samples


def _dataset():
    return create_sample_dataset(
        samples=_samples(),
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        dataset_name="synthetic_td",
    )


def _batch(dataset, size=6):
    loader = get_dataloader(dataset, batch_size=size, shuffle=False)
    return next(iter(loader))


class TestRealTimeTDCNNLSTM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.dataset = _dataset()

    def _model(self, **kwargs):
        torch.manual_seed(0)
        defaults = {"embedding_dim": 4, "lstm_hidden_dim": 8}
        defaults.update(kwargs)
        return RealTimeTDCNNLSTM(self.dataset, **defaults)

    def test_initialisation(self):
        model = self._model()
        self.assertEqual(model.num_features, 6)
        self.assertEqual(model.label_key, "mortality")
        self.assertEqual(model.encoder.feature_embedding.num_embeddings, 8)
        self.assertIsNotNone(model.target_encoder)
        self.assertTrue(
            all(not p.requires_grad for p in model.target_encoder.parameters())
        )
        self.assertEqual(model.value_mean.shape[0], 8)
        supervised = self._model(training_target="supervised")
        self.assertIsNone(supervised.target_encoder)
        with self.assertRaises(ValueError):
            self._model(training_target="mc")
        with self.assertRaises(ValueError):
            self._model(td_lambda=1.5)

    def test_forward_shapes(self):
        model = self._model()
        out = model(**_batch(self.dataset))
        self.assertEqual(out["y_prob"].shape, (6, 1))
        self.assertEqual(out["logit"].shape, (6, 1))
        self.assertEqual(out["y_true"].shape, (6, 1))
        self.assertEqual(out["loss"].dim(), 0)
        self.assertTrue(torch.all((out["y_prob"] >= 0) & (out["y_prob"] <= 1)))
        self.assertEqual(out["mortality_horizons"].shape, (6, 5))  # passed through

    def test_backward_only_updates_online_network(self):
        model = self._model()
        model(**_batch(self.dataset))["loss"].backward()
        grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))
        self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))

    def test_td_target_uses_reward_for_terminal_states(self):
        model = self._model()
        batch = _batch(self.dataset)
        target = model.td_target(batch)
        terminal = batch["terminal"].bool()
        torch.testing.assert_close(target[terminal], batch["terminal_reward"][terminal])
        self.assertTrue(torch.all((target >= 0) & (target <= 1)))

    def test_td_loss_matches_manual_computation(self):
        # Deterministic target network (eval mode) so the target can be recomputed.
        model = self._model(target_network_train_mode=False)
        model.train()
        batch = _batch(self.dataset)
        out = model(**batch)  # first step: no soft update owed yet
        target = model.td_target(batch).reshape(-1, 1)
        expected = F.binary_cross_entropy_with_logits(out["logit"], target)
        torch.testing.assert_close(out["loss"], expected)

    def test_td_lambda_one_is_supervised(self):
        model = self._model(td_lambda=1.0)
        model.train()
        out = model(**_batch(self.dataset))
        expected = F.binary_cross_entropy_with_logits(out["logit"], out["y_true"])
        torch.testing.assert_close(out["loss"], expected)

    def test_target_network_mode_follows_flag(self):
        model = self._model(target_network_train_mode=False)
        model.train()
        self.assertTrue(model.encoder.training)
        self.assertFalse(model.target_encoder.training)
        default = self._model()
        default.eval()
        self.assertTrue(default.target_encoder.training)  # authors' setting

    def test_eval_does_not_touch_target_network(self):
        model = self._model()
        model.eval()
        before = {k: v.clone() for k, v in model.target_encoder.state_dict().items()}
        with torch.no_grad():
            out = model(**_batch(self.dataset))
        for key, value in model.target_encoder.state_dict().items():
            torch.testing.assert_close(value, before[key])
        expected = F.binary_cross_entropy_with_logits(out["logit"], out["y_true"])
        torch.testing.assert_close(out["loss"], expected)

    def test_soft_update_once_per_optimiser_step(self):
        model = self._model(target_update_rate=0.5)
        optimizer = torch.optim.SGD(model.encoder.parameters(), lr=0.1)
        batch = _batch(self.dataset)
        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            model(**batch)["loss"].backward()
            optimizer.step()
        self.assertEqual(model.num_target_updates, 2)  # owed for steps 1 and 2
        before = model.target_encoder.head_fc2.weight.clone()
        online = model.encoder.head_fc2.weight.detach()
        model.update_target_network()
        torch.testing.assert_close(
            model.target_encoder.head_fc2.weight, 0.5 * before + 0.5 * online
        )
        model.eval()
        with torch.no_grad():
            model(**batch)
        self.assertEqual(model.num_target_updates, 3)  # eval never updates

    def test_supervised_class_balanced_loss(self):
        model = self._model(training_target="supervised", class_balanced=True)
        torch.testing.assert_close(model.class_weights, torch.tensor([0.5, 0.5]))
        model.eval()
        out = model(**_batch(self.dataset))
        expected = F.binary_cross_entropy_with_logits(
            out["logit"], out["y_true"], weight=torch.full_like(out["y_true"], 0.5)
        )
        torch.testing.assert_close(out["loss"], expected)

    def test_label_from_horizon_column(self):
        model = self._model(training_target="supervised", label_horizon_index=2)
        torch.testing.assert_close(model.class_weights, torch.tensor([0.5, 0.5]))
        batch = _batch(self.dataset)
        out = model(**batch)
        expected = batch["mortality_horizons"][:, 2:3].float()
        torch.testing.assert_close(out["y_true"], expected)

    def test_padding_invariance(self):
        model = self._model()
        model.eval()
        batch = _batch(self.dataset)
        with torch.no_grad():
            together = model(**batch)["logit"]
            for i in range(6):
                single = {
                    k: v[i : i + 1] for k, v in batch.items() if torch.is_tensor(v)
                }
                n = int((single["measurements"][0, :, 0] > 0).sum())
                single["measurements"] = single["measurements"][:, :n]
                alone = model(**single)["logit"]
                torch.testing.assert_close(
                    alone, together[i : i + 1], atol=1e-5, rtol=1e-4
                )

    def test_standardize_clips_and_handles_unknown_ids(self):
        model = self._model()
        x = torch.tensor([[[1.0, 0.0, 1e9, NAN, NAN], [99.0, 0.0, 1.0, NAN, NAN]]])
        ids, _, values, _, _, mask = model.standardize(x)
        self.assertEqual(ids[0, 1].item(), model.num_features + 1)  # unknown id
        self.assertTrue(mask.all())
        high = (model.value_high[1] - model.value_mean[1]) / model.value_std[1]
        self.assertAlmostEqual(values[0, 0].item(), high.item(), places=4)
        # Ids without statistics use the bounded fallback sign(x) * log1p(|x|).
        self.assertAlmostEqual(values[0, 1].item(), math.log1p(1.0), places=5)

    def test_delta_value_uses_clipped_values(self):
        model = self._model()
        self.assertEqual(model.has_delta_stats[1].item(), 1.0)
        # An outlier value of 1000 whose previous value was a normal 5.0.
        x = torch.tensor([[[1.0, 0.0, 1000.0, 30.0, 995.0]]])
        _, _, _, _, delta_values, _ = model.standardize(x)
        low, high = model.value_low[1], model.value_high[1]
        clean = high - torch.clamp(torch.tensor(5.0), low, high)
        expected = (clean - model.delta_value_mean[1]) / model.delta_value_std[1]
        self.assertAlmostEqual(delta_values[0, 0].item(), expected.item(), places=3)
        self.assertLess(abs(delta_values[0, 0].item()), 1e3)

    def test_paper_optimizer_params(self):
        model = self._model()
        params = model.paper_optimizer_params(batches_per_epoch=10)
        n = model.num_trainable_parameters()
        self.assertAlmostEqual(params["lr"], 1 / math.sqrt(n))
        self.assertAlmostEqual(params["weight_decay"], 1 / (params["lr"] * 10))

    def test_trains_with_stock_trainer(self):
        model = self._model()
        loader = get_dataloader(self.dataset, batch_size=3, shuffle=True)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                model=model, metrics=["roc_auc"], device="cpu", output_path=tmp
            )
            trainer.train(loader, loader, epochs=1, monitor="roc_auc")
            scores = trainer.evaluate(loader)
        self.assertIn("roc_auc", scores)
        self.assertEqual(model.num_target_updates, 1)  # 2 steps -> 1 update


if __name__ == "__main__":
    unittest.main()
