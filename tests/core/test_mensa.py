# Author: Zhenshuo Xu (zx65).
# MENSA: A Multi-Event Network for Survival Analysis with Trajectory-based
# Likelihood Estimation, Lillelund et al. https://arxiv.org/abs/2409.06525
"""MENSA pure functions, losses and model, on tiny synthetic data.

Hand-worked example used below (one component, k = 0, so each Weibull is an
exponential distribution with rate exp(b)). With b = log 2 in every state:

    log S(t) = -2t        log f(t) = log 2 - 2t

Patient A: rtime 1, recur 1; dtime 2, death 1.
    Extended times [2, 1, 2], indicators [0, 1, 1].
    log-likelihood = log S0(2) + log f1(1) + log f2(2)
                   = -4 + (log 2 - 2) + (log 2 - 4) = 2 log 2 - 10
Patient B: rtime 3, recur 0; dtime 3, death 0.
    Extended times [3, 3, 3], indicators [1, 0, 0].
    log-likelihood = log f0(3) + log S1(3) + log S2(3)
                   = (log 2 - 6) - 6 - 6 = log 2 - 18
L_surv (unit weights, N = 2) = -(3 log 2 - 28) / 2 = 14 - 1.5 log 2

Trajectory term (only patient A had both events):
    paper:     -(1/N) log S2(t_recur = 1) = -(-2) / 2 = 1
    reference: -mean over A of log S2(t_death = 2) = -(-4) = 4
"""

import math
import tempfile
import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MENSA
from pyhealth.models.mensa import (
    compute_state_weights,
    extend_labels,
    mensa_loss,
    mixture_log_density,
    mixture_log_survival,
    mixture_log_weights,
    survival_nll,
    trajectory_nll,
    weibull_log_density,
    weibull_log_survival,
)

LOG2 = math.log(2.0)


def worked_example() -> dict[str, torch.Tensor]:
    """Patients A and B from the module docstring, as model inputs."""
    times = torch.tensor([[1.0, 2.0], [3.0, 3.0]])
    indicators = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    ext_times, ext_indicators = extend_labels(times, indicators)
    k = torch.zeros(2, 1)  # one component, shape exp(0) = 1
    b = torch.full((2, 1), LOG2)  # rate 2
    gate = torch.zeros(2, 1)
    log_f = torch.stack(
        [mixture_log_density(ext_times[:, p], k, b, gate) for p in range(3)], 1
    )
    log_s = torch.stack(
        [mixture_log_survival(ext_times[:, p], k, b, gate) for p in range(3)], 1
    )
    return {
        "times": times,
        "ext_times": ext_times,
        "indicators": ext_indicators,
        "log_f": log_f,
        "log_s": log_s,
        # S_death at the recurrence time (paper) and at the death time (reference)
        "log_s_death_at_recur": mixture_log_survival(ext_times[:, 1], k, b, gate),
        "log_s_death_at_death": log_s[:, 2],
    }


class TestWeibullComponents(unittest.TestCase):
    def test_shape_one_is_exponential(self):
        t = torch.tensor([[0.5], [1.0], [3.0]])
        k, b = torch.zeros(3, 1), torch.full((3, 1), LOG2)
        torch.testing.assert_close(weibull_log_survival(t, k, b), -2 * t)
        torch.testing.assert_close(weibull_log_density(t, k, b), LOG2 - 2 * t)

    def test_matches_torch_weibull_distribution(self):
        torch.manual_seed(0)
        k, b = torch.randn(50, 1) * 0.5, torch.randn(50, 1) - 3
        t = torch.rand(50, 1) * 100 + 0.1
        # torch's Weibull uses scale = 1 / rate and concentration = shape.
        reference = torch.distributions.Weibull(torch.exp(-b), torch.exp(k))
        torch.testing.assert_close(
            weibull_log_density(t, k, b), reference.log_prob(t), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            weibull_log_survival(t, k, b).exp(),
            1 - reference.cdf(t),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_overflowing_component_does_not_produce_nan_gradients(self):
        # One component's survival exponent overflows float32 at t = 5000
        # (shape e^3 ~ 20, rate e^-1: 20 * (log 5000 - 1) ~ 150 > 88.7). Its
        # mixture weight is tiny, so the loss is finite, but without the cap
        # its gradient would be 0 * inf = NaN.
        k = torch.tensor([[3.0, -1.0, -1.0]], requires_grad=True)
        b = torch.tensor([[-1.0, -8.0, -8.0]], requires_grad=True)
        gate = torch.tensor([[-500.0, 0.0, 0.0]], requires_grad=True)
        t = torch.tensor([5000.0])
        log_s = mixture_log_survival(t, k, b, gate, temperature=1.0)
        log_f = mixture_log_density(t, k, b, gate, temperature=1.0)
        self.assertTrue(torch.isfinite(log_s).all() and torch.isfinite(log_f).all())
        (log_s.sum() + log_f.sum()).backward()
        for tensor in (k, b, gate):
            self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertLess(weibull_log_survival(t[:, None], k, b)[0, 0].item(), -1e30)

    def test_survival_is_one_at_zero_bounded_and_non_increasing(self):
        torch.manual_seed(1)
        k, b, gate = torch.randn(4, 3), torch.randn(4, 3) - 2, torch.randn(4, 3)
        grid = torch.tensor([0.0, 0.1, 1.0, 10.0, 100.0, 1000.0])
        curves = mixture_log_survival(
            grid.expand(4, -1), k.unsqueeze(1), b.unsqueeze(1), gate.unsqueeze(1)
        ).exp()
        self.assertEqual(curves.shape, (4, 6))
        self.assertTrue(torch.equal(curves[:, 0], torch.ones(4)))
        self.assertTrue(((curves >= 0) & (curves <= 1)).all())
        self.assertTrue((curves[:, 1:] <= curves[:, :-1]).all())

    def test_extreme_parameters_stay_finite_or_minus_infinity(self):
        t = torch.tensor([[1.0], [7000.0]])
        big = torch.full((2, 1), 50.0)  # capped at 20 before exponentiation
        log_s = weibull_log_survival(t, big, big)
        self.assertFalse(torch.isnan(log_s).any())
        self.assertFalse(torch.isnan(weibull_log_density(t, big, -big)).any())


class TestMixture(unittest.TestCase):
    def test_weights_sum_to_one_and_temperature_softens_them(self):
        gate = torch.tensor([[10.0, 0.0, -10.0], [1.0, 2.0, 3.0]])
        for temperature in (1.0, 1000.0):
            weights = mixture_log_weights(gate, temperature).exp()
            with self.subTest(temperature=temperature):
                torch.testing.assert_close(weights.sum(-1), torch.ones(2))
        nearly_uniform = mixture_log_weights(gate, 1000.0).exp()
        self.assertLess((nearly_uniform - 1 / 3).abs().max().item(), 0.01)
        with self.assertRaises(ValueError):
            mixture_log_weights(gate, 0.0)

    def test_identical_components_equal_a_single_component(self):
        t = torch.tensor([5.0, 50.0])
        k1, b1 = torch.tensor([[0.3], [0.3]]), torch.tensor([[-3.0], [-3.0]])
        k3, b3 = k1.expand(2, 3), b1.expand(2, 3)
        gate = torch.randn(2, 3)
        torch.testing.assert_close(
            mixture_log_survival(t, k3, b3, gate),
            weibull_log_survival(t.unsqueeze(-1), k1, b1).squeeze(-1),
        )
        torch.testing.assert_close(
            mixture_log_density(t, k3, b3, gate),
            weibull_log_density(t.unsqueeze(-1), k1, b1).squeeze(-1),
        )

    def test_rounding_never_pushes_survival_above_one(self):
        # A large shape and a tiny rate keep every component near 1 early on,
        # where float32 logsumexp can round to +1.2e-7 (S = 1.0000001).
        k = torch.tensor([[2.3756, 1.7242, 1.0063]])
        b = torch.tensor([[-8.7331, -8.3532, -7.5364]])
        gate = torch.tensor([[0.2784, 0.1588, -0.4333]])
        times = torch.tensor([0.0, 1.0, 2.0, 3.5215, 5.0, 10.0, 100.0, 1000.0])
        log_s = torch.stack(
            [mixture_log_survival(t.reshape(1), k, b, gate)[0] for t in times]
        )
        self.assertTrue((log_s <= 0).all())
        survival = log_s.exp()
        self.assertTrue((survival[1:] <= survival[:-1]).all())


class TestLabelsAndWeights(unittest.TestCase):
    def test_extend_labels_four_outcome_combinations(self):
        times = torch.tensor(
            [[900.0, 900.0], [1200.0, 2100.0], [365.5, 3000.0], [5000.0, 5000.0]]
        )
        indicators = torch.tensor([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        ext_times, ext_indicators = extend_labels(times, indicators)
        self.assertTrue(torch.equal(ext_times[:, 1:], times))  # times unchanged
        self.assertTrue(torch.equal(ext_indicators[:, 1:], indicators))
        self.assertEqual(ext_times[:, 0].tolist(), [900.0, 2100.0, 3000.0, 5000.0])
        self.assertEqual(ext_indicators[:, 0].tolist(), [0.0, 0.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            extend_labels(times, indicators[:, :1])

    def test_state_weights_are_inverse_frequency_and_sum_to_three(self):
        indicators = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
        )
        weights = compute_state_weights(indicators)  # counts [2, 1, 2]
        torch.testing.assert_close(weights, torch.tensor([0.75, 1.5, 0.75]))
        torch.testing.assert_close(weights.sum(), torch.tensor(3.0))


class TestLosses(unittest.TestCase):
    def setUp(self):
        self.ex = worked_example()

    def test_worked_example_survival_nll(self):
        self.assertEqual(self.ex["ext_times"].tolist(), [[2, 1, 2], [3, 3, 3]])
        self.assertEqual(self.ex["indicators"].tolist(), [[0, 1, 1], [1, 0, 0]])
        loss = survival_nll(self.ex["log_f"], self.ex["log_s"], self.ex["indicators"])
        self.assertAlmostEqual(loss.item(), 14 - 1.5 * LOG2, places=5)
        weighted = survival_nll(
            self.ex["log_f"],
            self.ex["log_s"],
            self.ex["indicators"],
            torch.ones(3),
        )
        self.assertAlmostEqual(weighted.item(), loss.item(), places=6)

    def test_censored_rows_ignore_an_infinite_density(self):
        log_f = torch.tensor([[-float("inf"), -1.0]])
        log_s = torch.tensor([[-2.0, -float("inf")]])
        loss = survival_nll(log_f, log_s, torch.tensor([[0.0, 1.0]]))
        self.assertEqual(loss.item(), 3.0)

    def test_worked_example_trajectory_variants(self):
        paper = trajectory_nll(
            self.ex["log_s_death_at_recur"], self.ex["indicators"], variant="paper"
        )
        reference = trajectory_nll(
            self.ex["log_s_death_at_death"], self.ex["indicators"], variant="reference"
        )
        self.assertAlmostEqual(paper.item(), 1.0, places=5)
        self.assertAlmostEqual(reference.item(), 4.0, places=5)
        with self.assertRaises(ValueError):
            trajectory_nll(
                self.ex["log_s_death_at_recur"], self.ex["indicators"], variant="x"
            )

    def test_trajectory_is_zero_without_patients_having_both_events(self):
        indicators = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        log_s = torch.tensor([-5.0, -6.0, -7.0], requires_grad=True)
        for variant in ("paper", "reference"):
            with self.subTest(variant=variant):
                value = trajectory_nll(log_s, indicators, variant=variant)
                self.assertEqual(value.item(), 0.0)
                value.backward()  # still part of the graph

    def test_total_loss_mixes_terms_with_lambda(self):
        args = (self.ex["log_f"], self.ex["log_s"], self.ex["indicators"])
        traj_in = self.ex["log_s_death_at_recur"]
        surv = 14 - 1.5 * LOG2
        for lam, expected in [(0.0, surv), (1.0, 1.0), (0.25, 0.75 * surv + 0.25)]:
            with self.subTest(lam=lam):
                total = mensa_loss(*args, traj_in, trajectory_weight=lam)
                self.assertAlmostEqual(total.item(), expected, places=5)
        for bad in (-0.1, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                mensa_loss(*args, traj_in, trajectory_weight=bad)

    def test_gradients_are_finite_through_mixture_and_loss(self):
        torch.manual_seed(2)
        params = [
            torch.randn(5, 3, 3, requires_grad=True),  # k
            (torch.randn(5, 3, 3) - 7).requires_grad_(),  # b, rate ~ 1/1000 days
            torch.randn(5, 3, 3, requires_grad=True),  # gate
        ]
        times = torch.tensor(
            [[900.0, 900.0], [1200, 2100], [365.5, 3000], [5000, 5000], [40, 50]]
        )
        indicators = torch.tensor([[1.0, 1], [0, 1], [1, 0], [0, 0], [1, 1]])
        ext_t, ext_e = extend_labels(times, indicators)
        k, b, g = params
        log_f = mixture_log_density(ext_t, k, b, g)
        log_s = mixture_log_survival(ext_t, k, b, g)
        later = mixture_log_survival(ext_t[:, 1], k[:, 2], b[:, 2], g[:, 2])
        loss = mensa_loss(log_f, log_s, ext_e, later, compute_state_weights(ext_e))
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for p in params:
            self.assertTrue(torch.isfinite(p.grad).all())


def synthetic_samples(n_features: int = 10) -> list[dict]:
    """Four patients, one per outcome combination, in the task's layout.

    Column 2 is constant, to check that a zero-variance feature gets scale 1.
    """
    rows = [
        ("21", [1985, 61, 1, 3, 0, 35, 291, 0, 0, 10], [900, 900], [1, 1]),
        ("22", [1988, 70, 1, 2, 4, 0, 12, 1, 0, 35], [1200, 2100], [0, 1]),
        ("23", [1979, 45, 1, 3, 2, 138, 0, 0, 1, 75], [365.5, 3000], [1, 0]),
        ("24", [1992, 52, 1, 2, 0, 60, 45, 0, 1, 10], [5000, 5000], [0, 0]),
    ]
    return [
        {
            "patient_id": pid,
            "features": [float(v) for v in features[:n_features]],
            "event_times": [float(v) for v in times],
            "event_indicators": [float(v) for v in indicators],
        }
        for pid, features, times, indicators in rows
    ]


def make_dataset(samples: list[dict]):
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"event_times": "tensor", "event_indicators": "tensor"},
        dataset_name="mensa_test",
    )


class TestMENSAModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = synthetic_samples()
        cls.dataset = make_dataset(cls.samples)
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=4)))

    def model(self, seed: int = 0, fit: bool = True, **kwargs) -> MENSA:
        torch.manual_seed(seed)
        model = MENSA(self.dataset, num_time_bins=8, **kwargs)
        if fit:
            model.fit_preprocessing(self.dataset)
        return model

    def test_instantiation_infers_widths_and_has_no_label_mode(self):
        model = self.model(fit=False)
        self.assertEqual((model.input_dim, model.n_events, model.n_states), (10, 2, 3))
        self.assertIsNone(model.mode)
        self.assertEqual(model.label_processors, {})
        self.assertFalse(bool(model.is_fitted))
        self.assertEqual(model.time_grid.shape, (9,))

    def test_default_hyperparameters(self):
        model = MENSA(self.dataset)
        self.assertEqual(model.hidden_dims, (32,))
        self.assertEqual(model.n_dists, 3)
        self.assertEqual(model.dropout, 0.25)
        self.assertEqual(model.trajectory_weight, 0.25)
        linear, _, _, dropout = model.network.embedding
        self.assertEqual((linear.in_features, linear.out_features), (10, 32))
        self.assertEqual(dropout.p, 0.25)
        first, _, last = model.network.adapters[0]
        self.assertEqual((first.in_features, first.out_features), (32, 16))
        self.assertEqual(last.out_features, 32)
        self.assertEqual(model.network.gate_heads[0].out_features, 3)

    def test_reference_rotterdam_config_can_be_passed(self):
        # thecml/mensa configs/mensa/rotterdam.yaml (commit 6b219cd):
        # layers [16], n_dists 3, dropout_rate 0.5, traj_lambda 0.25.
        model = MENSA(self.dataset, hidden_dims=(16,), dropout=0.5)
        linear, _, _, dropout = model.network.embedding
        self.assertEqual(linear.out_features, 16)
        self.assertEqual(dropout.p, 0.5)
        first, _, last = model.network.adapters[0]
        self.assertEqual((first.in_features, first.out_features), (16, 16))
        self.assertEqual(last.out_features, 16)

    def test_forward_shapes_finite_loss_and_gradients(self):
        model = self.model()
        output = model(**self.batch)
        self.assertEqual(output["y_prob"].shape, (4, 2, 9))
        self.assertEqual(output["y_true"].shape, (4, 4))
        torch.testing.assert_close(
            output["y_true"],
            torch.cat([self.batch["event_times"], self.batch["event_indicators"]], 1),
        )
        self.assertEqual(output["loss"].shape, ())
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        # BaseModel's empty _dummy_param only tracks the device.
        grads = [p.grad for n, p in model.named_parameters() if n != "_dummy_param"]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))

    def test_survival_curves_bounded_non_increasing_and_one_at_zero(self):
        model = self.model()
        curves = model.predict_survival(
            self.batch["features"], torch.tensor([0.0, 30, 365, 1000, 3000, 7000])
        )
        self.assertEqual(curves.shape, (4, 2, 6))
        self.assertTrue(torch.equal(curves[..., 0], torch.ones(4, 2)))
        self.assertTrue(((curves >= 0) & (curves <= 1)).all())
        self.assertTrue((curves[..., 1:] <= curves[..., :-1]).all())
        y_prob = model(**self.batch)["y_prob"]
        self.assertEqual(model.time_grid[0].item(), 0.0)
        self.assertTrue((y_prob[..., 1:] <= y_prob[..., :-1]).all())

    def test_fit_preprocessing_uses_only_the_given_split(self):
        train = self.samples[:2]  # patients 21 and 22
        model = self.model(fit=False)
        model.fit_preprocessing(make_dataset(train))
        x = torch.tensor([s["features"] for s in train], dtype=torch.float64)
        expected_scale = x.std(0, unbiased=False)
        expected_scale[expected_scale == 0] = 1.0
        torch.testing.assert_close(model.feature_mean, x.mean(0).float())
        torch.testing.assert_close(model.feature_scale, expected_scale.float())
        self.assertEqual(model.feature_scale[2].item(), 1.0)  # constant column
        # Extended indicators: 21 = [0, 1, 1], 22 = [0, 0, 1] -> counts [0, 1, 2].
        expected = compute_state_weights(torch.tensor([[0.0, 1, 1], [0, 0, 1]]))
        torch.testing.assert_close(model.state_weights, expected)
        times = torch.tensor([900.0, 900, 1200, 2100])
        self.assertEqual(model.time_grid[0].item(), 0.0)
        self.assertEqual(model.time_grid[1].item(), times.min().item())
        self.assertEqual(model.time_grid[-1].item(), times.max().item())
        self.assertTrue(bool(model.is_fitted))

    def test_using_the_model_before_fitting_raises(self):
        model = self.model(fit=False)
        with self.assertRaisesRegex(RuntimeError, "fit_preprocessing"):
            model(**self.batch)
        with self.assertRaisesRegex(RuntimeError, "fit_preprocessing"):
            model.predict_survival(self.batch["features"], torch.tensor([1.0]))

    def test_loss_matches_the_pure_functions_and_lambda(self):
        model = self.model(trajectory_weight=0.25)
        model.eval()
        output = model(**self.batch)
        x = (self.batch["features"] - model.feature_mean) / model.feature_scale
        k, b, g = model.network(x)
        t, e = extend_labels(self.batch["event_times"], self.batch["event_indicators"])
        log_f = mixture_log_density(t, k, b, g)
        log_s = mixture_log_survival(t, k, b, g)
        later = mixture_log_survival(t[:, 1], k[:, 2], b[:, 2], g[:, 2])
        surv = survival_nll(log_f, log_s, e, model.state_weights)
        traj = trajectory_nll(later, e)
        torch.testing.assert_close(output["loss"], 0.75 * surv + 0.25 * traj)
        model.trajectory_weight = 0.0
        torch.testing.assert_close(model(**self.batch)["loss"], surv)
        model.trajectory_weight, model.trajectory_variant = 1.0, "reference"
        torch.testing.assert_close(
            model(**self.batch)["loss"],
            trajectory_nll(log_s[:, 2], e, variant="reference"),
        )

    def test_state_dict_round_trip_gives_identical_predictions(self):
        model = self.model(seed=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            model(**self.batch)["loss"].backward()
            optimizer.step()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = f"{tmp}/mensa.pt"
            torch.save(model.state_dict(), path)
            restored = self.model(seed=1, fit=False)
            restored.load_state_dict(torch.load(path))
        self.assertTrue(bool(restored.is_fitted))
        times = torch.tensor([0.0, 100, 1000, 5000])
        self.assertTrue(
            torch.equal(
                model.predict_survival(self.batch["features"], times),
                restored.predict_survival(self.batch["features"], times),
            )
        )

    def test_training_mode_batch_of_two_and_mode_is_restored(self):
        model = self.model()
        model.train()
        batch = next(iter(get_dataloader(self.dataset, batch_size=2)))
        loss = model(**batch)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        model.predict_survival(batch["features"], torch.tensor([10.0]))
        self.assertTrue(all(m.training for m in model.modules()))

    def test_eight_feature_variant(self):
        dataset = make_dataset(synthetic_samples(n_features=8))
        torch.manual_seed(0)
        model = MENSA(dataset, num_time_bins=4)
        model.fit_preprocessing(dataset)
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        self.assertEqual(model.input_dim, 8)
        self.assertTrue(torch.isfinite(model(**batch)["loss"]))

    def test_invalid_configuration_and_labels_are_rejected(self):
        for kwargs in (
            {"feature_key": "x"},
            {"times_key": "t"},
            {"trajectory_weight": 1.5},
            {"trajectory_variant": "other"},
            {"trajectory": (1, 3)},
            {"trajectory": (2, 2)},
            {"dropout": 1.0},
            {"num_time_bins": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MENSA(self.dataset, **kwargs)
        model = self.model()
        for key, value in (("event_times", -1.0), ("event_indicators", 2.0)):
            batch = dict(self.batch)
            batch[key] = batch[key].clone()
            batch[key][0, 0] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                model(**batch)
        with self.assertRaises(ValueError):
            model(features=self.batch["features"][:, :5])

    def test_trainer_trains_and_reports_loss(self):
        from pyhealth.trainer import Trainer

        model = self.model()
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        # Capture the Trainer's INFO logs (it logs the whole model) to keep
        # test output readable.
        with self.assertLogs("pyhealth.trainer", level="INFO"):
            trainer = Trainer(model=model, device="cpu", enable_logging=False)
            trainer.train(
                train_dataloader=loader,
                val_dataloader=loader,
                epochs=2,
                monitor="loss",
                monitor_criterion="min",
            )
        scores = trainer.evaluate(get_dataloader(self.dataset, batch_size=4))
        self.assertEqual(set(scores), {"loss"})
        self.assertTrue(math.isfinite(scores["loss"]))


if __name__ == "__main__":
    unittest.main()
