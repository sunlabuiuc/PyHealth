# Authors: Murphy Crosby
# Deep Cox Mixtures for Survival Regression, Nagpal et al., MLHC 2021.
# https://proceedings.mlr.press/v149/nagpal21a.html
"""Deep Cox Mixtures survival model and its building blocks."""

import io
import math
import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SurvivalDCM
from pyhealth.models.survival_dcm import (
    breslow_baseline_survival,
    cox_partial_log_likelihood,
    smooth_baseline_survival,
)

NUM_PATIENTS = 24
NUM_FEATURES = 4


def make_dataset():
    """A small made-up survival dataset with tied times and censoring."""
    samples = []
    for i in range(NUM_PATIENTS):
        samples.append(
            {
                "patient_id": f"p{i}",
                "features": [math.sin(i), math.cos(2 * i), (i % 5) / 5, (-1.0) ** i],
                "duration": [5.0 + (i * 7) % 23],
                "event": [0.0 if i % 3 == 0 else 1.0],
            }
        )
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"duration": "tensor", "event": "tensor"},
        dataset_name="test",
    )


class TestCoxPartialLogLikelihood(unittest.TestCase):
    def setUp(self):
        # Hazard ratios 1, 2 and 3; the highest-risk patient dies first.
        self.log_hazard_ratio = torch.log(torch.tensor([1.0, 2.0, 3.0]))
        self.duration = torch.tensor([3.0, 2.0, 1.0])

    def test_hand_computed_value(self):
        # (3/6) * (2/3) * (1/1) = 1/3
        value = cox_partial_log_likelihood(
            self.log_hazard_ratio, self.duration, torch.ones(3)
        )
        self.assertAlmostEqual(value.item(), -math.log(3), places=5)

    def test_censored_patient_adds_no_term(self):
        # Without the first death's term: (2/3) * (1/1)
        event = torch.tensor([1.0, 1.0, 0.0])
        value = cox_partial_log_likelihood(self.log_hazard_ratio, self.duration, event)
        self.assertAlmostEqual(value.item(), math.log(2) - math.log(3), places=5)

    def test_no_events_gives_zero(self):
        value = cox_partial_log_likelihood(
            self.log_hazard_ratio, self.duration, torch.zeros(3)
        )
        self.assertEqual(value.item(), 0.0)

    def test_tied_times_share_a_risk_set(self):
        # Two deaths at the same time, hazard ratios 1 and 1, among three
        # patients with total hazard 3: each contributes log(1/3).
        value = cox_partial_log_likelihood(
            torch.zeros(3), torch.tensor([2.0, 2.0, 5.0]), torch.tensor([1.0, 1.0, 0.0])
        )
        self.assertAlmostEqual(value.item(), 2 * math.log(1 / 3), places=5)

    def test_unchanged_by_a_constant_shift(self):
        event = torch.ones(3)
        original = cox_partial_log_likelihood(
            self.log_hazard_ratio, self.duration, event
        )
        shifted = cox_partial_log_likelihood(
            self.log_hazard_ratio + 5.0, self.duration, event
        )
        self.assertAlmostEqual(original.item(), shifted.item(), places=5)

    def test_gradient_is_finite(self):
        log_hazard_ratio = self.log_hazard_ratio.clone().requires_grad_(True)
        cox_partial_log_likelihood(
            log_hazard_ratio, self.duration, torch.ones(3)
        ).backward()
        self.assertTrue(torch.isfinite(log_hazard_ratio.grad).all())


class TestBreslowBaselineSurvival(unittest.TestCase):
    def test_hand_computed_curve(self):
        # Cumulative hazard grows by 1/6, then 1/3, then 1/1.
        times, survival = breslow_baseline_survival(
            torch.log(torch.tensor([1.0, 2.0, 3.0])),
            torch.tensor([3.0, 2.0, 1.0]),
            torch.ones(3),
        )
        self.assertEqual(times.tolist(), [1.0, 2.0, 3.0])
        expected = torch.exp(-torch.tensor([1 / 6, 1 / 2, 3 / 2]))
        self.assertTrue(torch.allclose(survival, expected))

    def test_equal_hazards_with_censoring_and_ties(self):
        # With every hazard ratio 1 this is the Nelson-Aalen estimator.
        times, survival = breslow_baseline_survival(
            torch.zeros(5),
            torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0]),
            torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0]),
        )
        self.assertEqual(times.tolist(), [1.0, 2.0, 3.0])
        cumulative_hazard = -torch.log(survival)
        self.assertTrue(
            torch.allclose(cumulative_hazard, torch.tensor([0.2, 0.45, 0.95]))
        )

    def test_no_events_gives_empty_curve(self):
        times, survival = breslow_baseline_survival(
            torch.zeros(3), torch.tensor([1.0, 2.0, 3.0]), torch.zeros(3)
        )
        self.assertEqual(len(times), 0)
        self.assertEqual(len(survival), 0)

    def test_no_gradient_flows_through(self):
        log_hazard_ratio = torch.zeros(3, requires_grad=True)
        _, survival = breslow_baseline_survival(
            log_hazard_ratio, torch.tensor([1.0, 2.0, 3.0]), torch.ones(3)
        )
        self.assertFalse(survival.requires_grad)


class TestSmoothBaselineSurvival(unittest.TestCase):
    def setUp(self):
        self.grid = torch.linspace(0.0, 100.0, 201)

    def test_recovers_a_known_curve(self):
        times = torch.arange(1.0, 81.0)
        smooth, slope = smooth_baseline_survival(
            times, torch.exp(-0.02 * times), self.grid
        )
        inside = (self.grid >= 2) & (self.grid <= 78)
        truth = torch.exp(-0.02 * self.grid)
        self.assertLess((smooth - truth)[inside].abs().max().item(), 0.02)
        self.assertLess((slope + 0.02 * truth)[inside].abs().max().item(), 0.005)

    def test_result_is_a_valid_survival_curve(self):
        times = torch.tensor([3.0, 10.0, 11.0, 40.0, 41.0, 42.0, 90.0])
        survival = torch.tensor([0.9, 0.85, 0.6, 0.55, 0.5, 0.3, 0.1])
        smooth, slope = smooth_baseline_survival(times, survival, self.grid)
        self.assertEqual(smooth.shape, self.grid.shape)
        self.assertTrue(((smooth > 0) & (smooth <= 1)).all())
        self.assertTrue((smooth[1:] <= smooth[:-1]).all())
        self.assertTrue((slope <= 0).all())

    def test_no_events_gives_a_flat_curve(self):
        empty = torch.zeros(0)
        smooth, slope = smooth_baseline_survival(empty, empty, self.grid)
        self.assertTrue((smooth == 1).all())
        self.assertTrue((slope == 0).all())

    def test_few_event_times(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                times = torch.arange(1.0, count + 1.0)
                survival = torch.linspace(0.9, 0.7, count)
                smooth, slope = smooth_baseline_survival(times, survival, self.grid)
                self.assertTrue(torch.isfinite(smooth).all())
                self.assertTrue(torch.isfinite(slope).all())


class TestSurvivalDCM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = make_dataset()
        loader = get_dataloader(cls.dataset, batch_size=NUM_PATIENTS, shuffle=False)
        cls.batch = next(iter(loader))

    def setUp(self):
        torch.manual_seed(0)
        self.loader = get_dataloader(self.dataset, batch_size=8, shuffle=False)
        self.model = SurvivalDCM(
            self.dataset, hidden_dim=8, num_components=3, num_grid_points=65
        )

    def fitted_model(self, **kwargs):
        """A model whose baseline curves have been fitted once."""
        settings = {"hidden_dim": 8, "num_components": 3, "num_grid_points": 65}
        settings.update(kwargs)
        model = SurvivalDCM(self.dataset, **settings)
        model.fit_baselines(self.loader)
        return model

    def test_initialization(self):
        self.assertEqual(self.model.input_dim, NUM_FEATURES)
        self.assertEqual(self.model.num_components, 3)
        self.assertEqual(self.model.label_processors, {})
        self.assertIsNone(self.model.mode)
        self.assertIsNone(self.model.expert.bias)
        self.assertEqual(self.model.baseline_grid.shape, (65,))
        self.assertEqual(self.model.baseline_survival.shape, (3, 65))
        self.assertEqual(self.model.baseline_slope.shape, (3, 65))
        self.assertFalse(self.model.baselines_fitted)

    def test_invalid_arguments_raise(self):
        bad_arguments = {
            "unknown feature key": {"feature_key": "missing"},
            "unknown duration key": {"duration_key": "missing"},
            "unknown event key": {"event_key": "missing"},
            "no hidden units": {"hidden_dim": 0},
            "no hidden layers": {"num_hidden_layers": 0},
            "no components": {"num_components": 0},
            "one grid point": {"num_grid_points": 1},
            "zero hazard bound": {"max_log_hazard_ratio": 0.0},
        }
        for label, kwargs in bad_arguments.items():
            with self.subTest(label), self.assertRaises(ValueError):
                SurvivalDCM(self.dataset, **kwargs)

    def test_forward_output_shapes(self):
        self.model.eval()
        output = self.model(**self.batch)
        self.assertEqual(output["log_gate"].shape, (NUM_PATIENTS, 3))
        self.assertEqual(output["log_hazard_ratio"].shape, (NUM_PATIENTS, 3))
        self.assertEqual(output["loss"].ndim, 0)
        self.assertEqual(output["y_true"].shape, (NUM_PATIENTS, 2))
        self.assertEqual(output["y_prob"].shape, (NUM_PATIENTS, 65))

    def test_gate_rows_are_probabilities(self):
        self.model.eval()
        gate = self.model(**self.batch)["log_gate"].exp()
        self.assertTrue(torch.allclose(gate.sum(dim=1), torch.ones(NUM_PATIENTS)))

    def test_log_hazard_ratio_stays_within_bound(self):
        model = SurvivalDCM(self.dataset, hidden_dim=8, max_log_hazard_ratio=0.5)
        model.eval()
        extreme = {"features": self.batch["features"] * 1000}
        log_hazard_ratio = model(**extreme)["log_hazard_ratio"]
        self.assertLessEqual(log_hazard_ratio.abs().max().item(), 0.5)

    def test_features_only_returns_no_loss(self):
        self.model.eval()
        output = self.model(features=self.batch["features"])
        self.assertEqual(sorted(output), ["log_gate", "log_hazard_ratio"])

    def test_invalid_features_raise(self):
        self.model.eval()
        features = self.batch["features"]
        not_finite = features.clone()
        not_finite[0, 0] = float("nan")
        bad_features = {
            "nan value": not_finite,
            "wrong width": features[:, :2],
            "one dimension": features[0],
        }
        for label, bad in bad_features.items():
            with self.subTest(label), self.assertRaises(ValueError):
                self.model(features=bad)

    def test_training_before_fitting_raises(self):
        self.model.train()
        with self.assertRaises(RuntimeError):
            self.model(**self.batch)

    def test_fit_baselines(self):
        self.model.train()
        self.model.fit_baselines(self.loader)
        self.assertTrue(self.model.baselines_fitted)
        self.assertTrue(self.model.training)  # the mode is put back
        longest = self.batch["duration"].max().item()
        self.assertAlmostEqual(self.model.baseline_grid[-1].item(), longest, places=4)
        survival = self.model.baseline_survival
        self.assertTrue(((survival > 0) & (survival <= 1)).all())
        self.assertTrue((survival[:, 1:] <= survival[:, :-1]).all())
        self.assertTrue((self.model.baseline_slope <= 0).all())

    def test_loss_and_gradients(self):
        model = self.fitted_model()
        model.train()
        output = model(**self.batch)
        self.assertEqual(sorted(output), ["log_gate", "log_hazard_ratio", "loss"])
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        for name, parameter in model.named_parameters():
            if name == "_dummy_param":
                continue
            with self.subTest(name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_one_component_loss_is_the_partial_likelihood(self):
        model = self.fitted_model(num_components=1)
        model.train()
        output = model(**self.batch)
        duration = self.batch["duration"].reshape(-1)
        event = self.batch["event"].reshape(-1)
        expected = -cox_partial_log_likelihood(
            output["log_hazard_ratio"][:, 0], duration, event
        ) / len(duration)
        self.assertTrue(torch.allclose(output["loss"], expected))

    def test_evaluation_loss_is_repeatable(self):
        model = self.fitted_model()
        model.eval()
        first = model(**self.batch)["loss"].item()
        second = model(**self.batch)["loss"].item()
        self.assertEqual(first, second)

    def test_y_true_holds_duration_and_event(self):
        model = self.fitted_model()
        model.eval()
        y_true = model(**self.batch)["y_true"]
        self.assertTrue(torch.equal(y_true[:, 0], self.batch["duration"].reshape(-1)))
        self.assertTrue(torch.equal(y_true[:, 1], self.batch["event"].reshape(-1)))

    def test_baseline_lookup_interpolates(self):
        # Give every subgroup the straight line 1 - t / 200 on a 0..64 grid.
        self.model.baseline_grid.copy_(torch.linspace(0.0, 64.0, 65))
        self.model.baseline_survival.copy_(
            (1 - self.model.baseline_grid / 200).repeat(3, 1)
        )
        self.model.baseline_slope.fill_(-1 / 200)
        survival, slope = self.model.baseline_at(torch.tensor([0.0, 10.5, 64.0, 500.0]))
        self.assertEqual(survival.shape, (4, 3))
        expected = torch.tensor([1.0, 1 - 10.5 / 200, 1 - 64 / 200, 1 - 64 / 200])
        self.assertTrue(torch.allclose(survival[:, 0], expected))
        self.assertTrue(torch.allclose(slope, torch.full((4, 3), -1 / 200)))

    def test_posterior_matches_hand_computation(self):
        # Two subgroups with constant hazards 0.001 (slow) and 0.05 (fast).
        model = SurvivalDCM(
            self.dataset, hidden_dim=8, num_components=2, num_grid_points=4001
        )
        rate = torch.tensor([[0.001], [0.05]])
        model.baseline_grid.copy_(torch.linspace(0.0, 400.0, 4001))
        model.baseline_survival.copy_(torch.exp(-rate * model.baseline_grid))
        model.baseline_slope.copy_(-rate * torch.exp(-rate * model.baseline_grid))

        even_gate = torch.full((2, 2), math.log(0.5))
        posterior = model.subgroup_posterior(
            even_gate,
            torch.zeros(2, 2),
            torch.tensor([5.0, 300.0]),  # one early, one late
            torch.tensor([1.0, 0.0]),  # the early one died; the late one did not
        )
        self.assertTrue(torch.allclose(posterior.sum(dim=1), torch.ones(2)))

        # Early death: compare the two densities, rate * exp(-rate * 5).
        slow = 0.001 * math.exp(-0.005)
        fast = 0.05 * math.exp(-0.25)
        self.assertAlmostEqual(posterior[0, 1].item(), fast / (slow + fast), places=3)
        # Late censoring: surviving 300 is far likelier in the slow subgroup.
        self.assertGreater(posterior[1, 0].item(), 0.999)

    def test_sample_subgroup(self):
        certain = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        assigned = self.model.sample_subgroup(certain)
        self.assertEqual(assigned.tolist(), [1, 0, 2])
        self.assertEqual(assigned.dtype, torch.int64)

    def test_predicting_before_fitting_raises(self):
        with self.assertRaises(RuntimeError):
            self.model.predict_survival(self.batch["features"], torch.tensor([10.0]))

    def test_predict_survival(self):
        model = self.fitted_model()
        model.eval()
        times = torch.tensor([0.0, 8.0, 15.0, 27.0])
        survival = model.predict_survival(self.batch["features"], times)
        self.assertEqual(survival.shape, (NUM_PATIENTS, 4))
        self.assertTrue(((survival >= 0) & (survival <= 1)).all())
        self.assertTrue((survival[:, 1:] <= survival[:, :-1] + 1e-6).all())
        # At time 0 nobody has died yet. The smoothed curves start very close
        # to 1 but not exactly at it, so allow a small tolerance.
        self.assertGreater(survival[:, 0].min().item(), 0.99)

    def test_one_component_prediction_is_baseline_to_the_hazard_ratio(self):
        model = self.fitted_model(num_components=1)
        model.eval()
        times = torch.tensor([8.0, 15.0])
        features = self.batch["features"]
        hazard_ratio = model(features=features)["log_hazard_ratio"].exp()
        baseline, _ = model.baseline_at(times)
        expected = baseline[:, 0].unsqueeze(0) ** hazard_ratio
        self.assertTrue(
            torch.allclose(model.predict_survival(features, times), expected)
        )

    def test_save_and_load(self):
        model = self.fitted_model()
        model.eval()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)

        restored = SurvivalDCM(
            self.dataset, hidden_dim=8, num_components=3, num_grid_points=65
        )
        restored.load_state_dict(torch.load(buffer))
        restored.eval()
        self.assertTrue(restored.baselines_fitted)
        times = torch.tensor([8.0, 15.0])
        features = self.batch["features"]
        self.assertTrue(
            torch.allclose(
                restored.predict_survival(features, times),
                model.predict_survival(features, times),
            )
        )

    def test_training_lowers_the_loss(self):
        model = self.fitted_model(num_components=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        model.train()
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            loss = model(**self.batch)["loss"]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
