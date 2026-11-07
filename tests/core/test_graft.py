"""Small fabricated tensors only; full E0 training is a separate example."""

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import BaseModel, GRAFT
from pyhealth.models.graft import LocalKMImputer, _fingerprint

HAS_RANK = importlib.util.find_spec("torchsort") is not None
HAS_SURVIVAL = all(
    importlib.util.find_spec(name) is not None for name in ("lifelines", "sksurv")
)


def _ready(**kwargs):
    config = dict(n_features=2, use_gates=False, use_residual=False, l2=0)
    config.update(kwargs)
    model = GRAFT(**config)
    model._fit_preprocessing(np.array([[0.0, 0.0], [2.0, 2.0]]))
    with torch.no_grad():
        model.beta.copy_(torch.tensor([1.0, 0.0]))
        model.intercept.zero_()
    model.eval()
    return model


def _calibrated():
    model = _ready()
    model.ranking_fitted_ = True
    model.calibration_ = {
        "coefficient": -1.0,
        "score_mean": 0.25,
        "baseline_times": [1.0, 3.0, 5.0],
        "baseline_cumulative_hazard": [0.0, 0.4, 0.8],
        "training_max_time": 5.0,
        "censor_times": [-math.inf, 1.0, 2.0, 4.0, 5.0],
        "censor_probabilities": [1.0, 1.0, 2 / 3, 2 / 3, 0.0],
    }
    return model


class TestGRAFTInterface(unittest.TestCase):
    def test_base_model_schema_and_variants(self):
        samples = [
            {
                "patient_id": str(i),
                "features": [i, 1.0],
                "duration": i + 1.0,
                "event": i % 2,
            }
            for i in range(3)
        ]
        dataset = create_sample_dataset(
            samples, {"features": "tensor"}, {"duration": "tensor", "event": "tensor"}
        )
        model = GRAFT(dataset)
        self.assertIsInstance(model, BaseModel)
        self.assertIsNone(model.mode)
        self.assertTrue(model.fit_outside_trainer)
        self.assertEqual(model.n_features, 2)
        self.assertEqual(model.label_keys, ["duration", "event"])
        self.assertIsNone(GRAFT(n_features=2, use_gates=False).eta)
        linear = GRAFT(n_features=2, use_gates=False, use_residual=False)
        self.assertIsNone(linear.residual)

    def test_invalid_configuration(self):
        for arguments in (
            {},
            {"n_features": 0},
            {"n_features": 2, "sigma": 0},
            {"n_features": 2, "dropout": 1},
            {"n_features": 2, "l2": -1},
            {"n_features": 2, "rank_tau": float("nan")},
            {"n_features": 2, "mc_draws": 0},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                GRAFT(**arguments)

    def test_raw_inputs_require_training_preprocessing(self):
        with self.assertRaises(RuntimeError):
            GRAFT(n_features=2).predict_score([[1.0, 2.0]])
        model = _ready()
        for x in ([[1.0]], [[1.0, float("inf")]], []):
            with self.subTest(x=x), self.assertRaises(ValueError):
                model.predict_score(x)
        with self.assertRaises(ValueError):
            model._fit_preprocessing(np.array([[np.nan, 0.0], [np.nan, 1.0]]))

    def test_training_median_population_scale_and_constant_column(self):
        model = GRAFT(n_features=2)
        model._fit_preprocessing(np.array([[1.0, 7.0], [3.0, 7.0], [np.nan, 7.0]]))
        np.testing.assert_allclose(model.medians.numpy(), [2.0, 7.0])
        np.testing.assert_allclose(model.feature_mean.numpy(), [2.0, 7.0])
        np.testing.assert_allclose(model.feature_scale.numpy(), [np.sqrt(2 / 3), 1.0])
        np.testing.assert_allclose(
            model._transform([[np.nan, 7.0]]).numpy(), [[0.0, 0.0]]
        )

    def test_deterministic_prediction_restores_training_mode(self):
        model = _ready(use_gates=True, use_residual=True, dropout=0.7)
        model.train()
        x = [[0.2, 0.4], [1.2, 1.4]]
        np.testing.assert_array_equal(model.predict_score(x), model.predict_score(x))
        self.assertTrue(model.training)
        self.assertEqual(model.predict_score(x, batch_size=1).shape, (2,))
        with self.assertRaises(ValueError):
            model.predict_score(x, batch_size=0)

    def test_gate_bounds_and_single_global_mask(self):
        model = _ready(use_gates=True)
        with torch.no_grad():
            model.eta.copy_(torch.tensor([-2.0, 2.0]))
        torch.testing.assert_close(
            model.deterministic_gates(), torch.tensor([0.0, 1.0])
        )
        with torch.no_grad():
            model.eta.fill_(0.5)
            model.beta.fill_(1)
        model.train()
        model._gate_generator = torch.Generator().manual_seed(17)
        batch = torch.eye(2).repeat(2, 1)
        first = model._scores(batch)
        torch.testing.assert_close(first[:2], first[2:])
        self.assertTrue(bool(((first >= 0) & (first <= 1)).all()))
        model._gate_generator.manual_seed(17)
        torch.testing.assert_close(first, model._scores(batch))

    def test_held_out_labels_never_trigger_loss_or_change_state(self):
        model = _ready()
        before = deepcopy(model.state_dict())
        first = model(
            features=[[0.0, 1.0], [2.0, 1.0]], duration=[1.0, 5.0], event=[1, 0]
        )
        second = model(
            features=[[0.0, 1.0], [2.0, 1.0]], duration=[99.0, 3.0], event=[0, 1]
        )
        self.assertNotIn("loss", first)
        torch.testing.assert_close(first["score"], second["score"])
        torch.testing.assert_close(first["risk_score"], -first["score"])
        self.assertEqual(first["score"].shape, (2,))
        for name in ("feature_mean", "feature_scale", "medians", "beta"):
            torch.testing.assert_close(before[name], model.state_dict()[name])


@unittest.skipUnless(HAS_RANK, "Install the optional GRAFT torchsort dependency.")
class TestGRAFTRanking(unittest.TestCase):
    def test_objective_direction_and_nonzero_ranking_gradient(self):
        model = _ready()
        target = torch.arange(4, dtype=torch.float32)
        self.assertAlmostEqual(
            float(model._loss(target * 0.01, target)["ranking_loss"]), -1.0
        )
        self.assertAlmostEqual(
            float(model._loss(-target * 0.01, target)["ranking_loss"]), 1.0
        )
        scores = torch.tensor([0.0, 0.02, 0.01, 0.03], requires_grad=True)
        loss = model._loss(scores, target)["ranking_loss"]
        gradient = torch.autograd.grad(loss, scores)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.norm()), 0.0)

    def test_average_target_ranks_and_monte_carlo_mean(self):
        from scipy.stats import rankdata

        model = _ready()
        scores = torch.tensor([0.0, 0.01, 0.02, 0.03])
        targets = np.array([[1.0, 3.0], [1.0, 2.0], [4.0, 2.0], [2.0, 1.0]])
        expected = -np.mean(
            [
                np.corrcoef(scores.numpy(), rankdata(column))[0, 1]
                for column in targets.T
            ]
        )
        self.assertAlmostEqual(
            float(model._loss(scores, targets)["ranking_loss"]), expected, places=6
        )

    def test_constant_targets_and_predictions_have_finite_connected_gradients(self):
        for scores, targets in [
            (torch.arange(3, dtype=torch.float32), torch.ones(3)),
            (torch.zeros(3), torch.arange(3, dtype=torch.float32)),
        ]:
            scores.requires_grad_()
            output = _ready()._loss(scores, targets)
            output["loss"].backward()
            self.assertEqual(float(output["ranking_loss"].detach()), 0.0)
            self.assertTrue(bool(torch.isfinite(scores.grad).all()))

    def test_explicit_penalties_match_known_values(self):
        model = _ready(use_gates=True, sigma=0.5, gate_l0=0.01, l2=0.001)
        with torch.no_grad():
            model.eta.copy_(torch.tensor([-0.5, 0.5]))
            model.beta.copy_(torch.tensor([1.0, 2.0]))
            model.intercept.fill_(3.0)
        output = model._loss(torch.tensor([0.0, 0.01, 0.02]), torch.arange(3.0))
        self.assertAlmostEqual(float(output["gate_penalty"].detach()), 0.01, places=7)
        self.assertAlmostEqual(float(output["l2_penalty"].detach()), 0.0145, places=7)

    def test_real_parameter_update_reduces_fixed_batch_rank_loss(self):
        model = _ready()
        features = [[1.0, 1.0], [1.02, 1.01], [1.01, 1.02], [1.03, 1.03]]
        target = torch.tensor([0.0, 1.0, 2.0, 3.0])
        output = model(features=features, imputed_log_times=target)
        initial = float(output["ranking_loss"].detach())
        output["loss"].backward()
        self.assertGreater(float(model.beta.grad.norm()), 0.0)
        original = model.beta.detach().clone()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-1e-4)
        self.assertFalse(torch.equal(original, model.beta))
        self.assertLess(
            float(
                model(features=features, imputed_log_times=target)[
                    "ranking_loss"
                ].detach()
            ),
            initial,
        )

    def test_invalid_imputed_targets(self):
        model = _ready()
        for targets in (
            torch.ones(2),
            torch.empty(3, 0),
            torch.tensor([1.0, float("nan"), 2.0]),
            torch.ones(3, requires_grad=True),
        ):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                model._loss(torch.arange(3.0), targets)


class TestLocalKMImputer(unittest.TestCase):
    def test_known_conditional_quantiles_and_preserved_observed_times(self):
        km = LocalKMImputer(3, 2, 3).fit(
            np.zeros((4, 1)), [1.0, 2.0, 4.0, 6.0], [0, 1, 1, 0]
        )
        uniform = np.tile([0.0, 0.49, 0.5, 0.99], (4, 1))
        times = np.exp(km.sample(uniform))
        np.testing.assert_allclose(times[0], [2.0, 2.0, 4.0, 4.0])
        np.testing.assert_allclose(times[1], 2.0)
        np.testing.assert_allclose(times[2], 4.0)
        self.assertTrue(bool((times[0] > 1).all() and (times[3] > 6).all()))
        self.assertAlmostEqual(km.diagnostics_["conditional_residual_mass"][0], 1 / 3)

    def test_tied_event_censoring_risk_and_residual_mass(self):
        km = LocalKMImputer(4, 2, 4).fit(
            np.zeros((5, 1)), [2.0, 2.0, 2.0, 4.0, 6.0], [0, 1, 0, 1, 0]
        )
        self.assertEqual(km.diagnostics_["conditional_residual_mass"][0], 0.5)
        np.testing.assert_allclose(np.exp(km.sample(np.full((5, 2), 0.5)))[0], 4.0)
        self.assertTrue(
            all(size == 4 for size in km.diagnostics_["neighborhood_sizes"])
        )

    def test_self_exclusion_and_stable_source_id_distance_ties(self):
        km = LocalKMImputer(1, 1, 1).fit(
            np.zeros((3, 1)), [1.0, 3.0, 9.0], [0, 1, 1], row_ids=[30, 20, 10]
        )
        np.testing.assert_allclose(np.exp(km.sample(np.zeros((3, 1))))[0], 9.0)

    def test_zero_survival_tail_fallback_and_strict_failure(self):
        x, times, events = np.zeros((3, 1)), [6.0, 2.0, 4.0], [0, 1, 1]
        km = LocalKMImputer(2, 2, 2).fit(x, times, events)
        self.assertIsNone(km.diagnostics_["conditional_residual_mass"][0])
        self.assertEqual(km.diagnostics_["fallback_subjects"], 1)
        np.testing.assert_allclose(np.exp(km.sample(np.zeros((3, 2))))[0], 6.6)
        with self.assertRaisesRegex(ValueError, "Empty conditional"):
            LocalKMImputer(2, 2, 2, "raise").fit(x, times, events)

    def test_sparse_events_fail_instead_of_all_censored_fit(self):
        for events in ([0, 0, 0], [0, 1, 0]):
            with (
                self.subTest(events=events),
                self.assertRaisesRegex(ValueError, "Insufficient"),
            ):
                LocalKMImputer(2, 2, 2).fit(np.zeros((3, 1)), [1.0, 2.0, 3.0], events)

    def test_fixed_seed_uniforms_are_reproducible(self):
        km = LocalKMImputer(2, 1, 2).fit(np.zeros((3, 1)), [1.0, 2.0, 3.0], [0, 1, 1])
        first = np.random.default_rng(19).random((3, 5))
        second = np.random.default_rng(19).random((3, 5))
        np.testing.assert_array_equal(km.sample(first), km.sample(second))

    def test_invalid_uniforms_configuration_and_outcomes(self):
        with self.assertRaises(RuntimeError):
            LocalKMImputer().sample(np.ones((2, 2)))
        for args in ((0, 1, 2), (2, 3, 3), (2, 1, 1), (2, 1, 2, "other")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                LocalKMImputer(*args)
        km = LocalKMImputer(2, 1, 2).fit(np.zeros((3, 1)), [1.0, 2.0, 3.0], [0, 1, 1])
        for uniforms in (
            np.zeros((2, 2)),
            np.full((3, 2), 1.0),
            np.full((3, 2), np.nan),
            np.full((3, 2), -0.1),
        ):
            with self.subTest(uniforms=uniforms), self.assertRaises(ValueError):
                km.sample(uniforms)
        for times, events in (
            ([0.0, 1.0, 2.0], [0, 1, 1]),
            ([1.0, 2.0, 3.0], [0, 2, 1]),
        ):
            with (
                self.subTest(times=times, events=events),
                self.assertRaises(ValueError),
            ):
                LocalKMImputer(2, 1, 2).fit(np.zeros((3, 1)), times, events)


class TestGRAFTSurvivalState(unittest.TestCase):
    def test_unfitted_calibration_and_query_support(self):
        with self.assertRaises(RuntimeError):
            _ready().predict_survival([[0.0, 0.0]], [1.0, 2.0])
        for grid in ([], [-1.0, 1.0], [2.0, 1.0], [1.0, 1.0], [float("nan")], [6.0]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                _calibrated().predict_survival([[0.0, 0.0]], grid)

    def test_centered_cox_survival_matches_hand_calculation(self):
        model = _calibrated()
        result = model.predict_survival([[1.0, 1.0], [2.0, 1.0]], [0.0, 2.0, 5.0])
        expected = np.exp(
            -np.exp(-np.array([-0.25, 0.75]))[:, None]
            * np.array([0.0, 0.2, 0.8])[None, :]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12)
        self.assertTrue(bool(((result >= 0) & (result <= 1)).all()))
        self.assertTrue(bool((np.diff(result, axis=1) <= 0).all()))

    def test_checkpoint_and_state_dict_need_no_dataset(self):
        model = _calibrated()
        x, grid = [[1.0, 1.0], [2.0, 1.0]], [0.0, 2.0, 5.0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graft.pt"
            model.save(path)
            restored = GRAFT.load(path)
            self.assertIsNone(restored.dataset)
            np.testing.assert_array_equal(
                restored.predict_survival(x, grid), model.predict_survival(x, grid)
            )
            second = GRAFT(**model.config)
            second.load_state_dict(model.state_dict())
            np.testing.assert_array_equal(
                second.predict_score(x), model.predict_score(x)
            )
        with self.assertRaises(ValueError):
            GRAFT(**dict(model.config, rank_tau=0.2)).load_state_dict(
                model.state_dict()
            )


@unittest.skipUnless(HAS_SURVIVAL, "Install optional lifelines and scikit-survival.")
class TestGRAFTSurvivalMetrics(unittest.TestCase):
    def test_brier_ties_nonuniform_grid_and_admin_censoring(self):
        from sksurv.metrics import brier_score
        from sksurv.util import Surv

        train = Surv.from_arrays([1, 1, 0, 1, 0], [1.0, 2.0, 2.0, 4.0, 5.0])
        data = {
            "features": [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]],
            "duration": [1.0, 2.0, 2.0, 4.2, 7.0],
            "event": [1, 1, 0, 1, 0],
        }
        grid, horizon = np.array([1.25, 2.0, 2.7, 3.8]), 4.5
        predictions = np.full((5, 4), 0.5)
        model = _calibrated()
        with patch.object(model, "predict_survival", return_value=predictions):
            result = model.evaluate_survival(data, grid, horizon)
        test = Surv.from_arrays([1, 1, 0, 1, 0], [1.0, 2.0, 2.0, 4.2, horizon])
        _, reference = brier_score(train, test, predictions, grid)
        np.testing.assert_allclose(result["brier"], reference)
        self.assertAlmostEqual(result["brier"][1], 0.275)
        self.assertAlmostEqual(
            result["ibs"], np.trapezoid(reference, grid) / (grid[-1] - grid[0])
        )
        self.assertEqual(result["administratively_censored"], 1)
        self.assertEqual(result["subjects"], 5)

    def test_unsupported_censor_weights_fail_without_clipping(self):
        data = {
            "features": [[0.0, 1.0], [2.0, 1.0]],
            "duration": [1.0, 6.0],
            "event": [1, 0],
        }
        model = _calibrated()
        with self.assertRaisesRegex(ValueError, "censor survival below"):
            model.evaluate_survival(data, [1.5, 3.0], 5.0)
        with self.assertRaisesRegex(ValueError, "strictly below"):
            model.evaluate_survival(data, [1.5, 3.0], 3.0)

    def test_calibration_fits_training_only_and_verifies_centering(self):
        from lifelines import CoxPHFitter
        import pandas as pd

        data = {
            "features": [[0.0, 0.0], [0.6, 0.0], [0.1, 0.0], [-0.3, 0.0], [0.2, 0.0]],
            "duration": [1.0, 2.0, 3.0, 4.0, 5.0],
            "event": [1, 0, 1, 1, 0],
        }
        model = _ready()
        x, t, e, _ = model._cohort(data)
        model.ranking_fitted_ = True
        model._training_fingerprint = _fingerprint(x, t, e)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit_calibration(data)
        scores = model.predict_score(x)
        frame = pd.DataFrame({"score": scores.astype(float), "duration": t, "event": e})
        cox = CoxPHFitter(penalizer=0).fit(frame, "duration", "event")
        state = model.calibration_
        expected = cox.predict_partial_hazard(frame).to_numpy()
        np.testing.assert_allclose(
            np.exp(state["coefficient"] * (scores - state["score_mean"])),
            expected,
            rtol=1e-6,
        )
        self.assertEqual(state["subjects"], 5)
        changed = dict(data, event=[0, 1, 1, 1, 0])
        with self.assertRaisesRegex(ValueError, "identical training"):
            model.fit_calibration(changed)
        self.assertIsNone(model.calibration_)


@unittest.skipUnless(
    HAS_RANK and HAS_SURVIVAL, "Install the GRAFT optional dependencies."
)
class TestGRAFTFittingBoundary(unittest.TestCase):
    def test_train_only_state_best_restore_and_rng_isolation(self):
        train = {
            "features": [[0.0, 0.0], [0.6, 0.0], [0.1, 0.0], [-0.3, 0.0], [0.2, 0.0]],
            "duration": [1.0, 2.0, 3.0, 4.0, 5.0],
            "event": [1, 0, 1, 1, 0],
            "patient_id": [f"p{i}" for i in range(5)],
        }
        validation = {
            "features": [[100.0, 100.0], [200.0, 200.0]],
            "duration": [2.0, 9.0],
            "event": [1, 1],
            "patient_id": ["v0", "v1"],
        }
        model = GRAFT(
            n_features=2,
            hidden_dim=2,
            min_neighbors=2,
            min_events=1,
            max_neighbors=4,
            dropout=0,
        )
        checkpoints = []

        def validation_score(*args):
            checkpoints.append(model.beta.detach().clone())
            return [0.8, 0.6, 0.7][len(checkpoints) - 1]

        real_km_fit = LocalKMImputer.fit
        seen_rows = []

        def km_fit(instance, x, times, events, row_ids=None):
            seen_rows.append((len(x), np.asarray(times).copy()))
            return real_km_fit(instance, x, times, events, row_ids)

        rng_before = torch.random.get_rng_state().clone()
        with (
            patch("lifelines.utils.concordance_index", side_effect=validation_score),
            patch.object(LocalKMImputer, "fit", km_fit),
        ):
            model.fit(train, validation, epochs=3, patience=2, batch_size=2)
        torch.testing.assert_close(rng_before, torch.random.get_rng_state())
        self.assertEqual(seen_rows[0][0], 5)
        np.testing.assert_array_equal(seen_rows[0][1], train["duration"])
        np.testing.assert_allclose(model.feature_mean.numpy(), [0.12, 0.0])
        torch.testing.assert_close(model.beta, checkpoints[0])
        self.assertFalse(torch.equal(checkpoints[0], checkpoints[-1]))
        self.assertEqual(model.training_info_["best_epoch"], 1)
        self.assertEqual(model.training_info_["stop_reason"], "patience")
        self.assertIsNone(model.calibration_)
        self.assertFalse(model.training)

    def test_invalid_outcomes_overlap_and_row_ids(self):
        model = GRAFT(n_features=2)
        train = {
            "features": [[0.0, 0.0], [1.0, 1.0]],
            "duration": [1.0, 2.0],
            "event": [1, 0],
            "patient_id": ["a", "b"],
        }
        validation = dict(train, patient_id=["v0", "v1"])
        for changed in (
            dict(train, duration=[0.0, 2.0]),
            dict(train, event=[1, 2]),
            dict(train, event=[0, 0]),
            dict(train, patient_id=["a", "a"]),
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                model.fit(changed, validation, epochs=1)
        with self.assertRaisesRegex(ValueError, "overlap"):
            model.fit(train, train, epochs=1)
        with self.assertRaisesRegex(ValueError, "increasing"):
            model.fit(train, validation, row_ids=[1, 0], epochs=1)


if __name__ == "__main__":
    unittest.main()
