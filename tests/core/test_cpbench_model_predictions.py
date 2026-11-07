"""Explicit base-output artifacts reused by CP methods without cache policy."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pyhealth.benchmarks import Evaluation
from pyhealth.benchmarks.cpbench import (
    CPBench,
    load_model_predictions,
    required_model_outputs,
    save_model_predictions,
    save_predictions,
)
from pyhealth.benchmarks.cpbench.adapters import ADAPTERS
from pyhealth.benchmarks.cpbench.shared.preparation import prepare_evaluation
from pyhealth.calib.predictionset import (
    LABEL,
    RC3P,
    SCRIB,
    BaseConformal,
    ClusterLabel,
    CovariateLabel,
    FavMac,
    LabelShift,
    NeighborhoodLabel,
)
from pyhealth.models import MLP
from pyhealth.trainer import Trainer
from tests.core.test_cpbench import evaluation
from tests.core.test_cpbench_covariate import source_density, target_density


def saved_source(evaluation, path):
    params = copy.deepcopy(evaluation.params)
    params["model"] = {"predictions": str(path)}

    return Evaluation(params)


class TestSharedModelPredictions(unittest.TestCase):
    def test_adapter_can_supply_its_own_output_collection(self):
        ev = evaluation(dict)
        expected = ({"custom": {"values": np.array([1.0])}}, {"format": "custom"})
        calls = []

        def produce(evaluation, *, outputs):
            calls.append((evaluation, outputs))
            return expected

        adapter = SimpleNamespace(
            required_model_outputs=lambda evaluation: {"custom": ("values",)},
            predict_model_outputs=produce,
        )
        with (
            patch.dict(ADAPTERS, {dict: adapter}),
            patch(
                "pyhealth.benchmarks.cpbench.shared.preparation.prepare_evaluation",
                side_effect=AssertionError("must use the adapter's producer"),
            ),
        ):
            actual = CPBench().predict_model(ev)

        self.assertIs(actual, expected)
        self.assertEqual(calls, [(ev, {"custom": ("values",)})])

    def test_one_artifact_supplies_all_multiclass_methods(self):
        methods = (
            (BaseConformal, {"alpha": 0.2, "random_state": 42}),
            (BaseConformal, {"alpha": 0.9, "random_state": 42}),
            (LABEL, {"alpha": [0.1, 0.2, 0.3], "random_state": 42}),
            (RC3P, {"alpha": 0.2, "random_state": 42}),
            (SCRIB, {"risk": 0.2}),
            (LabelShift, {"alpha": 0.2, "random_state": 42}),
            (NeighborhoodLabel, {"alpha": 0.2, "k_neighbors": 3, "random_state": 42}),
            (ClusterLabel, {"alpha": 0.2, "n_clusters": 2, "random_state": 42}),
            (CovariateLabel, {"alpha": 0.2, "random_state": 42}),
        )
        evaluations = [
            evaluation(
                method,
                method_params=params,
                model_class=MLP,
                model_params={"embedding_dim": 4, "hidden_dim": 4},
                training={"epochs": 1},
                options={"calibration_kwargs": {"target_label_dist": [0.2, 0.3, 0.5]}}
                if method is LabelShift
                else None,
            )
            for method, params in methods
        ]
        requirements = {}
        for ev in evaluations:
            for role, fields in required_model_outputs(ev).items():
                requirements.setdefault(role, set()).update(fields)
        requirements = {
            role: tuple(sorted(fields)) for role, fields in requirements.items()
        }
        before = [ev.to_dict() for ev in evaluations]
        bench = CPBench()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.npz"
            values, metadata = bench.predict_model(evaluations[0], outputs=requirements)
            self.assertEqual(set(values["train"]), {"y_true", "embed"})
            self.assertTrue(all("y_predset" not in part for part in values.values()))
            save_model_predictions(values, path, **metadata)
            contents = path.read_bytes()
            expected = bench.evaluate(evaluations)
            cached = [saved_source(ev, path) for ev in evaluations]
            with (
                patch(
                    "pyhealth.benchmarks.cpbench.shared.preparation.load_splits",
                    side_effect=AssertionError("must not load a dataset"),
                ),
                patch.object(
                    MLP, "forward", side_effect=AssertionError("must not infer")
                ),
                patch.object(
                    Trainer, "train", side_effect=AssertionError("must not train")
                ),
            ):
                actual = bench.evaluate(cached, output_dir=Path(directory) / "results")
            self.assertEqual(contents, path.read_bytes())
        for live, stored in zip(evaluations, cached):
            self.assertEqual(expected[live].status, "success", expected[live].metadata)
            self.assertEqual(actual[stored].status, "success", actual[stored].metadata)
            np.testing.assert_equal(actual[stored].metrics, expected[live].metrics)
            self.assertEqual(actual[stored].metadata["model_predictions"], str(path))
            self.assertIsNone(actual[stored].artifact)
        self.assertEqual(before, [ev.to_dict() for ev in evaluations])
        self.assertNotEqual(
            actual[cached[0]].metrics["set_size"], actual[cached[1]].metrics["set_size"]
        )

    def test_saved_logits_reuse_favmac_with_different_cost_settings(self):
        evaluations = [
            evaluation(
                FavMac,
                mode="multilabel",
                model_class=MLP,
                model_params={"embedding_dim": 4, "hidden_dim": 4},
                training={"epochs": 1},
                method_params=params,
                metrics=[
                    "fp_cost",
                    "tp_value",
                    "fp_cost_violation_rate",
                    "miscoverage_ps",
                    "miscoverage_overall_ps",
                ],
            )
            for params in (
                {"target_cost": 0.8},
                {
                    "target_cost": 1.5,
                    "cost_weights": [2, 3, 5],
                    "value_weights": [7, 11, 13],
                },
                {"target_cost": 0.8, "delta": 0.2},
            )
        ]
        bench = CPBench()
        values, metadata = bench.predict_model(evaluations[0])
        self.assertEqual(set(values), {"calibration", "test"})
        self.assertEqual(set(values["calibration"]), {"y_true", "logit"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.npz"
            save_model_predictions(values, path, **metadata)
            expected = bench.evaluate(evaluations)
            cached = [saved_source(ev, path) for ev in evaluations]
            with patch.object(
                MLP, "forward", side_effect=AssertionError("no inference")
            ):
                actual = bench.evaluate(cached)
        for live, stored in zip(evaluations, cached):
            self.assertEqual(actual[stored].status, "success", actual[stored].metadata)
            self.assertEqual(expected[live].status, "success", expected[live].metadata)
            np.testing.assert_equal(actual[stored].metrics, expected[live].metrics)

    def test_missing_splits_and_fields_are_rejected_before_calibration(self):
        producer = evaluation()
        values, metadata = CPBench().predict_model(producer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.npz"
            save_model_predictions(values, path, **metadata)
            for method, options, message in (
                (ClusterLabel, {"alpha": 0.2, "n_clusters": 2}, "missing splits"),
                (NeighborhoodLabel, {"alpha": 0.2, "k_neighbors": 3}, "embed"),
            ):
                ev = saved_source(evaluation(method, method_params=options), path)
                # Cluster's source must declare its train role even though the file lacks it.
                if method is ClusterLabel:
                    params = copy.deepcopy(ev.params)
                    params["protocol"]["split"]["roles"].insert(0, "train")
                    ev = Evaluation(params)
                with (
                    self.subTest(method=method),
                    patch.object(
                        method,
                        "calibrate",
                        side_effect=AssertionError("must not calibrate"),
                    ),
                ):
                    result = CPBench().evaluate([ev])[ev]
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.metadata["error_type"], "ValueError")
                self.assertIn(message, result.metadata["error"])

    def test_requirements_depend_on_native_options(self):
        ev = evaluation(
            CovariateLabel,
            method_params={"alpha": 0.2},
            options={"calibration_kwargs": {"cal_weights": [1.0] * 24}},
        )
        self.assertEqual(
            required_model_outputs(ev),
            {
                "calibration": ("y_prob",),
                "test": ("y_prob",),
            },
        )
        ev = evaluation(
            CovariateLabel,
            method_params={
                "alpha": 0.2,
                "kde_cal": source_density,
                "kde_test": target_density,
            },
        )
        self.assertEqual(
            required_model_outputs(ev),
            {
                "calibration": ("y_prob", "embed"),
                "test": ("y_prob",),
            },
        )

    def test_cached_features_do_not_expose_test_targets_or_reuse_mutable_tensors(self):
        ev = evaluation()
        values, metadata = CPBench().predict_model(ev)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.npz"
            save_model_predictions(values, path, **metadata)
            _, splits, model = prepare_evaluation(
                saved_source(ev, path),
                required=("calibration", "test"),
                outputs=required_model_outputs(ev),
            )
            sample = splits["test"][0]
            features = {
                key: [sample[key]]
                if isinstance(sample[key], str)
                else sample[key].reshape(1)
                for key in model.feature_keys
            }
            output = model(**features)
            self.assertNotIn("y_true", output)
            expected = output["y_prob"].clone()
            output["y_prob"].fill_(0)
            torch.testing.assert_close(model(**features)["y_prob"], expected)
            self.assertIn(
                "y_true", model(**features, **{model.label_keys[0]: torch.tensor([0])})
            )


class TestModelPredictionFiles(unittest.TestCase):
    def test_round_trip_has_label_order_and_owned_numeric_arrays(self):
        values = {
            "calibration": {
                "y_true": np.array([0, 1]),
                "y_prob": np.array([[0.8, 0.2], [0.1, 0.9]]),
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.npz"
            save_model_predictions(
                values,
                path,
                mode="multiclass",
                label_key="outcome",
                class_labels=["negative", "positive"],
            )
            loaded, metadata = load_model_predictions(path)
            self.assertEqual(
                metadata,
                {
                    "mode": "multiclass",
                    "label_key": "outcome",
                    "class_labels": ["negative", "positive"],
                },
            )
            np.testing.assert_equal(loaded, values)
            loaded["calibration"]["y_prob"][0, 0] = 0
            again, _ = load_model_predictions(path)
            np.testing.assert_equal(again, values)

    def test_final_cp_predictions_cannot_be_mistaken_for_base_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cp.npz"
            save_predictions(
                {"y_true": np.array([0]), "y_predset": np.array([[1, 0]])}, path
            )
            with self.assertRaisesRegex(ValueError, "base-model prediction artifact"):
                load_model_predictions(path)

    def test_invalid_arrays_do_not_replace_an_existing_artifact(self):
        values = {"test": {"y_true": np.array([0]), "y_prob": np.array([[0.8, 0.2]])}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.npz"
            metadata = {
                "mode": "multiclass",
                "label_key": "label",
                "class_labels": [0, 1],
            }
            save_model_predictions(values, path, **metadata)
            before = path.read_bytes()
            for probabilities in (
                np.zeros((2, 2)),
                np.zeros((1, 3)),
                np.array([[np.nan, 0]]),
                np.array([["bad", "data"]]),
            ):
                with (
                    self.subTest(probabilities=probabilities),
                    self.assertRaises(ValueError),
                ):
                    save_model_predictions(
                        {"test": {"y_true": np.array([0]), "y_prob": probabilities}},
                        path,
                        **metadata,
                    )
                self.assertEqual(before, path.read_bytes())

    def test_object_arrays_are_never_unpickled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.npz"
            np.savez(
                path,
                __metadata__=np.asarray(
                    json.dumps(
                        {
                            "mode": "multiclass",
                            "label_key": "label",
                            "class_labels": [0, 1],
                        }
                    )
                ),
                **{
                    "test/y_true": np.array([0]),
                    "test/y_prob": np.array([[object(), object()]], dtype=object),
                },
            )
            with self.assertRaisesRegex(ValueError, "allow_pickle=False"):
                load_model_predictions(path)


if __name__ == "__main__":
    unittest.main()
