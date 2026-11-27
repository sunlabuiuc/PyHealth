"""CovariateLabel's native procedures through CPBench's existing adapter API."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from examples.cpbench.synthetic_data import SyntheticDataset
from pyhealth.benchmarks import Evaluation
from pyhealth.benchmarks.cpbench import CPBench
from pyhealth.benchmarks.cpbench.adapters import ADAPTERS
from pyhealth.benchmarks.cpbench.benchmark import _seeded
from pyhealth.benchmarks.cpbench.shared.collection import concatenate_predictions
from pyhealth.benchmarks.cpbench.shared.evaluation_preparation import (
    prediction_cache_path,
    prepare_evaluation,
)
from pyhealth.calib import predictionset
from pyhealth.calib.predictionset import CovariateLabel
from pyhealth.datasets import get_dataloader
from pyhealth.metrics import multiclass_metrics_fn
from pyhealth.models import MLP
from pyhealth.trainer import Trainer
from tests.core.test_cpbench import evaluation

METRICS = ["accuracy", "set_size", "miscoverage_overall_ps"]


def source_density(embeddings):
    """Importable example of an already-fitted density function."""
    return np.ones(len(embeddings))


def target_density(embeddings):
    return np.exp(np.clip(embeddings[:, 0], -5, 5))


class RelabelledTestDataset(SyntheticDataset):
    """Change held-out labels after generating inputs, keeping all features fixed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for sample in self.samples:
            if sample["patient_id"].startswith("test-"):
                sample["label"] = (sample["label"] + 1) % 3


def covariate_evaluation(arguments=None, calibration=None, **kwargs):
    return evaluation(
        CovariateLabel,
        model_class=MLP,
        model_params={"embedding_dim": 4, "hidden_dim": 4},
        method_params=arguments or {"alpha": 0.2, "random_state": 42},
        options={"calibration_kwargs": calibration or {}},
        metrics=METRICS,
        **kwargs,
    )


def native_embeddings(model, dataset, batch_size):
    """Reference extraction without calling the adapter's collection helper."""
    records = []
    with torch.no_grad():
        for batch in get_dataloader(dataset, batch_size, shuffle=False):
            features = {key: batch[key] for key in model.feature_keys}
            records.append(
                model(**features, embed=True)["embed"].detach().cpu().numpy()
            )

    return np.concatenate(records)


def compare_native(test, ev):
    before = ev.to_dict()
    observed = []
    original = CovariateLabel.forward

    def record(predictor, **features):
        test.assertEqual(set(features), {"x"})
        output = original(predictor, **features)
        observed.append(output["y_predset"].detach().cpu().numpy().copy())

        return output

    with patch.object(CovariateLabel, "forward", record):
        result = CPBench().evaluate([Evaluation.from_dict(ev.to_dict())])[ev]

    test.assertEqual(result.status, "success", result.metadata)
    test.assertEqual(ev.to_dict(), before)

    with _seeded(ev.params["protocol"]["seed"]):
        params, splits, model = prepare_evaluation(ev, required=("calibration", "test"))
        native = CovariateLabel(model, **params["method"]["params"])
        calibration = copy.deepcopy(
            params["method"].get("options", {}).get("calibration_kwargs", {})
        )
        if calibration.get("cal_weights") is None:
            calibration["cal_embeddings"] = native_embeddings(
                model, splits["calibration"], 5
            )
            if params["method"]["params"].get("kde_cal") is None:
                calibration["test_embeddings"] = native_embeddings(
                    model, splits["test"], 5
                )

        native.calibrate(splits["calibration"], **calibration)

        records = []
        with torch.no_grad():
            for batch in get_dataloader(splits["test"], 5, shuffle=False):
                output = native(x=batch["x"])
                records.append(
                    {
                        "y_true": batch["label"].numpy(),
                        "y_prob": output["y_prob"].detach().cpu().numpy(),
                        "y_predset": output["y_predset"].detach().cpu().numpy(),
                    }
                )

        predictions = concatenate_predictions(records)

    np.testing.assert_array_equal(np.concatenate(observed), predictions["y_predset"])
    expected = multiclass_metrics_fn(**predictions, metrics=METRICS)
    for name, value in expected.items():
        test.assertAlmostEqual(result.metrics[name], value)

    test.assertEqual(result.metadata, {"class_labels": [0, 1, 2]})
    test.assertIsNone(result.artifact)


class TestCovariateAdapter(unittest.TestCase):
    def test_all_exported_prediction_set_classes_are_explicitly_supported(self):
        methods = {
            getattr(predictionset, name)
            for name in predictionset.__all__
            if isinstance(getattr(predictionset, name), type)
        }
        self.assertEqual(methods - ADAPTERS.keys(), set())

    def test_kde_training_matches_native(self):
        for arguments in (
            {"alpha": 0.2, "random_state": 42},
            {"alpha": 0.2, "score_type": "aps", "random_state": 7},
            {"alpha": [0.1, 0.2, 0.3], "score_type": "aps", "random_state": 7},
        ):
            with self.subTest(arguments=arguments):
                compare_native(
                    self, covariate_evaluation(arguments, training={"epochs": 1})
                )

    def test_prefitted_densities_match_native_without_extracting_test_embeddings(self):
        ev = covariate_evaluation(
            {
                "alpha": 0.2,
                "random_state": 42,
                "kde_cal": source_density,
                "kde_test": target_density,
            },
            training={"epochs": 1},
        )
        from pyhealth.benchmarks.cpbench.adapters import covariate_label

        with patch.object(
            covariate_label,
            "_extract_embeddings",
            wraps=covariate_label._extract_embeddings,
        ) as extract:
            compare_native(self, ev)

        self.assertEqual(extract.call_count, 1)
        self.assertEqual(len(extract.call_args.args[1]), 24)

    def test_custom_weights_match_native_without_an_embedding_model(self):
        # FixedModel deliberately has no embedding output.
        weights = [1.0, 2.0, 0.5] * 8
        ev = evaluation(
            CovariateLabel,
            method_params={"alpha": 0.2, "random_state": 42},
            options={"calibration_kwargs": {"cal_weights": weights}},
            metrics=METRICS,
        )
        with patch(
            "pyhealth.benchmarks.cpbench.adapters.covariate_label._extract_embeddings",
            side_effect=AssertionError("weights must bypass embeddings"),
        ):
            compare_native(self, ev)

    def test_embedding_calls_never_receive_labels(self):
        ev = covariate_evaluation(training={"epochs": 1})
        original = MLP.forward
        counts = []

        def check(model, **batch):
            if batch.get("embed"):
                self.assertEqual(set(batch), {"x", "embed"})
                self.assertFalse(torch.is_grad_enabled())
                counts.append(len(batch["x"]))

            return original(model, **batch)

        with patch.object(MLP, "forward", check):
            result = CPBench().evaluate([ev])[ev]

        self.assertEqual(result.status, "success", result.metadata)
        self.assertEqual(counts, [5, 5, 5, 5, 4, 5, 5, 2])

    def test_managed_checkpoint_kde_matches_fresh_training(self):
        ev = covariate_evaluation(training={"epochs": 1})
        expected = CPBench().evaluate([ev])[ev]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with _seeded(42):
                prepare_evaluation(
                    ev, required=("train", "calibration", "test"), cache_dir=cache
                )
            with patch.object(
                Trainer, "train", side_effect=AssertionError("must not train")
            ):
                actual = CPBench().evaluate([ev], cache_dir=cache)[ev]
            self.assertEqual(actual.status, "success", actual.metadata)
            np.testing.assert_equal(actual.metrics, expected.metrics)

    def test_test_labels_do_not_affect_density_estimation_or_prediction_sets(self):
        original = covariate_evaluation(training={"epochs": 1})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params = copy.deepcopy(original.params)
            params["dataset"]["class"] = RelabelledTestDataset
            changed = Evaluation(params)

            results = CPBench().evaluate(
                [original, changed],
                cache_dir=root / "cache",
                output_dir=root / "outputs",
            )
            for result in results.values():
                self.assertEqual(result.status, "success", result.metadata)

            with (
                np.load(results[original].artifact) as first,
                np.load(results[changed].artifact) as second,
            ):
                self.assertFalse(np.array_equal(first["y_true"], second["y_true"]))
                np.testing.assert_array_equal(first["y_prob"], second["y_prob"])
                np.testing.assert_array_equal(first["y_predset"], second["y_predset"])

    def test_managed_outputs_are_reused_without_retraining(self):
        ev = covariate_evaluation(training={"epochs": 1})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = CPBench().evaluate(
                [ev], cache_dir=root / "cache", output_dir=root / "output"
            )[ev]
            with patch.object(
                Trainer, "train", side_effect=AssertionError("must reuse cache")
            ):
                second = CPBench().evaluate(
                    [ev], cache_dir=root / "cache", output_dir=root / "output"
                )[ev]

            self.assertEqual(first.status, "success", first.metadata)
            self.assertEqual(second.status, "success", second.metadata)
            self.assertEqual(first.metrics, second.metrics)
            self.assertEqual(
                second.artifact, prediction_cache_path(ev.params, root / "cache")
            )
            with np.load(second.artifact) as saved:
                self.assertEqual(set(saved.files), {"y_true", "y_prob", "y_predset"})
                self.assertEqual(saved["y_predset"].shape, (12, 3))

    def test_unsupported_options_and_rules_fail_before_preparation(self):
        base = covariate_evaluation(training={"epochs": 1})
        invalid = []
        for name in ("cal_embeddings", "test_embeddings"):
            params = copy.deepcopy(base.params)
            params["method"]["options"] = {"calibration_kwargs": {name: [[0.0]]}}
            invalid.append(Evaluation(params))

        for key, value in (("order", "chronological"), ("feedback", "immediate")):
            params = copy.deepcopy(base.params)
            params["protocol"][key] = value
            invalid.append(Evaluation(params))

        params = copy.deepcopy(base.params)
        params["method"]["options"] = {"unknown": 1}
        invalid.append(Evaluation(params))
        params = copy.deepcopy(base.params)
        params["method"]["params"]["kde_cal"] = source_density
        invalid.append(Evaluation(params))

        for ev in invalid:
            with (
                self.subTest(params=ev.params),
                patch(
                    "pyhealth.benchmarks.cpbench.adapters.covariate_label.prepare_evaluation",
                    side_effect=AssertionError("must fail before preparation"),
                ) as prepare,
            ):
                result = CPBench().evaluate([ev])[ev]

            prepare.assert_not_called()
            self.assertEqual(result.status, "failed")
            self.assertIn(result.metadata["error_type"], ("TypeError", "ValueError"))

    def test_runtime_failures_are_isolated(self):
        # Default density estimation needs embeddings, whereas custom weights do not.
        no_embeddings = evaluation(CovariateLabel)
        wrong_weights = evaluation(
            CovariateLabel, options={"calibration_kwargs": {"cal_weights": [1.0]}}
        )
        good = evaluation()
        results = CPBench().evaluate([no_embeddings, wrong_weights, good])
        self.assertIn("embed=True", results[no_embeddings].metadata["error"])
        self.assertIn(
            "cal_weights must have length", results[wrong_weights].metadata["error"]
        )
        self.assertEqual(results[good].status, "success", results[good].metadata)


if __name__ == "__main__":
    unittest.main()
