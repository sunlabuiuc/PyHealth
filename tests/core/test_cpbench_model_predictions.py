"""Managed base-output caches and low-level artifact file utilities."""

import copy
import json
import shutil
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
from pyhealth.benchmarks.cpbench.shared.evaluation_preparation import prepare_evaluation
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


class TestSharedModelPredictions(unittest.TestCase):
    def test_adapter_can_supply_its_own_output_collection(self):
        ev = evaluation(dict)
        expected = ({"custom": {"values": np.array([1.0])}}, {"format": "custom"})
        calls = []

        def produce(evaluation, *, outputs, cache_dir):
            calls.append((evaluation, outputs))
            return expected

        adapter = SimpleNamespace(
            required_model_outputs=lambda evaluation: {"custom": ("values",)},
            predict_model_outputs=produce,
        )
        with (
            patch.dict(ADAPTERS, {dict: adapter}),
            patch(
                "pyhealth.benchmarks.cpbench.shared.evaluation_preparation.prepare_evaluation",
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
            cache = Path(directory) / "cache"
            values, metadata = bench.predict_model(
                evaluations[0], outputs=requirements, cache_dir=cache
            )
            self.assertEqual(
                set(values["train"]), {"y_true", "embed", "__cpbench_sample_hash__"}
            )
            self.assertTrue(all("y_predset" not in part for part in values.values()))
            save_model_predictions(values, path, **metadata)
            contents = path.read_bytes()
            expected = bench.evaluate(evaluations)
            cached = evaluations
            with (
                patch(
                    "pyhealth.benchmarks.cpbench.shared.evaluation_preparation.load_splits",
                    side_effect=AssertionError("must not load a dataset"),
                ),
                patch.object(
                    MLP, "forward", side_effect=AssertionError("must not infer")
                ),
                patch.object(
                    Trainer, "train", side_effect=AssertionError("must not train")
                ),
            ):
                actual = bench.evaluate(
                    cached, cache_dir=cache, output_dir=Path(directory) / "results"
                )
            self.assertEqual(contents, path.read_bytes())
        for live, stored in zip(evaluations, cached):
            self.assertEqual(expected[live].status, "success", expected[live].metadata)
            self.assertEqual(actual[stored].status, "success", actual[stored].metadata)
            np.testing.assert_equal(actual[stored].metrics, expected[live].metrics)
            self.assertIn("model_predictions", actual[stored].metadata)
            self.assertIsNotNone(actual[stored].artifact)
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
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            values, _ = bench.predict_model(evaluations[0], cache_dir=cache)
            self.assertEqual(set(values), {"calibration", "test"})
            self.assertEqual(
                set(values["calibration"]),
                {"y_true", "logit", "__cpbench_sample_hash__"},
            )
            expected = bench.evaluate(evaluations)
            cached = evaluations
            with patch.object(
                MLP, "forward", side_effect=AssertionError("no inference")
            ):
                actual = bench.evaluate(cached, cache_dir=cache)
        for live, stored in zip(evaluations, cached):
            self.assertEqual(actual[stored].status, "success", actual[stored].metadata)
            self.assertEqual(expected[live].status, "success", expected[live].metadata)
            np.testing.assert_equal(actual[stored].metrics, expected[live].metrics)

    def test_missing_embeddings_and_train_split_are_added_without_retraining(self):
        basic = evaluation(
            model_class=MLP,
            model_params={"embedding_dim": 4, "hidden_dim": 4},
            training={"epochs": 1},
        )
        params = copy.deepcopy(basic.params)
        params["method"] = {
            "class": ClusterLabel,
            "params": {"alpha": 0.2, "n_clusters": 2, "random_state": 42},
        }
        cluster = Evaluation(params)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            first = CPBench().evaluate([basic], cache_dir=cache)[basic]
            self.assertEqual(first.status, "success", first.metadata)
            base_dir = Path(first.metadata["model_predictions"])
            before, _ = load_model_predictions(base_dir / "test.npz")
            self.assertNotIn("embed", before["test"])
            self.assertFalse((base_dir / "train.npz").exists())
            original = MLP.forward
            calls = []

            def record(model, **batch):
                if not model.training:
                    calls.append(batch.get("embed", False))
                return original(model, **batch)

            with (
                patch.object(
                    Trainer, "train", side_effect=AssertionError("must not retrain")
                ),
                patch.object(MLP, "forward", record),
            ):
                result = CPBench().evaluate([cluster], cache_dir=cache)[cluster]
            self.assertEqual(result.status, "success", result.metadata)
            self.assertTrue(calls)
            self.assertTrue(all(calls))
            after, _ = load_model_predictions(base_dir / "test.npz")
            np.testing.assert_array_equal(
                before["test"]["y_prob"], after["test"]["y_prob"]
            )
            self.assertIn("embed", after["test"])
            self.assertTrue((base_dir / "train.npz").exists())

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
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            _, splits, model = prepare_evaluation(
                ev,
                required=("calibration", "test"),
                outputs=required_model_outputs(ev),
                cache_dir=cache,
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

    def test_failed_output_write_keeps_checkpoint_and_existing_predictions(self):
        basic = evaluation(
            model_class=MLP,
            model_params={"embedding_dim": 4, "hidden_dim": 4},
            training={"epochs": 1},
        )
        params = copy.deepcopy(basic.params)
        params["method"] = {
            "class": NeighborhoodLabel,
            "params": {"alpha": 0.2, "k_neighbors": 3, "random_state": 42},
        }
        neighbor = Evaluation(params)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            first = CPBench().evaluate([basic], cache_dir=cache)[basic]
            base_dir = Path(first.metadata["model_predictions"])
            before = {path: path.read_bytes() for path in base_dir.glob("*.npz")}
            with (
                patch.object(
                    Trainer, "train", side_effect=AssertionError("must not retrain")
                ),
                patch(
                    "pyhealth.benchmarks.cpbench.shared.model_outputs.np.savez_compressed",
                    side_effect=OSError("interrupted write"),
                ),
            ):
                failed = CPBench().evaluate([neighbor], cache_dir=cache)[neighbor]
            self.assertEqual(failed.status, "failed")
            self.assertEqual(failed.metadata["error_type"], "OSError")
            self.assertEqual(
                before, {path: path.read_bytes() for path in base_dir.glob("*.npz")}
            )
            self.assertEqual(set(base_dir.iterdir()), set(before))
            with patch.object(
                Trainer, "train", side_effect=AssertionError("must reuse checkpoint")
            ):
                recovered = CPBench().evaluate([neighbor], cache_dir=cache)[neighbor]
            self.assertEqual(recovered.status, "success", recovered.metadata)

    def test_failed_checkpoint_write_can_be_retried(self):
        ev = evaluation(model_class=MLP, training={"epochs": 0})
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch.object(
                Trainer, "save_ckpt", side_effect=OSError("interrupted checkpoint")
            ):
                failed = CPBench().evaluate([ev], cache_dir=cache)[ev]
            self.assertEqual(failed.status, "failed")
            self.assertFalse(
                any(
                    path.is_dir()
                    for path in (cache / "cpbench" / "checkpoints").iterdir()
                )
            )
            recovered = CPBench().evaluate([ev], cache_dir=cache)[ev]
            self.assertEqual(recovered.status, "success", recovered.metadata)

    def test_new_outputs_cannot_be_joined_to_reordered_samples_with_identical_labels(
        self,
    ):
        from pyhealth.benchmarks.cpbench.shared import evaluation_preparation

        class Reordered(torch.utils.data.Dataset):
            def __init__(self, source):
                self.source = source
                self.output_processors = source.output_processors

            def __getattr__(self, name):
                return getattr(self.source, name)

            def __len__(self):
                return len(self.source)

            def __getitem__(self, index):
                # Class labels repeat every three rows. Swapping these rows
                # preserves y_true but changes which input receives an embedding.
                if index == 0:
                    index = 3
                elif index == 3:
                    index = 0
                return self.source[index]

        basic = evaluation(model_class=MLP, training={"epochs": 0})
        params = copy.deepcopy(basic.params)
        params["method"] = {
            "class": NeighborhoodLabel,
            "params": {"alpha": 0.2, "k_neighbors": 3},
        }
        neighbor = Evaluation(params)
        original_load = evaluation_preparation.load_splits

        def reordered(*args, **kwargs):
            return {
                role: Reordered(split)
                for role, split in original_load(*args, **kwargs).items()
            }

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            first = CPBench().evaluate([basic], cache_dir=cache)[basic]
            base_dir = Path(first.metadata["model_predictions"])
            before = {path: path.read_bytes() for path in base_dir.glob("*.npz")}
            with patch.object(
                evaluation_preparation, "load_splits", side_effect=reordered
            ):
                failed = CPBench().evaluate([neighbor], cache_dir=cache)[neighbor]
            self.assertEqual(failed.status, "failed")
            self.assertIn("sample ordering", failed.metadata["error"])
            self.assertEqual(
                before, {path: path.read_bytes() for path in base_dir.glob("*.npz")}
            )

    def test_model_inputs_change_cache_keys_but_storage_locations_do_not_change_ids(
        self,
    ):
        base = evaluation(model_class=MLP, training={"epochs": 0})
        variants = [base]
        for component, field, value in (
            ("model", "training", {"epochs": 0, "weight_decay": 0.1}),
            ("model", "params", {"embedding_dim": 4}),
            ("dataset", "params", {**base.params["dataset"]["params"], "seed": 1}),
            ("protocol", "batch_size", 6),
        ):
            params = copy.deepcopy(base.params)
            params[component][field] = value
            variants.append(Evaluation(params))
        params = copy.deepcopy(base.params)
        groups = params["protocol"]["split"]["params"]["subjects"]
        groups[0][0], groups[0][1] = groups[0][1], groups[0][0]
        variants.append(Evaluation(params))
        snapshots = [ev.to_dict() for ev in variants]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = CPBench().evaluate(variants, cache_dir=root / "first")
            self.assertTrue(
                all(result.status == "success" for result in first.values()), first
            )
            checkpoints = [
                path
                for path in (root / "first" / "cpbench" / "checkpoints").iterdir()
                if path.is_dir()
            ]
            self.assertEqual(len(checkpoints), len(variants))
            second = CPBench().evaluate([base], cache_dir=root / "second")[base]
            self.assertEqual(second.status, "success", second.metadata)
            np.testing.assert_equal(first[base].metrics, second.metrics)
        self.assertEqual(snapshots, [ev.to_dict() for ev in variants])

    def test_without_cache_models_are_trained_for_each_evaluation(self):
        ev = evaluation(model_class=MLP, training={"epochs": 0})
        original_train = Trainer.train
        with (
            patch.object(
                Trainer, "train", autospec=True, side_effect=original_train
            ) as train,
            patch(
                "pyhealth.benchmarks.cpbench.shared.evaluation_preparation.save_checkpoint",
                side_effect=AssertionError("must not persist"),
            ),
            patch(
                "pyhealth.benchmarks.cpbench.shared.evaluation_preparation.save_model_predictions",
                side_effect=AssertionError("must not persist"),
            ),
        ):
            first = CPBench().evaluate([ev])[ev]
            second = CPBench().evaluate([ev])[ev]
        self.assertEqual(train.call_count, 2)
        self.assertEqual(first.status, "success", first.metadata)
        self.assertEqual(second.status, "success", second.metadata)
        np.testing.assert_equal(first.metrics, second.metrics)

    def test_missing_checkpoint_does_not_mix_old_outputs_with_a_new_model(self):
        from pyhealth.benchmarks.cpbench.shared import evaluation_preparation

        basic = evaluation(model_class=MLP, training={"epochs": 0})
        params = copy.deepcopy(basic.params)
        params["method"] = {
            "class": NeighborhoodLabel,
            "params": {"alpha": 0.2, "k_neighbors": 3},
        }
        neighbor = Evaluation(params)
        original_prepare = evaluation_preparation.prepare_model

        def changed_model(*args, **kwargs):
            model = original_prepare(*args, **kwargs)
            with torch.no_grad():
                model.fc.bias[0].add_(5)
            return model

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            first = CPBench().evaluate([basic], cache_dir=cache)[basic]
            base_dir = Path(first.metadata["model_predictions"])
            before, _ = load_model_predictions(base_dir / "test.npz")
            checkpoint = next(
                path
                for path in (cache / "cpbench" / "checkpoints").iterdir()
                if path.is_dir()
            )
            shutil.rmtree(checkpoint)
            # Complete saved outputs remain usable without the weights.
            with patch.object(
                Trainer, "train", side_effect=AssertionError("must not train")
            ):
                warm = CPBench().evaluate([basic], cache_dir=cache)[basic]
            self.assertEqual(warm.status, "success", warm.metadata)
            # Missing fields require new weights. All outputs must then come
            # from that new model, including previously cached probabilities.
            with patch.object(
                evaluation_preparation, "prepare_model", side_effect=changed_model
            ):
                result = CPBench().evaluate([neighbor], cache_dir=cache)[neighbor]
            self.assertEqual(result.status, "success", result.metadata)
            after, _ = load_model_predictions(base_dir / "test.npz")
            self.assertFalse(
                np.allclose(before["test"]["y_prob"], after["test"]["y_prob"])
            )
            self.assertIn("embed", after["test"])

    def test_base_output_collection_handles_temporal_inputs(self):
        from pyhealth.benchmarks.cpbench.shared.model_outputs import (
            collect_model_predictions,
        )

        class TemporalModel:
            feature_keys, label_keys = ["x"], ["label"]

            def __call__(self, x, embed=False):
                _, values = x
                return {"embed": values} if embed else {"y_prob": values.softmax(dim=1)}

        samples = [
            {"x": (None, torch.tensor([1.0, 2.0])), "label": torch.tensor(0)},
            {"x": (None, torch.tensor([3.0, 4.0])), "label": torch.tensor(1)},
        ]
        probabilities = collect_model_predictions(
            TemporalModel(), {"test": samples}, {"test": ("y_prob",)}, 2
        )
        embeddings = collect_model_predictions(
            TemporalModel(), {"test": samples}, {"test": ("embed",)}, 2
        )
        np.testing.assert_array_equal(
            probabilities["test"]["__cpbench_sample_hash__"],
            embeddings["test"]["__cpbench_sample_hash__"],
        )
        np.testing.assert_array_equal(
            embeddings["test"]["embed"], [[1.0, 2.0], [3.0, 4.0]]
        )

    def test_checkpoint_creation_is_locked_across_concurrent_calls(self):
        from concurrent.futures import ThreadPoolExecutor

        ev = evaluation()
        original_train = Trainer.train
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)

            def prepare():
                return prepare_evaluation(ev, required=("test",), cache_dir=cache)

            with (
                patch.object(
                    Trainer, "train", autospec=True, side_effect=original_train
                ) as train,
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                results = list(executor.map(lambda _: prepare(), range(2)))
            self.assertEqual(train.call_count, 1)
            self.assertIsNot(results[0][2], results[1][2])


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
