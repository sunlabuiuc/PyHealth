"""Structured CPBench evaluations, native equivalence and model-source behavior."""

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask, make_splits
from pyhealth.benchmarks import Evaluation, Result
from pyhealth.benchmarks.cpbench import CPBench
from pyhealth.benchmarks.cpbench.adapters import ADAPTERS
from pyhealth.benchmarks.cpbench.adapters.favmac import validate_multilabel_predictions
from pyhealth.benchmarks.cpbench.adapters.fixed_calibration import (
    run as run_fixed_calibration,
)
from pyhealth.benchmarks.cpbench.benchmark import _seeded, validate_evaluation
from pyhealth.benchmarks.cpbench.shared.classification import (
    predict_classification_set_batch,
    score_multiclass_predictions,
    validate_label_encodings,
    validate_multiclass_predictions,
)
from pyhealth.benchmarks.cpbench.shared.prediction import (
    collect_predictions,
    concatenate_predictions,
    copy_predictions,
    prediction_cache_path,
)
from pyhealth.benchmarks.cpbench.shared.preparation import (
    model_cache_path,
    prepare_evaluation,
    require_protocol,
)
from pyhealth.calib.predictionset import (
    LABEL,
    RC3P,
    SCRIB,
    BaseConformal,
    ClusterLabel,
    FavMac,
    LabelShift,
    NeighborhoodLabel,
)
from pyhealth.datasets import (
    BaseDataset,
    PatientSplit,
    SampleBuilder,
    SubjectSplit,
    get_dataloader,
)
from pyhealth.metrics import multiclass_metrics_fn, multilabel_metrics_fn
from pyhealth.models import MLP, BaseModel
from pyhealth.tasks import BaseTask
from pyhealth.trainer import Trainer


class FixedModel(BaseModel):
    def forward(self, **batch):
        logits = batch["x"].to(self.device)
        probabilities = (
            logits.sigmoid() if self.mode == "multilabel" else logits.softmax(dim=1)
        )
        output = {"y_prob": probabilities, "logit": logits}
        if "label" in batch:
            output["y_true"] = batch["label"].to(self.device)
        return output


class CalibrationWithOptions(torch.nn.Module):
    def __init__(self, classifier, *, alpha, tag):
        super().__init__()
        self.predictor = BaseConformal(classifier, alpha=alpha)
        self.tag = tag

    def calibrate(self, dataset, *, expected_tag):
        if expected_tag != self.tag:
            raise ValueError("calibration argument was not passed through")
        self.predictor.calibrate(dataset)

    def forward(self, **features):
        return self.predictor(**features)


def method_factory(model, **kwargs):
    return CalibrationWithOptions(model, **kwargs)


def broken_model(dataset):
    random.random()
    np.random.rand()
    torch.rand(3)
    raise RuntimeError("model creation failed")


class EventTask(BaseTask):
    task_name = "cpbench_events"
    input_schema = {"x": "tensor"}
    output_schema = {"label": "multiclass"}

    def __call__(self, patient):
        return [
            {
                "patient_id": patient.patient_id,
                "record_id": patient.patient_id,
                "x": [float(event.value), 0.0, 1.0],
                "label": int(event.label),
            }
            for event in patient.get_events("events")
        ]


class FeedbackMethod:
    def __init__(self, model):
        self.model, self.last_label, self.pending = model, None, None

    def predict(self, **features):
        if self.pending is not None:
            raise RuntimeError("feedback must precede the next prediction")
        probabilities = self.model(**features)["y_prob"]
        sets = torch.ones_like(probabilities, dtype=torch.bool)
        if self.last_label is not None:
            sets.zero_()
            sets[:, self.last_label] = True
        self.pending = {"y_prob": probabilities, "y_predset": sets}
        return self.pending

    def update(self, label):
        self.last_label = int(label.item())
        # Deliberately reuse buffers: recorded predictions must remain snapshots.
        self.pending["y_prob"].zero_()
        self.pending["y_predset"].zero_()
        self.pending = None


def feedback_adapter(evaluation, cache_dir):
    require_protocol(
        evaluation.params["protocol"], order="dataset", feedback="immediate"
    )
    params, splits, model = prepare_evaluation(evaluation, required=("stream",))
    if model.mode != "multiclass":
        raise ValueError("feedback test requires a multiclass model")
    vocabulary = validate_label_encodings(model, splits)
    method = params["method"]["class"](model, **params["method"].get("params", {}))
    records = []
    for batch in get_dataloader(splits["stream"], batch_size=1):
        records.append(
            copy_predictions(
                predict_classification_set_batch(
                    method.predict, batch, model.feature_keys, "label"
                )
            )
        )
        method.update(batch["label"])
    predictions = concatenate_predictions(records)
    validate_multiclass_predictions(predictions, len(vocabulary))
    return score_multiclass_predictions(predictions, params["metrics"], vocabulary)


def directory_adapter(evaluation, cache_dir):
    return Result("success", {}, {"id": evaluation.id, "cache": cache_dir})


def broken_adapter(evaluation, cache_dir):
    raise RuntimeError("adapter failed")


def invalid_adapter(evaluation, cache_dir):
    return {"not": "a Result"}


# Real native checkpoint files, shared by immutable test specifications.
_CHECKPOINT_DIRECTORY = tempfile.TemporaryDirectory(prefix="cpbench-checkpoints-")
_CHECKPOINTS = {}


def write_checkpoint(path, model, train_data):
    path.mkdir(parents=True, exist_ok=True)
    Trainer(model, enable_logging=False).save_ckpt(str(path / "model.ckpt"))
    builder = SampleBuilder(
        train_data.input_schema,
        train_data.output_schema,
        input_processors=train_data.input_processors,
        output_processors=train_data.output_processors,
    )
    builder.fit(train_data)
    builder.save(str(path / "schema.pkl"))


def checkpoint_for(mode):
    if mode not in _CHECKPOINTS:
        train = make_splits(mode=mode)["train"]
        path = Path(_CHECKPOINT_DIRECTORY.name) / mode
        write_checkpoint(path, FixedModel(train), train)
        _CHECKPOINTS[mode] = str(path)
    return _CHECKPOINTS[mode]


def evaluation(
    method=BaseConformal,
    *,
    mode="multiclass",
    method_params=None,
    options=None,
    dataset_params=None,
    model_class=None,
    model_params=None,
    training=None,
    checkpoint=None,
    metrics=None,
):
    data = {
        "n_train": 18,
        "n_calibration": 36 if mode == "multilabel" else 24,
        "n_test": 12,
        "mode": mode,
        **(dataset_params or {}),
    }
    model = {"class": model_class or FixedModel, "params": model_params or {}}
    if training is None:
        model["checkpoint"] = checkpoint or checkpoint_for(mode)
        roles = ["calibration", "test"]
    else:
        model["training"] = training
        roles = ["train", "calibration", "test"]
    if method_params is None:
        method_params = (
            {"target_cost": 0.8}
            if method is FavMac
            else {"alpha": 0.2, "random_state": 42}
        )
    return Evaluation(
        {
            "dataset": {"class": SyntheticDataset, "params": data},
            "task": {"class": SyntheticTask, "params": {"mode": mode}},
            "model": model,
            "method": {
                "class": method,
                "params": method_params,
                "options": options or {},
            },
            "metrics": metrics
            or (
                ["tp", "fp", "roc_auc_micro"]
                if mode == "multilabel"
                else [
                    "accuracy",
                    "set_size",
                    "miscoverage_overall_ps",
                    "miscoverage_ps",
                    "error_overall_ps",
                    "error_ps",
                ]
            ),
            "protocol": {
                "split": {
                    "class": SubjectSplit,
                    "params": {
                        "subjects": [
                            [f"{role}-{i}" for i in range(data[f"n_{role}"])]
                            for role in roles
                        ]
                    },
                    "roles": roles,
                },
                "seed": 42,
                "batch_size": 5,
                "device": "cpu",
                "order": "dataset",
                "feedback": "none",
            },
        }
    )


class TestEvaluationContract(unittest.TestCase):
    def test_serialization_round_trip(self):
        for ev in (evaluation(), evaluation(FavMac, mode="multilabel")):
            restored = Evaluation.from_dict(json.loads(json.dumps(ev.to_dict())))
            self.assertEqual(restored, ev)
            self.assertIs(
                restored.params["method"]["class"], ev.params["method"]["class"]
            )
            self.assertEqual(
                set(restored.params),
                {"dataset", "task", "model", "method", "metrics", "protocol"},
            )
            validate_evaluation(**restored.params)

    def test_invalid_schema_rejects_entire_batch_before_dispatch(self):
        good = evaluation()
        invalid = []
        for path, value in (
            (("dataset",), None),
            (("task", "class"), "SyntheticTask"),
            (("method", "params"), []),
            (("method", "options"), None),
            (("model", "params"), None),
            (("model", "training"), {}),  # Both checkpoint and training.
            (("model", "checkpoint"), None),
            (("metrics",), []),
            (("metrics",), [42]),
            (("protocol", "batch_size"), True),
            (("protocol", "seed"), -1),
            (("protocol", "seeds"), [0, 1]),
            (("protocol", "split", "roles"), ["test", "test"]),
        ):
            params = copy.deepcopy(good.params)
            target = params
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            invalid.append(Evaluation(params))
        for key in ("metrics", "task"):
            params = copy.deepcopy(good.params)
            del params[key]
            invalid.append(Evaluation(params))
        params = copy.deepcopy(good.params)
        del params["model"]["checkpoint"]
        invalid.append(Evaluation(params))
        for bad in invalid:
            with (
                self.subTest(params=bad.params),
                patch("pyhealth.benchmarks.cpbench.benchmark.adapters.run") as run,
                self.assertRaises((TypeError, ValueError)),
            ):
                # Even a valid evaluation preceding the bad one must not run.
                CPBench().evaluate([good, bad])
            run.assert_not_called()

    def test_unknown_method_and_no_inheritance_guessing(self):
        ev = evaluation(LabelShift, method_params={"alpha": 0.2})
        with (
            patch.dict(ADAPTERS),
            patch("pyhealth.benchmarks.cpbench.benchmark.adapters.run") as run,
            self.assertRaisesRegex(ValueError, "No CPBench adapter"),
        ):
            # Removing its explicit mapping must not select BaseConformal's adapter.
            del ADAPTERS[LabelShift]
            CPBench().evaluate([evaluation(), ev])
        run.assert_not_called()

    def test_adapter_receives_evaluation_and_cache_while_results_save_centrally(self):
        ev = evaluation(FeedbackMethod, method_params={})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(ADAPTERS, {FeedbackMethod: directory_adapter}):
                result = CPBench().evaluate(
                    [ev], cache_dir=root / "cache", output_dir=root / "output"
                )[ev]

            self.assertEqual(result.metadata, {"id": ev.id, "cache": root / "cache"})
            saved = json.loads((root / "output" / "results.json").read_text())
            self.assertEqual(saved[ev.id]["metadata"]["cache"], str(root / "cache"))

    def test_runtime_failures_do_not_discard_other_results(self):
        good = evaluation()
        bad = evaluation(FeedbackMethod, method_params={})
        with patch.dict(ADAPTERS, {FeedbackMethod: broken_adapter}):
            results = CPBench().evaluate([bad, good])
        self.assertEqual(results[bad].status, "failed")
        self.assertEqual(results[bad].metadata["error_type"], "RuntimeError")
        self.assertEqual(results[good].status, "success", results[good].metadata)

    def test_invalid_adapter_return_is_rejected_by_result_validation(self):
        ev = evaluation(FeedbackMethod, method_params={})
        with (
            patch.dict(ADAPTERS, {FeedbackMethod: invalid_adapter}),
            self.assertRaisesRegex(TypeError, "adapter must return a Result"),
        ):
            CPBench().evaluate([ev])

    def test_result_status_and_metrics_are_checked(self):
        ev = evaluation()
        for result, error, message in (
            (Result("pending", {}), ValueError, "invalid result status"),
            (Result("success", []), TypeError, "metrics must be a mapping"),
            (Result("failed", None), TypeError, "metrics must be a mapping"),
        ):
            with (
                self.subTest(result=result),
                patch(
                    "pyhealth.benchmarks.cpbench.benchmark.adapters.run",
                    return_value=result,
                ),
                self.assertRaisesRegex(error, message),
            ):
                CPBench().evaluate([ev])

    def test_result_validation_preserves_base_correspondence_checks(self):
        ev = evaluation()
        other = evaluation(method_params={"alpha": 0.3})
        for results, message in (
            ({}, "missing result"),
            (
                {ev: Result("success", {}), other: Result("success", {})},
                "unexpected result",
            ),
        ):
            with (
                self.subTest(results=results),
                self.assertRaisesRegex(ValueError, message),
            ):
                CPBench().validate_results([ev], results)

    def test_duplicate_evaluations_are_rejected(self):
        ev = evaluation()
        with self.assertRaises(ValueError):
            CPBench().evaluate([ev, ev])


class TestModelSources(unittest.TestCase):
    def test_checkpoint_needs_no_training_split_or_processor_fitting(self):
        ev = evaluation()
        self.assertNotIn("train", ev.params["protocol"]["split"]["roles"])
        schema = SampleBuilder.load(
            str(Path(ev.params["model"]["checkpoint"]) / "schema.pkl")
        )
        with (
            patch.object(
                Trainer, "train", side_effect=AssertionError("must not train")
            ),
            patch.object(
                type(schema.input_processors["x"]),
                "fit",
                side_effect=AssertionError("must not fit"),
            ),
            patch.object(
                type(schema.output_processors["label"]),
                "fit",
                side_effect=AssertionError("must not fit"),
            ),
        ):
            result = CPBench().evaluate([ev])[ev]
        self.assertEqual(result.status, "success", result.metadata)

    def test_checkpoint_restores_actual_weights(self):
        train = make_splits(n_train=18, n_calibration=24, n_test=12)["train"]
        model = MLP(train, embedding_dim=4, hidden_dim=4).eval()
        with torch.no_grad():
            model.fc.bias.copy_(torch.tensor([3.0, -2.0, 1.0]))
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(Path(directory), model, train)
            ev = evaluation(
                model_class=MLP,
                model_params={"embedding_dim": 4, "hidden_dim": 4},
                checkpoint=directory,
            )
            params, splits, loaded = prepare_evaluation(
                ev, required=("calibration", "test")
            )
            self.assertNotIn("train", splits)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
            self.assertEqual(params, ev.params)

    def test_training_requires_train_first(self):
        ev = evaluation(model_class=MLP, training={"epochs": 1})
        params = copy.deepcopy(ev.params)
        params["protocol"]["split"]["roles"] = ["calibration", "train", "test"]
        bad = Evaluation(params)
        with self.assertRaisesRegex(ValueError, "first split role"):
            CPBench().evaluate([bad])

    def test_wrong_checkpoint_schema_is_rejected(self):
        params = copy.deepcopy(evaluation().params)
        params["task"]["params"]["mode"] = "multilabel"
        bad = Evaluation(params)
        result = CPBench().evaluate([bad])[bad]
        self.assertEqual(result.status, "failed")
        self.assertIn("schema differs", result.metadata["error"])

    def test_cached_predictions_are_explicitly_unsupported_by_current_adapters(self):
        for method, mode in ((BaseConformal, "multiclass"), (FavMac, "multilabel")):
            params = copy.deepcopy(evaluation(method, mode=mode).params)
            del params["model"]["checkpoint"]
            params["model"]["predictions"] = "/not-read/predictions.npz"
            ev = Evaluation(params)
            validate_evaluation(**ev.params)
            with patch(
                "pyhealth.benchmarks.cpbench.shared.preparation.load_splits",
                side_effect=AssertionError("must not load"),
            ):
                result = CPBench().evaluate([ev])[ev]
            self.assertEqual(result.metadata["error_type"], "NotImplementedError")
            self.assertIn("cached predictions", result.metadata["error"])

    def test_patient_split_is_executable(self):
        params = copy.deepcopy(
            evaluation(
                model_class=MLP,
                model_params={"embedding_dim": 4, "hidden_dim": 4},
                training={"epochs": 1},
            ).params
        )
        params["protocol"]["split"] = {
            "class": PatientSplit,
            "params": {"ratios": [0.5, 0.25, 0.25], "seed": 7},
            "roles": ["train", "calibration", "test"],
        }
        ev = Evaluation(params)
        result = CPBench().evaluate([ev])[ev]
        self.assertEqual(result.status, "success", result.metadata)

    def test_real_base_dataset_and_task_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "events.csv").write_text(
                "patient_id,time,value,label\n"
                + "\n".join(
                    f"p{i:02d},2020-01-01 00:00:00,{float(i % 3)},{i % 3}"
                    for i in range(24)
                )
            )
            (root / "config.yaml").write_text(
                'version: "1.0"\ntables:\n  events:\n'
                '    file_path: "events.csv"\n    patient_id: "patient_id"\n'
                '    timestamp: "time"\n    attributes:\n'
                '      - "value"\n      - "label"\n'
            )
            params = copy.deepcopy(evaluation().params)
            params["dataset"] = {
                "class": BaseDataset,
                "params": {
                    "root": directory,
                    "tables": ["events"],
                    "config_path": str(root / "config.yaml"),
                    "cache_dir": str(root / "cache"),
                    "num_workers": 1,
                },
            }
            params["task"] = {"class": EventTask}
            params["protocol"]["split"]["params"] = {
                "subjects": [
                    [f"p{i:02d}" for i in range(12)],
                    [f"p{i:02d}" for i in range(12, 24)],
                ]
            }
            ev = Evaluation(params)
            result = CPBench().evaluate([ev])[ev]
            self.assertEqual(result.status, "success", result.metadata)


class TestNativeMethods(unittest.TestCase):
    def test_multiclass_native_equivalence(self):
        for method, arguments in (
            (BaseConformal, {"alpha": 0.2, "random_state": 42}),
            (BaseConformal, {"alpha": 0.2, "score_type": "aps", "random_state": 7}),
            (LABEL, {"alpha": [0.1, 0.2, 0.3], "random_state": 42}),
            (RC3P, {"alpha": 0.2, "random_state": 42}),
        ):
            with self.subTest(method=method, arguments=arguments):
                self._native_equivalence(
                    evaluation(method, method_params=arguments),
                    method,
                    arguments,
                    multiclass_metrics_fn,
                )

    def test_label_shift_native_equivalence(self):
        for score_type in ("threshold", "aps"):
            for source in (None, [0.2, 0.3, 0.5]):
                arguments = {
                    "alpha": 0.2,
                    "score_type": score_type,
                    "random_state": 42,
                }
                calibration = {"target_label_dist": [0.0, 1.0, 0.0]}
                if source is not None:
                    calibration["source_label_dist"] = source

                with self.subTest(score_type=score_type, source=source):
                    ev = evaluation(
                        LabelShift,
                        method_params=arguments,
                        options={"calibration_kwargs": calibration},
                        dataset_params={"test_class": 1},
                    )
                    self._native_equivalence(
                        ev,
                        LabelShift,
                        arguments,
                        multiclass_metrics_fn,
                        calibration_kwargs=calibration,
                    )

    def test_label_shift_requires_valid_distributions(self):
        good = evaluation()
        for calibration in (
            {},
            {"target_label_dist": [0.5, 0.5]},
            {"target_label_dist": [0.5, -0.1, 0.6]},
            {"target_label_dist": [0.2, 0.3, 0.5], "source_label_dist": [0, 1, 0]},
        ):
            with self.subTest(calibration=calibration):
                bad = evaluation(
                    LabelShift, options={"calibration_kwargs": calibration}
                )
                results = CPBench().evaluate([bad, good])

                self.assertEqual(results[bad].status, "failed")
                self.assertIn(
                    results[bad].metadata["error_type"], ("TypeError", "ValueError")
                )
                self.assertEqual(
                    results[good].status, "success", results[good].metadata
                )

    def test_label_shift_training_cache_and_outputs(self):
        ev = evaluation(
            LabelShift,
            model_class=MLP,
            model_params={"embedding_dim": 4, "hidden_dim": 4},
            training={"epochs": 1},
            options={"calibration_kwargs": {"target_label_dist": [0.2, 0.3, 0.5]}},
        )
        before = ev.to_dict()

        with tempfile.TemporaryDirectory() as directory:
            cache, output = Path(directory) / "cache", Path(directory) / "output"
            cold = CPBench().evaluate([ev], cache_dir=cache, output_dir=output)[ev]

            with patch.object(
                Trainer, "train", side_effect=AssertionError("must reuse")
            ):
                warm = CPBench().evaluate([ev], cache_dir=cache, output_dir=output)[ev]

            self.assertEqual(cold.status, "success", cold.metadata)
            self.assertEqual(warm.status, "success", warm.metadata)
            self.assertEqual(cold.metrics, warm.metrics)
            self.assertEqual(warm.artifact, prediction_cache_path(ev.params, cache))

            with np.load(warm.artifact) as predictions:
                self.assertEqual(
                    set(predictions.files), {"y_true", "y_prob", "y_predset"}
                )
                self.assertEqual(predictions["y_predset"].shape, (12, 3))

        self.assertEqual(ev.to_dict(), before)

    def test_favmac_native_equivalence(self):
        for arguments in (
            {"target_cost": 0.8},
            {"target_cost": 0.8, "delta": 0.2},
            {"target_cost": 0.8, "cost_weights": [1, 2, 1], "value_weights": [2, 1, 3]},
        ):
            with self.subTest(arguments=arguments):
                self._native_equivalence(
                    evaluation(FavMac, mode="multilabel", method_params=arguments),
                    FavMac,
                    arguments,
                    multilabel_metrics_fn,
                )

    def _native_equivalence(
        self, ev, method, arguments, metric_function, *, calibration_kwargs=None
    ):
        before = ev.to_dict()
        observed = []
        original = method.forward

        def record(predictor, **features):
            self.assertEqual(set(features), {"x"})
            output = original(predictor, **features)
            observed.append(output["y_predset"].cpu().numpy().copy())
            return output

        with patch.object(method, "forward", record):
            result = CPBench().evaluate([Evaluation.from_dict(ev.to_dict())])[ev]
        self.assertEqual(result.status, "success", result.metadata)
        splits = make_splits(**ev.params["dataset"]["params"])
        native_arguments = {
            key: np.asarray(value)
            if key.endswith("weights") and isinstance(value, list)
            else value
            for key, value in arguments.items()
        }
        native = method(FixedModel(splits["train"]).eval(), **native_arguments)
        native.calibrate(splits["calibration"], **(calibration_kwargs or {}))
        records = []
        with torch.no_grad():
            for batch in get_dataloader(splits["test"], 5):
                output = native(x=batch["x"])
                records.append(
                    {
                        "y_true": batch["label"].numpy(),
                        "y_prob": output["y_prob"].cpu().numpy(),
                        "y_predset": output["y_predset"].cpu().numpy(),
                    }
                )
        predictions = concatenate_predictions(records)
        np.testing.assert_array_equal(
            np.concatenate(observed), predictions["y_predset"]
        )
        names = (
            ["tp", "fp", "roc_auc_micro"]
            if method is FavMac
            else ["accuracy", "set_size", "miscoverage_overall_ps"]
        )
        expected = metric_function(**predictions, metrics=names)
        for name, value in expected.items():
            self.assertAlmostEqual(result.metrics[name], value)
        self.assertEqual(ev.to_dict(), before)
        self.assertEqual(result.metadata, {"class_labels": [0, 1, 2]})

    def test_all_existing_adapter_methods_can_train_mlp(self):
        for method, arguments, options, mode in (
            (BaseConformal, {"alpha": 0.2}, {}, "multiclass"),
            (SCRIB, {"risk": 0.2}, {}, "multiclass"),
            (
                NeighborhoodLabel,
                {"alpha": 0.2, "k_neighbors": 3, "random_state": 9},
                {"calibration_kwargs": {"batch_size": 7}},
                "multiclass",
            ),
            (FavMac, {"target_cost": 0.8}, {}, "multilabel"),
        ):
            with self.subTest(method=method):
                ev = evaluation(
                    method,
                    mode=mode,
                    model_class=MLP,
                    model_params={"embedding_dim": 4, "hidden_dim": 4},
                    training={"epochs": 1},
                    method_params=arguments,
                    options=options,
                )
                result = CPBench().evaluate([ev])[ev]
                self.assertEqual(result.status, "success", result.metadata)

    def test_native_factory_and_calibration_arguments(self):
        ev = evaluation(
            method_factory,
            method_params={"alpha": 0.2, "tag": "configured"},
            options={"calibration_kwargs": {"expected_tag": "configured"}},
        )
        with patch.dict(ADAPTERS, {method_factory: run_fixed_calibration}):
            result = CPBench().evaluate([Evaluation.from_dict(ev.to_dict())])[ev]
        self.assertEqual(result.status, "success", result.metadata)

    def test_unsupported_protocol_and_options_fail_before_preparation(self):
        for method, mode, adapter in (
            (BaseConformal, "multiclass", "fixed_calibration"),
            (ClusterLabel, "multiclass", "cluster_label"),
            (FavMac, "multilabel", "favmac"),
        ):
            base = evaluation(method, mode=mode)
            invalid = []
            for key, value in (("feedback", "immediate"), ("order", "chronological")):
                params = copy.deepcopy(base.params)
                params["protocol"][key] = value
                invalid.append((Evaluation(params), "ValueError"))
            for key in ("unknown", "evaluation", "cache_dir", "output_dir"):
                params = copy.deepcopy(base.params)
                params["method"]["options"] = {key: "unsupported"}
                invalid.append((Evaluation(params), "TypeError"))
            for ev, error in invalid:
                with (
                    self.subTest(method=method, params=ev.params),
                    patch(
                        f"pyhealth.benchmarks.cpbench.adapters.{adapter}.prepare_evaluation"
                    ) as prepare,
                ):
                    result = CPBench().evaluate([ev])[ev]
                prepare.assert_not_called()
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.metadata["error_type"], error)

    def test_unknown_metrics_are_not_silently_omitted(self):
        for method, mode in ((BaseConformal, "multiclass"), (FavMac, "multilabel")):
            ev = evaluation(method, mode=mode, metrics=["not_a_metric"])
            result = CPBench().evaluate([ev])[ev]
            self.assertEqual(result.status, "failed")
            self.assertIn("metric", result.metadata["error"].lower())

    def test_default_output_and_rng_restoration_including_failure(self):
        good = evaluation()
        params = copy.deepcopy(good.params)
        params["model"]["class"] = broken_model
        bad = Evaluation(params)
        py_state, np_state, torch_state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        results = CPBench().evaluate([bad, good])
        self.assertEqual(results[bad].status, "failed")
        self.assertEqual(results[good].status, "success", results[good].metadata)
        self.assertIsNone(results[good].artifact)
        self.assertEqual(random.getstate(), py_state)
        now = np.random.get_state()
        self.assertEqual(now[0], np_state[0])
        np.testing.assert_array_equal(now[1], np_state[1])
        self.assertEqual(now[2:], np_state[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_different_families_in_one_call(self):
        multiclass, multilabel = evaluation(), evaluation(FavMac, mode="multilabel")
        results = CPBench().evaluate([multiclass, multilabel])
        self.assertTrue(
            all(result.status == "success" for result in results.values()), results
        )

    def test_feedback_loop_reuses_preparation_and_collection(self):
        params = copy.deepcopy(
            evaluation(
                FeedbackMethod,
                method_params={},
                metrics=["set_size", "miscoverage_overall_ps", "error_overall_ps"],
            ).params
        )
        params["protocol"]["feedback"] = "immediate"
        params["protocol"]["split"]["roles"] = ["calibration", "stream"]
        ev = Evaluation(params)
        with patch.dict(ADAPTERS, {FeedbackMethod: feedback_adapter}):
            result = CPBench().evaluate([ev])[ev]
        self.assertEqual(result.status, "success", result.metadata)
        self.assertAlmostEqual(result.metrics["set_size"], 14 / 12)
        self.assertAlmostEqual(result.metrics["miscoverage_overall_ps"], 11 / 12)
        self.assertEqual(result.metrics["error_overall_ps"], 1.0)


class TestCacheAndOutput(unittest.TestCase):
    def trained_evaluation(self, method=BaseConformal, mode="multiclass"):
        arguments = {
            BaseConformal: {"alpha": 0.2, "random_state": 42},
            ClusterLabel: {"alpha": 0.2, "n_clusters": 2, "random_state": 42},
            FavMac: {"target_cost": 0.8},
        }[method]
        return evaluation(
            method,
            mode=mode,
            model_class=MLP,
            model_params={"embedding_dim": 4, "hidden_dim": 4},
            training={"epochs": 1},
            method_params=arguments,
            metrics=["tp", "fp", "roc_auc_micro"]
            if mode == "multilabel"
            else ["accuracy", "set_size", "miscoverage_overall_ps"],
        )

    def test_all_adapters_save_outputs_and_share_model_cache(self):
        evaluations = [
            self.trained_evaluation(),
            self.trained_evaluation(ClusterLabel),
            self.trained_evaluation(FavMac, "multilabel"),
        ]
        original_train = Trainer.train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                Trainer, "train", autospec=True, side_effect=original_train
            ) as train:
                results = CPBench().evaluate(
                    evaluations, cache_dir=root / "cache", output_dir=root / "output"
                )
            # ClusterLabel and BaseConformal use the same fitted base model.
            self.assertEqual(train.call_count, 2)
            for ev, result in results.items():
                self.assertEqual(result.status, "success", result.metadata)
                self.assertEqual(
                    result.artifact, prediction_cache_path(ev.params, root / "cache")
                )
                with np.load(result.artifact, allow_pickle=False) as artifact:
                    self.assertEqual(
                        set(artifact.files), {"y_true", "y_prob", "y_predset"}
                    )
                    predictions = {name: artifact[name] for name in artifact.files}
                self.assertEqual(len(predictions["y_true"]), 12)
                scorer = (
                    multilabel_metrics_fn
                    if ev.params["method"]["class"] is FavMac
                    else multiclass_metrics_fn
                )
                expected = scorer(**predictions, metrics=ev.params["metrics"])
                for name, value in expected.items():
                    self.assertAlmostEqual(result.metrics[name], value)
            self.assertEqual(
                list((root / "output").iterdir()), [root / "output" / "results.json"]
            )
            saved = json.loads((root / "output" / "results.json").read_text())
            self.assertEqual(set(saved), {ev.id for ev in evaluations})
            for ev, result in results.items():
                self.assertEqual(saved[ev.id]["metrics"], result.metrics)
                self.assertEqual(saved[ev.id]["artifact"], str(result.artifact))

    def test_cache_reuse_invalidation_and_numerical_parity(self):
        first = self.trained_evaluation()
        plain = CPBench().evaluate([first])[first]
        alpha_params = copy.deepcopy(first.params)
        alpha_params["method"]["params"]["alpha"] = 0.3
        other_alpha = Evaluation(alpha_params)
        metric_params = copy.deepcopy(first.params)
        metric_params["metrics"] = ["accuracy"]
        other_metrics = Evaluation(metric_params)
        seed_params = copy.deepcopy(first.params)
        seed_params["protocol"]["seed"] = 43
        other_seed = Evaluation(seed_params)
        original_train = Trainer.train
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch.object(
                Trainer, "train", autospec=True, side_effect=original_train
            ) as train:
                cold = CPBench().evaluate([first], cache_dir=cache)[first]
                warm = CPBench().evaluate([first], cache_dir=cache)[first]
                rest = CPBench().evaluate(
                    [other_alpha, other_metrics, other_seed], cache_dir=cache
                )
            self.assertEqual(train.call_count, 2)
            for result in [plain, cold, warm, *rest.values()]:
                self.assertEqual(result.status, "success", result.metadata)
            self.assertEqual(plain.metrics, cold.metrics)
            self.assertEqual(cold.metrics, warm.metrics)
            self.assertEqual(
                rest[other_metrics].metrics["accuracy"], cold.metrics["accuracy"]
            )
            self.assertEqual(cold.artifact, prediction_cache_path(first.params, cache))
            self.assertEqual(rest[other_metrics].artifact, cold.artifact)
            self.assertNotEqual(rest[other_alpha].artifact, cold.artifact)
            self.assertNotEqual(rest[other_seed].artifact, cold.artifact)

    def test_preparation_rng_and_cached_model_ownership(self):
        ev = self.trained_evaluation()
        before = ev.to_dict()
        with tempfile.TemporaryDirectory() as directory, _seeded(42):
            py_state, np_state, torch_state = (
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
            )
            _, _, cold = prepare_evaluation(
                ev, required=("train", "calibration", "test"), cache_dir=Path(directory)
            )
            saved = {
                key: value.detach().clone() for key, value in cold.state_dict().items()
            }
            with torch.no_grad():
                cold.fc.bias.add_(10)
            _, _, warm = prepare_evaluation(
                ev, required=("train", "calibration", "test"), cache_dir=Path(directory)
            )
            self.assertIsNot(cold, warm)
            for key, value in saved.items():
                self.assertTrue(torch.equal(value, warm.state_dict()[key]), key)
            self.assertEqual(random.getstate(), py_state)
            now = np.random.get_state()
            self.assertEqual(now[0], np_state[0])
            np.testing.assert_array_equal(now[1], np_state[1])
            self.assertEqual(now[2:], np_state[2:])
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(ev.to_dict(), before)

    def test_cache_publication_failure_does_not_leave_an_entry(self):
        ev = self.trained_evaluation()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            target = model_cache_path(ev.params, cache)
            with patch.object(
                SampleBuilder, "save", side_effect=OSError("interrupted write")
            ):
                result = CPBench().evaluate([ev], cache_dir=cache)[ev]
            self.assertEqual(result.status, "failed")
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.glob(f".{target.name}.*")), [])
            recovered = CPBench().evaluate([ev], cache_dir=cache)[ev]
            self.assertEqual(recovered.status, "success", recovered.metadata)
            self.assertTrue((target / "model.ckpt").is_file())
            self.assertTrue((target / "schema.pkl").is_file())

    def test_prediction_publication_failure_preserves_existing_cache_file(self):
        ev = evaluation()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = CPBench().evaluate([ev], cache_dir=root / "cache")[ev]
            self.assertEqual(first.status, "success", first.metadata)
            contents = first.artifact.read_bytes()

            with patch(
                "pyhealth.benchmarks.cpbench.shared.prediction.np.savez_compressed",
                side_effect=OSError("interrupted write"),
            ):
                failed = CPBench().evaluate(
                    [ev], cache_dir=root / "cache", output_dir=root / "output"
                )[ev]

            self.assertEqual(failed.status, "failed")
            self.assertEqual(first.artifact.read_bytes(), contents)
            self.assertEqual(list(first.artifact.parent.iterdir()), [first.artifact])
            saved = json.loads((root / "output" / "results.json").read_text())
            self.assertEqual(saved[ev.id]["status"], "failed")

    def test_output_without_cache_saves_results_only(self):
        good = evaluation()
        bad = evaluation(FeedbackMethod, method_params={})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.dict(ADAPTERS, {FeedbackMethod: broken_adapter}):
                results = CPBench().evaluate([good, bad], output_dir=output)

            self.assertTrue(all(result.artifact is None for result in results.values()))
            self.assertEqual(list(output.iterdir()), [output / "results.json"])
            saved = json.loads((output / "results.json").read_text())
            self.assertEqual(set(saved), {good.id, bad.id})
            self.assertEqual(saved[good.id]["status"], "success")
            self.assertEqual(saved[bad.id]["status"], "failed")
            self.assertEqual(saved[bad.id]["metadata"]["error_type"], "RuntimeError")


class TestPredictionAndScoring(unittest.TestCase):
    def test_invalid_multiclass_outputs_remain_rejected(self):
        original = BaseConformal.forward
        for key in ("y_prob", "y_predset"):

            def invalid(predictor, **features):
                output = original(predictor, **features)
                output[key] = torch.full_like(output[key], 0.5, dtype=torch.float32)
                return output

            ev = evaluation()
            with patch.object(BaseConformal, "forward", invalid):
                result = CPBench().evaluate([ev])[ev]
            self.assertEqual(result.status, "failed")

    def test_multilabel_shapes_and_probabilities(self):
        valid = {
            "y_true": np.array([[1, 1, 0], [0, 1, 0]]),
            "y_prob": np.array([[0.8, 0.9, 0.1], [0.2, 0.7, 0.4]]),
            "y_predset": np.array([[1, 1, 0], [0, 1, 1]]),
        }
        validate_multilabel_predictions(valid, 3)
        for name, value in (
            ("y_true", np.array([0, 1])),
            ("y_true", np.full((2, 3), 0.5)),
            ("y_prob", np.full((2, 3), np.nan)),
            ("y_prob", np.full((2, 3), 1.1)),
            ("y_predset", np.ones((2, 2))),
            ("y_predset", np.full((2, 3), 0.5)),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_multilabel_predictions({**valid, name: value}, 3)

    def test_collection_handles_arbitrary_output_names_and_retains_gradient_context(
        self,
    ):
        dataset = make_splits(n_train=18, n_calibration=24, n_test=12)["test"]
        expected = np.concatenate(
            [batch["label"].numpy() for batch in get_dataloader(dataset, 5)]
        )

        def intervals(batch):
            self.assertTrue(torch.is_grad_enabled())
            truth = batch["label"].numpy()
            return {"target": truth, "lower": truth - 1.0, "upper": truth + 2.0}

        predictions = collect_predictions(dataset, 5, intervals)
        self.assertEqual(set(predictions), {"target", "lower", "upper"})
        np.testing.assert_array_equal(predictions["target"], expected)
        np.testing.assert_array_equal(predictions["lower"], expected - 1.0)
        np.testing.assert_array_equal(predictions["upper"], expected + 2.0)

    def test_collection_snapshots_reused_buffers(self):
        dataset = make_splits(n_train=18, n_calibration=24, n_test=12)["test"]
        buffer = np.empty((5, 2))
        calls = []

        def reuse(batch):
            size = len(batch["label"])
            calls.append(size)
            buffer[:size] = len(calls)
            return {"values": buffer[:size]}

        predictions = collect_predictions(dataset, 5, reuse)
        self.assertEqual(calls, [5, 5, 2])
        np.testing.assert_array_equal(
            predictions["values"][:, 0], [1] * 5 + [2] * 5 + [3] * 2
        )

    def test_collection_checks_batch_rows_and_output_names(self):
        dataset = make_splits(n_train=18, n_calibration=24, n_test=12)["test"]
        with self.assertRaisesRegex(ValueError, "one row per input sample"):
            collect_predictions(dataset, 5, lambda batch: {"values": np.ones((1, 2))})
        with self.assertRaisesRegex(ValueError, "matching row counts"):
            concatenate_predictions([{"target": np.ones(2), "values": np.ones((1, 2))}])
        with self.assertRaisesRegex(ValueError, "same output names"):
            concatenate_predictions([{"a": np.ones(2)}, {"b": np.ones(1)}])
        with self.assertRaisesRegex(ValueError, "sample dimension"):
            copy_predictions({"scalar": np.array(1.0)})
        with self.assertRaisesRegex(ValueError, "empty"):
            concatenate_predictions([])

    def test_conditional_metrics_keep_zero_and_mark_absence(self):
        result = score_multiclass_predictions(
            {
                "y_true": np.array([0, 0, 1]),
                "y_prob": np.array([[0.8, 0.1, 0.1], [0.4, 0.5, 0.1], [0.1, 0.8, 0.1]]),
                "y_predset": np.array([[1, 0, 0], [1, 1, 0], [1, 1, 0]]),
            },
            ["miscoverage_ps", "error_ps", "error_overall_ps"],
            {0: 0, 1: 1, 2: 2},
        )
        self.assertEqual(
            result.metrics,
            {
                "miscoverage_ps": [0.0, 0.0, None],
                "error_ps": [0.0, None, None],
                "error_overall_ps": 0.0,
            },
        )

    def test_absent_classes_and_no_singletons_are_undefined(self):
        ev = evaluation(
            method_params={"alpha": [0.1, 0.1, 0.1]},
            dataset_params={"uniform": True, "test_class": 0},
        )
        result = CPBench().evaluate([ev])[ev]
        self.assertEqual(result.status, "success", result.metadata)
        self.assertEqual(result.metrics["miscoverage_ps"], [0.0, None, None])
        self.assertEqual(result.metrics["error_ps"], [None, None, None])
        self.assertIsNone(result.metrics["error_overall_ps"])

    def test_existing_metrics_with_multiple_output_keys(self):
        ev = evaluation(metrics=["hits@1", "mean_rank"])
        result = CPBench().evaluate([ev])[ev]
        self.assertEqual(result.status, "success", result.metadata)
        self.assertEqual(
            set(result.metrics), {"HITS@1", "mean_rank", "mean_reciprocal_rank"}
        )


if __name__ == "__main__":
    unittest.main()
