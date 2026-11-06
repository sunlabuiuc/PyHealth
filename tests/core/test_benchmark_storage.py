"""Public benchmark writers and automatic validated batch persistence."""

import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from examples.cpbench.synthetic_data import make_splits
from pyhealth.benchmarks import Evaluation, Result, save_results
from pyhealth.benchmarks.cpbench import save_checkpoint, save_predictions
from pyhealth.benchmarks.cpbench.shared.preparation import prepare_evaluation
from pyhealth.datasets import SampleBuilder
from pyhealth.models import MLP
from tests.core.test_benchmark_base import _CountingBenchmark
from tests.core.test_cpbench import evaluation


class TestResultStorage(unittest.TestCase):
    def test_evaluate_saves_all_results_and_evaluation_parameters(self):
        first, second = Evaluation({"x": 1}), Evaluation({"x": 2})
        results = {
            first: Result("success", {"score": 0.5}),
            second: Result(
                "failed", {}, {"error_type": "ValueError", "error": "bad input"}
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            returned = _CountingBenchmark(results).evaluate(
                [first, second], output_dir=output
            )
            self.assertIs(returned, results)
            self.assertEqual(list(output.iterdir()), [output / "results.json"])
            saved = json.loads((output / "results.json").read_text())

        self.assertEqual(set(saved), {first.id, second.id})
        self.assertEqual(saved[first.id]["evaluation"], first.to_dict())
        self.assertEqual(saved[first.id]["metrics"], {"score": 0.5})
        self.assertEqual(saved[second.id]["status"], "failed")
        self.assertEqual(saved[second.id]["metadata"], results[second].metadata)
        self.assertIsNone(saved[first.id]["artifact"])

    def test_no_output_dir_means_no_result_writes(self):
        with patch("pyhealth.benchmarks.base.save_results") as save:
            _CountingBenchmark().evaluate([Evaluation({"x": 1})])
        save.assert_not_called()

    def test_result_validation_happens_before_persistence(self):
        ev = Evaluation({"x": 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text("existing contents")
            with self.assertRaisesRegex(ValueError, "missing result"):
                _CountingBenchmark({}).evaluate([ev], output_dir=Path(directory))
            self.assertEqual(path.read_text(), "existing contents")

    def test_public_writer_can_save_combined_calls(self):
        first, second = Evaluation({"x": 1}), Evaluation({"x": 2})
        bench = _CountingBenchmark()
        results = {**bench.evaluate([first]), **bench.evaluate([second])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "combined.json"
            self.assertIsNone(save_results(results, path))
            self.assertEqual(set(json.loads(path.read_text())), {first.id, second.id})

    def test_numpy_paths_and_nonfinite_values_are_converted_only_on_disk(self):
        ev = Evaluation({"class": OrderedDict})
        values = np.array([0.0, np.nan, np.inf, -np.inf])
        result = Result(
            "success",
            {"values": values, "count": np.int64(2), "flag": np.bool_(True)},
            {"nested": (np.float32(0.5), Path("cache"))},
            artifact=Path("cache/predictions.npz"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            save_results({ev: result}, path)
            saved = json.loads(path.read_text())[ev.id]

        self.assertEqual(saved["evaluation"], ev.to_dict())
        self.assertEqual(
            saved["metrics"],
            {"values": [0.0, None, None, None], "count": 2, "flag": True},
        )
        self.assertEqual(saved["metadata"], {"nested": [0.5, "cache"]})
        self.assertEqual(saved["artifact"], "cache/predictions.npz")
        self.assertIs(result.metrics["values"], values)
        self.assertTrue(np.isnan(values[1]))
        self.assertTrue(np.isposinf(values[2]))
        self.assertIsInstance(result.metadata["nested"], tuple)
        self.assertIsInstance(result.artifact, Path)

    def test_reusing_output_dir_replaces_the_batch_without_implicit_merging(self):
        first, second = Evaluation({"x": 1}), Evaluation({"x": 2})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _CountingBenchmark().evaluate([first], output_dir=output)
            _CountingBenchmark().evaluate([second], output_dir=output)
            self.assertEqual(
                set(json.loads((output / "results.json").read_text())), {second.id}
            )

    def test_empty_evaluation_sequence_saves_an_empty_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            result = _CountingBenchmark().evaluate([], output_dir=Path(directory))
            self.assertEqual(result, {})
            self.assertEqual(
                json.loads((Path(directory) / "results.json").read_text()), {}
            )

    def test_failed_publication_preserves_existing_results_and_propagates(self):
        ev = Evaluation({"x": 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text("existing contents")
            with patch(
                "pyhealth.benchmarks.base.os.replace",
                side_effect=OSError("interrupted write"),
            ):
                with self.assertRaisesRegex(OSError, "interrupted write"):
                    _CountingBenchmark().evaluate([ev], output_dir=Path(directory))
            self.assertEqual(path.read_text(), "existing contents")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_unsupported_values_and_nonstring_keys_do_not_replace_existing_file(self):
        ev = Evaluation({"x": 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text("existing contents")
            for metrics in ({"bad": object()}, {1: "ambiguous key"}):
                with self.subTest(metrics=metrics), self.assertRaises(TypeError):
                    save_results({ev: Result("success", metrics)}, path)
                self.assertEqual(path.read_text(), "existing contents")


class TestArtifactStorage(unittest.TestCase):
    def test_public_prediction_writer_only_saves_arrays_and_returns_none(self):
        predictions = {
            "target": np.array([1, 2]),
            "lower": np.array([0, 1]),
            "upper": np.array([2, 3]),
        }
        before = {name: value.copy() for name, value in predictions.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "predictions.npz"
            self.assertIsNone(save_predictions(predictions, path))
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(predictions))
                for name, value in predictions.items():
                    np.testing.assert_array_equal(saved[name], value)
            self.assertEqual(list(path.parent.iterdir()), [path])
        for name, value in before.items():
            np.testing.assert_array_equal(predictions[name], value)

    def test_prediction_writer_preserves_existing_file_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.npz"
            save_predictions({"target": np.array([1])}, path)
            original = path.read_bytes()
            with patch(
                "pyhealth.benchmarks.cpbench.shared.prediction.np.savez_compressed",
                side_effect=OSError("interrupted write"),
            ):
                with self.assertRaisesRegex(OSError, "interrupted write"):
                    save_predictions({"target": np.array([2])}, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_public_checkpoint_writer_round_trips_weights_and_preprocessing(self):
        train = make_splits(n_train=18, n_calibration=24, n_test=12)["train"]
        model = MLP(train, embedding_dim=4, hidden_dim=4).eval()
        with torch.no_grad():
            model.fc.bias.copy_(torch.tensor([3.0, -2.0, 1.0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "checkpoint"
            self.assertIsNone(save_checkpoint(model, train, path))
            self.assertEqual(
                {item.name for item in path.iterdir()}, {"model.ckpt", "schema.pkl"}
            )
            spec = evaluation(
                model_class=MLP,
                model_params={"embedding_dim": 4, "hidden_dim": 4},
                checkpoint=str(path),
            )
            _, _, loaded = prepare_evaluation(spec, required=("calibration", "test"))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[name]), name)
            self.assertEqual(
                loaded.dataset.output_processors["label"].label_vocab,
                train.output_processors["label"].label_vocab,
            )

    def test_checkpoint_writer_refuses_existing_destination(self):
        train = make_splits(n_train=18, n_calibration=24, n_test=12)["train"]
        model = MLP(train, embedding_dim=4, hidden_dim=4).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            save_checkpoint(model, train, path)
            before = {file.name: file.read_bytes() for file in path.iterdir()}
            with self.assertRaises(FileExistsError):
                save_checkpoint(model, train, path)
            self.assertEqual(
                {file.name: file.read_bytes() for file in path.iterdir()}, before
            )

    def test_failed_checkpoint_write_does_not_publish_a_partial_directory(self):
        train = make_splits(n_train=18, n_calibration=24, n_test=12)["train"]
        model = MLP(train, embedding_dim=4, hidden_dim=4).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "checkpoint"
            with patch.object(
                SampleBuilder, "save", side_effect=OSError("interrupted write")
            ):
                with self.assertRaisesRegex(OSError, "interrupted write"):
                    save_checkpoint(model, train, path)
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
