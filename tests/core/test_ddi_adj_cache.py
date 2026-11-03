"""Tests for the shared ``ddi_adj.npy`` cache used by the ``"ddi"`` metric.

GAMENet, MICRON, MoleRec and SafeDrug write their drug-drug interaction
matrix to one file in the PyHealth cache directory, and the ``"ddi"`` metric
used to read it back. Concurrent runs (or two models with different label
vocabularies) overwrote each other's file, and a reader could catch a
partially written one. These tests cover the three parts of the fix:

* ``multilabel_metrics_fn(..., ddi_adj=...)`` never touches the file;
* the four models expose ``ddi_adj_matrix``, and ``Trainer.evaluate`` (and
  ``pyhealth.autoimprove`` scoring) use it for ``"ddi"``;
* ``save_ddi_adj`` replaces the file atomically.
"""

import importlib.metadata as importlib_metadata
import os
import stat
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch
from packaging import version

import pyhealth.models
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.metrics import multilabel_metrics_fn
from pyhealth.models.utils import save_ddi_adj
from pyhealth.trainer import Trainer


def _patch_cache_dir(testcase, *targets):
    tmpdir = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmpdir.cleanup)
    for target in targets:
        patcher = patch(target, tmpdir.name)
        patcher.start()
        testcase.addCleanup(patcher.stop)
    return tmpdir.name


class TestDDIMetricArgument(unittest.TestCase):
    def setUp(self):
        self.cache_dir = _patch_cache_dir(
            self, "pyhealth.metrics.multilabel.CACHE_PATH"
        )
        self.y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        self.y_prob = np.array(
            [[0.9, 0.4, 0.8], [0.2, 0.7, 0.6], [0.8, 0.9, 0.1], [0.1, 0.2, 0.7]]
        )
        # Labels 0 and 1 interact, label 2 interacts with nothing.
        self.ddi_adj = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)

    def test_passed_matrix_wins_over_cache_file(self):
        """A different (all-interacting) matrix in the file must be ignored."""
        np.save(os.path.join(self.cache_dir, "ddi_adj.npy"), np.ones((3, 3)))
        from_file = multilabel_metrics_fn(self.y_true, self.y_prob, ["ddi"])
        passed = multilabel_metrics_fn(
            self.y_true, self.y_prob, ["ddi"], ddi_adj=self.ddi_adj
        )
        self.assertAlmostEqual(passed["ddi_score"], 0.4)
        self.assertAlmostEqual(from_file["ddi_score"], 1.0)

    def test_passed_matrix_needs_no_cache_file(self):
        """Wrong-shaped or missing file: the passed matrix is still used."""
        out = multilabel_metrics_fn(
            self.y_true, self.y_prob, ["ddi"], ddi_adj=self.ddi_adj
        )
        self.assertAlmostEqual(out["ddi_score"], 0.4)
        np.save(os.path.join(self.cache_dir, "ddi_adj.npy"), np.zeros((1, 1)))
        out = multilabel_metrics_fn(
            self.y_true, self.y_prob, ["ddi"], ddi_adj=self.ddi_adj
        )
        self.assertAlmostEqual(out["ddi_score"], 0.4)

    def test_cache_file_fallback_is_kept(self):
        np.save(os.path.join(self.cache_dir, "ddi_adj.npy"), self.ddi_adj)
        out = multilabel_metrics_fn(self.y_true, self.y_prob, ["ddi"])
        self.assertAlmostEqual(out["ddi_score"], 0.4)

    def test_missing_cache_file_has_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "ddi_adj="):
            multilabel_metrics_fn(self.y_true, self.y_prob, ["ddi"])


def _require_ogb(testcase):
    """MoleRec needs ogb>=1.3.5; skip like tests/core/test_molerec.py does."""
    try:
        installed = importlib_metadata.version("ogb")
    except importlib_metadata.PackageNotFoundError:
        testcase.skipTest("ogb package not available")
    if version.parse(installed) < version.parse("1.3.5"):
        testcase.skipTest(f"ogb>=1.3.5 required, but found {installed}")
    try:
        from ogb.utils import smiles2graph  # noqa: F401
    except ImportError as exc:
        testcase.skipTest(f"ogb import failed: {exc}")


class _ModelDDIMatrixMixin:
    """Each drug recommendation model must expose the matrix it cached, and
    Trainer.evaluate must score "ddi" with it, not with the shared file."""

    model_name = ""  # class in pyhealth.models, module pyhealth.models.<lower>
    uses_drugs_hist = False

    def build_model(self, dataset):
        model_cls = getattr(pyhealth.models, self.model_name)
        return model_cls(dataset=dataset)

    def setUp(self):
        self.cache_dir = _patch_cache_dir(
            self,
            f"pyhealth.models.{self.model_name.lower()}.CACHE_PATH",
            "pyhealth.metrics.multilabel.CACHE_PATH",
        )
        samples = []
        for i, drugs in enumerate(
            [["drug-1", "drug-2"], ["drug-2", "drug-3"], ["drug-1", "drug-3"]]
        ):
            sample = {
                "patient_id": f"patient-{i}",
                "visit_id": f"visit-{i}",
                "conditions": [["cond-1", "cond-2"], ["cond-3"]],
                "procedures": [["proc-1"], ["proc-2"]],
                "drugs": drugs,
            }
            if self.uses_drugs_hist:
                sample["drugs_hist"] = [["drug-1"], []]
            samples.append(sample)
        input_schema = {
            "conditions": "nested_sequence",
            "procedures": "nested_sequence",
        }
        if self.uses_drugs_hist:
            input_schema["drugs_hist"] = "nested_sequence"
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema=input_schema,
            output_schema={"drugs": "multilabel"},
            dataset_name="ddi_cache_test",
        )
        torch.manual_seed(0)
        self.model = self.build_model(self.dataset)
        self.loader = get_dataloader(self.dataset, batch_size=3, shuffle=False)
        self.cache_file = os.path.join(self.cache_dir, "ddi_adj.npy")

    def test_model_exposes_the_matrix_it_cached(self):
        n = len(self.dataset.output_processors["drugs"].label_vocab)
        self.assertIsInstance(self.model.ddi_adj_matrix, np.ndarray)
        self.assertEqual(self.model.ddi_adj_matrix.shape, (n, n))
        np.testing.assert_array_equal(
            np.load(self.cache_file), self.model.ddi_adj_matrix
        )


class _EvaluateDDIMixin:
    def _expected_ddi(self):
        trainer = Trainer(model=self.model, metrics=["ddi"], enable_logging=False)
        y_true, y_prob, _ = trainer.inference(self.loader)
        expected = multilabel_metrics_fn(
            y_true, y_prob, ["ddi"], ddi_adj=self.model.ddi_adj_matrix
        )["ddi_score"]
        return trainer, y_true, y_prob, expected

    def test_evaluate_ignores_a_clobbered_cache_file(self):
        """Another process overwrites the file after the model was built."""
        trainer, y_true, y_prob, expected = self._expected_ddi()

        # A matrix of the right shape that says every pair interacts.
        n = self.model.ddi_adj_matrix.shape[0]
        np.save(self.cache_file, np.ones((n, n)))
        from_file = multilabel_metrics_fn(y_true, y_prob, ["ddi"])["ddi_score"]
        self.assertNotAlmostEqual(from_file, expected)
        scores = trainer.evaluate(self.loader)
        self.assertAlmostEqual(scores["ddi_score"], expected)

        # A matrix for a smaller vocabulary would index out of range.
        np.save(self.cache_file, np.zeros((1, 1)))
        scores = trainer.evaluate(self.loader)
        self.assertAlmostEqual(scores["ddi_score"], expected)


class TestGAMENetDDIMatrix(_ModelDDIMatrixMixin, _EvaluateDDIMixin, unittest.TestCase):
    model_name = "GAMENet"
    uses_drugs_hist = True

    def build_model(self, dataset):
        return pyhealth.models.GAMENet(dataset, embedding_dim=16, hidden_dim=16)

    def test_autoimprove_score_uses_the_model_matrix(self):
        from pyhealth.autoimprove.evaluate import score

        _, _, _, expected = self._expected_ddi()
        n = self.model.ddi_adj_matrix.shape[0]
        np.save(self.cache_file, np.ones((n, n)))
        out = score(self.model, self.dataset, ["ddi"])
        self.assertAlmostEqual(out["ddi_score"], expected)


class TestSafeDrugDDIMatrix(_ModelDDIMatrixMixin, _EvaluateDDIMixin, unittest.TestCase):
    model_name = "SafeDrug"


class TestMICRONDDIMatrix(_ModelDDIMatrixMixin, _EvaluateDDIMixin, unittest.TestCase):
    model_name = "MICRON"


class TestMoleRecDDIMatrix(_ModelDDIMatrixMixin, unittest.TestCase):
    model_name = "MoleRec"

    def setUp(self):
        _require_ogb(self)
        super().setUp()


class TestSaveDDIAdj(unittest.TestCase):
    def setUp(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.dir = tmpdir.name
        self.path = os.path.join(self.dir, "ddi_adj.npy")

    def test_writes_loadable_file_and_no_temp_files(self):
        matrix = np.array([[0, 1], [1, 0]], dtype=float)
        returned = save_ddi_adj(matrix, self.path)
        np.testing.assert_array_equal(np.load(self.path), matrix)
        np.testing.assert_array_equal(returned, matrix)
        self.assertEqual(os.listdir(self.dir), ["ddi_adj.npy"])

    def test_file_mode_follows_umask_like_np_save(self):
        """The saved file gets the same permissions np.save gives a file."""
        reference = os.path.join(self.dir, "reference.npy")
        np.save(reference, np.eye(2))
        save_ddi_adj(np.eye(2), self.path)
        self.assertEqual(
            stat.S_IMODE(os.stat(self.path).st_mode),
            stat.S_IMODE(os.stat(reference).st_mode),
        )

    def test_accepts_torch_tensor_and_overwrites(self):
        save_ddi_adj(np.zeros((4, 4)), self.path)
        returned = save_ddi_adj(torch.eye(2), self.path)
        self.assertIsInstance(returned, np.ndarray)
        np.testing.assert_array_equal(np.load(self.path), np.eye(2))
        self.assertEqual(os.listdir(self.dir), ["ddi_adj.npy"])

    def test_failed_write_keeps_old_file_and_cleans_up(self):
        save_ddi_adj(np.eye(3), self.path)
        failing_save = patch(
            "pyhealth.models.utils.np.save", side_effect=OSError("disk full")
        )
        with failing_save, self.assertRaises(OSError):
            save_ddi_adj(np.zeros((5, 5)), self.path)
        np.testing.assert_array_equal(np.load(self.path), np.eye(3))
        self.assertEqual(os.listdir(self.dir), ["ddi_adj.npy"])

    def test_concurrent_readers_never_see_a_partial_file(self):
        small, large = np.eye(3), np.ones((400, 400))
        save_ddi_adj(small, self.path)
        errors = []
        stop = threading.Event()

        def writer():
            for i in range(100):
                save_ddi_adj(large if i % 2 else small, self.path)
            stop.set()

        def reader():
            while not stop.is_set():
                try:
                    shape = np.load(self.path).shape
                    if shape not in {(3, 3), (400, 400)}:
                        errors.append(shape)
                except (OSError, EOFError, ValueError) as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=writer)] + [
            threading.Thread(target=reader) for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(os.listdir(self.dir), ["ddi_adj.npy"])


if __name__ == "__main__":
    unittest.main()
