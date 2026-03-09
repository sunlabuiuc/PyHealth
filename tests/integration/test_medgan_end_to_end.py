"""End-to-end integration tests for the MedGAN synthetic EHR generation pipeline.

Category A tests use InMemorySampleDataset with synthetic data — no external
data required and must always pass.

Category B tests require actual MIMIC-III data and are skipped gracefully when
the data is unavailable.

The bootstrap pattern mirrors test_corgan_end_to_end.py: load MedGAN and
InMemorySampleDataset via importlib while stubbing out heavy optional
dependencies (einops, litdata, etc.) that are not yet in the venv.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Bootstrap: load MedGAN, BaseModel, and InMemorySampleDataset without
# triggering pyhealth.models.__init__ or pyhealth.datasets.__init__.
# ---------------------------------------------------------------------------


def _bootstrap():
    """Load MedGAN, BaseModel, and InMemorySampleDataset via importlib.

    Returns:
        (BaseModel, MedGAN, InMemorySampleDataset)
    """
    import pyhealth  # noqa: F401  — top-level __init__ has no heavy deps

    # Stub pyhealth.datasets so that base_model.py's
    # "from ..datasets import SampleDataset" resolves cleanly.
    if "pyhealth.datasets" not in sys.modules:
        ds_stub = MagicMock()

        class _FakeSampleDataset:
            pass

        ds_stub.SampleDataset = _FakeSampleDataset
        sys.modules["pyhealth.datasets"] = ds_stub

    # Stub pyhealth.models so we can control loading without the real __init__.
    if "pyhealth.models" not in sys.modules or isinstance(
        sys.modules["pyhealth.models"], MagicMock
    ):
        models_stub = MagicMock()
        sys.modules["pyhealth.models"] = models_stub
    else:
        models_stub = sys.modules["pyhealth.models"]

    # Processors are safe to import normally.
    from pyhealth.processors import PROCESSOR_REGISTRY  # noqa: F401

    def _load_file(mod_name, filepath):
        spec = importlib.util.spec_from_file_location(mod_name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    base = os.path.join(root, "pyhealth", "models")

    # Load base_model and expose via stub.
    bm_mod = _load_file(
        "pyhealth.models.base_model", os.path.join(base, "base_model.py")
    )
    BaseModel = bm_mod.BaseModel
    models_stub.BaseModel = BaseModel

    gen_stub = MagicMock()
    sys.modules.setdefault("pyhealth.models.generators", gen_stub)

    # Load MedGAN directly.
    medgan_mod = _load_file(
        "pyhealth.models.generators.medgan",
        os.path.join(base, "generators", "medgan.py"),
    )
    MedGAN = medgan_mod.MedGAN

    # Stub litdata so sample_dataset.py can be loaded.
    if "litdata" not in sys.modules:
        litdata_pkg = MagicMock()
        litdata_pkg.StreamingDataset = type(
            "StreamingDataset", (), {"__init__": lambda self, *a, **kw: None}
        )
        litdata_utilities = MagicMock()
        litdata_utilities_train_test = MagicMock()
        litdata_utilities_train_test.deepcopy_dataset = lambda x: x
        litdata_utilities.train_test_split = litdata_utilities_train_test
        litdata_pkg.utilities = litdata_utilities
        sys.modules["litdata"] = litdata_pkg
        sys.modules["litdata.utilities"] = litdata_utilities
        sys.modules["litdata.utilities.train_test_split"] = (
            litdata_utilities_train_test
        )

    # Load sample_dataset.py directly (bypasses datasets/__init__.py).
    ds_file_mod = _load_file(
        "pyhealth.datasets.sample_dataset",
        os.path.join(root, "pyhealth", "datasets", "sample_dataset.py"),
    )
    InMemorySampleDataset = ds_file_mod.InMemorySampleDataset

    return BaseModel, MedGAN, InMemorySampleDataset


BaseModel, MedGAN, InMemorySampleDataset = _bootstrap()

import torch  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SMALL_SAMPLES = [
    {"patient_id": "p1", "visits": ["A", "B", "C"]},
    {"patient_id": "p2", "visits": ["A", "C", "D"]},
    {"patient_id": "p3", "visits": ["B", "D", "E"]},
    {"patient_id": "p4", "visits": ["A", "B", "C", "D"]},
    {"patient_id": "p5", "visits": ["C", "E"]},
    {"patient_id": "p6", "visits": ["A", "D", "E"]},
    {"patient_id": "p7", "visits": ["B", "C", "D"]},
    {"patient_id": "p8", "visits": ["A", "E"]},
]

_SMALL_MODEL_KWARGS = dict(
    latent_dim=4,
    hidden_dim=4,
    autoencoder_hidden_dim=4,
    discriminator_hidden_dim=8,
    batch_size=4,
    ae_epochs=1,
    gan_epochs=1,
    save_dir=None,
)


def _make_dataset(samples=None):
    if samples is None:
        samples = _SMALL_SAMPLES
    return InMemorySampleDataset(
        samples=samples,
        input_schema={"visits": "multi_hot"},
        output_schema={},
    )


# ---------------------------------------------------------------------------
# Category A: In-Memory Integration Tests (must always pass)
# ---------------------------------------------------------------------------


class TestMedGANIsBaseModelInstance(unittest.TestCase):
    """MedGAN model is an instance of BaseModel."""

    def test_model_is_basemodel_instance(self):
        dataset = _make_dataset()
        model = MedGAN(dataset, **_SMALL_MODEL_KWARGS)
        self.assertIsInstance(model, BaseModel)


class TestMedGANFeatureKeys(unittest.TestCase):
    """model.feature_keys equals ['visits']."""

    def test_feature_keys(self):
        dataset = _make_dataset()
        model = MedGAN(dataset, **_SMALL_MODEL_KWARGS)
        self.assertEqual(model.feature_keys, ["visits"])


class TestMedGANVocabSize(unittest.TestCase):
    """MedGAN.input_dim matches processor.size()."""

    def test_vocab_size_matches_processor(self):
        dataset = _make_dataset()
        expected = dataset.input_processors["visits"].size()
        model = MedGAN(dataset, **_SMALL_MODEL_KWARGS)
        self.assertEqual(model.input_dim, expected)


class TestMedGANForwardRaisesNotImplementedError(unittest.TestCase):
    """Calling forward() raises NotImplementedError."""

    def test_forward_not_implemented(self):
        dataset = _make_dataset()
        model = MedGAN(dataset, **_SMALL_MODEL_KWARGS)
        with self.assertRaises(NotImplementedError):
            model.forward()


class TestMedGANTrainModelRuns(unittest.TestCase):
    """train_model completes one epoch without error."""

    def test_train_model_runs_one_epoch(self):
        dataset = _make_dataset()
        model = MedGAN(dataset, **_SMALL_MODEL_KWARGS)
        try:
            model.train_model(dataset, val_dataset=None)
        except Exception as exc:
            self.fail(f"train_model raised an unexpected exception: {exc}")


class TestMedGANSynthesizeCount(unittest.TestCase):
    """synthesize_dataset(num_samples=5) returns exactly 5 dicts."""

    def setUp(self):
        dataset = _make_dataset()
        self.model = MedGAN(dataset, **_SMALL_MODEL_KWARGS)

    def test_synthesize_returns_correct_count(self):
        result = self.model.synthesize_dataset(num_samples=5)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)


class TestMedGANSynthesizeOutputStructure(unittest.TestCase):
    """Each synthesized dict has patient_id (str) and visits (flat list of str)."""

    def setUp(self):
        dataset = _make_dataset()
        self.model = MedGAN(dataset, **_SMALL_MODEL_KWARGS)

    def test_synthesize_output_structure(self):
        result = self.model.synthesize_dataset(num_samples=3)
        for i, item in enumerate(result):
            self.assertIsInstance(item, dict, f"Item {i} is not a dict")
            self.assertIn("patient_id", item, f"Item {i} missing 'patient_id'")
            self.assertIn("visits", item, f"Item {i} missing 'visits'")
            self.assertIsInstance(
                item["patient_id"], str, f"patient_id in item {i} is not a str"
            )
            self.assertIsInstance(
                item["visits"], list, f"visits in item {i} is not a list"
            )
            for code in item["visits"]:
                self.assertIsInstance(
                    code, str, f"code '{code}' in item {i} is not a str"
                )


class TestMedGANSaveLoadRoundtrip(unittest.TestCase):
    """save_model then load_model; synthesize_dataset still returns correct count."""

    def test_save_load_roundtrip(self):
        dataset = _make_dataset()
        model = MedGAN(dataset, **_SMALL_MODEL_KWARGS)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "medgan_test.pt")
            model.save_model(ckpt_path)
            self.assertTrue(
                os.path.exists(ckpt_path),
                f"Expected checkpoint at {ckpt_path}",
            )
            model.load_model(ckpt_path)
            result = model.synthesize_dataset(num_samples=3)
            self.assertEqual(len(result), 3)


# ---------------------------------------------------------------------------
# Category B: MIMIC-III Integration Tests (skipped if data unavailable)
# ---------------------------------------------------------------------------

_MIMIC3_PATH = os.environ.get(
    "PYHEALTH_MIMIC3_PATH",
    "/srv/local/data/physionet.org/files/mimiciii/1.4",
)


class TestMedGANMIMIC3Integration(unittest.TestCase):
    """End-to-end pipeline test with actual MIMIC-III data."""

    @classmethod
    def setUpClass(cls):
        cls.skip_integration = False
        cls.skip_reason = ""
        try:
            _saved_stub = sys.modules.pop("pyhealth.datasets", None)
            try:
                import importlib as _il

                _il.invalidate_caches()
                from pyhealth.datasets import MIMIC3Dataset as _MIMIC3Dataset
                from pyhealth.tasks.medgan_generation import (
                    MedGANGenerationMIMIC3,
                )
            except (ImportError, ModuleNotFoundError) as exc:
                if _saved_stub is not None:
                    sys.modules["pyhealth.datasets"] = _saved_stub
                raise ImportError(str(exc)) from exc

            cls.dataset = _MIMIC3Dataset(
                root=_MIMIC3_PATH,
                tables=["diagnoses_icd"],
            )
            task = MedGANGenerationMIMIC3()
            cls.sample_dataset = cls.dataset.set_task(task)
        except (FileNotFoundError, OSError, ImportError, ValueError) as exc:
            cls.skip_integration = True
            cls.skip_reason = str(exc)

    def setUp(self):
        if self.skip_integration:
            self.skipTest(
                f"MIMIC-III integration test skipped: {self.skip_reason}"
            )

    def test_mimic3_set_task_returns_nonempty_dataset(self):
        self.assertGreater(len(self.sample_dataset), 0)

    def test_mimic3_sample_keys(self):
        for sample in self.sample_dataset:
            self.assertIn("patient_id", sample)
            self.assertIn("visits", sample)

    def test_mimic3_visits_are_flat_multihot_tensors(self):
        processor = self.sample_dataset.input_processors["visits"]
        vocab_size = processor.size()
        for sample in self.sample_dataset:
            visits = sample["visits"]
            self.assertIsInstance(visits, torch.Tensor)
            self.assertEqual(visits.shape, (vocab_size,))
            self.assertEqual(visits.dtype, torch.float32)
            self.assertTrue(
                torch.all((visits == 0.0) | (visits == 1.0)),
                "visits tensor contains values outside {0, 1}",
            )

    def test_mimic3_full_pipeline_train_and_synthesize(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MedGAN(
                self.sample_dataset,
                latent_dim=64,
                hidden_dim=64,
                batch_size=32,
                ae_epochs=1,
                gan_epochs=1,
                save_dir=tmpdir,
            )
            model.train_model(self.sample_dataset, val_dataset=None)
            synthetic = model.synthesize_dataset(num_samples=10)
            self.assertEqual(len(synthetic), 10)
            for item in synthetic:
                self.assertIn("patient_id", item)
                self.assertIn("visits", item)
                self.assertIsInstance(item["visits"], list)


if __name__ == "__main__":
    unittest.main()
