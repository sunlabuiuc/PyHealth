"""End-to-end integration tests for the PromptEHR synthetic EHR generation pipeline.

Category A tests use InMemorySampleDataset with synthetic data — no external
data required and must always pass.

Category B tests require actual MIMIC-III data and are skipped gracefully when
the data is unavailable.

The bootstrap pattern mirrors test_corgan_end_to_end.py: load PromptEHR and
InMemorySampleDataset via importlib while stubbing out heavy optional
dependencies (litdata, pyarrow) that are not yet in the venv. transformers IS
available in the venv and is loaded normally.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Bootstrap: load PromptEHR, BaseModel, and InMemorySampleDataset without
# triggering pyhealth.models.__init__ (many models have unavailable deps) or
# pyhealth.datasets.__init__ (requires litdata, pyarrow, ...).
# ---------------------------------------------------------------------------


def _bootstrap():
    """Load PromptEHR, BaseModel, and InMemorySampleDataset via importlib.

    Returns:
        (BaseModel, PromptEHR, InMemorySampleDataset)
    """
    import pyhealth  # noqa: F401  — top-level __init__ has no heavy deps

    # Stub pyhealth.datasets so that base_model.py's
    # "from ..datasets import SampleDataset" resolves cleanly.
    if "pyhealth.datasets" not in sys.modules:
        ds_stub = MagicMock()

        class _FakeSampleDataset:  # noqa: N801
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
    models_dir = os.path.join(root, "pyhealth", "models")
    promptehr_dir = os.path.join(models_dir, "promptehr")

    # Load base_model and expose via stub.
    bm_mod = _load_file(
        "pyhealth.models.base_model", os.path.join(models_dir, "base_model.py")
    )
    BaseModel = bm_mod.BaseModel
    models_stub.BaseModel = BaseModel

    # Create a package stub for pyhealth.models.promptehr so that
    # model.py's relative imports (from .conditional_prompt import ...) work.
    promptehr_pkg_stub = MagicMock()
    sys.modules.setdefault("pyhealth.models.promptehr", promptehr_pkg_stub)

    # Load each PromptEHR submodule in dependency order.
    # Each is standalone (only torch + transformers, no cross-module imports).
    for mod_name in (
        "conditional_prompt",
        "bart_encoder",
        "bart_decoder",
        "visit_sampler",
        "generation",
    ):
        _load_file(
            f"pyhealth.models.promptehr.{mod_name}",
            os.path.join(promptehr_dir, f"{mod_name}.py"),
        )

    # Load model.py last (depends on the submodules loaded above + BaseModel).
    model_mod = _load_file(
        "pyhealth.models.promptehr.model",
        os.path.join(promptehr_dir, "model.py"),
    )
    PromptEHR = model_mod.PromptEHR

    # Stub litdata so sample_dataset.py can be loaded without the full package.
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

    return BaseModel, PromptEHR, InMemorySampleDataset


BaseModel, PromptEHR, InMemorySampleDataset = _bootstrap()

import torch  # noqa: E402
from transformers import BartConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Nested lists of code strings — PromptEHR uses nested_sequence schema.
# 8 samples with ≥2 visits each, plus demographics.
_SMALL_SAMPLES = [
    {"patient_id": "p1", "visits": [["A", "B"], ["C", "D"]], "age": 65.0, "gender": 0},
    {"patient_id": "p2", "visits": [["E"], ["F", "G"]], "age": 45.0, "gender": 1},
    {"patient_id": "p3", "visits": [["A", "C"], ["B", "E"]], "age": 55.0, "gender": 0},
    {"patient_id": "p4", "visits": [["D"], ["A"]], "age": 70.0, "gender": 1},
    {"patient_id": "p5", "visits": [["B", "F"], ["C", "G"]], "age": 40.0, "gender": 0},
    {"patient_id": "p6", "visits": [["E", "A"], ["D"]], "age": 60.0, "gender": 1},
    {"patient_id": "p7", "visits": [["G", "B"], ["F", "A"]], "age": 50.0, "gender": 0},
    {"patient_id": "p8", "visits": [["C"], ["D", "E"]], "age": 35.0, "gender": 1},
]

# Tiny BART config to keep tests fast (avoids downloading/using 768-dim bart-base).
_TINY_BART_CONFIG = BartConfig(
    d_model=32,
    encoder_layers=1,
    decoder_layers=1,
    encoder_ffn_dim=64,
    decoder_ffn_dim=64,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
    max_position_embeddings=128,
)

# Minimal model kwargs — tiny architecture and 1 epoch to keep tests fast.
_SMALL_MODEL_KWARGS = dict(
    n_num_features=1,
    cat_cardinalities=[2],
    d_hidden=32,
    prompt_length=1,
    bart_config_name=_TINY_BART_CONFIG,
    epochs=1,
    batch_size=4,
    warmup_steps=0,
    max_seq_length=64,
)


def _make_dataset(samples=None):
    if samples is None:
        samples = _SMALL_SAMPLES
    return InMemorySampleDataset(
        samples=samples,
        input_schema={"visits": "nested_sequence"},
        output_schema={},
    )


def _make_trained_model():
    """Return a PromptEHR model trained for 1 epoch on _SMALL_SAMPLES."""
    dataset = _make_dataset()
    tmpdir = tempfile.mkdtemp()
    model = PromptEHR(dataset, save_dir=tmpdir, **_SMALL_MODEL_KWARGS)
    model.train_model(dataset)
    return model, tmpdir


# ---------------------------------------------------------------------------
# Category A: In-Memory Integration Tests (must always pass)
# ---------------------------------------------------------------------------


class TestPromptEHRIsBaseModelInstance(unittest.TestCase):
    """PromptEHR model is an instance of BaseModel."""

    def test_model_is_basemodel_instance(self):
        dataset = _make_dataset()
        model = PromptEHR(dataset, **_SMALL_MODEL_KWARGS)
        self.assertIsInstance(model, BaseModel)


class TestPromptEHRFeatureKeys(unittest.TestCase):
    """model.feature_keys equals ['visits']."""

    def test_feature_keys(self):
        dataset = _make_dataset()
        model = PromptEHR(dataset, **_SMALL_MODEL_KWARGS)
        self.assertEqual(model.feature_keys, ["visits"])


class TestPromptEHRVocabSize(unittest.TestCase):
    """_vocab.total_size equals processor.vocab_size() + 5."""

    def test_vocab_size_matches_processor(self):
        dataset = _make_dataset()
        processor = dataset.input_processors["visits"]
        model = PromptEHR(dataset, **_SMALL_MODEL_KWARGS)
        expected = processor.vocab_size() + 5
        self.assertEqual(model._vocab.total_size, expected)


class TestPromptEHRForwardRaisesNotImplementedError(unittest.TestCase):
    """Calling forward() raises NotImplementedError.

    PromptEHR is a generative model; the discriminative forward pass is not
    applicable.
    """

    def test_forward_not_implemented(self):
        dataset = _make_dataset()
        model = PromptEHR(dataset, **_SMALL_MODEL_KWARGS)
        with self.assertRaises(NotImplementedError):
            model.forward()


class TestPromptEHRTrainModelRuns(unittest.TestCase):
    """train_model completes one epoch without error."""

    def test_train_model_runs_one_epoch(self):
        dataset = _make_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            model = PromptEHR(dataset, save_dir=tmpdir, **_SMALL_MODEL_KWARGS)
            try:
                model.train_model(dataset, val_dataset=None)
            except Exception as exc:  # noqa: BLE001
                self.fail(f"train_model raised an unexpected exception: {exc}")
            # A checkpoint must be saved after training
            ckpt = os.path.join(tmpdir, "checkpoint.pt")
            self.assertTrue(os.path.exists(ckpt), f"Expected checkpoint at {ckpt}")


class TestPromptEHRSynthesizeCount(unittest.TestCase):
    """synthesize_dataset(num_samples=3) returns exactly 3 dicts."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tmpdir = _make_trained_model()

    def test_synthesize_returns_correct_count(self):
        result = self.model.synthesize_dataset(num_samples=3)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)


class TestPromptEHRSynthesizeOutputStructure(unittest.TestCase):
    """Each synthesized dict has patient_id (str) and visits (nested list of str).

    PromptEHR outputs nested visit lists — each patient is a list of visits,
    each visit is a list of diagnosis code strings.
    """

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tmpdir = _make_trained_model()

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
            # visits is a nested list: list of visits, each visit a list of strings
            for visit_idx, visit in enumerate(item["visits"]):
                self.assertIsInstance(
                    visit, list,
                    f"visit {visit_idx} in item {i} is not a list"
                )
                for code in visit:
                    self.assertIsInstance(
                        code, str,
                        f"code '{code}' in visit {visit_idx}, item {i} is not str"
                    )


class TestPromptEHRSaveLoadRoundtrip(unittest.TestCase):
    """save_model then load_model; synthesize_dataset returns correct count."""

    def test_save_load_roundtrip(self):
        dataset = _make_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            model = PromptEHR(dataset, save_dir=tmpdir, **_SMALL_MODEL_KWARGS)
            model.train_model(dataset)
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
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


class TestPromptEHRMIMIC3Integration(unittest.TestCase):
    """End-to-end pipeline test with actual MIMIC-III data.

    Skipped automatically when MIMIC-III is not present on this machine.
    """

    @classmethod
    def setUpClass(cls):
        cls.skip_integration = False
        cls.skip_reason = ""
        try:
            # Remove bootstrap stubs so we can attempt a real import.
            _saved_ds_stub = sys.modules.pop("pyhealth.datasets", None)
            try:
                import importlib as _il
                _il.invalidate_caches()
                from pyhealth.datasets import MIMIC3Dataset as _MIMIC3Dataset
                from pyhealth.tasks.ehr_generation import PromptEHRGenerationMIMIC3
            except (ImportError, ModuleNotFoundError) as exc:
                if _saved_ds_stub is not None:
                    sys.modules["pyhealth.datasets"] = _saved_ds_stub
                raise ImportError(str(exc)) from exc

            cls.dataset = _MIMIC3Dataset(
                root=_MIMIC3_PATH,
                tables=["patients", "admissions", "diagnoses_icd"],
            )
            task = PromptEHRGenerationMIMIC3()
            cls.sample_dataset = cls.dataset.set_task(task)
        except (FileNotFoundError, OSError, ImportError, ValueError) as exc:
            cls.skip_integration = True
            cls.skip_reason = str(exc)

    def setUp(self):
        if self.skip_integration:
            self.skipTest(f"MIMIC-III integration test skipped: {self.skip_reason}")

    def test_mimic3_set_task_returns_nonempty_dataset(self):
        """set_task produces at least one sample from MIMIC-III."""
        self.assertGreater(len(self.sample_dataset), 0)

    def test_mimic3_sample_keys(self):
        """Every sample must contain patient_id, visits, age, and gender keys."""
        for sample in self.sample_dataset:
            self.assertIn("patient_id", sample)
            self.assertIn("visits", sample)
            self.assertIn("age", sample)
            self.assertIn("gender", sample)

    def test_mimic3_visits_are_nested_tensors(self):
        """visits must be a list of 1-D int64 tensors (NestedSequenceProcessor output).

        NestedSequenceProcessor encodes each visit as a 1-D LongTensor of
        code indices. This verifies the nested_sequence schema round-trips
        correctly through set_task.
        """
        for sample in self.sample_dataset:
            visits = sample["visits"]
            self.assertIsInstance(visits, list)
            self.assertGreater(len(visits), 0)
            for visit in visits:
                self.assertIsInstance(visit, torch.Tensor)
                self.assertEqual(visit.dtype, torch.long)

    def test_mimic3_full_pipeline_train_and_synthesize(self):
        """Train one epoch on MIMIC-III data and synthesize a small batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = PromptEHR(
                self.sample_dataset,
                d_hidden=64,
                prompt_length=1,
                bart_config_name=_TINY_BART_CONFIG,
                epochs=1,
                batch_size=16,
                warmup_steps=0,
                save_dir=tmpdir,
            )
            model.train_model(self.sample_dataset, val_dataset=None)
            synthetic = model.synthesize_dataset(num_samples=5)
            self.assertEqual(len(synthetic), 5)
            for item in synthetic:
                self.assertIn("patient_id", item)
                self.assertIn("visits", item)
                self.assertIsInstance(item["visits"], list)


if __name__ == "__main__":
    unittest.main()
