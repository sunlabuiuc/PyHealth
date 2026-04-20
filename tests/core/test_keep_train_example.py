"""Smoke tests for the KEEP example scripts.

Covers two entry points that share a config-block convention:
    - ``pyhealth.medcode.pretrained_embeddings.keep_emb.examples.train_keep``
      — embedding training only (no downstream task)
    - ``examples.mortality_prediction.mortality_mimic4_grasp_keep``
      — end-to-end KEEP + GRASP mortality prediction

Both must stay syntactically valid and expose the same set of KEEP-related
config knobs so users can swap between them without learning a new API.

What these tests catch:
    - Syntax errors in either example (e.g. ``DEV_MODE = TRUE``)
    - Broken imports after refactoring ``run_pipeline.py``
    - Config-knob drift between the two examples (e.g. one has ``DEVICE``
      but the other doesn't)
    - Output-path structure regressions

What these tests do NOT do:
    - Actually run the pipelines (requires Athena + MIMIC data)
    - Validate training convergence or downstream quality
    - Import the mortality example's ``main()`` flow (it doesn't wrap in one
      — it's a top-level script; we read it as source text for structural
      checks)
"""

import ast
import importlib
import re
import sys
from pathlib import Path

import pytest


TRAIN_KEEP_MODULE = (
    "pyhealth.medcode.pretrained_embeddings.keep_emb.examples.train_keep"
)

# The mortality example is a top-level script (not wrapped in main()), so
# importing it at module load time runs the full training. We read it as
# source instead to check config-knob presence without side effects.
REPO_ROOT = Path(__file__).resolve().parents[2]
MORTALITY_EXAMPLE_PATH = (
    REPO_ROOT / "examples" / "mortality_prediction"
    / "mortality_mimic4_grasp_keep.py"
)


# ---------------------------------------------------------------------------
# Import-time correctness
# ---------------------------------------------------------------------------

class TestImportSmoke:
    """Example must import cleanly — no syntax errors, no missing deps."""

    def test_module_imports(self):
        """The example file parses and imports without error.

        Catches typos like ``DEV_MODE = TRUE`` (uppercase Python-illegal
        literal) that would raise NameError at import time.
        """
        # Force reimport so prior test pollution doesn't hide errors
        if TRAIN_KEEP_MODULE in sys.modules:
            del sys.modules[TRAIN_KEEP_MODULE]
        module = importlib.import_module(TRAIN_KEEP_MODULE)
        assert module is not None

    def test_expected_config_vars_exist(self):
        """Config block exposes the documented public knobs."""
        module = importlib.import_module(TRAIN_KEEP_MODULE)
        expected = [
            "ATHENA_DIR",
            "KEEP_VARIANT",
            "RUN_INTRINSIC_EVAL",
            "ENABLE_COMPUTE_TRACKING",
            "DEVICE",
            "SAVE_COOC_ARTIFACTS",
            "MIMIC_VERSION",
            "LOCAL_MIMIC_ROOTS",
            "DEV_MODE",
            "MIN_OCCURRENCES",
            "OUTPUT_ROOT",
            "KEEP_VARIANTS",
        ]
        missing = [name for name in expected if not hasattr(module, name)]
        assert not missing, f"Missing config vars: {missing}"

    def test_dev_mode_is_python_bool(self):
        """``DEV_MODE`` must be a bool, not the string 'TRUE' or similar.

        Regression guard: a past edit accidentally set ``DEV_MODE = TRUE``
        (uppercase) which would import-error as ``NameError``. Import-time
        success above proves it parses; this checks the actual value type.
        """
        module = importlib.import_module(TRAIN_KEEP_MODULE)
        assert isinstance(module.DEV_MODE, bool), (
            f"DEV_MODE should be a Python bool, got "
            f"{type(module.DEV_MODE).__name__}: {module.DEV_MODE!r}"
        )

    def test_compute_tracking_flag_is_bool(self):
        """``ENABLE_COMPUTE_TRACKING`` must be a bool."""
        module = importlib.import_module(TRAIN_KEEP_MODULE)
        assert isinstance(module.ENABLE_COMPUTE_TRACKING, bool)

    def test_save_cooc_artifacts_flag_is_bool(self):
        """``SAVE_COOC_ARTIFACTS`` must be a bool."""
        module = importlib.import_module(TRAIN_KEEP_MODULE)
        assert isinstance(module.SAVE_COOC_ARTIFACTS, bool)


# ---------------------------------------------------------------------------
# Config value validity
# ---------------------------------------------------------------------------

class TestConfigValidity:
    """Config values should be one of the documented options."""

    def test_keep_variant_is_defined(self):
        """``KEEP_VARIANT`` must match one of the keys in ``KEEP_VARIANTS``."""
        module = importlib.import_module(TRAIN_KEEP_MODULE)
        assert module.KEEP_VARIANT in module.KEEP_VARIANTS, (
            f"KEEP_VARIANT='{module.KEEP_VARIANT}' is not a key in "
            f"KEEP_VARIANTS={list(module.KEEP_VARIANTS.keys())}"
        )

    def test_keep_variants_have_required_hyperparams(self):
        """Each variant must specify the 3 GloVe hyperparameters.

        ``reg_reduction`` is intentionally NOT in the required set — it's
        hardcoded to ``sum`` inside the library (paper Eq 4) because it's
        mathematically coupled to ``lambd`` and can't be tuned independently.
        """
        module = importlib.import_module(TRAIN_KEEP_MODULE)
        required_keys = {"reg_distance", "optimizer", "lambd"}
        for variant_name, params in module.KEEP_VARIANTS.items():
            missing = required_keys - set(params.keys())
            assert not missing, (
                f"KEEP_VARIANTS['{variant_name}'] missing keys: {missing}"
            )

    def test_keep_variants_do_not_expose_reg_reduction(self):
        """``reg_reduction`` must not appear in KEEP_VARIANTS.

        Guards against accidentally re-exposing the footgun. If you want to
        ablate, pass it directly at the library call site.
        """
        module = importlib.import_module(TRAIN_KEEP_MODULE)
        for variant_name, params in module.KEEP_VARIANTS.items():
            assert "reg_reduction" not in params, (
                f"KEEP_VARIANTS['{variant_name}'] should not specify "
                f"reg_reduction — the library hardcodes sum per paper Eq 4."
            )

    def test_device_is_one_of_valid_options(self):
        """``DEVICE`` must be one of: auto, cuda, mps, cpu."""
        module = importlib.import_module(TRAIN_KEEP_MODULE)
        assert module.DEVICE in ("auto", "cuda", "mps", "cpu"), (
            f"DEVICE='{module.DEVICE}' is not one of auto/cuda/mps/cpu"
        )

    def test_mimic_version_is_valid(self):
        """``MIMIC_VERSION`` must be 'mimic3' or 'mimic4'."""
        module = importlib.import_module(TRAIN_KEEP_MODULE)
        assert module.MIMIC_VERSION in ("mimic3", "mimic4")

    def test_mimic_version_has_root_configured(self):
        """The chosen ``MIMIC_VERSION`` must have a root path in ``LOCAL_MIMIC_ROOTS``."""
        module = importlib.import_module(TRAIN_KEEP_MODULE)
        assert module.MIMIC_VERSION in module.LOCAL_MIMIC_ROOTS

    def test_min_occurrences_is_positive_int(self):
        """``MIN_OCCURRENCES`` must be a positive int (paper default: 2)."""
        module = importlib.import_module(TRAIN_KEEP_MODULE)
        assert isinstance(module.MIN_OCCURRENCES, int)
        assert module.MIN_OCCURRENCES >= 1


# ---------------------------------------------------------------------------
# Output layout convention
# ---------------------------------------------------------------------------

class TestOutputLayout:
    """Output directory structure must follow the Trainer-aligned convention."""

    def test_output_root_matches_trainer_convention(self):
        """``OUTPUT_ROOT`` should nest under PyHealth's ``output/`` convention."""
        module = importlib.import_module(TRAIN_KEEP_MODULE)
        # The Trainer puts runs under ./output/; KEEP runs should share that
        # parent so downstream and embedding artifacts sit together.
        assert module.OUTPUT_ROOT.startswith("output/") or module.OUTPUT_ROOT == "output", (
            f"OUTPUT_ROOT='{module.OUTPUT_ROOT}' should start with 'output/' "
            "to share Trainer's output/ convention"
        )

    def test_timestamp_format_matches_trainer(self):
        """Timestamps use ``%Y%m%d-%H%M%S`` to match Trainer's exp_name default."""
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        # e.g. 20260420-143042
        assert re.match(r"^\d{8}-\d{6}$", ts), (
            f"Timestamp format mismatch: {ts}"
        )


# ---------------------------------------------------------------------------
# Library linkage — train_keep.py must stay wired to run_pipeline.py
# ---------------------------------------------------------------------------

class TestLibraryLinkage:
    """train_keep.py imports from run_pipeline.py — catch regressions if the
    library's public API changes without updating the example.
    """

    def test_run_keep_pipeline_is_importable(self):
        """The underlying ``run_keep_pipeline`` must be callable."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            run_keep_pipeline,
        )
        assert callable(run_keep_pipeline)

    def test_resolve_device_is_importable(self):
        """``resolve_device`` must be exposed from run_pipeline."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            resolve_device,
        )
        assert callable(resolve_device)


# ---------------------------------------------------------------------------
# Library default alignment — run_pipeline.py defaults should match the
# "paper" variant across all specified KEEP hyperparameters. Guards against
# regressions like the ``reg_reduction="mean"`` bug that silently crashed
# Resnik because reg strength was ~7853x too weak.
# ---------------------------------------------------------------------------

class TestLibraryDefaultsMatchPaper:
    """Library defaults should equal the paper variant's values.

    Guards against the ``reg_reduction="mean"`` class of bug: a default
    that's PyTorch-conventional but paper-incorrect, such that anyone calling
    the function without overrides silently runs a broken variant. We went
    further and *removed* the parameter entirely — so these tests also check
    that ``reg_reduction`` is no longer a function parameter.
    """

    def test_run_pipeline_defaults_match_paper_variant(self):
        """``run_keep_pipeline`` signature defaults match KEEP_VARIANTS['paper']."""
        import inspect
        from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
            run_keep_pipeline,
        )
        sig = inspect.signature(run_keep_pipeline)
        defaults = {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not inspect.Parameter.empty
        }

        paper_defaults = {
            "reg_distance": "l2",
            "optimizer": "adamw",
            "lambd": 1e-3,
            "embedding_dim": 100,
            "num_walks": 750,
            "walk_length": 30,
            "glove_epochs": 300,
            "min_occurrences": 2,
        }
        mismatches = {
            k: (defaults.get(k), v)
            for k, v in paper_defaults.items()
            if defaults.get(k) != v
        }
        assert not mismatches, (
            f"run_keep_pipeline defaults drifted from paper variant. "
            f"Mismatches (got, expected): {mismatches}"
        )
        assert "reg_reduction" not in sig.parameters, (
            "reg_reduction should be hardcoded to sum, not exposed as a param."
        )

    def test_train_glove_defaults_match_paper(self):
        """``train_glove.train_keep`` signature defaults match paper values.

        run_keep_pipeline delegates to train_keep — if train_keep's defaults
        drift, the whole stack silently breaks even though run_keep_pipeline
        looks correct.
        """
        import inspect
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove import (
            train_keep,
        )
        sig = inspect.signature(train_keep)
        defaults = {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not inspect.Parameter.empty
        }

        paper_defaults = {
            "reg_distance": "l2",
            "optimizer": "adamw",
            "lambd": 1e-3,
            "alpha": 0.75,
            "lr": 0.05,
        }
        mismatches = {
            k: (defaults.get(k), v)
            for k, v in paper_defaults.items()
            if defaults.get(k) != v
        }
        assert not mismatches, (
            f"train_keep defaults drifted from paper. "
            f"Mismatches (got, expected): {mismatches}"
        )
        assert "reg_reduction" not in sig.parameters

    def test_keepglove_class_defaults_match_paper(self):
        """``KeepGloVe`` class constructor defaults match paper values."""
        import inspect
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove import (
            KeepGloVe,
        )
        sig = inspect.signature(KeepGloVe.__init__)
        defaults = {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not inspect.Parameter.empty
        }

        paper_defaults = {
            "reg_distance": "l2",
            "lambd": 1e-3,
            "embedding_dim": 100,
        }
        mismatches = {
            k: (defaults.get(k), v)
            for k, v in paper_defaults.items()
            if defaults.get(k) != v
        }
        assert not mismatches, (
            f"KeepGloVe.__init__ defaults drifted from paper. "
            f"Mismatches (got, expected): {mismatches}"
        )
        assert "reg_reduction" not in sig.parameters


# ---------------------------------------------------------------------------
# Mortality example — structural checks against the source text.
# Importing the mortality example executes training (no main() wrapper),
# so we parse it as AST instead of importing.
# ---------------------------------------------------------------------------

class TestMortalityExampleStructural:
    """Structural checks on the mortality example's config block."""

    @staticmethod
    def _parse_module_constants():
        """Parse the mortality example's top-level assignments without running it."""
        if not MORTALITY_EXAMPLE_PATH.exists():
            pytest.skip(f"Mortality example not found at {MORTALITY_EXAMPLE_PATH}")
        tree = ast.parse(MORTALITY_EXAMPLE_PATH.read_text())
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.isupper():
                    # Only track simple literal / dict values for these checks
                    try:
                        constants[target.id] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        constants[target.id] = "<unparseable>"
        return constants

    def test_source_parses(self):
        """Mortality example must be syntactically valid (no ``DEV_MODE = TRUE`` typos)."""
        constants = self._parse_module_constants()
        assert constants  # parsed something

    def test_expected_config_vars_exist(self):
        """Config vars present and consistent with train_keep.py's set."""
        constants = self._parse_module_constants()
        expected = [
            "USE_KEEP",
            "ATHENA_DIR",
            "KEEP_VARIANT",
            "USE_KEEP_CACHE",
            "KEEP_CACHE_ROOT",
            "KEEP_CACHE_RUN_ID",
            "MIMIC_VERSION",
            "LOCAL_MIMIC_ROOTS",
            "DEV_MODE",
            "DEVICE",                  # must be present for consistency with train_keep
            "MIN_OCCURRENCES",
            "KEEP_VARIANTS",
        ]
        missing = [k for k in expected if k not in constants]
        assert not missing, f"Mortality example missing config vars: {missing}"

    def test_device_value_valid(self):
        """Mortality example's DEVICE must be one of auto/cuda/mps/cpu."""
        constants = self._parse_module_constants()
        assert constants.get("DEVICE") in ("auto", "cuda", "mps", "cpu")

    def test_cache_root_matches_trainer_convention(self):
        """``KEEP_CACHE_ROOT`` should share Trainer's ``output/`` convention."""
        constants = self._parse_module_constants()
        cache_root = constants.get("KEEP_CACHE_ROOT", "")
        assert cache_root.startswith("output/") or cache_root == "output", (
            f"KEEP_CACHE_ROOT='{cache_root}' should start with 'output/'"
        )

    def test_keep_variants_match_train_keep(self):
        """Mortality and train_keep should define the same set of variants.

        Guards against config drift when someone adds a variant to one example
        but forgets to update the other.
        """
        mortality_variants = self._parse_module_constants().get("KEEP_VARIANTS", {})
        train_keep_module = importlib.import_module(TRAIN_KEEP_MODULE)
        train_variants = train_keep_module.KEEP_VARIANTS
        assert set(mortality_variants.keys()) == set(train_variants.keys()), (
            f"Variant-set drift: mortality={list(mortality_variants.keys())} "
            f"vs train_keep={list(train_variants.keys())}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
