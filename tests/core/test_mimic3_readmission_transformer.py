"""
Lightweight integration tests for examples/mimic3_readmission_transformer.py.

These tests verify that:

1. The example pipeline runs end-to-end without errors.
2. The returned metrics dictionary has expected keys and sane values.

They are intentionally small and configurable so they can be used both in local
development and in CI environments.

Environment variables
---------------------
PYHEALTH_MIMIC3_ROOT         (optional)
    If set, points to a local MIMIC-III root directory. When present and not
    starting with "http", tests will use this path without skipping.

PYHEALTH_RUN_ONLINE_EXAMPLES (optional)
    If set to "1" / "true" / "True" and PYHEALTH_MIMIC3_ROOT is a remote URL
    (e.g., Synthetic MIMIC-III), tests are allowed to download from that URL.

    If PYHEALTH_MIMIC3_ROOT is not set, the tests will fall back to the default
    Synthetic MIMIC-III URL from the example module. In that case, a remote URL
    also requires PYHEALTH_RUN_ONLINE_EXAMPLES to be enabled; otherwise tests
    are skipped to avoid unexpected network access in CI.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict

import pytest

def _load_example_module():
    """Dynamically load examples/mimic3_readmission_transformer.py."""
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"
    module_path = examples_dir / "mimic3_readmission_transformer.py"

    if not module_path.exists():
        pytest.skip(
            f"Example script not found at expected path: {module_path!s}. "
            "Ensure you are running tests from the PyHealth repository root."
        )

    spec = importlib.util.spec_from_file_location(
        "mimic3_readmission_transformer", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot create spec for mimic3_readmission_transformer")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]

    return module


_example_mod = _load_example_module()
run_mimic3_readmission_transformer = _example_mod.run_mimic3_readmission_transformer
DEFAULT_MIMIC3_SYNTHETIC_ROOT = _example_mod.DEFAULT_MIMIC3_SYNTHETIC_ROOT


_ENV_MIMIC3_ROOT = "PYHEALTH_MIMIC3_ROOT"
_ENV_RUN_ONLINE = "PYHEALTH_RUN_ONLINE_EXAMPLES"


def _resolve_mimic3_root() -> str:
    """
    Resolve which MIMIC-III root to use for tests.

    Preference:
        1. PYHEALTH_MIMIC3_ROOT if set
        2. DEFAULT_MIMIC3_SYNTHETIC_ROOT from the example module

    If the resolved root is remote (starts with http/https), tests will only run
    when PYHEALTH_RUN_ONLINE_EXAMPLES is set; otherwise they will be skipped to
    avoid unexpected network I/O in automated test environments.
    """
    root = os.getenv(_ENV_MIMIC3_ROOT, DEFAULT_MIMIC3_SYNTHETIC_ROOT)

    if root.startswith("http"):
        run_online = os.getenv(_ENV_RUN_ONLINE, "0")
        if run_online not in {"1", "true", "True"}:
            pytest.skip(
                "Using a remote MIMIC-III source (Synthetic demo). "
                "Set PYHEALTH_RUN_ONLINE_EXAMPLES=1 to enable online example tests."
            )

    return root


@pytest.mark.integration
def test_mimic3_readmission_transformer_pipeline_runs_end_to_end() -> None:
    """
    Ensure the example pipeline runs end-to-end and produces basic metrics.

    This test uses:
        - dev=True to keep runtime small
        - epochs=1 and small batch_size for quick feedback

    It does not assert specific metric values; instead, it focuses on:
        - correct wiring of the dataset/task/model/trainer
        - existence and sanity of the returned metrics
    """
    root = _resolve_mimic3_root()

    metrics: Dict[str, Any] = run_mimic3_readmission_transformer(
        root=root,
        batch_size=8,
        epochs=1,
        dev=True,
        num_workers=0,
    )

    # Basic structure checks
    assert isinstance(metrics, dict)
    for key in ("loss", "pr_auc", "roc_auc", "num_samples"):
        assert key in metrics, f"Missing expected metric key: {key}"

    # Sanity checks on values
    assert metrics["num_samples"] > 0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


@pytest.mark.integration
def test_mimic3_readmission_transformer_accepts_custom_tables() -> None:
    """
    Verify that overriding the `tables` argument still yields a valid run.

    This is primarily a smoke test for the `tables` parameter wiring.
    """
    root = _resolve_mimic3_root()

    metrics: Dict[str, Any] = run_mimic3_readmission_transformer(
        root=root,
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        batch_size=8,
        epochs=1,
        dev=True,
        num_workers=0,
    )

    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
