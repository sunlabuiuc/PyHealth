"""Test configuration for ``tests/core``.

Registers a ``slow`` pytest marker that is skipped by default so the
default test run stays in the sub-second-per-test budget recommended by
the PyHealth contribution guide. Tests that genuinely exercise the full
``BaseDataset`` → ``set_task`` → model-training pipeline (which takes a
few seconds even on synthetic data) are marked ``slow`` and only run
when ``--run-slow`` is passed to pytest.

Usage:
    pytest tests/core/test_eol_mistrust_model.py              # fast only
    pytest tests/core/test_eol_mistrust_model.py --run-slow   # include integration
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (end-to-end pipeline tests).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: end-to-end integration test that exercises the full PyHealth "
        "pipeline (dataset → set_task → model). Skipped unless --run-slow.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(
        reason="Slow integration test; pass --run-slow to include."
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
