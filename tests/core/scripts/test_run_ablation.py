"""Tests for scripts/run_ablation.sh.

We pass ``DRY_RUN=1`` so the script echoes its resolved config and
exits before spawning Python — this lets us verify:

1. Usage / error paths.
2. That dataset-appropriate defaults are echoed correctly.
3. That RUN_* env overrides pass through.

Earlier iterations of these tests stubbed ``PYTHON=/bin/true`` but the
behavior of that trick varied across systems (seen failing on CI); the
DRY_RUN hook in the script is portable and self-documenting.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_ablation.sh"


def _run(args, env_extra, cwd=None):
    env = os.environ.copy()
    env.update(env_extra)
    return subprocess.run(
        [str(SCRIPT)] + list(args),
        env=env, cwd=cwd or str(REPO_ROOT),
        capture_output=True, text=True,
    )


def test_usage_on_missing_dataset():
    r = _run([], {})
    assert r.returncode != 0
    assert "Usage" in r.stderr


def test_usage_on_unknown_dataset():
    r = _run(["unknown"], {})
    assert r.returncode != 0
    assert "Usage" in r.stderr


def test_missing_data_root_errors_with_hint(tmp_path):
    # Point at an empty path that doesn't have the data laid out.
    nonexistent = tmp_path / "no_data"
    r = _run(
        ["hippocampus"],
        {"RUN_DATA_ROOT": str(nonexistent), "DRY_RUN": "1"},
    )
    assert r.returncode != 0
    assert "data root not found" in r.stderr
    assert "download_data.sh" in r.stderr  # the [hint] line


def test_hippocampus_defaults_echoed(tmp_path):
    # Fake data root so the existence check passes.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    r = _run(
        ["hippocampus", "5"],
        {
            "RUN_DATA_ROOT": str(data_dir),
            "DRY_RUN": "1",
            "LOG_DIR": str(tmp_path / "logs"),
        },
    )
    assert r.returncode == 0, f"stderr={r.stderr}"
    assert "dataset=hippocampus" in r.stdout
    assert "epochs=5" in r.stdout
    assert "min/max_size=128/128" in r.stdout
    assert "4,8,16,32,64" in r.stdout
    assert "<none>" in r.stdout  # hippocampus default: no HU window


def test_spleen_defaults_echoed(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    r = _run(
        ["spleen", "3", "1.0"],
        {
            "RUN_DATA_ROOT": str(data_dir),
            "DRY_RUN": "1",
            "LOG_DIR": str(tmp_path / "logs"),
        },
    )
    assert r.returncode == 0
    assert "dataset=spleen" in r.stdout
    assert "epochs=3" in r.stdout
    assert "lambdas=1.0" in r.stdout
    assert "min/max_size=256/256" in r.stdout
    assert "16,32,64,128,256" in r.stdout
    assert "-160,240" in r.stdout


def test_env_overrides_win_over_defaults(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    r = _run(
        ["spleen"],
        {
            "RUN_DATA_ROOT": str(data_dir),
            "DRY_RUN": "1",
            "LOG_DIR": str(tmp_path / "logs"),
            "RUN_MIN_SIZE": "192",
            "RUN_MAX_SIZE": "192",
            "RUN_ANCHORS": "8,16,32,64,128",
            "RUN_HU_WINDOW": "-50,150",
            "RUN_LR": "1e-3",
            "EPOCHS": "7",
        },
    )
    assert r.returncode == 0
    assert "epochs=7" in r.stdout
    assert "min/max_size=192/192" in r.stdout
    assert "8,16,32,64,128" in r.stdout
    assert "-50,150" in r.stdout
