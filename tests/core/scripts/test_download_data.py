"""Tests for scripts/download_data.sh.

These tests build a tiny fake MSD tar with matching internal layout,
seed it into ``TAR_CACHE_DIR`` so the script's download step is
skipped, and let the script run its extraction + conversion on the
fake data. Assertions verify output layout, zero-padding preservation,
and axis transposition.
"""

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "download_data.sh"


def _make_nifti(path: Path, shape, fill=0.0, box=None) -> None:
    data = np.full(shape, fill_value=fill, dtype=np.float32)
    if box is not None:
        s = tuple(slice(a, b) for a, b in box)
        data[s] = 1.0
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, path)


def _build_fake_tar(tar_path: Path, subdir: str, stem: str, pids, vol_shape) -> None:
    """Build a tar with MSD-style imagesTr/labelsTr/<stem>_<pid>.nii.gz entries."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    staging = tar_path.with_suffix(".stage")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / subdir / "imagesTr").mkdir(parents=True)
    (staging / subdir / "labelsTr").mkdir(parents=True)

    for pid in pids:
        img_path = staging / subdir / "imagesTr" / f"{stem}_{pid}.nii.gz"
        lbl_path = staging / subdir / "labelsTr" / f"{stem}_{pid}.nii.gz"
        _make_nifti(img_path, vol_shape, fill=50.0)
        # Stick a small "lesion" in the first label slice (always non-empty)
        # so the convert path produces at least one foreground voxel.
        slice_box = [(0, 1)] + [(1, 3), (1, 3)] if len(vol_shape) == 3 else None
        _make_nifti(lbl_path, vol_shape, fill=0.0, box=slice_box)

    with tarfile.open(tar_path, "w") as tf:
        tf.add(staging / subdir, arcname=subdir)

    shutil.rmtree(staging)


def _run(script_args, env_extra, cwd=None):
    env = os.environ.copy()
    env.update(env_extra)
    return subprocess.run(
        [str(SCRIPT)] + list(script_args),
        env=env, cwd=cwd or str(REPO_ROOT),
        capture_output=True, text=True,
    )


def test_usage_on_missing_dataset():
    r = _run([], {})
    assert r.returncode != 0
    assert "Usage" in r.stderr


def test_usage_on_unknown_dataset():
    r = _run(["lungs"], {})
    assert r.returncode != 0
    assert "Usage" in r.stderr


def test_hippocampus_preserves_zero_padded_ids(tmp_path):
    cache = tmp_path / "cache"
    out_dir = tmp_path / "hippo_out"
    # IDs "001", "003", "042" — the padded strings used in the real MSD tar.
    _build_fake_tar(
        cache / "Task04_Hippocampus.tar",
        "Task04_Hippocampus",
        "hippocampus",
        pids=["001", "003", "042"],
        vol_shape=(4, 8, 8),  # axis 0 is slice axis; no transpose needed
    )

    r = _run(
        ["hippocampus", "2"],
        {"TAR_CACHE_DIR": str(cache), "DATA_OUT_DIR": str(out_dir)},
    )
    assert r.returncode == 0, f"stderr={r.stderr}\nstdout={r.stdout}"
    # First TWO patients by numeric order: 001, 003
    assert (out_dir / "patient_001" / "volume.npy").exists()
    assert (out_dir / "patient_001" / "mask.npy").exists()
    assert (out_dir / "patient_003" / "volume.npy").exists()
    # Third should NOT be extracted
    assert not (out_dir / "patient_042").exists()

    vol = np.load(out_dir / "patient_001" / "volume.npy")
    mask = np.load(out_dir / "patient_001" / "mask.npy")
    # Hippocampus path: no transpose, shape preserved.
    assert vol.shape == (4, 8, 8)
    assert mask.shape == (4, 8, 8)
    assert vol.dtype == np.float32
    assert mask.dtype == np.int8


def test_spleen_transposes_slice_axis_to_front(tmp_path):
    cache = tmp_path / "cache"
    out_dir = tmp_path / "spleen_out"
    # MSD spleen stores (H, W, D) = (8, 8, 4). After transpose: (4, 8, 8).
    _build_fake_tar(
        cache / "Task09_Spleen.tar",
        "Task09_Spleen",
        "spleen",
        pids=["2", "3"],
        vol_shape=(8, 8, 4),
    )

    r = _run(
        ["spleen", "2"],
        {"TAR_CACHE_DIR": str(cache), "DATA_OUT_DIR": str(out_dir)},
    )
    assert r.returncode == 0, f"stderr={r.stderr}\nstdout={r.stdout}"

    vol = np.load(out_dir / "patient_2" / "volume.npy")
    # Transposed from (8, 8, 4) -> (4, 8, 8); axis 0 is now the slice axis.
    assert vol.shape == (4, 8, 8)


def test_idempotent_second_run_adds_nothing(tmp_path):
    cache = tmp_path / "cache"
    out_dir = tmp_path / "hippo_out"
    _build_fake_tar(
        cache / "Task04_Hippocampus.tar",
        "Task04_Hippocampus",
        "hippocampus",
        pids=["001", "002"],
        vol_shape=(4, 8, 8),
    )

    first = _run(
        ["hippocampus", "2"],
        {"TAR_CACHE_DIR": str(cache), "DATA_OUT_DIR": str(out_dir)},
    )
    assert first.returncode == 0
    assert "added 2 patients" in first.stdout

    second = _run(
        ["hippocampus", "2"],
        {"TAR_CACHE_DIR": str(cache), "DATA_OUT_DIR": str(out_dir)},
    )
    assert second.returncode == 0
    # Second run should no-op — tar is cached, patients already exist.
    assert "cached tar" in second.stdout
    assert "added 0 patients" in second.stdout
