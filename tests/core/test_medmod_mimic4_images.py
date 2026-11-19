"""MedMod in-hospital mortality: chest X-ray presets (in a child process: some
conda envs abort on importing torchvision) and MedFuse's 512-px resize."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.core.medmod_test_utils import SID
from pyhealth.datasets.medmod.resize_cxr import (
    BASEWIDTH,
    medfuse_resize,
    original_path,
    resize_cxr,
)


ROOT = Path(__file__).resolve().parents[2]  # the repository root


CHILD = textwrap.dedent(
    """
    import sys; sys.path.insert(0, {root!r})
    import torch
    from PIL import Image
    from torchvision.transforms import CenterCrop, InterpolationMode, Normalize
    from torchvision.transforms import Resize, ToTensor
    from pyhealth.datasets.medmod.image_presets import (
        get_preset, IMAGENET_MEAN, IMAGENET_STD,
    )

    a, b = get_preset("early_fusion"), get_preset("convirt")
    ta = [type(t) for t in a.transforms]
    tb = [type(t) for t in b.transforms]
    assert ta == [Resize, CenterCrop, ToTensor, Normalize], ta
    assert tb == [Resize, CenterCrop, Resize, ToTensor], tb
    assert a.transforms[0].size == 256 and list(a.transforms[1].size) == [224, 224]
    assert list(a.transforms[3].mean) == IMAGENET_MEAN
    assert list(a.transforms[3].std) == IMAGENET_STD
    assert list(b.transforms[0].size) == [256, 256]
    assert list(b.transforms[1].size) == [224, 224]
    assert list(b.transforms[2].size) == [224, 224]
    assert b.transforms[2].interpolation == InterpolationMode.BICUBIC

    img = Image.new("RGB", (2544, 3056), (128, 128, 128))   # portrait CXR-sized
    xa, xb = a(img), b(img)
    assert tuple(xa.shape) == (3, 224, 224) and tuple(xb.shape) == (3, 224, 224)
    # (b): no normalization
    assert 0.0 <= float(xb.min()) and float(xb.max()) <= 1.0
    exp = (128 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    # (a): ImageNet normalized
    assert abs(float(xa[0].mean()) - exp) < 1e-3, float(xa[0].mean())

    # training presets: MedFuse cxr_dataset.py:65-71, MedMod utils.py:80-101
    from torchvision.transforms import RandomAffine, RandomHorizontalFlip
    from pyhealth.datasets.medmod.image_presets import RandomColorDistortion, RandomCrop
    at, bt = get_preset("early_fusion", train=True), get_preset("convirt", train=True)
    try:
        get_preset("a")
        raise AssertionError("unknown preset name accepted")
    except ValueError as e:
        assert "early_fusion" in str(e)
    assert [type(t) for t in at.transforms] == [
        Resize, RandomHorizontalFlip, RandomAffine, CenterCrop, ToTensor, Normalize
    ]
    aff = at.transforms[2]
    assert list(aff.degrees) == [-45.0, 45.0] and aff.scale == (0.85, 1.15)
    assert aff.translate == (0.15, 0.15) and at.transforms[0].size == 256
    assert [type(t) for t in bt.transforms] == [
        Resize, RandomAffine, RandomCrop, Resize, RandomHorizontalFlip,
        RandomColorDistortion, ToTensor,
    ]
    aff = bt.transforms[1]
    assert aff.scale == (0.7, 1.5) and aff.translate == (0.1, 0.1)
    assert list(aff.shear) == [-25.0, 25.0]
    assert bt.transforms[3].interpolation == InterpolationMode.BICUBIC
    import numpy as np
    np.random.seed(0); torch.manual_seed(0)
    pixels = np.random.randint(0, 255, (600, 512), np.uint8)
    noisy = Image.fromarray(pixels).convert("RGB")
    xat, xbt1, xbt2 = at(noisy), bt(noisy), bt(noisy)
    assert tuple(xat.shape) == (3, 224, 224) and tuple(xbt1.shape) == (3, 224, 224)
    assert 0.0 <= float(xbt1.min()) and float(xbt1.max()) <= 1.0  # no Normalize
    assert not torch.equal(xbt1, xbt2)  # random augmentation
    sizes = set()
    for _ in range(200):
        sizes.add(RandomCrop()(Image.new("RGB", (256, 256))).size[0])
    assert min(sizes) >= 153 and max(sizes) <= 255  # int(U(153.6, 256))
    print("OK")
"""
)


def test_image_presets_child_process():
    proc = subprocess.run(
        [sys.executable, "-c", CHILD.format(root=str(ROOT))],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 and "OMP: Error #15" in proc.stderr:
        pytest.skip(
            "torchvision cannot be imported in this environment (duplicate libomp)"
        )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("OK")


def _jpg(path: Path, size=(1000, 800), mode="L") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    arr = rng.integers(
        0,
        256,
        size=(size[1], size[0]) + ((3,) if mode == "RGB" else ()),
        dtype=np.uint8,
    )
    Image.fromarray(arr, mode).save(path, format="JPEG")
    return path


def test_original_path_layout(tmp_path):
    p = original_path(tmp_path, "10000032", "50414267", "02aa804e")
    assert p == tmp_path / "files/p10/p10000032/s50414267/02aa804e.jpg"


@pytest.mark.parametrize(
    "size,mode",
    [((1000, 800), "L"), ((2544, 3056), "L"), ((3056, 2544), "RGB"), ((300, 200), "L")],
)
def test_resize_width_512_keeps_aspect(tmp_path, size, mode):
    src = _jpg(original_path(tmp_path / "cxr", SID, "50000001", "d1"), size, mode)
    out = tmp_path / "resized"
    report = resize_cxr([src], out)
    assert (report.resized, report.skipped, report.missing, report.failed) == (
        1,
        0,
        0,
        0,
    )
    dst = out / "d1.jpg"  # flat folder, same basename
    with Image.open(dst) as img:
        assert img.format == "JPEG" and img.mode == mode
        w, h = img.size
    assert w == BASEWIDTH
    assert h == int(size[1] * BASEWIDTH / size[0])  # resize.py:32-34 (truncation)
    assert abs(w / h - size[0] / size[1]) < 2 / h  # aspect ratio kept to within a pixel


def test_resize_matches_medfuse_formula_and_bicubic():
    img = Image.fromarray(
        np.arange(80 * 100, dtype=np.uint32).reshape(80, 100).astype(np.uint8), "L"
    )
    out = medfuse_resize(img)
    assert out.size == (512, int(80 * 512 / 100))
    expected = img.resize((512, int(80 * 512 / 100)), Image.Resampling.BICUBIC)
    assert np.array_equal(np.asarray(out), np.asarray(expected))


def test_resize_skip_missing_and_report(tmp_path):
    root = tmp_path / "cxr"
    a = _jpg(original_path(root, SID, "50000001", "a"))
    b = _jpg(original_path(root, SID, "50000002", "b"))
    gone = original_path(root, SID, "50000003", "gone")  # never written
    out = tmp_path / "resized"
    # an existing file anywhere under resized/ counts as done (resize.py:14, 22-24)
    _jpg(out / "old_batch" / "b.jpg", (512, 400))
    r = resize_cxr([a, b, gone], out)
    assert (r.resized, r.skipped, r.missing, r.failed) == (1, 1, 1, 0)
    assert r.missing_paths == [str(gone)]
    assert not (out / "b.jpg").exists()  # skipped, not rewritten
    r2 = resize_cxr([a, b, gone], out)  # second run: nothing left to do
    assert (r2.resized, r2.skipped, r2.missing) == (0, 2, 1)
    assert not list(out.glob(".tmp_*"))
    assert str(r2) == "resized=0 skipped=2 missing=1 failed=0"


def test_resize_counts_unreadable_file_as_failed(tmp_path):
    bad = original_path(tmp_path / "cxr", SID, "50000001", "bad")
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a jpeg")
    r = resize_cxr([bad], tmp_path / "resized")
    assert (r.resized, r.failed) == (0, 1)
    assert not (tmp_path / "resized" / "bad.jpg").exists()


def test_failed_paths_go_to_a_file_not_the_terminal(tmp_path, capsys):
    from pyhealth.datasets.medmod.resize_cxr import FAILED_FILE, ResizeReport

    assert ResizeReport().write_failed(tmp_path) is None
    report = ResizeReport(failed=1, failed_paths=["files/p1/s2/d3.jpg: bad"])
    path = report.write_failed(tmp_path)
    assert path == tmp_path / FAILED_FILE
    assert path.read_text() == "files/p1/s2/d3.jpg: bad\n"
    assert "d3" not in str(report) and "d3" not in capsys.readouterr().out
