"""Resize MIMIC-CXR-JPG images the way MedFuse's resize.py does.

    from pyhealth.datasets.medmod.resize_cxr import paths_from_metadata, resize_cxr

    root = "/data/mimic-cxr-jpg/2.0.0"
    report = resize_cxr(paths_from_metadata(root), f"{root}/resized")
    print(report)  # counts: resized, skipped, missing, failed

``paths_from_metadata`` lists every image in the MIMIC-CXR metadata CSV. Any
list of original ``.jpg`` paths works too, for example only the cohort's.

Settings, with sources (MedFuse = github.com/nyuad-cai/MedFuse, resize.py;
Pillow = github.com/python-pillow/Pillow at tag 8.0.0, the version MedFuse
and MedMod pin, MedFuse environment.yml:129, MedMod environment.yml:145):

* width 512, height = int(h * 512 / w), aspect ratio kept: resize.py:28, 32-35
* output: flat folder ``resized/<dicom_id>.jpg`` (basename of the input):
  resize.py:29, 37
* resample: none passed (resize.py:35), so Pillow's default. In Pillow 8.0.0
  that is BICUBIC (src/PIL/Image.py:1831), NEAREST only for modes "1"/"P"
  (Image.py:1895-1896). Passed explicitly here so newer Pillow gives the same.
* JPEG quality: none passed (resize.py:37), so ``quality=-1``
  (src/PIL/JpegImagePlugin.py:619) and Pillow skips jpeg_set_quality
  (src/libImaging/JpegEncode.c:175-176), leaving libjpeg's default. That this
  default is 75 is from memory. This module also passes no quality, so it
  gets the same libjpeg default.
* format from the ``.jpg`` extension (resize.py:37); here format="JPEG" is
  explicit because the file is first written under a temporary name.
* skip an input if a file with the same basename already exists anywhere
  under the output folder (recursive glob): resize.py:14, 22-24
* 10 threads: resize.py:40-47 (speed only, no effect on output)

Differences from resize.py, none of which change the pixels:

* writes to a temporary file and renames it, so an interrupted run cannot leave
  a truncated JPEG that a later run would then skip;
* a failing image is counted as ``failed`` instead of crashing the pool;
* inputs can be a list of expected paths (e.g. only the cohort's images), so
  files that are not on disk are reported as ``missing``.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image

try:  # Pillow >= 9.1
    BICUBIC = Image.Resampling.BICUBIC
    NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover
    BICUBIC = Image.BICUBIC
    NEAREST = Image.NEAREST

BASEWIDTH = 512  # MedFuse resize.py:28


def original_path(
    cxr_root: str | Path,
    subject_id: str | int,
    study_id: str | int,
    dicom_id: str,
) -> Path:
    """files/p<first 2 digits>/p<subject_id>/s<study_id>/<dicom_id>.jpg
    (the layout PyHealth builds in ``process_image_path``, pyhealth/datasets/mimic4.py).

    Examples:
        >>> from pyhealth.datasets.medmod.resize_cxr import original_path
        >>> str(original_path("/cxr", "123", "456", "abc"))
        '/cxr/files/p12/p123/s456/abc.jpg'
    """
    sid = str(subject_id)
    return (
        Path(cxr_root)
        / "files"
        / f"p{sid[:2]}"
        / f"p{sid}"
        / f"s{study_id}"
        / f"{dicom_id}.jpg"
    )


def medfuse_resize(img: Image.Image, basewidth: int = BASEWIDTH) -> Image.Image:
    """resize.py:32-35 with Pillow 8.0.0's default resample made explicit.

    Examples:
        >>> from PIL import Image
        >>> from pyhealth.datasets.medmod.resize_cxr import medfuse_resize
        >>> medfuse_resize(Image.new("L", (1024, 1200))).size
        (512, 600)
    """
    wpercent = basewidth / float(img.size[0])
    hsize = int(float(img.size[1]) * float(wpercent))
    resample = NEAREST if img.mode in ("1", "P") else BICUBIC
    return img.resize((basewidth, hsize), resample)


FAILED_FILE = "medmod_ihm_images_failed.txt"  # in <cxr-root>; holds MIMIC IDs


@dataclass
class ResizeReport:
    """Counts from ``resize_cxr``. Paths of missing or failed files are
    kept apart (they contain MIMIC IDs); ``str()`` shows counts only.

    Examples:
        >>> from pyhealth.datasets.medmod.resize_cxr import ResizeReport
        >>> print(ResizeReport(resized=2, skipped=1))
        resized=2 skipped=1 missing=0 failed=0
    """
    resized: int = 0
    skipped: int = 0
    missing: int = 0
    failed: int = 0
    missing_paths: list[str] = field(default_factory=list)
    failed_paths: list[str] = field(default_factory=list)

    def write_failed(self, folder: str | Path) -> Path | None:
        """Failed paths contain MIMIC IDs, so they go to a file next to the
        data (FAILED_FILE), never to the terminal. Returns the file, if any."""
        if not self.failed_paths:
            return None
        path = Path(folder) / FAILED_FILE
        path.write_text("".join(f"{p}\n" for p in self.failed_paths))
        return path

    def __str__(self) -> str:
        return (
            f"resized={self.resized} skipped={self.skipped} "
            f"missing={self.missing} failed={self.failed}"
        )


def _resize_one(src: Path, out_dir: Path) -> None:
    dst = out_dir / src.name  # resize.py:29, 37 (flat folder, same basename)
    with Image.open(src) as img:
        small = medfuse_resize(img)
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".tmp_", suffix=".jpg")
    os.close(fd)
    try:
        small.save(
            tmp, format="JPEG"
        )  # no quality: Pillow/libjpeg default, as resize.py:37
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def existing_basenames(out_dir: Path) -> set:
    """resize.py:14, 22: basenames of every .jpg under the output folder.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from pyhealth.datasets.medmod.resize_cxr import existing_basenames
        >>> existing_basenames(Path(tempfile.mkdtemp()))
        set()
    """
    return {p.name for p in out_dir.glob("**/*.jpg") if not p.name.startswith(".tmp_")}


def resize_cxr(
    paths: Iterable[str | Path], out_dir: str | Path, threads: int = 10
) -> ResizeReport:
    """MedFuse resize.py: each image to 512 px wide, saved flat as
    ``out_dir/<basename>``; basenames already there are skipped.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from PIL import Image
        >>> from pyhealth.datasets.medmod.resize_cxr import resize_cxr
        >>> src = Path(tempfile.mkdtemp()) / "abc.jpg"
        >>> Image.new("L", (1024, 1200)).save(src)
        >>> print(resize_cxr([src], src.parent / "resized", threads=1))
        resized=1 skipped=0 missing=0 failed=0
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = existing_basenames(out_dir)
    report = ResizeReport()
    todo: list[Path] = []
    seen: set = set()
    for p in map(Path, paths):
        if p.name in done or p.name in seen:  # resize.py:24
            report.skipped += 1
        elif not p.is_file():
            report.missing += 1
            report.missing_paths.append(str(p))
        else:
            todo.append(p)
            seen.add(p.name)

    def work(p: Path) -> str | None:
        try:
            _resize_one(p, out_dir)
            return None
        except Exception as e:  # noqa: BLE001
            return f"{p}: {e}"

    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:  # resize.py:40-47
        for p, err in zip(todo, pool.map(work, todo)):
            if err is None:
                report.resized += 1
            else:
                report.failed += 1
                report.failed_paths.append(err)
    return report


def paths_from_metadata(
    cxr_root: str | Path, metadata_csv: str | Path | None = None
) -> list[Path]:
    """Expected original path of every dicom in the MIMIC-CXR metadata CSV.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from pyhealth.datasets.medmod.resize_cxr import paths_from_metadata
        >>> root = Path(tempfile.mkdtemp())
        >>> meta = root / "meta.csv"
        >>> _ = meta.write_text("dicom_id,subject_id,study_id\\nabc,123,456\\n")
        >>> [str(p.relative_to(root)) for p in paths_from_metadata(root, meta)]
        ['files/p12/p123/s456/abc.jpg']
    """

    root = Path(cxr_root)
    if metadata_csv is None:
        gz = root / "mimic-cxr-2.0.0-metadata.csv.gz"
        metadata_csv = gz if gz.exists() else root / "mimic-cxr-2.0.0-metadata.csv"
    meta = pd.read_csv(
        metadata_csv, dtype=str, usecols=["dicom_id", "subject_id", "study_id"]
    )
    return [
        original_path(root, r.subject_id, r.study_id, r.dicom_id)
        for r in meta.itertuples()
    ]
