"""Proof that sunlab CXR metadata can be written to cache, not the data root.

The loader required a directory named ``images`` and wrote the derived CSV
into the PhysioNet root. The complete resized set lives under
``resized_images``, and the cluster root is read-only.

Measured on the cluster:

  Complete resized cohort: 377,110 / 377,110 images, 0 dropped.
  256x256 greyscale, 3.3 GB. PhysioNet ``images/`` was incomplete (p13-p19
  absent). Hardcoded ``images/`` raised ``FileNotFoundError`` on the complete
  set. Default CXR config raised ``KeyError: 'studytime_normalized'``.
  ``chmod 0o555`` on the root: CSV lands under ``cache_dir``.

After the layout worked (one seed, 6 epochs, 18,542 train / 2,285 test,
prevalence 0.0565, split seed 42). These compare with each other, not with
the primary notes_labs table:

  cxr_only          PR-AUC 0.0602  ROC 0.5267
  cxr + labs        PR-AUC 0.3082  ROC 0.8047
  cxr + notes+labs  PR-AUC 0.4096  ROC 0.8625

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/test_p2_sunlab_cache.py -q
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class TestP2SunlabLayout(unittest.TestCase):
    def _fake_cxr_root(self, tmp: str, image_dirname: str) -> str:
        root = Path(tmp) / "cxr"
        (root / image_dirname).mkdir(parents=True)
        pd.DataFrame(
            {"dicom_id": ["abc"], "StudyTime": ["93000"], "subject_id": ["1"]}
        ).to_csv(root / "mimic-cxr-2.0.0-metadata.csv", index=False)
        return str(root)

    def test_resized_images_and_cache_write(self):
        from pyhealth.datasets.mimic4 import MIMIC4CXRSunlabDataset

        host = MIMIC4CXRSunlabDataset.__new__(MIMIC4CXRSunlabDataset)
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fake_cxr_root(tmp, "resized_images")
            cache = Path(tmp) / "cache"
            cache.mkdir()
            dest = host.prepare_metadata(root, cache_dir=str(cache))
            self.assertTrue(dest.startswith(str(cache)))
            self.assertTrue(Path(dest).is_file())
            root_csv = Path(root) / "mimic-cxr-2.0.0-metadata-pyhealth-sunlab.csv"
            self.assertFalse(root_csv.exists())
            written = pd.read_csv(dest)
            self.assertTrue(
                str(written.loc[0, "image_path"]).endswith(
                    os.path.join("resized_images", "abc.jpg")
                )
            )

    def test_unwritable_root_falls_back_to_cache(self):
        from pyhealth.datasets.mimic4 import MIMIC4CXRSunlabDataset

        host = MIMIC4CXRSunlabDataset.__new__(MIMIC4CXRSunlabDataset)
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fake_cxr_root(tmp, "images")
            cache = Path(tmp) / "cache"
            cache.mkdir()
            os.chmod(root, 0o555)
            try:
                dest = host.prepare_metadata(root, cache_dir=str(cache))
            finally:
                os.chmod(root, 0o755)
            self.assertTrue(Path(dest).is_file())
            self.assertTrue(dest.startswith(str(cache)))

    def test_writes_to_root_when_no_cache_dir(self):
        from pyhealth.datasets.mimic4 import MIMIC4CXRSunlabDataset

        host = MIMIC4CXRSunlabDataset.__new__(MIMIC4CXRSunlabDataset)
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fake_cxr_root(tmp, "images")
            dest = host.prepare_metadata(root, cache_dir=None)
            self.assertTrue(dest.startswith(root))
            self.assertTrue(Path(dest).is_file())
