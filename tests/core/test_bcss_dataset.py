"""Tests for the BCSS tiling dataset loader (synthetic data only)."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pyhealth.datasets import BCSSDataset, SubjectSplit
from pyhealth.datasets.bcss import parse_bcss_roi_name, tile_label

ROIS = {
    # name: 8x8 mask of four 4x4 quadrants (top-left, top-right, bottom-left, bottom-right)
    "TCGA-A1-AAAA-DX1_xmin0_ymin0_MPP-0.2500.png": (1, 2, 0, 9),
    "TCGA-A2-BBBB-DX1_xmin10_ymin20_MPP-0.2500.png": (3, 3, 4, 1),
    "TCGA-E2-CCCC-DX1_xmin5_ymin5_MPP-0.2500.png": (2, 1, 4, 0),
}


def write_fixture(root: Path, image_dir: str = "rgbs_colorNormalized") -> None:
    """Write 8x8 BCSS-style mask/RGB pairs; each 4x4 quadrant is pure."""
    (root / "masks").mkdir(parents=True, exist_ok=True)
    (root / image_dir).mkdir(parents=True, exist_ok=True)
    for name, (a, b, c, d) in ROIS.items():
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:4, :4], mask[:4, 4:], mask[4:, :4], mask[4:, 4:] = a, b, c, d
        Image.fromarray(mask, mode="L").save(root / "masks" / name)
        rgb = np.stack([mask * 20] * 3, axis=-1).astype(np.uint8)
        Image.fromarray(rgb, mode="RGB").save(root / image_dir / name)


class TestBCSSHelpers(unittest.TestCase):
    def test_parse_name(self):
        info = parse_bcss_roi_name("TCGA-OL-A5RW-DX1_xmin1_ymin2_MPP-0.2500.png")
        self.assertEqual(info["patient_id"], "TCGA-OL-A5RW")
        self.assertEqual(info["site"], "OL")
        self.assertIsNone(parse_bcss_roi_name("gtruth_codes.tsv"))

    def test_dont_care_counts_against_purity(self):
        tile = np.array([[0, 0], [2, 2]])
        self.assertIsNone(tile_label(tile, (1, 2), 0.6))
        self.assertEqual(tile_label(tile, (1, 2), 0.5), (2, 0.5))
        self.assertIsNone(tile_label(np.zeros((2, 2), dtype=int), (1, 2), 0.1))


class TestBCSSDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "bcss"
        self.cache = self.tmp / "cache"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_tiling_and_labels(self):
        write_fixture(self.root)
        ds = BCSSDataset(root=str(self.root), tile_size=4, cache_dir=str(self.cache))
        self.assertEqual(
            sorted(ds.unique_patient_ids),
            ["TCGA-A1-AAAA", "TCGA-A2-BBBB", "TCGA-E2-CCCC"],
        )
        events = ds.get_patient("TCGA-A1-AAAA").get_events(event_type="bcss")
        # Code 0 (don't care) and code 9 (fat, not a main class) are dropped.
        self.assertEqual(sorted(e.label for e in events), ["stroma", "tumor"])
        self.assertTrue(all(Path(e.path).is_file() for e in events))
        for e in events:
            mask = np.asarray(Image.open(e.mask_path))
            self.assertEqual(mask.shape, (4, 4))
            self.assertEqual(int(mask[0, 0]), int(e.label_code))
        self.assertTrue(all(e.split == "train" for e in events))
        test_events = ds.get_patient("TCGA-E2-CCCC").get_events(event_type="bcss")
        self.assertEqual({e.split for e in test_events}, {"test"})
        self.assertFalse(any("tiles" in p.name for p in self.root.iterdir()))
        # The default task keeps the recipe's tile size (no silent resize).
        self.assertEqual(ds.default_task.image_size, 4)

    def test_purity_and_stride_change_recipe(self):
        write_fixture(self.root)
        # The default purity 0.9 keeps no 8x8 tile (see the no-kept-tiles
        # test), so compare two purities that both keep TCGA-A2-BBBB.
        strict = BCSSDataset(
            root=str(self.root), tile_size=8, min_purity=0.45, cache_dir=str(self.cache)
        )
        loose = BCSSDataset(
            root=str(self.root), tile_size=8, min_purity=0.5, cache_dir=str(self.cache)
        )
        self.assertNotEqual(strict.tile_dir, loose.tile_dir)
        # Only TCGA-A2-BBBB has one class (code 3) on half of the 8x8 ROI.
        self.assertEqual(loose.unique_patient_ids, ["TCGA-A2-BBBB"])
        overlap = BCSSDataset(
            root=str(self.root), tile_size=4, stride=2, cache_dir=str(self.cache)
        )
        events = overlap.get_patient("TCGA-A2-BBBB").get_events(event_type="bcss")
        # 4 pure quadrants plus the tile spanning the two code-3 quadrants;
        # tiles that mix codes fail min_purity=0.9.
        self.assertEqual(len(events), 5)

    def test_class_codes_and_images_folder(self):
        write_fixture(self.root, image_dir="images")
        ds = BCSSDataset(
            root=str(self.root),
            tile_size=4,
            class_codes=(9,),
            cache_dir=str(self.cache),
        )
        self.assertEqual(ds.unique_patient_ids, ["TCGA-A1-AAAA"])
        self.assertEqual(ds.default_task.classes, ["fat"])

    def test_official_split(self):
        write_fixture(self.root)
        ds = BCSSDataset(root=str(self.root), tile_size=4, cache_dir=str(self.cache))
        split = ds.official_split()
        self.assertIsInstance(split, SubjectSplit)
        self.assertEqual(split.subjects[1], ("TCGA-E2-CCCC",))
        train, val, test = ds.official_split(val_ratio=0.5, seed=0).subjects
        self.assertEqual(len(train) + len(val), 2)
        self.assertEqual(test, ("TCGA-E2-CCCC",))

    def test_bad_arguments(self):
        write_fixture(self.root)
        with self.assertRaises(ValueError):
            BCSSDataset(root=str(self.root), min_purity=0, cache_dir=str(self.cache))
        with self.assertRaises(ValueError):
            BCSSDataset(
                root=str(self.root), class_codes=(0,), cache_dir=str(self.cache)
            )
        with self.assertRaises(FileNotFoundError):
            BCSSDataset(root=str(self.tmp / "missing"), cache_dir=str(self.cache))

    def test_no_kept_tiles_raises_and_writes_no_index(self):
        write_fixture(self.root)
        # Every 8x8 ROI mixes codes, so none reaches the default purity 0.9.
        for _ in range(2):  # the second call must not reuse an empty index
            with self.assertRaisesRegex(ValueError, "no tile"):
                BCSSDataset(root=str(self.root), tile_size=8, cache_dir=str(self.cache))
        self.assertEqual(list(self.cache.rglob("*.csv")), [])

    def test_unpaired_roi_raises(self):
        write_fixture(self.root)
        name = next(iter(ROIS))
        (self.root / "rgbs_colorNormalized" / name).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
            BCSSDataset(root=str(self.root), tile_size=4, cache_dir=str(self.cache))
        write_fixture(self.root)
        (self.root / "masks" / name).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
            BCSSDataset(root=str(self.root), tile_size=4, cache_dir=str(self.cache))

    def test_metadata_dir_and_resume_after_interruption(self):
        write_fixture(self.root)
        meta = self.tmp / "tiles"
        ds = BCSSDataset(
            root=str(self.root),
            tile_size=4,
            cache_dir=str(self.cache),
            metadata_dir=str(meta),
        )
        self.assertEqual(ds.tile_dir.parent, meta)
        n_patients = len(ds.unique_patient_ids)
        # Simulate an interrupted run: tiles exist but the index CSV does not.
        ds.metadata_path.unlink()
        tile = next(ds.tile_dir.rglob("x0_y0.png"))
        tile.unlink()
        again = BCSSDataset(
            root=str(self.root),
            tile_size=4,
            cache_dir=str(self.cache / "again"),
            metadata_dir=str(meta),
        )
        self.assertEqual(again.tile_dir, ds.tile_dir)
        self.assertTrue(again.metadata_path.is_file())
        self.assertTrue(tile.is_file())
        self.assertEqual(len(again.unique_patient_ids), n_patients)

    def test_refresh_metadata_and_source_change(self):
        write_fixture(self.root)
        ds = BCSSDataset(root=str(self.root), tile_size=4, cache_dir=str(self.cache))
        stamp = ds.metadata_path.stat().st_mtime_ns
        refreshed = BCSSDataset(
            root=str(self.root),
            tile_size=4,
            cache_dir=str(self.cache / "r"),
            metadata_dir=str(ds.tile_dir.parent),
            refresh_metadata=True,
        )
        self.assertEqual(refreshed.tile_dir, ds.tile_dir)
        self.assertGreater(refreshed.metadata_path.stat().st_mtime_ns, stamp)
        # Rewriting a source file with the same size changes its mtime, and
        # so the tile directory.
        name = next(iter(ROIS))
        path = self.root / "masks" / name
        data = path.read_bytes()
        path.write_bytes(data)
        os.utime(path, ns=(stamp + 10**9, stamp + 10**9))
        changed = BCSSDataset(
            root=str(self.root),
            tile_size=4,
            cache_dir=str(self.cache / "c"),
            metadata_dir=str(ds.tile_dir.parent),
        )
        self.assertNotEqual(changed.tile_dir, ds.tile_dir)


if __name__ == "__main__":
    unittest.main()
