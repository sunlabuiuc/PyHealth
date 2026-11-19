"""Tests for the LNDb dataset loader, on a tiny synthetic LNDb tree."""

import importlib.util
import shutil
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import LNDbDataset
from pyhealth.datasets.lndb import (
    SMALL_NODULE_VOLUME,
    MetaImage,
    equivalent_diameter,
    read_mhd,
)

SHAPE_ZYX = (12, 16, 16)
ORIGIN = (-8.0, -8.0, -6.0)
SPACING = (1.0, 1.0, 1.0)


def write_mhd(
    folder: Path,
    name: str,
    array: np.ndarray,
    origin=ORIGIN,
    compressed: bool = False,
    msb: bool = False,
    matrix: str = "1 0 0 0 1 0 0 0 1",
) -> Path:
    """Write ``array`` (z, y, x) as an int16 MetaImage.

    By default uncompressed and little-endian; ``compressed`` zlib-compresses
    the data (as LNDb's ``.raw`` files are) and ``msb`` writes big-endian.
    """
    folder.mkdir(parents=True, exist_ok=True)
    data = array.astype(">i2" if msb else "<i2").tobytes()
    if compressed:
        data = zlib.compress(data)
    (folder / f"{name}.raw").write_bytes(data)
    z, y, x = array.shape
    header = (
        "ObjectType = Image\nNDims = 3\nBinaryData = True\n"
        f"BinaryDataByteOrderMSB = {msb}\nCompressedData = {compressed}\n"
        + (f"CompressedDataSize = {len(data)}\n" if compressed else "")
        + f"TransformMatrix = {matrix}\n"
        f"Offset = {origin[0]} {origin[1]} {origin[2]}\n"
        f"ElementSpacing = {SPACING[0]} {SPACING[1]} {SPACING[2]}\n"
        f"DimSize = {x} {y} {z}\nElementType = MET_SHORT\n"
        f"ElementDataFile = {name}.raw\n"
    )
    path = folder / f"{name}.mhd"
    path.write_text(header)
    return path


def build_lndb_root(root: Path, with_scans=(1, 2, 3)) -> None:
    """Three patients, five findings, one scan missing when asked."""
    csv_dir = root / "trainset_csv"
    csv_dir.mkdir(parents=True)
    rows = [
        # LNDbID, RadID, RadFindingID, FindingID, x, y, z, AgrLevel, Nodule, Volume, Text
        (1, "1,2", "1,1", 1, 0.0, 0.0, 0.0, 2, 1, 250.0, 5.0),
        (1, "1", "2", 2, 3.0, -2.0, 1.0, 1, 1, SMALL_NODULE_VOLUME, 3.0),
        (2, "1", "1", 1, -1.0, 1.0, -1.0, 1, 0, 0.0, None),
        (3, "1,2,3", "1,1,1", 1, 2.0, 2.0, 2.0, 3, 1, 400.0, 4.0),
        (3, "2", "1", 2, -4.0, 4.0, 0.0, 1, 0, 0.0, None),
    ]
    pd.DataFrame(
        rows,
        columns=[
            "LNDbID", "RadID", "RadFindingID", "FindingID", "x", "y", "z",
            "AgrLevel", "Nodule", "Volume", "Text",
        ],
    ).to_csv(csv_dir / "trainNodules_gt.csv", index=False)
    pd.DataFrame(
        {"Fold0": [1], "Fold1": [2], "Fold2": [3], "Fold3": [4]}
    ).to_csv(csv_dir / "trainFolds.csv", index=False)
    rng = np.random.default_rng(0)
    for lndb_id in with_scans:
        array = rng.integers(-1000, 400, size=SHAPE_ZYX).astype(np.int16)
        write_mhd(root / "data0", f"LNDb-{lndb_id:04d}", array)


class TestLNDbReader(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_read_mhd_numpy_layout_and_geometry(self):
        array = np.arange(np.prod(SHAPE_ZYX), dtype=np.int16).reshape(SHAPE_ZYX)
        path = write_mhd(self.tmp, "LNDb-0009", array)
        image = read_mhd(path, backend="numpy")
        self.assertEqual(image.array.shape, SHAPE_ZYX)
        np.testing.assert_array_equal(np.asarray(image.array), array)
        np.testing.assert_allclose(image.origin, ORIGIN)
        # World (x, y, z) = origin maps to voxel (0, 0, 0).
        self.assertEqual(image.world_to_voxel(ORIGIN).tolist(), [0, 0, 0])
        self.assertEqual(
            image.world_to_voxel([-5.0, -7.0, -4.0]).tolist(), [2, 1, 3]
        )

    def test_read_mhd_rejects_bad_backend_and_header(self):
        path = write_mhd(self.tmp, "a", np.zeros((2, 2, 2), dtype=np.int16))
        with self.assertRaises(ValueError):
            read_mhd(path, backend="nope")
        path.write_text(path.read_text().replace("MET_SHORT", "MET_WEIRD"))
        with self.assertRaises(ValueError):
            read_mhd(path, backend="numpy")

    def test_read_mhd_zlib_compressed_and_big_endian(self):
        array = np.arange(np.prod(SHAPE_ZYX), dtype=np.int16).reshape(SHAPE_ZYX)
        array = array - 1000  # negative HU values exercise the sign bit
        for msb in (False, True):
            path = write_mhd(
                self.tmp, f"LNDb-c{int(msb)}", array, compressed=True, msb=msb
            )
            image = read_mhd(path, backend="numpy")
            np.testing.assert_array_equal(np.asarray(image.array), array)
        path = write_mhd(self.tmp, "LNDb-msb", array, msb=True)
        np.testing.assert_array_equal(
            np.asarray(read_mhd(path, backend="numpy").array), array
        )

    def test_direction_matrix_is_applied(self):
        image = MetaImage(
            np.zeros((4, 4, 4)), np.zeros(3), np.ones(3),
            np.diag([-1.0, -1.0, 1.0]),
        )
        self.assertEqual(image.world_to_voxel([-2.0, -1.0, 3.0]).tolist(), [3, 1, 2])

    def test_non_symmetric_transform_matrix_from_header(self):
        # MetaIO rows are the image axes: image x points along world y,
        # image y along world z and image z along world x (a cyclic
        # permutation, which is not symmetric, so a wrong transpose fails).
        matrix = "0 1 0 0 0 1 1 0 0"
        path = write_mhd(
            self.tmp, "LNDb-perm", np.zeros(SHAPE_ZYX, dtype=np.int16),
            origin=(0.0, 0.0, 0.0), matrix=matrix,
        )
        image = read_mhd(path, backend="numpy")
        # Voxel (x=1, y=2, z=3) sits at world (x=3, y=1, z=2).
        self.assertEqual(image.world_to_voxel([3.0, 1.0, 2.0]).tolist(), [3, 2, 1])
        if importlib.util.find_spec("SimpleITK") is not None:
            sitk_image = read_mhd(path, backend="sitk")
            np.testing.assert_allclose(sitk_image.direction, image.direction)
            self.assertEqual(
                sitk_image.world_to_voxel([3.0, 1.0, 2.0]).tolist(), [3, 2, 1]
            )

    def test_small_nodule_volume_is_3mm(self):
        self.assertAlmostEqual(equivalent_diameter(SMALL_NODULE_VOLUME), 3.0)


class TestLNDbDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "lndb"
        build_lndb_root(self.root, with_scans=(1, 3))
        self.cache = self.tmp / "cache"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_metadata_rows_classes_and_folds(self):
        dataset = LNDbDataset(root=str(self.root), cache_dir=str(self.cache))
        meta = pd.read_csv(dataset.metadata_path)
        self.assertEqual(len(meta), 5)
        self.assertTrue(str(dataset.metadata_path).startswith(str(self.cache)))
        self.assertFalse(any(self.root.rglob("*pyhealth*")))
        by_record = meta.set_index("record_id")
        self.assertEqual(by_record.loc["LNDb-0001_1", "finding_class"], 2)
        self.assertEqual(by_record.loc["LNDb-0001_2", "finding_class"], 1)
        self.assertEqual(by_record.loc["LNDb-0002_1", "finding_class"], 0)
        self.assertEqual(by_record.loc["LNDb-0003_1", "finding_class"], 2)
        self.assertEqual(by_record.loc["LNDb-0003_1", "fold"], 2)
        # LNDb-0002 has no CT on disk: kept, with an empty image path.
        self.assertTrue(pd.isna(by_record.loc["LNDb-0002_1", "image_path"]))
        self.assertEqual(len(dataset.unique_patient_ids), 3)

    def test_metadata_rebuilds_when_scans_change(self):
        LNDbDataset(root=str(self.root), cache_dir=str(self.cache))
        write_mhd(
            self.root / "data1", "LNDb-0002",
            np.zeros(SHAPE_ZYX, dtype=np.int16),
        )
        dataset = LNDbDataset(root=str(self.root), cache_dir=str(self.cache))
        meta = pd.read_csv(dataset.metadata_path).set_index("record_id")
        self.assertTrue(meta.loc["LNDb-0002_1", "image_path"].endswith(".mhd"))

    def test_missing_csv_raises(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            LNDbDataset(root=str(empty), cache_dir=str(self.cache))

    def test_default_task(self):
        dataset = LNDbDataset(root=str(self.root), cache_dir=str(self.cache))
        self.assertEqual(
            dataset.default_task.task_name, "LNDbFindingClassification"
        )


if __name__ == "__main__":
    unittest.main()
