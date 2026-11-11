"""set_task(compression=...), subset() decoding and the pytree-layout patch."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dask.dataframe as dd
import pandas as pd
import torch
from litdata.streaming.writer import _ARROW_FOOTER_MAGIC

from pyhealth.datasets import PatientSplit, create_sample_dataset
from pyhealth.datasets.base_dataset import BaseDataset, _litdata_pytree_layout
from pyhealth.datasets.sample_dataset import SampleDataset
from pyhealth.tasks.base_task import BaseTask

N_PATIENTS = 20
PER_PATIENT = 3


class _Task(BaseTask):
    task_name = "compression_task"
    input_schema = {"values": "tensor", "note": "raw"}
    output_schema = {"label": "binary"}

    def __call__(self, patient):
        return [
            {
                "patient_id": patient.patient_id,
                "values": [float(row["test/value"])] * 8,
                "note": f"note {row['test/value']}",
                "label": int(row["test/value"]) % 2,
            }
            for row in patient.data_source.iter_rows(named=True)
        ]


class _Dataset(BaseDataset):
    def __init__(self, cache_dir: str):
        super().__init__(
            root="", tables=[], dataset_name="CompressionDataset", cache_dir=cache_dir
        )

    def load_data(self) -> dd.DataFrame:
        n = N_PATIENTS * PER_PATIENT
        return dd.from_pandas(
            pd.DataFrame(
                {
                    "patient_id": [f"p{i % N_PATIENTS:02d}" for i in range(n)],
                    "event_type": ["test"] * n,
                    "timestamp": [None] * n,
                    "test/value": list(range(n)),
                }
            ),
            npartitions=1,
        )


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _items(dataset) -> list:
    return [dataset[i] for i in range(len(dataset))]


def _one(result: SampleDataset | tuple[SampleDataset, ...]) -> SampleDataset:
    """The dataset set_task returns without a split."""
    assert isinstance(result, SampleDataset)
    return result


def _parts(
    result: SampleDataset | tuple[SampleDataset, ...],
) -> tuple[SampleDataset, ...]:
    """The parts set_task returns with a split."""
    assert isinstance(result, tuple)
    return result


def _index_config(cache_dir: Path) -> dict:
    return json.loads((cache_dir / "index.json").read_text())["config"]


def _config(dataset: SampleDataset) -> dict:
    return _index_config(Path(dataset.path))


def _has_arrow_footer(cache_dir: Path) -> bool:
    return any(
        chunk.read_bytes()[-len(_ARROW_FOOTER_MAGIC) :] == _ARROW_FOOTER_MAGIC
        for chunk in cache_dir.glob("chunk-*.bin")
    )


class TestSetTaskCompression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _set_task(self, name: str, **kwargs):
        dataset = _Dataset(cache_dir=str(self.root / name))
        return dataset.set_task(_Task(), num_workers=1, **kwargs)

    def _set_task_one(self, name: str, **kwargs) -> SampleDataset:
        return _one(self._set_task(name, **kwargs))

    def test_zstd_writes_per_sample_zstd(self):
        samples = self._set_task_one("zstd", compression="zstd")
        task_df = Path(samples.path).parent / "task_df.ld"
        for config in (_config(samples), _index_config(task_df)):
            self.assertEqual(config["compression"], "zstd")
            self.assertEqual(config["compression_level"], "sample")
        self.assertEqual(samples.compression, "zstd")

    def test_zstd_reads_back_the_uncompressed_samples(self):
        plain = self._set_task_one("plain")
        zstd = self._set_task_one("zstd", compression="zstd")
        self.assertEqual(len(zstd), N_PATIENTS * PER_PATIENT)
        self.assertTrue(_same(_items(zstd), _items(plain)))

    def test_default_is_uncompressed(self):
        samples = self._set_task_one("plain")
        self.assertIsNone(_config(samples)["compression"])
        task_df = Path(samples.path).parent / "task_df.ld"
        self.assertIsNone(_index_config(task_df)["compression"])
        self.assertIsNone(samples.compression)

    def test_compression_does_not_change_the_cache_path(self):
        dataset = _Dataset(cache_dir=str(self.root / "shared"))
        plain = _one(dataset.set_task(_Task(), num_workers=1))
        with self.assertLogs("pyhealth.datasets.base_dataset", "INFO") as logs:
            reused = _one(
                dataset.set_task(_Task(), num_workers=1, compression="zstd")
            )
        # The existing uncompressed cache is reused as written.
        self.assertEqual(reused.path, plain.path)
        self.assertIsNone(reused.compression)
        self.assertTrue(any("Reusing cached samples" in m for m in logs.output))

    def test_unknown_compression_raises(self):
        dataset = _Dataset(cache_dir=str(self.root / "bad"))
        for value in ("gzip", "zstd:4", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                dataset.set_task(_Task(), num_workers=1, compression=value)

    def test_split_parts_match_uncompressed(self):
        split = PatientSplit(ratios=(0.6, 0.2, 0.2), seed=0)
        plain = _parts(self._set_task("plain", split=split))
        zstd = _parts(self._set_task("zstd", split=split, compression="zstd"))
        self.assertEqual([len(p) for p in zstd], [len(p) for p in plain])
        for zstd_part, plain_part in zip(zstd, plain):
            self.assertEqual(zstd_part.batch_decode, 0)
            self.assertTrue(_same(_items(zstd_part), _items(plain_part)))


class TestSubsetBatchDecode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_records_with_tensors_decode_one_item(self):
        full = _one(_Dataset(cache_dir=self.tmp.name).set_task(_Task(), num_workers=1))
        indices = list(range(0, len(full), 3))
        subset = full.subset(indices)
        self.assertEqual(full.batch_decode, "auto")
        self.assertEqual(subset.batch_decode, 0)
        self.assertTrue(_same(_items(subset), [full[i] for i in indices]))

    def test_records_without_tensors_decode_one_item(self):
        # No tensor field: unpatched, litdata would store these records in its
        # Arrow-footer layout, which cannot be read item by item with
        # POSIX-fast off (Lightning-AI/litData#954).
        samples = [
            {"patient_id": f"p{i}", "note": f"note {i}", "y": f"class {i % 3}"}
            for i in range(60)
        ]
        full = create_sample_dataset(
            samples, {"note": "raw"}, {"y": "raw"}, in_memory=False
        )
        try:
            self.assertEqual(_config(full)["data_format"], ["bytes"])
            self.assertFalse(_has_arrow_footer(Path(full.path)))
            indices = list(range(0, 60, 3))
            subset = full.subset(indices)
            self.assertEqual(subset.batch_decode, 0)
            self.assertTrue(_same(_items(subset), [full[i] for i in indices]))
        finally:
            full.close()


class _RawTask(BaseTask):
    """A task with no tensor field: every sample is one bytes value on disk."""

    task_name = "raw_compression_task"
    input_schema = {"note": "raw"}
    output_schema = {"tag": "raw"}

    def __call__(self, patient):
        return [
            {
                "patient_id": patient.patient_id,
                "note": f"note {row['test/value']} " * 10,
                "tag": f"tag {int(row['test/value']) % 3}",
            }
            for row in patient.data_source.iter_rows(named=True)
        ]


class TestPytreeLayout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_records_without_tensors_are_compressed_and_split(self):
        split = PatientSplit(ratios=(0.6, 0.2, 0.2), seed=0)
        dataset = _Dataset(cache_dir=self.tmp.name)
        parts = _parts(
            dataset.set_task(_RawTask(), num_workers=1, split=split, compression="zstd")
        )
        config = _config(parts[0])
        self.assertEqual(config["data_format"], ["bytes"])
        self.assertEqual(config["compression"], "zstd")
        self.assertEqual(parts[0].compression, "zstd")
        task_df = Path(parts[0].path).parent / "task_df.ld"
        self.assertEqual(_index_config(task_df)["compression"], "zstd")
        for path in (Path(parts[0].path), task_df):
            self.assertFalse(_has_arrow_footer(path), path)
        # Every part reads one item at a time.
        self.assertEqual(sum(len(p) for p in parts), N_PATIENTS * PER_PATIENT)
        for part in parts:
            self.assertEqual(part.batch_decode, 0)
            self.assertTrue(all(i["note"].startswith("note ") for i in _items(part)))

    def test_patch_fails_loudly_without_the_litdata_name(self):
        import litdata.streaming.writer as writer

        # patch.object restores the attribute on exit, even once deleted.
        with mock.patch.object(writer, "is_arrow_footer_type"):
            delattr(writer, "is_arrow_footer_type")
            with self.assertRaisesRegex(RuntimeError, "is_arrow_footer_type"):
                with _litdata_pytree_layout():
                    pass


if __name__ == "__main__":
    unittest.main()
