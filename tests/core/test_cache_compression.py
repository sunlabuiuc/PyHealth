"""set_task(compression=...) and the batch_decode setting of subset()."""

import json
import tempfile
import unittest
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import torch

from pyhealth.datasets import PatientSplit, create_sample_dataset
from pyhealth.datasets.base_dataset import BaseDataset
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


def _config(dataset: SampleDataset) -> dict:
    return json.loads((Path(dataset.path) / "index.json").read_text())["config"]


class TestSetTaskCompression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _set_task(self, name: str, **kwargs):
        dataset = _Dataset(cache_dir=str(self.root / name))
        return dataset.set_task(_Task(), num_workers=1, **kwargs)

    def test_zstd_writes_per_sample_zstd(self):
        samples = self._set_task("zstd", compression="zstd")
        config = _config(samples)
        self.assertEqual(config["compression"], "zstd")
        self.assertEqual(config["compression_level"], "sample")
        self.assertEqual(samples.compression, "zstd")

    def test_zstd_reads_back_the_uncompressed_samples(self):
        plain = self._set_task("plain")
        zstd = self._set_task("zstd", compression="zstd")
        self.assertEqual(len(zstd), N_PATIENTS * PER_PATIENT)
        self.assertTrue(_same(_items(zstd), _items(plain)))

    def test_default_is_uncompressed(self):
        samples = self._set_task("plain")
        self.assertIsNone(_config(samples)["compression"])
        self.assertIsNone(samples.compression)

    def test_compression_does_not_change_the_cache_path(self):
        dataset = _Dataset(cache_dir=str(self.root / "shared"))
        plain = dataset.set_task(_Task(), num_workers=1)
        with self.assertLogs("pyhealth.datasets.base_dataset", "INFO") as logs:
            reused = dataset.set_task(_Task(), num_workers=1, compression="zstd")
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
        plain = self._set_task("plain", split=split)
        zstd = self._set_task("zstd", split=split, compression="zstd")
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
        full = _Dataset(cache_dir=self.tmp.name).set_task(_Task(), num_workers=1)
        indices = list(range(0, len(full), 3))
        subset = full.subset(indices)
        self.assertEqual(full.batch_decode, "auto")
        self.assertEqual(subset.batch_decode, 0)
        self.assertTrue(_same(_items(subset), [full[i] for i in indices]))

    def test_bytes_only_records_keep_the_default(self):
        # No tensor field: litdata stores these records in its Arrow-footer
        # layout, which litdata <= 0.2.76 cannot read item by item with
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
            indices = list(range(0, 60, 3))
            subset = full.subset(indices)
            self.assertEqual(subset.batch_decode, "auto")
            self.assertTrue(_same(_items(subset), [full[i] for i in indices]))
        finally:
            full.close()


if __name__ == "__main__":
    unittest.main()
