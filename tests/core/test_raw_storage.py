"""Variable-shape (non-tensor) fields must round-trip through the disk cache.

litdata infers one serializer per value from the first sample it writes and
pairs later samples against it, so a raw list whose length or shape varies
between samples used to fail to write, or be misread. Every non-tensor field is
now stored as one opaque value per sample; these tests assert exact equality of
what is read back, on every write and read path.
"""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import BaseDataset, PatientSplit, create_sample_dataset, get_dataloader
from pyhealth.datasets.sample_dataset import STORAGE_FORMAT, SampleBuilder, _decode_record
from pyhealth.processors.base_processor import FeatureProcessor
from pyhealth.tasks import BaseTask

LONG_NOTE = "x" * 100_000

# Each case: values for consecutive samples. Orders are permuted in the tests so
# the first sample is the shortest, the longest, or empty.
CASES = {
    "scalar str": ["a", "bb", "ccc"],
    "scalar int": [1, 22, 333],
    "scalar float": [0.5, 1.25, -3.0],
    "None and values": [None, "x", None],
    "empty list": [[], [], []],
    "flat str, varying": [["n1"], ["n1", "n2", "n3"], ["n1", "n2"]],
    "flat int, varying": [[1], [1, 2, 3, 4], [1, 2]],
    "flat float, varying": [[0.5], [0.5, 1.5, 2.5], [0.5, 1.5]],
    "nested, varying": [[["a"], ["b", "c"]], [["d"]], [["e", "f", "g"], []]],
    "tuples": [("a", 1), ("b", 2, 3.0), ("c",)],
    "dict, fixed keys": [{"k": 1, "v": [1]}, {"k": 2, "v": [1, 2]}, {"k": 3, "v": []}],
    "dict, varying keys": [{"a": 1}, {"b": [2, 3]}, {}],
    "list with None": [[None, 1], [2, None, 3], [None]],
    "long string": ["short", LONG_NOTE, "mid" * 100],
    "bytes": [b"\x00\x01", b"", b"abc" * 10],
    "numpy arrays": [np.arange(3), np.arange(0), np.ones((2, 2))],
}


def _orders(values):
    """The case as given, shortest first, longest first, and empty first if any."""
    size = lambda v: len(v) if hasattr(v, "__len__") else 0
    orders = {"given": list(values)}
    orders["shortest first"] = sorted(values, key=size)
    orders["longest first"] = sorted(values, key=size, reverse=True)
    empty = [v for v in values if hasattr(v, "__len__") and len(v) == 0]
    if empty:
        orders["empty first"] = empty[:1] + [v for v in values if v is not empty[0]]
    return orders


def _equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return type(a) is type(b) and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return type(b) is dict and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


ORDER_NAMES = ("given", "shortest first", "longest first", "empty first")
CASE_FIELDS = {case: f"raw_{k}" for k, case in enumerate(CASES)}


def _all_cases_samples(order):
    """One sample list holding every case as its own raw field, in `order`."""
    columns = {}
    for case, values in CASES.items():
        orders = _orders(values)
        columns[CASE_FIELDS[case]] = orders.get(order, orders["given"])
    return [
        {"patient_id": f"p{i}", "codes": ["c1", "c2", "c3"][: 1 + i % 3], "label": i % 2,
         "extra": ["e"] * (i + 1), **{f: col[i] for f, col in columns.items()}}
        for i in range(3)
    ]


ALL_INPUT = {**{f: "raw" for f in CASE_FIELDS.values()}, "codes": "sequence"}


class NotATensor(FeatureProcessor):
    """Declares tensor output but returns a list (a misconfigured processor)."""

    stores_tensor = True

    def process(self, value):
        return list(value)


def _samples(values):
    return [
        {
            "patient_id": f"p{i}",
            "raw_field": v,
            "codes": ["c1", "c2", "c3"][: 1 + i % 3],
            "label": i % 2,
            "extra": ["e"] * (i + 1),  # not in any schema, varying length
        }
        for i, v in enumerate(values)
    ]


INPUT = {"raw_field": "raw", "codes": "sequence"}
OUTPUT = {"label": "binary"}


class TestCreateSampleDatasetRoundTrip(unittest.TestCase):
    def test_every_shape_and_order(self):
        for order in ORDER_NAMES:
            samples = _all_cases_samples(order)
            disk = create_sample_dataset(samples, ALL_INPUT, OUTPUT, in_memory=False)
            memory = create_sample_dataset(samples, ALL_INPUT, OUTPUT, in_memory=True)
            for case, field in CASE_FIELDS.items():
                with self.subTest(case=case, order=order):
                    for i, sample in enumerate(samples):
                        self.assertTrue(_equal(disk[i][field], sample[field]))
                        # disk and in-memory datasets return the same objects
                        self.assertTrue(_equal(disk[i][field], memory[i][field]))
            for i, sample in enumerate(samples):
                self.assertEqual(list(disk[i]), list(memory[i]))  # same keys, same order
                self.assertTrue(_equal(disk[i]["extra"], sample["extra"]))
                self.assertEqual(disk[i]["patient_id"], sample["patient_id"])
                self.assertTrue(torch.equal(disk[i]["codes"], memory[i]["codes"]))

    def test_tensor_fields_stay_tensors(self):
        disk = create_sample_dataset(_samples(CASES["flat str, varying"]), INPUT, OUTPUT, in_memory=False)
        self.assertIsInstance(disk[0]["codes"], torch.Tensor)
        self.assertIsInstance(disk[0]["label"], torch.Tensor)
        self.assertEqual(disk.storage_format, STORAGE_FORMAT)

    def test_read_paths(self):
        values = CASES["nested, varying"] * 4
        samples = _samples(values)
        disk = create_sample_dataset(samples, INPUT, OUTPUT, in_memory=False)
        by_pid = {s["patient_id"]: s["raw_field"] for s in samples}
        for shuffle in (False, True):
            with self.subTest(read="iteration", shuffle=shuffle):
                disk.set_shuffle(shuffle)
                seen = {s["patient_id"]: s["raw_field"] for s in disk}
                self.assertEqual(seen, by_pid)
        disk.set_shuffle(False)
        with self.subTest(read="subset"):
            part = disk.subset([1, 3, 5])
            self.assertEqual([part[i]["raw_field"] for i in range(3)], [values[1], values[3], values[5]])
        for shuffle in (False, True):
            with self.subTest(read="dataloader", shuffle=shuffle):
                got = {}
                for batch in get_dataloader(disk, batch_size=4, shuffle=shuffle):
                    got.update(zip(batch["patient_id"], batch["raw_field"]))
                self.assertEqual(got, by_pid)

    def test_misdeclared_tensor_processor_raises_with_field_name(self):
        with self.assertRaisesRegex(ValueError, "codes"):
            create_sample_dataset(
                _samples(CASES["scalar str"]),
                {"raw_field": "raw", "codes": NotATensor},
                OUTPUT,
                in_memory=False,
            )

    def test_disk_write_starts_no_processes(self):
        # litdata.optimize left helper processes running when the transform
        # raised; they deadlocked a later resource-tracker shutdown on Linux.
        from unittest import mock

        with mock.patch(
            "multiprocessing.process.BaseProcess.start",
            side_effect=AssertionError("create_sample_dataset started a process"),
        ):
            create_sample_dataset(_samples(CASES["scalar str"]), INPUT, OUTPUT, in_memory=False)
            with self.assertRaisesRegex(ValueError, "codes"):
                create_sample_dataset(
                    _samples(CASES["scalar str"]),
                    {"raw_field": "raw", "codes": NotATensor},
                    OUTPUT,
                    in_memory=False,
                )

    def test_old_records_are_returned_unchanged(self):
        old = {"patient_id": "p1", "raw_field": "v"}
        self.assertIs(_decode_record(old), old)

    def test_encode_keeps_one_opaque_item_per_sample(self):
        builder = SampleBuilder(input_schema=INPUT, output_schema=OUTPUT)
        samples = _samples(CASES["flat str, varying"])
        builder.fit(samples)
        records = [builder.transform_for_storage({"sample": pickle.dumps(s)}) for s in samples]
        self.assertEqual({tuple(sorted(r)) for r in records}, {("_pyhealth_objects", "codes", "label")})


class EventTask(BaseTask):
    task_name = "raw_storage_task"
    input_schema = {"raw_field": "raw", "codes": "nested_sequence", "value": "tensor"}
    output_schema = {"label": "binary"}

    def __call__(self, patient):
        k = int(patient.patient_id[1:])
        samples = []
        for i, event in enumerate(patient.get_events("events")):
            # varying shapes and types across patients and samples
            raw = [["note"] * (1 + (k + i) % 4) for _ in range(1 + k % 3)] if k % 5 else None
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": f"{patient.patient_id}-{i}",
                    "raw_field": raw,
                    "codes": [["a"] * (1 + j % 3) for j in range(1 + (k + i) % 3)],
                    "value": [float(k), float(i)],
                    "label": (k + i) % 2,
                    "times": [0.5 * j for j in range(k % 6)],  # extra, not in the schema
                }
            )
        return samples


class TestSetTaskRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / "data"
        root.mkdir()
        rows = ["patient_id,time,code"] + [
            f"p{k},2020-01-0{1 + v} 00:00:00,c" for k in range(30) for v in range(1 + k % 3)
        ]
        (root / "events.csv").write_text("\n".join(rows) + "\n")
        (root / "config.yaml").write_text(
            'version: "1.0"\ntables:\n  events:\n    file_path: "events.csv"\n'
            '    patient_id: "patient_id"\n    timestamp: "time"\n    attributes:\n      - "code"\n'
        )
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _dataset(self, workers):
        return BaseDataset(
            root=str(self.root), tables=["events"], dataset_name=f"RawStorage{workers}",
            config_path=str(self.root / "config.yaml"),
            cache_dir=str(Path(self.tmp.name) / f"cache{workers}"), num_workers=workers,
        )

    @staticmethod
    def _expected(sample_id):
        pid, i = sample_id.split("-")
        k, i = int(pid[1:]), int(i)
        raw = [["note"] * (1 + (k + i) % 4) for _ in range(1 + k % 3)] if k % 5 else None
        return raw, [0.5 * j for j in range(k % 6)]

    def test_set_task_round_trip(self):
        for workers in (1, 2):
            with self.subTest(num_workers=workers):
                samples = self._dataset(workers).set_task(EventTask(), num_workers=workers)
                self.assertGreater(len(samples), 0)
                for i in range(len(samples)):
                    s = samples[i]
                    raw, times = self._expected(s["visit_id"])
                    self.assertEqual(s["raw_field"], raw)
                    self.assertEqual(s["times"], times)
                    self.assertIsInstance(s["codes"], torch.Tensor)

    def test_split_parts_round_trip(self):
        parts = self._dataset(1).set_task(EventTask(), split=PatientSplit((0.6, 0.2, 0.2), seed=1))
        for part in parts:
            for s in part:
                raw, times = self._expected(s["visit_id"])
                self.assertEqual((s["raw_field"], s["times"]), (raw, times))


if __name__ == "__main__":
    unittest.main()
