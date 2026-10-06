"""set_task(task, split=PatientSplit(...)): fit processors on training patients only.

The synthetic data makes leakage visible: test patients' values are shifted by
+100 and they carry a code that never occurs in training.
"""

import json
import logging
import tempfile
import tracemalloc
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import BaseDataset, PatientSplit, Split
from pyhealth.datasets.sample_dataset import SampleBuilder
from pyhealth.processors import SequenceProcessor
from pyhealth.processors.base_processor import FeatureProcessor
from pyhealth.tasks import BaseTask

N_PATIENTS = 40
SPLIT = PatientSplit(ratios=(0.6, 0.2, 0.2), seed=3)

CONFIG = """version: "1.0"
tables:
  events:
    file_path: "events.csv"
    patient_id: "patient_id"
    timestamp: "time"
    attributes:
      - "code"
      - "value"
"""


class MeanCenter(FeatureProcessor):
    """Learns the mean of a numeric feature and subtracts it (a statistic)."""

    learns_statistics = True

    def __init__(self):
        self.mean = None

    def fit(self, samples, field):
        values = [float(s[field][0]) for s in samples]
        self.mean = sum(values) / len(values)

    def process(self, value):
        return torch.tensor([float(value[0]) - self.mean])


class EventTask(BaseTask):
    task_name = "split_test_task"
    input_schema = {"codes": "sequence", "value": MeanCenter}
    output_schema = {"label": "binary"}

    def __call__(self, patient):
        samples = []
        for i, event in enumerate(patient.get_events("events")):
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": f"{patient.patient_id}-{i}",
                    "codes": event.code.split("|"),
                    "value": [float(event.value)],
                    "label": i % 2 if len(patient.get_events("events")) > 1 else int(patient.patient_id[-1]) % 2,
                }
            )
        return samples


class HoldoutSplit(Split):
    """Custom split: the given patients form part 1, everyone else part 0."""

    def __init__(self, test_patients):
        self.test_patients = sorted(test_patients)

    def split_indices(self, patient_to_index):
        test = set(self.test_patients)
        parts = ([], [])
        for pid, indices in patient_to_index.items():
            parts[pid in test].extend(indices)
        return [np.array(sorted(p), dtype=np.int64) for p in parts]

    def to_dict(self):
        return {"kind": "holdout", "test_patients": self.test_patients}


class OverlapSplit(Split):
    """Custom split whose part 1 repeats part 0: set_task must accept it as given."""

    def split_indices(self, patient_to_index):
        everything = np.array(sorted(i for ix in patient_to_index.values() for i in ix), dtype=np.int64)
        return [everything, everything[:5]]

    def to_dict(self):
        return {"kind": "overlap"}


def _patient_ids():
    return [f"p{i:03d}" for i in range(N_PATIENTS)]


def _parts(split):
    """Patients in each split part, from one-sample-per-patient indices."""
    ids = _patient_ids()
    parts = split.split_indices({pid: [k] for k, pid in enumerate(ids)})
    return [{ids[int(k)] for k in part} for part in parts]


def _write_dataset(root: Path, test_patients: set) -> None:
    rows = ["patient_id,time,code,value"]
    for k, pid in enumerate(_patient_ids()):
        for v in range(1 + k % 3):  # 1-3 samples per patient
            shift = 100.0 if pid in test_patients else 0.0
            code = "A|TESTONLY" if pid in test_patients else ("A|B" if v % 2 else "A")
            rows.append(f"{pid},2020-01-{1 + v:02d} 00:00:00,{code},{10.0 + v + shift}")
    (root / "events.csv").write_text("\n".join(rows) + "\n")
    (root / "config.yaml").write_text(CONFIG)


class TestSplitSetTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / "data"
        root.mkdir()
        cls.parts = _parts(SPLIT)
        _write_dataset(root, test_patients=cls.parts[2])
        cls.dataset = BaseDataset(
            root=str(root),
            tables=["events"],
            dataset_name="SplitDataset",
            config_path=str(root / "config.yaml"),
            cache_dir=str(Path(cls.tmp.name) / "cache"),
            num_workers=1,
        )
        cls.train, cls.val, cls.test = cls.dataset.set_task(EventTask(), split=SPLIT)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def _patients(ds):
        return {ds[i]["patient_id"] for i in range(len(ds))}

    def test_parts_are_patient_disjoint_and_complete(self):
        parts = [self._patients(d) for d in (self.train, self.val, self.test)]
        self.assertEqual(parts, self.parts)
        all_samples = self.dataset.set_task(EventTask())
        self.assertEqual(sum(len(d) for d in (self.train, self.val, self.test)), len(all_samples))

    def test_statistics_come_from_training_patients(self):
        train_values = [
            10.0 + v
            for k, pid in enumerate(_patient_ids())
            if pid in self.parts[0]
            for v in range(1 + k % 3)
        ]
        mean = self.train.input_processors["value"].mean
        self.assertAlmostEqual(mean, float(np.mean(train_values)))
        self.assertLess(mean, 20.0)  # fitted on all samples it would include +100 values

    def test_test_only_codes_map_to_unknown(self):
        codes = self.train.input_processors["codes"]
        self.assertNotIn("TESTONLY", codes.code_vocab)
        sample = self.test[0]
        self.assertIn(codes.code_vocab["<unk>"], sample["codes"].tolist())

    def test_fit_split_is_recorded(self):
        for part in (self.train, self.val, self.test):
            self.assertEqual(part.fit_split, SPLIT.to_dict())
        self.assertIsNone(self.dataset.set_task(EventTask()).fit_split)

    def test_rerun_reuses_the_cache_and_the_same_split(self):
        with self.assertLogs("pyhealth.datasets.base_dataset", "INFO") as logs:
            again = self.dataset.set_task(EventTask(), split=SPLIT)
        self.assertTrue(any("Found cached processed samples" in m for m in logs.output))
        self.assertFalse(any("Fitting processors" in m for m in logs.output))
        self.assertEqual([self._patients(d) for d in again], self.parts)

    def test_other_seed_gives_other_split_and_cache(self):
        other = PatientSplit(ratios=(0.6, 0.2, 0.2), seed=4)
        train, _, _ = self.dataset.set_task(EventTask(), split=other)
        self.assertEqual(self._patients(train), _parts(other)[0])
        self.assertNotEqual(train.path, self.train.path)

    def test_two_part_split(self):
        parts = self.dataset.set_task(EventTask(), split=PatientSplit(ratios=(0.8, 0.2)))
        self.assertEqual(len(parts), 2)

    def test_four_part_split(self):
        split = PatientSplit(ratios=(0.5, 0.1, 0.1, 0.3), seed=3)
        parts = self.dataset.set_task(EventTask(), split=split)
        self.assertEqual(len(parts), 4)
        patients = [self._patients(d) for d in parts]
        self.assertEqual(patients, _parts(split))
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(patients[i] & patients[j])
        self.assertEqual(set().union(*patients), set(_patient_ids()))
        self.assertEqual(sum(len(d) for d in parts), len(self.dataset.set_task(EventTask())))
        self.assertEqual(parts[3].fit_split["ratios"], [0.5, 0.1, 0.1, 0.3])

    def test_custom_split(self):
        split = HoldoutSplit(self.parts[2])
        train, held = self.dataset.set_task(EventTask(), split=split)
        self.assertEqual(self._patients(held), self.parts[2])
        self.assertEqual(self._patients(train), set(_patient_ids()) - self.parts[2])
        self.assertEqual(held.fit_split, split.to_dict())
        codes = train.input_processors["codes"]
        self.assertNotIn("TESTONLY", codes.code_vocab)
        self.assertIn(codes.code_vocab["<unk>"], held[0]["codes"].tolist())

    def test_custom_split_parts_are_used_as_given(self):
        everything, first = self.dataset.set_task(EventTask(), split=OverlapSplit())
        self.assertEqual(len(everything), len(self.dataset.set_task(EventTask())))
        self.assertEqual(len(first), 5)
        self.assertEqual(
            [first[i]["visit_id"] for i in range(5)],
            [everything[i]["visit_id"] for i in range(5)],
        )

    def test_unsplit_path_and_cache_key_are_unchanged(self):
        samples = self.dataset.set_task(EventTask())
        proc_key = json.dumps(
            {"input_processors": None, "output_processors": None}, sort_keys=True, default=str
        )
        self.assertTrue(samples.path.endswith(f"samples_{uuid.uuid5(uuid.NAMESPACE_DNS, proc_key)}.ld"))

    def test_supplied_processor_is_kept_with_split(self):
        codes = SequenceProcessor()
        codes.fit([{"codes": ["A", "B", "Z"]}], "codes")
        vocab = dict(codes.code_vocab)
        train, _, test = self.dataset.set_task(
            EventTask(), input_processors={"codes": codes}, split=SPLIT
        )
        self.assertEqual(train.input_processors["codes"].code_vocab, vocab)
        self.assertLess(train.input_processors["value"].mean, 20.0)


class TestSplitBuilder(unittest.TestCase):
    def _samples(self, n=6):
        return [
            {"patient_id": f"p{i}", "codes": ["A"], "value": [float(i)], "label": i % 2}
            for i in range(n)
        ]

    def test_warns_when_statistics_are_fitted_on_all_samples(self):
        builder = SampleBuilder(
            input_schema={"codes": "sequence", "value": MeanCenter},
            output_schema={"label": "binary"},
        )
        with self.assertLogs("pyhealth.datasets.sample_dataset", "WARNING") as logs:
            builder.fit(self._samples())
        self.assertIn("value", "\n".join(logs.output))

    def test_no_warning_with_a_split(self):
        builder = SampleBuilder(
            input_schema={"codes": "sequence", "value": MeanCenter},
            output_schema={"label": "binary"},
        )
        with self.assertNoLogs("pyhealth.datasets.sample_dataset", "WARNING"):
            builder.fit(self._samples(), split=PatientSplit(ratios=(0.5, 0.5)))

    def test_no_warning_for_vocabulary_only(self):
        builder = SampleBuilder(
            input_schema={"codes": "sequence"}, output_schema={"label": "binary"}
        )
        with self.assertNoLogs("pyhealth.datasets.sample_dataset", "WARNING"):
            builder.fit(self._samples())

    def test_split_fitting_streams_without_collecting_samples(self):
        n, payload = 400, 4000  # ~50 MB if every sample were held at once

        class LazyStream:
            def __iter__(self):
                for i in range(n):
                    yield {
                        "patient_id": f"p{i % 100}",
                        "codes": ["A", "B"],
                        "value": [float(i)],
                        "label": i % 2,
                        "payload": [float(i)] * payload,
                    }

            def __len__(self):
                raise AssertionError("fit must not take len() of the stream")

        builder = SampleBuilder(
            input_schema={"codes": "sequence", "value": MeanCenter},
            output_schema={"label": "binary"},
        )
        tracemalloc.start()
        builder.fit(LazyStream(), split=PatientSplit(ratios=(0.7, 0.3), seed=0))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertLess(peak, 5 * 2**20, f"fit peaked at {peak / 2**20:.1f} MiB")
        train_idx = builder.split_indices[0]
        self.assertEqual(sum(len(p) for p in builder.split_indices), n)
        expected = float(np.mean([float(i) for i in train_idx]))
        self.assertAlmostEqual(builder.input_processors["value"].mean, expected)

    def test_invalid_ratios(self):
        for ratios in ((0.5, 0.6), (1.0,), (0.5, -0.1, 0.6), (0.2, 0.2, 0.2, 0.3)):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                PatientSplit(ratios=ratios)


class TestSplitSetTaskManyWorkers(unittest.TestCase):
    """More than 10 workers: merged samples must stay in task order.

    patient_to_index is built before processing, so a merge that reorders the
    worker blocks (10.index.json before 2.index.json) hands every part the wrong samples.
    """

    N_WORKERS = 12

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / "data"
        root.mkdir()
        cls.parts = _parts(SPLIT)
        _write_dataset(root, test_patients=cls.parts[2])
        cls.dataset = BaseDataset(
            root=str(root),
            tables=["events"],
            dataset_name="SplitDatasetManyWorkers",
            config_path=str(root / "config.yaml"),
            cache_dir=str(Path(cls.tmp.name) / "cache"),
            num_workers=1,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_split_parts_hold_their_patients(self):
        parts = self.dataset.set_task(EventTask(), num_workers=self.N_WORKERS, split=SPLIT)
        got = [{d[i]["patient_id"] for i in range(len(d))} for d in parts]
        self.assertEqual(got, self.parts)

    def test_patient_to_index_points_at_own_samples(self):
        samples = self.dataset.set_task(EventTask(), num_workers=self.N_WORKERS)
        self.assertGreater(len(samples), self.N_WORKERS * 2)
        for pid, indices in samples.patient_to_index.items():
            for i in indices:
                self.assertEqual(samples[i]["patient_id"], pid)


if __name__ == "__main__":
    unittest.main()
