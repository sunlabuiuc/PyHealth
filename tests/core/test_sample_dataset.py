import pickle
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Iterator, List

from litdata import StreamingDataset, optimize

from pyhealth.datasets.sample_dataset import SampleDataset, SampleSubset
from pyhealth.processors.base_processor import FeatureProcessor

# Top-level identity function for litdata.optimize (must be picklable).
def _identity(sample: Dict[str, Any]) -> Dict[str, Any]:
    return sample


class RecordingProcessor(FeatureProcessor):
    """Processor that records fit/process calls and prefixes outputs."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.fit_called = False
        self.fit_seen: List[Any] = []
        self.process_seen: List[Any] = []

    def fit(self, samples: Iterator[Dict[str, Any]], field: str) -> None:
        self.fit_called = True
        for sample in samples:
            self.fit_seen.append(pickle.loads(sample[field]))

    def process(self, value: Any) -> Any:
        self.process_seen.append(value)
        return f"{self.prefix}-{value}"


class TestSampleDatasetAndSubset(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name) / "stream"

        raw_samples = [
            {"patient_id": "p1", "record_id": "r1", "x": 1, "y": 10},
            {"patient_id": "p2", "record_id": "r2", "x": 2, "y": 20},
            {"patient_id": "p3", "record_id": "r3", "x": 3, "y": 30},
        ]
        pickled_samples = [{k: pickle.dumps(v) for k, v in sample.items()} for sample in raw_samples]

        optimize(
            fn=_identity,
            inputs=pickled_samples,
            output_dir=str(self.output_dir),
            chunk_size=len(pickled_samples),
            num_workers=1,
            verbose=False,
        )

        streaming_dataset = StreamingDataset(
            input_dir=str(self.output_dir),
            cache_dir=str(self.output_dir),
        )

        self.sample_dataset = SampleDataset(
            dataset=streaming_dataset,
            input_schema={"x": (RecordingProcessor, {"prefix": "in"})},
            output_schema={"y": (RecordingProcessor, {"prefix": "out"})},
            dataset_name="test_dataset",
            task_name="task",
        )

        self.raw_samples = raw_samples

    def tearDown(self) -> None:
        self.tmpdir.cleanup()
        super().tearDown()

    def test_sample_dataset_builds_processors(self) -> None:
        self.assertIn("x", self.sample_dataset.input_processors)
        self.assertIn("y", self.sample_dataset.output_processors)

        input_proc: RecordingProcessor = self.sample_dataset.input_processors["x"] # type: ignore
        output_proc: RecordingProcessor = self.sample_dataset.output_processors["y"] # type: ignore

        self.assertTrue(input_proc.fit_called)
        self.assertTrue(output_proc.fit_called)
        self.assertEqual(input_proc.fit_seen, [1, 2, 3])
        self.assertEqual(output_proc.fit_seen, [10, 20, 30])

    def test_sample_dataset_returns_processed_items(self) -> None:
        # __getitem__ path
        item = self.sample_dataset[0]
        self.assertEqual(item["x"], "in-1")
        self.assertEqual(item["y"], "out-10")
        self.assertEqual(item["patient_id"], "p1")
        self.assertEqual(item["record_id"], "r1")

        # __iter__ path
        self.sample_dataset.dataset.reset()
        items = list(iter(self.sample_dataset))
        self.assertEqual([s["x"] for s in items], ["in-1", "in-2", "in-3"])
        self.assertEqual([s["y"] for s in items], ["out-10", "out-20", "out-30"])
        self.assertEqual([s["patient_id"] for s in items], ["p1", "p2", "p3"])

    def test_sample_subset_respects_indices_and_processing(self) -> None:
        subset_indices = [1, 2]
        subset = SampleSubset(self.sample_dataset, subset_indices)

        self.assertEqual(len(subset), len(subset_indices))

        # __getitem__ path
        second = subset[0]
        third = subset[1]

        self.assertEqual(second["x"], "in-2")
        self.assertEqual(second["y"], "out-20")
        self.assertEqual(second["patient_id"], "p2")

        self.assertEqual(third["x"], "in-3")
        self.assertEqual(third["y"], "out-30")
        self.assertEqual(third["patient_id"], "p3")

        # __iter__ path
        subset.dataset.reset()
        iter_items = list(iter(subset))
        self.assertEqual(len(iter_items), 2)
        self.assertEqual([s["x"] for s in iter_items], ["in-2", "in-3"])
        self.assertEqual([s["y"] for s in iter_items], ["out-20", "out-30"])
        self.assertEqual([s["patient_id"] for s in iter_items], ["p2", "p3"])

    def test_shuffle_behavior_and_isolation(self) -> None:
        # Baseline (no shuffle)
        baseline = [s["patient_id"] for s in iter(self.sample_dataset)]
        self.sample_dataset.dataset.reset()

        # Shuffle affects iteration but not __getitem__
        self.sample_dataset.set_shuffle(True)
        shuffled_iter = [s["patient_id"] for s in iter(self.sample_dataset)]
        self.assertCountEqual(shuffled_iter, baseline)
        if len(baseline) > 1:
            self.assertNotEqual(shuffled_iter, baseline)
        self.sample_dataset.dataset.reset()
        self.assertEqual(self.sample_dataset[0]["patient_id"], "p1")

        # Subset created from shuffled dataset should disable shuffle during construction
        subset = SampleSubset(self.sample_dataset, [0, 1])
        self.assertFalse(subset.dataset.shuffle)
        subset_items = [s["patient_id"] for s in iter(subset)]
        self.assertEqual(subset_items, ["p1", "p2"])
        subset.dataset.reset()
        self.assertEqual(subset[0]["patient_id"], "p1")

        # Shuffling one subset doesn't affect dataset or other subsets
        subset2 = SampleSubset(self.sample_dataset, [1, 2])
        subset.set_shuffle(True)
        shuffled_subset_iter = [s["patient_id"] for s in iter(subset)]
        self.assertCountEqual(shuffled_subset_iter, ["p1", "p2"])
        if len(shuffled_subset_iter) > 1:
            self.assertNotEqual(shuffled_subset_iter, ["p1", "p2"])
        self.assertFalse(subset2.dataset.shuffle)
        self.assertEqual(subset2[0]["patient_id"], "p2")
        self.assertTrue(self.sample_dataset.dataset.shuffle)
