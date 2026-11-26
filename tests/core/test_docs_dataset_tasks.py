"""Tests for the docs extension that lists each dataset's tasks."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

EXTENSION = (
    Path(__file__).resolve().parents[2] / "docs" / "extensions" / "dataset_tasks.py"
)


def load_extension():
    spec = importlib.util.spec_from_file_location("dataset_tasks", EXTENSION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _App:
    pass


class TestDatasetTasks(unittest.TestCase):
    """Check the task -> dataset mapping and the page rewrite."""

    @classmethod
    def setUpClass(cls):
        cls.ext = load_extension()
        cls.mapping = cls.ext.tasks_by_dataset()

    def test_maps_real_tasks_to_their_datasets(self):
        self.assertIn("MortalityPredictionMIMIC3", self.mapping["MIMIC3Dataset"])
        self.assertIn("SleepStagingSleepEDF", self.mapping["SleepEDFDataset"])

    def test_rollups_include_related_classes(self):
        # MIMIC4Dataset also lists tasks written for its EHR module.
        for task in self.mapping["MIMIC4EHRDataset"]:
            self.assertIn(task, self.mapping["MIMIC4Dataset"])
        # Each MSD task page lists tasks written for the base loader.
        for page in ("MSDHippocampusDataset", "MSDProstateDataset", "MSDSpleenDataset"):
            self.assertIn("MSDSegmentationTask", self.mapping[page])

    def test_private_classes_are_skipped(self):
        for tasks in self.mapping.values():
            self.assertFalse(any(task.startswith("_") for task in tasks))

    def test_section_goes_above_autoclass(self):
        self.ext.init_mapping(None)
        source = ["Title\n=====\n\nIntro.\n\n.. autoclass:: pyhealth.datasets.MIMIC3Dataset\n"]
        self.ext.add_task_list(_App(), "api/datasets/pyhealth.datasets.MIMIC3Dataset", source)
        text = source[0]
        self.assertIn("Tasks for this dataset", text)
        self.assertIn("- :class:`.MortalityPredictionMIMIC3`", text)
        self.assertLess(text.index("Tasks for this dataset"), text.index(".. autoclass::"))

    def test_other_pages_are_untouched(self):
        self.ext.init_mapping(None)
        source = ["Tasks\n=====\n"]
        self.ext.add_task_list(_App(), "api/tasks", source)
        self.assertEqual(source[0], "Tasks\n=====\n")

    def test_unparseable_module_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "broken.py").write_text("class Oops(:\n", encoding="utf-8")
            Path(tmp, "ok.py").write_text(
                'class T:\n    """Works with :class:`~pyhealth.datasets.X`."""\n',
                encoding="utf-8",
            )
            original = self.ext.TASKS_DIR
            self.ext.TASKS_DIR = Path(tmp)
            try:
                with self.assertLogs(level="WARNING"):
                    mapping = self.ext.tasks_by_dataset()
            finally:
                self.ext.TASKS_DIR = original
        self.assertEqual(mapping, {"X": ["T"]})


if __name__ == "__main__":
    unittest.main()
