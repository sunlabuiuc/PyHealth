import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pyhealth.datasets import eICUDataset
from pyhealth.tasks import CARTVitalsGenerationEICU
from pyhealth.tasks.vitals_generation_eicu import _parse_age


class TestCARTVitalsGenerationEICU(unittest.TestCase):
    """CARTVitalsGenerationEICU on the eICU demo subset in test-resources."""

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parent.parent.parent / "test-resources/core/eicudemo"
        # litdata keeps its chunk files memory-mapped; on Windows the handles
        # are still open when the directory is removed.
        cls.cache = tempfile.TemporaryDirectory(
            ignore_cleanup_errors=sys.platform == "win32"
        )
        cls.dataset = eICUDataset(
            root=str(root),
            tables=["vitalperiodic", "vitalaperiodic"],
            cache_dir=cls.cache.name,
        )
        cls.task = CARTVitalsGenerationEICU()
        cls.samples = cls.dataset.set_task(cls.task)

    @classmethod
    def tearDownClass(cls):
        cls.cache.cleanup()

    def test_schema(self):
        """A generative task: one tensor input and no label."""
        self.assertEqual(self.task.input_schema, {"vitals": "tensor"})
        self.assertEqual(self.task.output_schema, {})

    def test_one_sample_per_patient(self):
        """Each patient contributes at most their first stay."""
        self.assertGreater(len(self.samples), 0)
        patient_ids = [
            self.samples[i]["patient_id"] for i in range(len(self.samples))
        ]
        self.assertEqual(len(patient_ids), len(set(patient_ids)))

    def test_vitals_shape_and_ranges(self):
        """Every sample is age plus four vitals within the plausible ranges."""
        ranges = list(self.task.vitals.values())
        for i in range(len(self.samples)):
            vitals = self.samples[i]["vitals"]
            self.assertEqual(vitals.shape, (len(self.task.feature_names),))
            self.assertTrue(0 <= vitals[0] <= 90)
            for value, (low, high) in zip(vitals[1:].tolist(), ranges):
                self.assertTrue(low <= value <= high)

    def test_first_stay_is_chosen(self):
        """A patient with several stays keeps the earliest one."""
        stays = [
            SimpleNamespace(
                hospitaldischargeyear="2015",
                patienthealthsystemstayid="20",
                unitvisitnumber="1",
            ),
            SimpleNamespace(
                hospitaldischargeyear="2014",
                patienthealthsystemstayid="30",
                unitvisitnumber="2",
            ),
            SimpleNamespace(
                hospitaldischargeyear="2014",
                patienthealthsystemstayid="30",
                unitvisitnumber="1",
            ),
        ]
        self.assertIs(min(stays, key=self.task._stay_order), stays[2])

    def test_window_mean_drops_out_of_window_and_implausible(self):
        """Only plausible readings in the first 24 hours are averaged."""
        events = [
            SimpleNamespace(observationoffset="10", heartrate="80"),
            SimpleNamespace(observationoffset="20", heartrate="100"),
            SimpleNamespace(observationoffset="30", heartrate="0"),
            SimpleNamespace(observationoffset="2000", heartrate="150"),
            SimpleNamespace(observationoffset="40", heartrate=None),
        ]
        self.assertEqual(self.task._window_mean(events, "heartrate", 20, 250), 90)
        self.assertIsNone(self.task._window_mean(events[2:], "heartrate", 20, 250))

    def test_parse_age(self):
        """eICU's "> 89" becomes 90; blanks are missing."""
        self.assertEqual(_parse_age("> 89"), 90.0)
        self.assertEqual(_parse_age("67"), 67.0)
        self.assertIsNone(_parse_age(""))
        self.assertIsNone(_parse_age(None))


if __name__ == "__main__":
    unittest.main()
