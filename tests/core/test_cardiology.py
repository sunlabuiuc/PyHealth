"""CardiologyDataset and CardiologyDetection, on synthetic Challenge 2020 files."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from pyhealth.datasets import CardiologyDataset
from pyhealth.datasets.cardiology import parse_header
from pyhealth.tasks import CardiologyDetection, cardiology_detect
from pyhealth.tasks.cardiology_detect import CARDIOLOGY_CATEGORIES

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
AR_CODE, WA_CODE, OTHER_CODE = "164889003", "164934002", "426783006"


def write_record(folder, name, fs, n_samples, dx, sex="Male", age="60", sx=True):
    """Writes a 12-lead record as the Challenge does: ``.mat`` plus ``.hea``."""
    folder.mkdir(parents=True, exist_ok=True)
    signal = np.tile(np.arange(n_samples, dtype=np.int16) % 1000, (12, 1))
    savemat(folder / f"{name}.mat", {"val": signal})
    lines = [f"{name} 12 {fs} {n_samples}"]
    lines += [f"{name}.mat 16x1+24 1000.0(0)/mV 16 0 0 0 0 {lead}" for lead in LEADS]
    lines += [f"# Age: {age}", f"# Sex: {sex}", f"# Dx: {dx}"]
    lines += ["# Rx: Unknown", "# Hx: Unknown"]
    if sx:  # Georgia headers have no Sx line.
        lines.append("# Sx: Unknown")
    (folder / f"{name}.hea").write_text("\n".join(lines) + "\n")


def write_challenge(root):
    """Two 500 Hz sources, PTB at 1000 Hz and INCART at 257 Hz."""
    write_record(root / "cpsc_2018" / "g1", "A0001", 500, 7500, AR_CODE)
    write_record(root / "cpsc_2018" / "g1", "A0002", 500, 4000, WA_CODE)  # < 10 s
    write_record(
        root / "georgia" / "g1",
        "E00001",
        500,
        5000,
        f"{OTHER_CODE},{WA_CODE}",
        age="NaN",
        sx=False,
    )
    write_record(root / "ptb" / "g1", "S0001", 1000, 20000, OTHER_CODE)
    write_record(
        root / "st_petersburg_incart" / "g1", "I0001", 257, 257 * 20, AR_CODE, sex="F"
    )


class Patient:
    """A stand-in for a PyHealth patient with one recording."""

    def __init__(self, root, source, header):
        info = parse_header(header)
        info.update(source=source, signal_file=str(header.with_suffix(".mat")))
        self.patient_id = info["record_id"]
        self._event = type("Event", (), info)()

    def get_events(self):
        return [self._event]


class TestParseHeader(unittest.TestCase):
    def test_reads_rate_demographics_and_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_record(
                folder, "I0001", 257, 600, f"{AR_CODE},{WA_CODE}", sex="F", age="NaN"
            )
            header = parse_header(folder / "I0001.hea")
        self.assertEqual(header["record_id"], "I0001")
        self.assertEqual(header["fs"], 257.0)
        self.assertEqual(header["n_samples"], 600)
        self.assertEqual(header["n_leads"], 12)
        self.assertEqual(header["sex"], "Female")
        self.assertEqual(header["age"], "")
        self.assertEqual(header["dx"], f"{AR_CODE},{WA_CODE}")

    def test_record_name_drops_a_file_extension(self):
        # Some copies of the Georgia headers name the record "E00001.mat".
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "E00001.hea"
            path.write_text("E00001.mat 12 500 5000 05-May-2020 14:50:55\n#Dx: 426783006\n")
            self.assertEqual(parse_header(path)["record_id"], "E00001")


class TestCardiologyDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        write_challenge(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _samples(self, source, name, task):
        header = next((self.root / source).glob(f"*/{name}.hea"))
        return task(Patient(self.root, source, header))

    def test_500_hz_windows_match_the_1x_layout(self):
        samples = self._samples("cpsc_2018", "A0001", CardiologyDetection("AR"))
        # 15 s at 500 Hz: windows at 0 s and 5 s, as in 1.x.
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["signal"].shape, (12, 5000))
        self.assertEqual([s["window"] for s in samples], [0, 1])
        self.assertEqual(samples[1]["signal"][0, 0].item(), 2500 % 1000)
        self.assertEqual(samples[0]["label"], 1)
        self.assertEqual(samples[0]["source"], "cpsc_2018")

    def test_other_rates_are_resampled_to_500_hz(self):
        for source, name in (("ptb", "S0001"), ("st_petersburg_incart", "I0001")):
            with self.subTest(source=source):
                samples = self._samples(source, name, CardiologyDetection("AR"))
                # 20 s of signal: windows at 0, 5 and 10 s.
                self.assertEqual(len(samples), 3)
                self.assertEqual(samples[0]["signal"].shape, (12, 5000))

    def test_short_recordings_give_no_windows(self):
        self.assertEqual(
            self._samples("cpsc_2018", "A0002", CardiologyDetection("WA")), []
        )

    def test_labels_follow_the_category_on_every_header_layout(self):
        # Georgia headers lack the Sx line; 1.x read their Sex line as Dx.
        for category, label in (("WA", 1), ("AR", 0)):
            with self.subTest(category=category):
                samples = self._samples(
                    "georgia", "E00001", CardiologyDetection(category)
                )
                self.assertEqual([s["label"] for s in samples], [label])
        incart = self._samples(
            "st_petersburg_incart", "I0001", CardiologyDetection("AR")
        )
        self.assertEqual(incart[0]["sex"], "Female")

    def test_window_length_follows_epoch_sec(self):
        task = CardiologyDetection("AR", epoch_sec=4, shift=4, target_fs=250)
        samples = self._samples("cpsc_2018", "A0001", task)
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0]["signal"].shape, (12, 1000))

    def test_500_hz_windows_and_labels_equal_the_1x_functions(self):
        # With the defaults, a 500 Hz recording gives the 1.x windows and
        # labels exactly (the 1.x functions pickle each window to a file).
        header = next((self.root / "cpsc_2018").glob("*/A0001.hea"))
        for category in CARDIOLOGY_CATEGORIES:
            with self.subTest(category=category), tempfile.TemporaryDirectory() as tmp:
                ours = self._samples(
                    "cpsc_2018", "A0001", CardiologyDetection(category)
                )
                legacy_fn = getattr(cardiology_detect, f"cardiology_is{category}_fn")
                legacy = legacy_fn(
                    [
                        {
                            "load_from_path": str(header.parent),
                            "patient_id": "A0001",
                            "signal_file": "A0001.mat",
                            "label_file": "A0001.hea",
                            "save_to_path": tmp,
                        }
                    ]
                )
                self.assertEqual(len(ours), len(legacy))
                for new, old in zip(ours, legacy):
                    with open(old["epoch_path"], "rb") as handle:
                        old_window = pickle.load(handle)
                    np.testing.assert_array_equal(
                        new["signal"].numpy(), old_window["signal"]
                    )
                    self.assertEqual(new["label"], int(old["label"]))

    def test_rejects_bad_settings(self):
        with self.assertRaises(ValueError):
            CardiologyDetection("XX")
        with self.assertRaises(ValueError):
            CardiologyDetection("AR", shift=0)
        # Positive, but rounds to zero samples at target_fs.
        with self.assertRaises(ValueError):
            CardiologyDetection("AR", shift=0.001, target_fs=100)
        with self.assertRaises(ValueError):
            CardiologyDetection("AR", epoch_sec=0.001, target_fs=100)

    def test_missing_age_and_sex_are_empty_strings(self):
        # pandas hands the empty age/sex cells of the metadata CSV to the
        # task as NaN, which must not become the string "nan".
        header = next((self.root / "georgia").glob("*/E00001.hea"))
        patient = Patient(self.root, "georgia", header)
        patient._event.age = float("nan")
        patient._event.sex = float("nan")
        samples = CardiologyDetection("WA")(patient)
        self.assertEqual((samples[0]["age"], samples[0]["sex"]), ("", ""))

    def test_resampling_preserves_signal_values(self):
        task = CardiologyDetection("AR")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "const.mat")
            savemat(path, {"val": np.full((12, 1000), 7, dtype=np.int16)})
            out = task._load(path, 1000.0)  # 1000 Hz -> 500 Hz
        self.assertEqual(out.shape, (12, 500))
        # Ignore filter edge effects at both ends.
        np.testing.assert_allclose(out[:, 50:-50], 7.0, atol=1e-3)


class TestCardiologyDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "training"
        write_challenge(cls.root)
        cls.cache = Path(cls.tmp.name) / "cache"
        cls.dataset = CardiologyDataset(root=str(cls.root), cache_dir=cls.cache)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_recording_is_one_patient(self):
        self.assertEqual(
            self.dataset.sources,
            ["cpsc_2018", "georgia", "ptb", "st_petersburg_incart"],
        )
        self.assertEqual(
            sorted(self.dataset.unique_patient_ids),
            ["A0001", "A0002", "E00001", "I0001", "S0001"],
        )
        event = self.dataset.get_patient("S0001").get_events()[0]
        self.assertEqual(float(event.fs), 1000.0)
        self.assertEqual(event.source, "ptb")
        self.assertTrue(str(event.signal_file).endswith("S0001.mat"))
        self.assertTrue((self.root / "cardiology-ptb-pyhealth.csv").exists())

    def test_set_task_builds_windows_of_every_source(self):
        samples = self.dataset.set_task(self.dataset.default_task)
        # A0001: 2, A0002: 0, E00001: 1, S0001: 3, I0001: 3.
        self.assertEqual(len(samples), 9)
        self.assertEqual(tuple(samples[0]["signal"].shape), (12, 5000))

    def test_sources_and_chosen_dataset_select_tables(self):
        cache = self.cache / "subset"
        by_name = CardiologyDataset(
            root=str(self.root), sources=["ptb"], cache_dir=cache
        )
        self.assertEqual(by_name.sources, ["ptb"])
        self.assertEqual(by_name.unique_patient_ids, ["S0001"])
        by_flags = CardiologyDataset(
            root=str(self.root), chosen_dataset=[0, 0, 0, 1, 0, 0], cache_dir=cache
        )
        self.assertEqual(by_flags.sources, ["ptb"])

    def test_missing_requested_source_is_skipped_with_a_warning(self):
        with self.assertLogs("pyhealth.datasets.cardiology", "WARNING") as logs:
            dataset = CardiologyDataset(
                root=str(self.root),
                sources=["ptb", "ptb-xl"],
                cache_dir=self.cache / "missing",
            )
        self.assertEqual(dataset.sources, ["ptb"])
        self.assertIn("ptb-xl", "".join(logs.output))

    def test_rejects_bad_source_selections(self):
        bad = (
            {"sources": ["ptb"], "chosen_dataset": [0, 0, 0, 1, 0, 0]},
            {"sources": ["mitbih"]},
            {"chosen_dataset": [1, 1]},
            {"chosen_dataset": [2, 0, 0, 0, 0, 0]},
            {"sources": ["ptb-xl"]},  # not on disk
        )
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CardiologyDataset(root=str(self.root), **kwargs)


if __name__ == "__main__":
    unittest.main()
