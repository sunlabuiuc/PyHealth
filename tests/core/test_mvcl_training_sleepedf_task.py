import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from collections import Counter

import concurrent.futures
from mne import create_info, io, Annotations
import numpy as np
import pandas as pd

from pyhealth.datasets import SleepEDFDataset
from pyhealth.tasks import MVCLTrainingSleepEEG


class TestMVCLTrainingSleepEEGTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.dummy_dataset_dir = Path(cls.temp_dir) / "dummy_dataset"
        cls.cassette_dir = cls.dummy_dataset_dir / "sleep-cassette"
        cls.cassette_dir.mkdir(parents=True, exist_ok=True)

        cls._create_dummy_subject_spreadsheets()
        cls._create_dummy_patient_files()
        cls._create_dummy_metadata_csv()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _create_dummy_subject_spreadsheets(cls):
        """Create required SC/ST files in both requested locations."""
        df = pd.DataFrame(
            {
                "subject": [1, 2],
                "night": [1, 2],
                "age": [25, 30],
                "sex (F=1)": [1, 2],
                "LightsOff": ["22:00", "22:30"],
            }
        )
        spreadsheet_targets = [
            cls.dummy_dataset_dir / "SC-subjects.xls",
            cls.dummy_dataset_dir / "ST-subjects.xls",
            cls.cassette_dir / "SC-subjects.xls",
            cls.cassette_dir / "ST-subjects.xls",
        ]
        for path in spreadsheet_targets:
            # These files are only test placeholders; metadata is loaded from
            # sleepedf-cassette-pyhealth.csv created below.
            df.to_csv(path, index=False)

    @classmethod
    def _create_dummy_patient_files(cls):
        """Create two patients with 2 x 3000 dummy points each."""
        # Expected cassette metadata rows in `sleepedf-cassette-pyhealth.csv` look like:
        # subject,night,age,sex,lights_off,signal_file,label_file
        # 1,1,25,F,22:00,<...>/SC4011E0-PSG.edf,<...>/SC4011E0-Hypnogram.edf
        # 2,2,30,M,22:30,<...>/SC4022E0-PSG.edf,<...>/SC4022E0-Hypnogram.edf
        #
        # This helper creates those referenced PSG/Hypnogram files so SleepEDFDataset
        # can load events and MVCLTrainingSleepEEG can read per-patient signal/labels.
        patient_records = [
            ("SC4011E0", 1),  # subject 01, night 1
            ("SC4022E0", 2),  # subject 02, night 2
        ]
        def write_patient_file(record):
            stem, seed = record
            signal = np.full(6000, fill_value=seed, dtype=np.float32)
            for suffix in ("-PSG.edf", "-Hypnogram.edf"):
                file_path = cls.cassette_dir / f"{stem}{suffix}"
                signal.tofile(file_path)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(write_patient_file, patient_records)

    @classmethod
    def _create_dummy_metadata_csv(cls):
        rows = [
            {
                "subject": 1,
                "night": 1,
                "age": 25,
                "sex": "F",
                "lights_off": "22:00",
                "signal_file": str(cls.cassette_dir / "SC4011E0-PSG.edf"),
                "label_file": str(cls.cassette_dir / "SC4011E0-Hypnogram.edf"),
            },
            {
                "subject": 2,
                "night": 2,
                "age": 30,
                "sex": "M",
                "lights_off": "22:30",
                "signal_file": str(cls.cassette_dir / "SC4022E0-PSG.edf"),
                "label_file": str(cls.cassette_dir / "SC4022E0-Hypnogram.edf"),
            },
        ]
        pd.DataFrame(rows).to_csv(
            cls.dummy_dataset_dir / "sleepedf-cassette-pyhealth.csv", index=False
        )

    @staticmethod
    def _mock_read_raw_edf(signal_file, *args, **kwargs):
        """Load the binary dummy payload from .edf placeholder file."""
        signal = np.fromfile(signal_file, dtype=np.float32)
        if signal.size != 6000:
            raise ValueError(f"Expected 6000 points in {signal_file}, got {signal.size}")
        data = signal.reshape(1, -1)
        info = create_info(["EEG Fpz-Cz"], sfreq=100, ch_types=["eeg"])
        return io.RawArray(data, info, verbose="error")

    @staticmethod
    def _mock_read_annotations(label_file, *args, **kwargs):
        """Return two 30-second sleep-stage annotations per patient."""
        name = Path(label_file).name
        if "SC4011E0" in name:
            descriptions = ["Sleep stage W", "Sleep stage R"]
        elif "SC4022E0" in name:
            descriptions = ["Sleep stage 2", "Sleep stage 4"]
        else:
            raise ValueError(f"Unexpected label file: {label_file}")
        return Annotations(
            onset=[0.0, 30.0],
            duration=[30.0, 30.0],
            description=descriptions,
        )

    def test_import_from_pyhealth_tasks(self):
        """Matches notebook usage: from pyhealth.tasks import MVCLTrainingSleepEEG."""
        self.assertTrue(callable(MVCLTrainingSleepEEG))

    def test_sleepedf_dummy_dataset_label_mapping(self):
        dataset = SleepEDFDataset(root=str(self.dummy_dataset_dir), subset="cassette")
        task = MVCLTrainingSleepEEG(window_size=200, crop_length=178, eeg_channel="EEG Fpz-Cz")

        with patch(
            "pyhealth.tasks.mvcl_training_sleepedf_task.mne.io.read_raw_edf",
            side_effect=self._mock_read_raw_edf,
        ), patch(
            "pyhealth.tasks.mvcl_training_sleepedf_task.mne.read_annotations",
            side_effect=self._mock_read_annotations,
        ):
            sample_dataset = dataset.set_task(task, num_workers=1)

        # 2 patients x 2 epochs each x (3000 / 200) windows = 60 windows
        self.assertEqual(len(sample_dataset), 60)
        self.assertEqual(sample_dataset.input_schema, {"xt": "tensor", "xd": "tensor", "xf": "tensor"})
        self.assertEqual(sample_dataset.output_schema, {"label": "multiclass"})

        sample = sample_dataset[0]
        for key in ("xt", "xd", "xf"):
            self.assertIn(key, sample)
        # MV views are stored as [L, C] in the task; enforce equivalent 1x178 content.
        for key in ("xt", "xd", "xf"):
            self.assertEqual(sample[key].ndim, 2)
            self.assertIn(1, sample[key].shape)
            self.assertIn(178, sample[key].shape)

        # set_task() encodes multiclass labels to contiguous ids, but class balance
        # should still match the four injected stages (W, R, 2, 4) => 15 windows each.
        label_counts = Counter(int(s["label"]) for s in sample_dataset)
        self.assertEqual(len(label_counts), 4)
        self.assertTrue(all(count == 15 for count in label_counts.values()))


if __name__ == "__main__":
    unittest.main()
