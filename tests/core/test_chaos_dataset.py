"""Synthetic tests for the CHAOS dataset loader."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyhealth.datasets import CHAOSDataset


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


class TestCHAOSDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir) / "CHAOS"
        self.cache_dir = Path(self.temp_dir) / "cache"

        for index in range(2):
            _touch(
                self.root
                / "Train_Sets/CT/1/DICOM_anon"
                / f"slice_{index}.dcm"
            )
            _touch(
                self.root
                / "Train_Sets/CT/1/Ground"
                / f"mask_{index}.png"
            )
        _touch(self.root / "Test_Sets/CT/3/DICOM_anon/slice.dcm")

        for phase in ("InPhase", "OutPhase"):
            for index in range(2):
                _touch(
                    self.root
                    / "Train_Sets/MR/2/T1DUAL/DICOM_anon"
                    / phase
                    / f"slice_{index}.dcm"
                )
            _touch(
                self.root
                / "Test_Sets/MR/4/T1DUAL/DICOM_anon"
                / phase
                / "slice.dcm"
            )
        _touch(self.root / "Train_Sets/MR/2/T1DUAL/Ground/mask.png")
        _touch(self.root / "Train_Sets/MR/2/T2SPIR/DICOM_anon/slice.dcm")
        _touch(self.root / "Train_Sets/MR/2/T2SPIR/Ground/mask.png")
        _touch(self.root / "Test_Sets/MR/4/T2SPIR/DICOM_anon/slice.dcm")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_metadata(self):
        metadata_dir = Path(self.temp_dir) / "metadata"
        dataset = CHAOSDataset(
            root=str(self.root),
            cache_dir=str(self.cache_dir),
            metadata_dir=str(metadata_dir),
        )
        self.assertEqual(
            set(dataset.unique_patient_ids),
            {"ct_1", "ct_3", "mr_2", "mr_4"},
        )

        ct_events = dataset.get_patient("ct_1").get_events()
        self.assertEqual(len(ct_events), 1)
        self.assertEqual(ct_events[0].sequence, "CT")
        self.assertEqual(int(ct_events[0].n_slices), 2)
        self.assertEqual(int(ct_events[0].n_masks), 2)

        mr_events = dataset.get_patient("mr_2").get_events()
        self.assertEqual(len(mr_events), 3)
        by_record = {event.record_id: event for event in mr_events}
        in_phase = by_record["mr_2_t1dual_inphase"]
        out_phase = by_record["mr_2_t1dual_outphase"]
        t2spir = by_record["mr_2_t2spir"]
        self.assertEqual(in_phase.phase, "InPhase")
        self.assertEqual(out_phase.phase, "OutPhase")
        self.assertEqual(in_phase.mask_dir, out_phase.mask_dir)
        self.assertEqual(int(in_phase.n_slices), 2)
        self.assertEqual(int(in_phase.n_masks), 1)
        self.assertEqual(t2spir.sequence, "T2SPIR")
        self.assertEqual(int(t2spir.n_slices), 1)

        test_events = dataset.get_patient("mr_4").get_events()
        self.assertEqual(len(test_events), 3)
        self.assertTrue(all(not event.mask_dir for event in test_events))
        self.assertTrue(
            all(str(event.has_mask) == "False" for event in test_events)
        )

        self.assertTrue(
            (metadata_dir / "chaos-metadata-pyhealth.csv").exists()
        )
        self.assertFalse((self.root / "chaos-metadata-pyhealth.csv").exists())

    def test_missing_train_sets_and_download_and_default_metadata_dir(self):
        missing_root = Path(self.temp_dir) / "missing_chaos"
        missing_root.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            CHAOSDataset(
                root=str(missing_root),
                cache_dir=str(self.cache_dir),
            )

        def _populate_on_download(*_args, **_kwargs):
            _touch(
                missing_root / "Train_Sets/CT/1/DICOM_anon/slice_0.dcm"
            )
            _touch(
                missing_root / "Train_Sets/CT/1/Ground/mask_0.png"
            )

        with mock.patch(
            "pyhealth.datasets.chaos._download_archives",
            side_effect=_populate_on_download,
        ) as mocked:
            dataset = CHAOSDataset(
                root=str(missing_root),
                cache_dir=str(self.cache_dir),
                download=True,
            )
            mocked.assert_called_once()
            self.assertEqual(set(dataset.unique_patient_ids), {"ct_1"})

        # Collision between Train_Sets and Test_Sets case IDs must raise
        _touch(missing_root / "Test_Sets/CT/1/DICOM_anon/slice_0.dcm")
        with self.assertRaises(ValueError):
            CHAOSDataset(
                root=str(missing_root),
                cache_dir=str(self.cache_dir),
                refresh_metadata=True,
            )


if __name__ == "__main__":
    unittest.main()
