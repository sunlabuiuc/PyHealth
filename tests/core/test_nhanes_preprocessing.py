"""Tests for NHANES raw data preprocessing."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from pyhealth.datasets.nhanes import (
    iter_participants,
    prepare_activity,
    prepare_clinical,
    prepare_mortality,
    prepare_nhanes,
)


class TestNHANESPreprocessing(unittest.TestCase):
    """Tests raw data parsing and preprocessing."""

    def test_iter_participants_across_chunks(self) -> None:
        """Participants spanning chunks should remain intact."""
        chunk1 = pd.DataFrame(
            {
                "SEQN": [1, 1, 2],
                "PAXN": [1, 2, 1],
                "PAXINTEN": [10.0, 20.0, 30.0],
            }
        )

        chunk2 = pd.DataFrame(
            {
                "SEQN": [2, 2, 3],
                "PAXN": [2, 3, 1],
                "PAXINTEN": [40.0, 50.0, 60.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activity.xpt"
            path.touch()

            reader = MagicMock()
            reader.__enter__.return_value = iter([chunk1, chunk2])

            with patch(
                "pyhealth.datasets.nhanes.pd.read_sas",
                return_value=reader,
            ):
                participants = list(iter_participants(path, chunksize=3))

            self.assertEqual(len(participants), 3)

            self.assertEqual(
                [len(person) for person in participants],
                [2, 3, 1],
            )

            self.assertEqual(
                participants[1]["PAXN"].tolist(),
                [1, 2, 3],
            )

            self.assertEqual(
                participants[1]["PAXINTEN"].tolist(),
                [30.0, 40.0, 50.0],
            )

    def test_prepare_activity(self) -> None:
        """Activity records should become participant Parquet files."""
        participant = pd.DataFrame(
            {
                "SEQN": np.full(10080, 1),
                "PAXN": np.arange(1, 10081),
                "PAXINTEN": np.full(10080, 100.0),
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with patch(
                "pyhealth.datasets.nhanes.iter_participants",
                return_value=(p for p in [participant]),
            ):
                metadata = prepare_activity(root, "2003-2004")

            self.assertEqual(len(metadata), 1)

            row = metadata.iloc[0]

            self.assertEqual(row["patient_id"], "1")
            self.assertEqual(row["num_epochs"], 2016)
            self.assertEqual(row["total_wear_minutes"], 10080)

            path = root / row["activity_file"]
            self.assertTrue(path.is_file())

            epochs = pd.read_parquet(path)

            self.assertEqual(epochs.shape, (2016, 7))

            np.testing.assert_allclose(epochs["intensity"], 100.0)

    def test_prepare_clinical(self) -> None:
        """Clinical tables should merge by participant identifier."""
        demo = pd.DataFrame(
            {
                "SEQN": [1, 2],
                "RIDAGEYR": [50, 60],
                "RIAGENDR": [1, 2],
                "RIDRETH1": [3, 4],
            }
        )

        bmx = pd.DataFrame(
            {
                "SEQN": [1],
                "BMXWT": [70.0],
                "BMXHT": [170.0],
                "BMXBMI": [24.2],
            }
        )

        bpx = pd.DataFrame(
            {
                "SEQN": [1],
                "BPXSY1": [120.0],
                "BPXSY2": [130.0],
                "BPXSY3": [np.nan],
                "BPXSY4": [np.nan],
                "BPXDI1": [80.0],
                "BPXDI2": [90.0],
                "BPXDI3": [np.nan],
                "BPXDI4": [np.nan],
            }
        )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "pyhealth.datasets.nhanes.pd.read_sas",
                side_effect=[demo, bmx, bpx],
            ),
        ):
            clinical = prepare_clinical(tmp, "2003-2004")

        clinical = clinical.set_index("patient_id")

        self.assertEqual(len(clinical), 2)
        self.assertEqual(clinical.loc["1", "age"], 50)
        self.assertAlmostEqual(clinical.loc["1", "bmi"], 24.2)

        self.assertEqual(clinical.loc["1", "systolic_bp"], 125.0)

        self.assertEqual(clinical.loc["1", "diastolic_bp"], 85.0)

        # Left joins must preserve participants with missing data.
        self.assertTrue(pd.isna(clinical.loc["2", "bmi"]))

    def test_prepare_mortality(self) -> None:
        """Fixed-width mortality records should parse correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mortality_dir = root / "mortality"
            mortality_dir.mkdir()

            path = mortality_dir / "NHANES_2003_2004_MORT_2019_PUBLIC.dat"

            lines = []

            for pid, mortstat, followup in [
                (1, 0, 120),
                (2, 1, 30),
            ]:
                record = [" "] * 48

                record[0:6] = f"{pid:06d}"
                record[14] = "1"
                record[15] = str(mortstat)
                record[42:45] = f"{followup:03d}"
                record[45:48] = f"{followup:03d}"

                lines.append("".join(record))

            path.write_text(
                "\n".join(lines) + "\n",
                encoding="ascii",
            )

            mortality = prepare_mortality(root, "2003-2004")

        self.assertEqual(len(mortality), 2)

        self.assertEqual(
            mortality["patient_id"].tolist(),
            ["1", "2"],
        )

        self.assertEqual(
            mortality["eligstat"].tolist(),
            [1, 1],
        )

        self.assertEqual(
            mortality["mortstat"].tolist(),
            [0, 1],
        )

        self.assertEqual(
            mortality["permth_int"].tolist(),
            [120, 30],
        )

        self.assertEqual(
            mortality["permth_exm"].tolist(),
            [120, 30],
        )

    def test_prepare_nhanes_cached_metadata(self) -> None:
        """Existing processed metadata should be reused."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed/pyhealth"
            processed.mkdir(parents=True)

            expected = pd.DataFrame(
                {
                    "patient_id": ["1", "2"],
                    "cycle": ["2003-2004", "2005-2006"],
                }
            )

            expected.to_parquet(
                processed / "nhanes-metadata.parquet",
                index=False,
            )

            with patch("pyhealth.datasets.nhanes.prepare_activity") as mock_activity:
                actual = prepare_nhanes(root)

                mock_activity.assert_not_called()

            pd.testing.assert_frame_equal(actual, expected)

    def test_invalid_cycle(self) -> None:
        """Unsupported survey cycles should raise errors."""
        with tempfile.TemporaryDirectory() as tmp:
            for function in [
                prepare_activity,
                prepare_clinical,
                prepare_mortality,
            ]:
                with (
                    self.subTest(function=function.__name__),
                    self.assertRaises(ValueError),
                ):
                    function(tmp, "invalid-cycle")

    def test_missing_activity_file(self) -> None:
        """Missing raw activity files should raise errors."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.xpt"

            with self.assertRaises(FileNotFoundError):
                list(iter_participants(path))

    def test_invalid_chunksize(self) -> None:
        """Chunk size must be positive."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activity.xpt"
            path.touch()

            with self.assertRaises(ValueError):
                list(iter_participants(path, chunksize=0))


if __name__ == "__main__":
    unittest.main()
