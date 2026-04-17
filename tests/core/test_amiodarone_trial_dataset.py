"""Tests for AmiodaroneTrialDataset.

The raw trial data is embedded in the module, so these tests don't
need a network or external files.
"""

import os
import tempfile
import unittest

import pandas as pd


class TestAmiodaroneTrialDataset(unittest.TestCase):
    """Tests for AmiodaroneTrialDataset."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_prepare_metadata_creates_csv(self):
        """prepare_metadata produces the expected CSV."""
        from pyhealth.datasets.amiodarone_trial_dataset import (
            AmiodaroneTrialDataset,
        )

        AmiodaroneTrialDataset.prepare_metadata(self.tmpdir)
        csv_path = os.path.join(
            self.tmpdir, "amiodarone_trials-metadata-pyhealth.csv"
        )
        self.assertTrue(os.path.exists(csv_path))

        df = pd.read_csv(csv_path)
        # 21 original trials from Letelier et al. 2003
        self.assertEqual(len(df), 21)
        for col in (
            "patient_id",
            "visit_id",
            "trial_name",
            "log_relative_risk",
            "variance",
            "split",
        ):
            self.assertIn(col, df.columns)

    def test_split_counts(self):
        """Dataset splits into 10 placebo-controlled and 11 non-placebo."""
        from pyhealth.datasets.amiodarone_trial_dataset import (
            AmiodaroneTrialDataset,
        )

        AmiodaroneTrialDataset.prepare_metadata(self.tmpdir)
        df = pd.read_csv(
            os.path.join(
                self.tmpdir, "amiodarone_trials-metadata-pyhealth.csv"
            )
        )
        self.assertEqual((df["split"] == "trusted").sum(), 10)
        self.assertEqual((df["split"] == "untrusted").sum(), 11)

    def test_effect_calculation_plausible(self):
        """Log relative risk values fall within a clinically plausible range."""
        from pyhealth.datasets.amiodarone_trial_dataset import (
            AmiodaroneTrialDataset,
        )

        AmiodaroneTrialDataset.prepare_metadata(self.tmpdir)
        df = pd.read_csv(
            os.path.join(
                self.tmpdir, "amiodarone_trials-metadata-pyhealth.csv"
            )
        )
        # Amiodarone is effective => most log RR should be positive
        self.assertGreater((df["log_relative_risk"] > 0).sum(), 15)
        # Variances must be non-negative
        self.assertTrue((df["variance"] >= 0).all())

    def test_dose_values_in_expected_range(self):
        """Manually computed doses are in a sensible mg range."""
        from pyhealth.datasets.amiodarone_trial_dataset import (
            AmiodaroneTrialDataset,
        )

        AmiodaroneTrialDataset.prepare_metadata(self.tmpdir)
        df = pd.read_csv(
            os.path.join(
                self.tmpdir, "amiodarone_trials-metadata-pyhealth.csv"
            )
        )
        # Paper doses span roughly 350-3000 mg; accept a wider window.
        self.assertTrue((df["amiodarone_total_24h_mg"] >= 100).all())
        self.assertTrue((df["amiodarone_total_24h_mg"] <= 5000).all())

    def test_all_feature_columns_numeric(self):
        """All declared feature columns are numeric and have no NaNs."""
        from pyhealth.datasets.amiodarone_trial_dataset import (
            AmiodaroneTrialDataset,
            FEATURE_COLUMNS,
        )

        AmiodaroneTrialDataset.prepare_metadata(self.tmpdir)
        df = pd.read_csv(
            os.path.join(
                self.tmpdir, "amiodarone_trials-metadata-pyhealth.csv"
            )
        )
        for col in FEATURE_COLUMNS:
            self.assertIn(col, df.columns)
            self.assertTrue(
                pd.api.types.is_numeric_dtype(df[col]),
                f"Column {col} is not numeric",
            )
            self.assertFalse(df[col].isna().any(), f"{col} has NaN")


if __name__ == "__main__":
    unittest.main()
