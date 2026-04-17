"""Tests for PMLBMetaAnalysisDataset.

Uses synthetic CSV data that mimics the PMLB format so tests run
in milliseconds without needing the ``pmlb`` package or a network.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


def _make_synthetic_pmlb_frame(n: int = 30, d: int = 4) -> pd.DataFrame:
    """Build a tiny DataFrame shaped like a PMLB fetch result."""
    rng = np.random.RandomState(0)
    cols = {f"feat_{i}": rng.randn(n) for i in range(d)}
    cols["target"] = rng.randn(n) * 10
    return pd.DataFrame(cols)


class TestPMLBMetaAnalysisDataset(unittest.TestCase):
    """Tests for PMLBMetaAnalysisDataset."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_prepare_metadata_creates_csv(self):
        """prepare_metadata writes a CSV with expected columns."""
        from pyhealth.datasets.pmlb_meta_analysis_dataset import (
            PMLBMetaAnalysisDataset,
        )

        synthetic = _make_synthetic_pmlb_frame()
        with patch(
            "pyhealth.datasets.pmlb_meta_analysis_dataset.fetch_data",
            return_value=synthetic,
            create=True,
        ):
            # Direct call to the underlying fetch path
            with patch("pmlb.fetch_data", return_value=synthetic):
                PMLBMetaAnalysisDataset.prepare_metadata(
                    root=self.tmpdir,
                    pmlb_dataset_name="1196_BNG_pharynx",
                    synthesize_noise=False,
                )

        expected = os.path.join(
            self.tmpdir, "pmlb_meta_analysis-metadata-pyhealth.csv"
        )
        self.assertTrue(os.path.exists(expected))
        df = pd.read_csv(expected)
        self.assertIn("patient_id", df.columns)
        self.assertIn("visit_id", df.columns)
        self.assertIn("true_effect", df.columns)
        self.assertEqual(len(df), 30)

    def test_synthesize_noise_adds_columns(self):
        """synthesize_noise=True adds observed_effect, variance, prior_mean."""
        from pyhealth.datasets.pmlb_meta_analysis_dataset import (
            PMLBMetaAnalysisDataset,
        )

        synthetic = _make_synthetic_pmlb_frame()
        with patch("pmlb.fetch_data", return_value=synthetic):
            PMLBMetaAnalysisDataset.prepare_metadata(
                root=self.tmpdir,
                pmlb_dataset_name="1196_BNG_pharynx",
                synthesize_noise=True,
                prior_error=0.5,
                effect_noise=0.2,
                seed=42,
            )

        noisy_path = os.path.join(
            self.tmpdir, "pmlb_meta_analysis_noisy-metadata-pyhealth.csv"
        )
        self.assertTrue(os.path.exists(noisy_path))
        df = pd.read_csv(noisy_path)
        for col in ("observed_effect", "variance", "prior_mean"):
            self.assertIn(col, df.columns)

    def test_invalid_dataset_name_raises(self):
        """Invalid pmlb_dataset_name raises ValueError."""
        from pyhealth.datasets.pmlb_meta_analysis_dataset import (
            PMLBMetaAnalysisDataset,
        )

        with self.assertRaises(ValueError):
            PMLBMetaAnalysisDataset(
                root=self.tmpdir,
                pmlb_dataset_name="not_a_real_dataset",
            )

    def test_noise_generation_reproducible(self):
        """Same seed produces the same synthetic noise columns."""
        from pyhealth.datasets.pmlb_meta_analysis_dataset import (
            PMLBMetaAnalysisDataset,
        )

        df = _make_synthetic_pmlb_frame()
        df.insert(0, "patient_id", [f"t_{i}" for i in range(len(df))])
        df.insert(1, "visit_id", [f"v_{i}" for i in range(len(df))])
        df = df.rename(columns={"target": "true_effect"})

        a = PMLBMetaAnalysisDataset._add_synthetic_noise(
            df, prior_error=0.5, effect_noise=0.2, seed=7
        )
        b = PMLBMetaAnalysisDataset._add_synthetic_noise(
            df, prior_error=0.5, effect_noise=0.2, seed=7
        )
        np.testing.assert_array_equal(
            a["observed_effect"].to_numpy(), b["observed_effect"].to_numpy()
        )
        np.testing.assert_array_equal(
            a["variance"].to_numpy(), b["variance"].to_numpy()
        )


if __name__ == "__main__":
    unittest.main()
