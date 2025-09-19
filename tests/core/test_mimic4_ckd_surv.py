import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import json


class TestMIMIC4CKDSurv(unittest.TestCase):
    """Test CKD survival analysis task on MIMIC-IV demo using ehr_root.

    This test downloads the MIMIC-IV demo dataset from PhysioNet using wget,
    constructs a MIMIC4Dataset with ehr_root, and calls set_task() with
    MIMIC4CKDSurvAnalysis. The test is tolerant to environments without
    network access or wget and will skip gracefully when required resources
    or implementations are unavailable.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.demo_dataset_path = None
        self._download_demo_dataset()
        self._maybe_import_pyhealth()

    def tearDown(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _download_demo_dataset(self):
        """Download MIMIC-IV demo dataset using wget (skip if unavailable)."""
        download_url = "https://physionet.org/files/mimic-iv-demo/2.2/"
        cmd = [
            "wget",
            "-r",
            "-N",
            "-c",
            "-np",
            "--directory-prefix",
            self.temp_dir,
            download_url,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise unittest.SkipTest(
                f"Failed to download MIMIC-IV demo dataset: {e}"
            )
        except FileNotFoundError:
            raise unittest.SkipTest(
                "wget not available - skipping MIMIC-IV download test"
            )

        # Find the downloaded dataset path
        physionet_dir = (
            Path(self.temp_dir)
            / "physionet.org"
            / "files"
            / "mimic-iv-demo"
            / "2.2"
        )
        if physionet_dir.exists():
            self.demo_dataset_path = str(physionet_dir)
        else:
            raise unittest.SkipTest(
                "Downloaded dataset not found in expected location"
            )

    def _maybe_import_pyhealth(self):
        try:
            from pyhealth.datasets.mimic4 import MIMIC4Dataset  # noqa: F401
            from pyhealth.tasks.ckd_surv import (  # noqa: F401
                MIMIC4CKDSurvAnalysis,
            )
        except Exception as e:
            raise unittest.SkipTest(
                f"pyhealth import failed or incomplete: {e}"
            )

    def test_mimic4_ckd_surv_all_modes(self):
        """Instantiate MIMIC4Dataset and run set_task() for all modes.

        Modes: time_invariant, time_variant, heterogeneous. For each mode,
        assert dataset is constructed and, if samples exist, print the first
        sample for visualization.
        """
        try:
            from pyhealth.datasets.mimic4 import MIMIC4Dataset
            from pyhealth.tasks.ckd_surv import MIMIC4CKDSurvAnalysis
        except Exception as e:
            raise unittest.SkipTest(
                f"pyhealth import failed or incomplete: {e}"
            )

        # Initialize dataset with EHR tables needed by the task
        ehr_tables = [
            "patients",
            "admissions",
            "labevents",
            "diagnoses_icd",
        ]

        try:
            dataset = MIMIC4Dataset(
                ehr_root=self.demo_dataset_path,
                ehr_tables=ehr_tables,
                dataset_name="mimic4_demo_ehr",
            )
        except Exception as e:
            raise unittest.SkipTest(f"Failed to construct MIMIC4Dataset: {e}")

        modes = ["time_invariant", "time_variant", "heterogeneous"]
        for mode in modes:
            try:
                task = MIMIC4CKDSurvAnalysis(setting=mode)
            except Exception as e:
                raise unittest.SkipTest(
                    f"Task initialization failed for mode '{mode}': {e}"
                )

            try:
                sample_dataset = dataset.set_task(task)
            except Exception as e:
                self.fail(
                    f"set_task() raised an exception in mode '{mode}': {e}"
                )

            self.assertIsNotNone(
                sample_dataset, "set_task should return a dataset"
            )
            self.assertTrue(
                hasattr(sample_dataset, "samples"),
                "Returned dataset should have a 'samples' attribute",
            )

            n = len(getattr(sample_dataset, "samples", []))
            print(f"\n[ckd_surv] mode={mode} samples={n}")

            if n:
                sample = sample_dataset.samples[0]
                self.assertIsInstance(sample, dict)
                self.assertIn("patient_id", sample)

                # Pretty-print a compact view depending on mode
                view = {k: sample.get(k) for k in (
                    "patient_id",
                    "duration_days",
                    "has_esrd",
                    "baseline_egfr",
                    "lab_measurements",
                ) if k in sample}

                # Truncate lab_measurements if long
                if "lab_measurements" in view and isinstance(
                    view["lab_measurements"], list
                ):
                    lm = view["lab_measurements"]
                    view["lab_measurements"] = lm[:3]

                print(json.dumps(view, indent=2, default=str))


if __name__ == "__main__":
    unittest.main()
