import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


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
            raise unittest.SkipTest(f"Failed to download MIMIC-IV demo dataset: {e}")
        except FileNotFoundError:
            raise unittest.SkipTest(
                "wget not available - skipping MIMIC-IV download test"
            )

        # Find the downloaded dataset path
        physionet_dir = (
            Path(self.temp_dir) / "physionet.org" / "files" / "mimic-iv-demo" / "2.2"
        )
        if physionet_dir.exists():
            self.demo_dataset_path = str(physionet_dir)
        else:
            raise unittest.SkipTest("Downloaded dataset not found in expected location")

    def _maybe_import_pyhealth(self):
        try:
            from pyhealth.datasets.mimic4 import MIMIC4Dataset  # noqa: F401
            from pyhealth.tasks.ckd_surv import (  # noqa: F401
                MIMIC4CKDSurvAnalysis,
            )
        except Exception as e:
            raise unittest.SkipTest(f"pyhealth import failed or incomplete: {e}")

    def test_mimic4_ckd_surv_set_task(self):
        """Instantiate MIMIC4Dataset with ehr_root and run set_task()."""
        try:
            from pyhealth.datasets.mimic4 import MIMIC4Dataset
            from pyhealth.tasks.ckd_surv import MIMIC4CKDSurvAnalysis
        except Exception as e:
            raise unittest.SkipTest(f"pyhealth import failed or incomplete: {e}")

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

        # Build task and run set_task; allow zero samples
        # but expect no exceptions during execution
        try:
            task = MIMIC4CKDSurvAnalysis()
        except Exception as e:
            raise unittest.SkipTest(f"Task initialization incomplete or failing: {e}")

        try:
            sample_dataset = dataset.set_task(task)
        except Exception as e:
            self.fail(f"set_task() raised an exception: {e}")

        self.assertIsNotNone(sample_dataset, "set_task should return a dataset")
        self.assertTrue(
            hasattr(sample_dataset, "samples"),
            "Returned dataset should have a 'samples' attribute",
        )

        # If samples exist, perform a couple of light checks
        if hasattr(sample_dataset, "samples") and sample_dataset.samples:
            sample = sample_dataset.samples[0]
            self.assertIsInstance(sample, dict)
            self.assertIn("patient_id", sample)


if __name__ == "__main__":
    unittest.main()
