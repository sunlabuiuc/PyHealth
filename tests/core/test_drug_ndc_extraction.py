"""Tests that MIMIC-3/4 tasks extract NDC codes (not drug names) from prescriptions.

The fix changed event.drug -> event.ndc in mortality and readmission tasks
so that CrossMap NDC->ATC mapping actually receives valid NDC codes.
"""

import re
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks.mortality_prediction import MortalityPredictionMIMIC3
from pyhealth.tasks.readmission_prediction import ReadmissionPredictionMIMIC3


# NDC codes are numeric strings (with possible leading zeros), and may include hyphens,
# but should not contain letters (e.g., "0002-3227-30" is a valid NDC format).
NDC_PATTERN = re.compile(r"^[0-9-]+$")


class TestDrugNDCExtraction(unittest.TestCase):
    """Verify that task classes extract NDC codes from prescriptions, not drug names."""

    @classmethod
    def setUpClass(cls):
        cls.cache_dir = tempfile.TemporaryDirectory()
        demo_path = str(
            Path(__file__).parent.parent.parent
            / "test-resources"
            / "core"
            / "mimic3demo"
        )
        cls.dataset = MIMIC3Dataset(
            root=demo_path,
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=cls.cache_dir.name,
        )

    @classmethod
    def tearDownClass(cls):
        cls.cache_dir.cleanup()

    def _get_drug_vocab(self, sample_dataset):
        """Get the drug vocabulary from the processor, excluding special tokens."""
        proc = sample_dataset.input_processors["drugs"]
        return {
            k for k in proc.code_vocab.keys()
            if k not in ("<pad>", "<unk>")
        }

    def test_mortality_drugs_are_ndc_codes(self):
        """MortalityPredictionMIMIC3 drug vocabulary should contain NDC codes, not drug names."""
        task = MortalityPredictionMIMIC3()
        sample_dataset = self.dataset.set_task(task)
        drug_vocab = self._get_drug_vocab(sample_dataset)

        self.assertGreater(len(drug_vocab), 0, "Should have at least one drug code")

        # NDC codes are numeric; drug names contain letters
        non_ndc = [d for d in drug_vocab if not NDC_PATTERN.match(str(d))]
        self.assertEqual(
            len(non_ndc),
            0,
            f"Found drug names instead of NDC codes: {non_ndc[:5]}",
        )

    def test_readmission_drugs_are_ndc_codes(self):
        """ReadmissionPredictionMIMIC3 drug vocabulary should contain NDC codes, not drug names."""
        task = ReadmissionPredictionMIMIC3()
        sample_dataset = self.dataset.set_task(task)
        drug_vocab = self._get_drug_vocab(sample_dataset)

        self.assertGreater(len(drug_vocab), 0, "Should have at least one drug code")

        non_ndc = [d for d in drug_vocab if not NDC_PATTERN.match(str(d))]
        self.assertEqual(
            len(non_ndc),
            0,
            f"Found drug names instead of NDC codes: {non_ndc[:5]}",
        )

    def test_drug_vocab_not_drug_names(self):
        """Drug vocabulary should not contain common drug names."""
        task = MortalityPredictionMIMIC3()
        sample_dataset = self.dataset.set_task(task)
        drug_vocab = self._get_drug_vocab(sample_dataset)

        # These are drug names that would appear if event.drug was used
        drug_names = {"Aspirin", "Bisacodyl", "Senna", "Heparin", "Insulin"}
        overlap = drug_vocab & drug_names
        self.assertEqual(
            len(overlap),
            0,
            f"Vocabulary contains drug names (should be NDC codes): {overlap}",
        )


if __name__ == "__main__":
    unittest.main()
