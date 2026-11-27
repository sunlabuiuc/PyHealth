"""VariantClassificationClinVar label mapping for ClinVar's own values.

ClinVar's ``ClinicalSignificance`` column holds values such as
``Conflicting classifications of pathogenicity``. A conflicting variant has
no agreed class, so the task must drop it, as its docstring says, rather
than label it Pathogenic because the text contains "pathogenic".
"""

import unittest
from types import SimpleNamespace

from pyhealth.tasks import VariantClassificationClinVar


class _Variant:
    """Minimal stand-in for a ClinVar ``Patient`` with one variant event."""

    def __init__(self, clinical_significance):
        self.patient_id = "0"
        self.event = SimpleNamespace(
            gene_symbol="BRCA1",
            variant_type="single nucleotide variant",
            chromosome="17",
            clinical_significance=clinical_significance,
        )

    def get_events(self, event_type=None):
        return [self.event]


class TestClinVarConflictingSignificance(unittest.TestCase):
    def setUp(self):
        self.task = VariantClassificationClinVar()

    def test_conflicting_values_are_dropped(self):
        for value in (
            # ClinVar's current wording, and the wording in older releases.
            "Conflicting classifications of pathogenicity",
            "Conflicting interpretations of pathogenicity",
            "Conflicting classifications of pathogenicity; other",
            "Conflicting classifications of pathogenicity; risk factor",
        ):
            with self.subTest(value=value):
                self.assertIsNone(self.task._normalize_clinical_significance(value))
                self.assertEqual(self.task(_Variant(value)), [])

    def test_standard_values_keep_their_class(self):
        expected = {
            "Pathogenic": "Pathogenic",
            "Likely pathogenic": "Likely pathogenic",
            "Pathogenic/Likely pathogenic": "Likely pathogenic",
            "Uncertain significance": "Uncertain significance",
            "Likely benign": "Likely benign",
            "Benign/Likely benign": "Likely benign",
            "Benign": "Benign",
        }
        for value, label in expected.items():
            with self.subTest(value=value):
                samples = self.task(_Variant(value))
                self.assertEqual(len(samples), 1)
                self.assertEqual(samples[0]["clinical_significance"], label)


if __name__ == "__main__":
    unittest.main()
