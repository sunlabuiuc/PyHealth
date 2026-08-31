import unittest
from pathlib import Path

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.length_of_stay_stagenet_mimic4 import LengthOfStayStageNetMIMIC4
from pyhealth.tasks.mortality_prediction_stagenet_mimic4 import (
    MortalityPredictionStageNetMIMIC4,
)


class TestStageNetTaskLeakagePrevention(unittest.TestCase):
    """Regression tests for the target-admission leakage fix.

    ``diagnoses_icd``/``procedures_icd`` events are timestamped at
    ``dischtime`` (see pyhealth/datasets/configs/mimic4_ehr.yaml), so codes
    recorded for the admission whose own outcome (mortality or LOS) is being
    predicted are only known at-or-after that outcome. These tests verify
    that MortalityPredictionStageNetMIMIC4 and LengthOfStayStageNetMIMIC4
    exclude that admission's codes while leaving earlier, already-resolved
    admissions unaffected -- and, for mortality specifically, that this
    restriction applies identically to death and survivor cases. An
    earlier version of the fix restricted only the death class, which let
    a model learn "richer features -> survived" as a shortcut from the
    asymmetric amount of information available per class, rather than any
    real clinical signal.
    """

    @classmethod
    def setUpClass(cls):
        test_dir = Path(__file__).parent.parent.parent
        root = str(test_dir / "test-resources" / "core" / "mimic4demo")
        tables = ["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"]
        cls.dataset = MIMIC4Dataset(ehr_root=root, ehr_tables=tables)

    def test_mortality_excludes_terminal_admission_codes(self):
        """Patient 10003 has two admissions (20005, then terminal 20006).

        Codes from the non-terminal admission (20005) must be present;
        codes from the terminal admission (20006), which are only known at
        its own dischtime, must not leak into the features.
        """
        patient = self.dataset.get_patient("10003")
        samples = MortalityPredictionStageNetMIMIC4()(patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["mortality"], 1)

        _, icd_codes = sample["icd_codes"]
        flat_codes = [code for visit in icd_codes for code in visit]

        for code in ["E1010", "I10", "5A1955Z"]:
            self.assertIn(code, flat_codes)
        for code in ["E1011", "N170", "I509", "5A1D70Z", "02HV33Z"]:
            self.assertNotIn(
                code,
                flat_codes,
                f"terminal-admission code {code} leaked into features",
            )

    def test_los_excludes_target_admission_codes(self):
        """Patient 10001 has three admissions (19999, 20001, then 20002),
        all survived. The LOS label comes from the most recent (target)
        admission (20002), whose codes must be excluded; codes unique to
        the two earlier admissions must still be present.
        """
        patient = self.dataset.get_patient("10001")
        samples = LengthOfStayStageNetMIMIC4()(patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]

        _, icd_codes = sample["icd_codes"]
        # Only the two non-target admissions should contribute code lists.
        self.assertEqual(len(icd_codes), 2)

        flat_codes = [code for visit in icd_codes for code in visit]
        for code in ["E1010", "E1165", "I10", "5A1955Z", "3E0G76Z"]:
            self.assertIn(code, flat_codes)
        for code in ["E1011", "N179", "5A1D70Z"]:
            self.assertNotIn(
                code,
                flat_codes,
                f"target-admission code {code} leaked into LOS features",
            )

    def test_mortality_survivor_target_admission_also_excluded(self):
        """Regression test for the class-asymmetry leak: a survivor's most
        recent admission must be windowed the same way a death case's
        terminal admission is, not left fully unrestricted.

        Patient 10001 has three admissions (19999, 20001, then 20002), all
        survived -- the same fixture used by the LOS test above, which
        already (correctly) excludes 20002 as its target admission. Before
        this fix, MortalityPredictionStageNetMIMIC4 only restricted the
        death class, so this same patient's 20002 codes would leak in here
        while being correctly excluded for LOS -- an inconsistency that is
        itself evidence of the asymmetry.
        """
        patient = self.dataset.get_patient("10001")
        samples = MortalityPredictionStageNetMIMIC4()(patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["mortality"], 0)

        _, icd_codes = sample["icd_codes"]
        # Only the two non-target (19999, 20001) admissions should
        # contribute code lists; the target (20002) must not.
        self.assertEqual(len(icd_codes), 2)

        flat_codes = [code for visit in icd_codes for code in visit]
        for code in ["E1010", "E1165", "I10", "5A1955Z", "3E0G76Z"]:
            self.assertIn(code, flat_codes)
        for code in ["E1011", "N179", "5A1D70Z"]:
            self.assertNotIn(
                code,
                flat_codes,
                f"target-admission code {code} leaked into survivor features",
            )

    def test_mortality_and_los_treat_same_survivor_identically(self):
        """Direct symmetry check: for the same survivor, mortality and LOS
        must agree on which admission is the target and exclude the exact
        same code set from it -- confirming the mortality task no longer
        gives survivors a privileged, unrestricted view of their own most
        recent admission relative to what LOS already does correctly.
        """
        patient = self.dataset.get_patient("10001")
        mortality_codes = MortalityPredictionStageNetMIMIC4()(patient)[0]["icd_codes"][1]
        los_codes = LengthOfStayStageNetMIMIC4()(patient)[0]["icd_codes"][1]

        mortality_flat = sorted(code for visit in mortality_codes for code in visit)
        los_flat = sorted(code for visit in los_codes for code in visit)
        self.assertEqual(mortality_flat, los_flat)


if __name__ == "__main__":
    unittest.main()
