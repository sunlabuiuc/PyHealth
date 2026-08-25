import unittest
from pathlib import Path

from pyhealth.datasets import OMOPDataset
from pyhealth.tasks import DrugRecommendationOMOP
from pyhealth.tasks.drug_recommendation import drug_recommendation_omop_fn


class _MockVisit:
    """Minimal stand-in for the legacy pyhealth.data.Visit interface that
    drug_recommendation_omop_fn expects (visit_id, get_code_list(table)).
    """

    def __init__(self, visit_id, codes):
        self.visit_id = visit_id
        self._codes = codes

    def get_code_list(self, table):
        return self._codes[table]


class _MockPatient:
    """Minimal stand-in for the legacy indexable/len-able Patient interface
    that drug_recommendation_omop_fn expects (patient_id, len(), [i]).
    """

    def __init__(self, patient_id, visits):
        self.patient_id = patient_id
        self._visits = visits

    def __len__(self):
        return len(self._visits)

    def __getitem__(self, i):
        return self._visits[i]


class TestDrugRecommendationOMOPLeakage(unittest.TestCase):
    """Regression test for the drugs_all self-leakage bug.

    drug_recommendation_omop_fn built a "drugs_all" history feature by
    accumulating each visit's own drugs without ever excluding the current
    visit -- unlike its class-based siblings (DrugRecommendationMIMIC3/4/
    EICU) and the drug_recommendation_mimic3_fn/mimic4_fn functions, all of
    which zero out the current visit's slot in the history sequence. That
    meant the last entry of "drugs_all" was identical to the "drugs" target
    for every sample, so a model could trivially copy it instead of
    predicting from history.
    """

    def setUp(self):
        self.visits = [
            _MockVisit(
                "v1",
                {
                    "condition_occurrence": ["C1"],
                    "procedure_occurrence": ["P1"],
                    "drug_exposure": ["D1"],
                },
            ),
            _MockVisit(
                "v2",
                {
                    "condition_occurrence": ["C2"],
                    "procedure_occurrence": ["P2"],
                    "drug_exposure": ["D2"],
                },
            ),
            _MockVisit(
                "v3",
                {
                    "condition_occurrence": ["C3"],
                    "procedure_occurrence": ["P3"],
                    "drug_exposure": ["D3"],
                },
            ),
        ]
        self.patient = _MockPatient("pt1", self.visits)

    def test_drugs_all_excludes_current_visit_drugs(self):
        samples = drug_recommendation_omop_fn(self.patient)
        self.assertEqual(len(samples), 3)

        for i, sample in enumerate(samples):
            with self.subTest(visit=sample["visit_id"]):
                self.assertEqual(
                    sample["drugs_all"][i],
                    [],
                    "current visit's own drugs must not leak into its own "
                    "history slot",
                )

    def test_drugs_all_preserves_prior_visit_history(self):
        samples = drug_recommendation_omop_fn(self.patient)

        # visit 2's history should still contain visit 1's drugs
        self.assertEqual(samples[1]["drugs_all"][0], ["D1"])
        # visit 3's history should still contain visits 1 and 2's drugs
        self.assertEqual(samples[2]["drugs_all"][0], ["D1"])
        self.assertEqual(samples[2]["drugs_all"][1], ["D2"])

    def test_drugs_target_unaffected(self):
        samples = drug_recommendation_omop_fn(self.patient)
        self.assertEqual(samples[0]["drugs"], ["D1"])
        self.assertEqual(samples[1]["drugs"], ["D2"])
        self.assertEqual(samples[2]["drugs"], ["D3"])


class TestDrugRecommendationOMOP(unittest.TestCase):
    """DrugRecommendationOMOP is the current-API, leak-free replacement for
    the legacy drug_recommendation_omop_fn (which cannot even run under the
    current dataset API -- see the docs note on this task family). Verified
    against real demo OMOP data.
    """

    @classmethod
    def setUpClass(cls):
        root = str(Path(__file__).parents[2] / "test-resources" / "omop")
        tables = ["condition_occurrence", "procedure_occurrence", "drug_exposure"]
        cls.dataset = OMOPDataset(root=root, tables=tables)

    def test_drugs_hist_excludes_current_visit_and_preserves_history(self):
        # person_id "1" has 4 chronological visits (ids "1".."4"), each with
        # exactly one condition/procedure/drug code (all coded "1").
        patient = self.dataset.get_patient("1")
        samples = DrugRecommendationOMOP()(patient)
        self.assertEqual(len(samples), 4)

        for i, sample in enumerate(samples):
            with self.subTest(visit=sample["visit_id"]):
                self.assertEqual(sample["drugs"], ["1"])
                self.assertEqual(
                    sample["drugs_hist"][i],
                    [],
                    "current visit's own drugs must not leak into its own "
                    "history slot",
                )
                for j in range(i):
                    self.assertEqual(sample["drugs_hist"][j], ["1"])


if __name__ == "__main__":
    unittest.main()
