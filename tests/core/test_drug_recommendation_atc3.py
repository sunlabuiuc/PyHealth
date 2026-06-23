import csv
import gzip
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyhealth.tasks.drug_recommendation as drug_rec
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from pyhealth.tasks import DrugRecommendationMIMIC3, DrugRecommendationMIMIC4


class FakeNDCToATC3Map:
    def __init__(self):
        self.calls = []
        self.mapping = {
            "11111111111": ["A10B"],
            "22222222222": ["C03C", "C03C"],
            "33333333333": ["N02B"],
        }

    def map(self, ndc, target_kwargs=None):
        self.calls.append((ndc, target_kwargs))
        return self.mapping.get(ndc, [])


class TestDrugRecommendationATC3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resources_root = Path(__file__).parents[2] / "test-resources" / "core"

    def setUp(self):
        drug_rec._NDC_TO_ATC3_MAPPER = None
        drug_rec._NDC_TO_ATC3_CACHE.clear()
        self.mapper = FakeNDCToATC3Map()
        patcher = patch(
            "pyhealth.tasks.drug_recommendation.CrossMap.load",
            return_value=self.mapper,
        )
        self.addCleanup(patcher.stop)
        self.crossmap_load = patcher.start()
        self.temp_dirs = []

    def tearDown(self):
        drug_rec._NDC_TO_ATC3_MAPPER = None
        drug_rec._NDC_TO_ATC3_CACHE.clear()
        for temp_dir in self.temp_dirs:
            temp_dir.cleanup()

    def _copy_demo(self, demo_name):
        temp_dir = tempfile.TemporaryDirectory()
        self.temp_dirs.append(temp_dir)
        source = self.resources_root / demo_name
        target = Path(temp_dir.name) / demo_name
        shutil.copytree(source, target)
        return target, temp_dir

    def _rewrite_prescription_ndcs(self, path, replacements):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")

        counts = {hadm_id: 0 for hadm_id in replacements}
        templates = {}
        rewritten_rows = []
        for row in rows:
            hadm_id = str(row["hadm_id"])
            if hadm_id in replacements:
                templates.setdefault(hadm_id, row.copy())
                index = counts[hadm_id]
                ndcs = replacements[hadm_id]
                row["ndc"] = ndcs[index] if index < len(ndcs) else "99999999999"
                counts[hadm_id] += 1
            rewritten_rows.append(row)

        for hadm_id, ndcs in replacements.items():
            if hadm_id not in templates:
                raise ValueError(f"No prescription rows found for hadm_id={hadm_id}")
            while counts[hadm_id] < len(ndcs):
                row = templates[hadm_id].copy()
                if "row_id" in row:
                    row["row_id"] = str(10_000_000 + len(rewritten_rows))
                row["ndc"] = ndcs[counts[hadm_id]]
                rewritten_rows.append(row)
                counts[hadm_id] += 1

        with opener(path, "wt", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rewritten_rows)

    def _assert_atc3_samples(self, samples, first_hadm_id, second_hadm_id):
        by_visit = {str(sample["visit_id"]): sample for sample in samples}
        self.assertIn(first_hadm_id, by_visit)
        self.assertIn(second_hadm_id, by_visit)

        first_sample = by_visit[first_hadm_id]
        second_sample = by_visit[second_hadm_id]
        self.assertEqual(first_sample["drugs"], ["A10B", "C03C"])
        self.assertEqual(second_sample["drugs"], ["N02B"])
        self.assertNotIn("1111", first_sample["drugs"])
        self.assertNotIn("2222", first_sample["drugs"])
        self.assertNotIn("3333", second_sample["drugs"])
        self.assertNotIn("0", first_sample["drugs"])
        self.assertNotIn("9999", first_sample["drugs"])

    def test_mimic3_demo_drug_recommendation_maps_ndc_to_atc3(self):
        demo_path, cache_dir = self._copy_demo("mimic3demo")
        self._rewrite_prescription_ndcs(
            demo_path / "PRESCRIPTIONS.csv.gz",
            {
                "142582": [
                    "11111111111",
                    "22222222222",
                    "11111111111",
                    "0",
                    "99999999999",
                ],
                "122098": ["33333333333", "", "<NA>"],
            },
        )
        dataset = MIMIC3Dataset(
            root=str(demo_path),
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=cache_dir.name,
        )

        samples = DrugRecommendationMIMIC3()(dataset.get_patient("10059"))

        self.crossmap_load.assert_called_once_with("NDC", "ATC")
        self._assert_atc3_samples(samples, "142582", "122098")
        self.assertTrue(
            all(kwargs == {"level": 3} for _, kwargs in self.mapper.calls)
        )

    def test_mimic4_demo_drug_recommendation_maps_ndc_to_atc3(self):
        demo_path, cache_dir = self._copy_demo("mimic4demo")
        self._rewrite_prescription_ndcs(
            demo_path / "hosp" / "prescriptions.csv",
            {
                "20001": [
                    "11111111111",
                    "22222222222",
                    "11111111111",
                    "0",
                    "99999999999",
                ],
                "20002": ["33333333333", "", "<NA>"],
            },
        )
        dataset = MIMIC4Dataset(
            ehr_root=str(demo_path),
            ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=cache_dir.name,
            num_workers=1,
        )

        samples = DrugRecommendationMIMIC4()(dataset.get_patient("10001"))

        self.crossmap_load.assert_called_once_with("NDC", "ATC")
        self._assert_atc3_samples(samples, "20001", "20002")
        self.assertTrue(
            all(kwargs == {"level": 3} for _, kwargs in self.mapper.calls)
        )


if __name__ == "__main__":
    unittest.main()
