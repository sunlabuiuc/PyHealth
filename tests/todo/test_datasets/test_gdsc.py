import unittest

from pyhealth.datasets import GDSCDataset
import polars as pl
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
sys.path.append(repo_root)


class TestsGDSCDataset(unittest.TestCase):
    DATASET_NAME = "gdsc-demo"
    ROOT = "https://github.com/svshah4/extending-cadre/blob/main/data/input/"
    TABLES = ["drug_info"]
    REFRESH_CACHE = True

    dataset = GDSCDataset(
        dataset_name=DATASET_NAME,
        root=ROOT,
        tables=TABLES,
    )

    def setUp(self):
        pass

    def test_drug_info(self):
        """Tests that a drug entry from drug_info_gdsc.csv is parsed correctly."""

        # Pick a deterministic row that should always exist
        selected_drug_id = "1242"

        expected_name = "(5Z)-7-Oxozeaenol"
        expected_synonyms = "5Z-7-Oxozeaenol, LL-Z1640-2"
        expected_targets = "TAK1"
        expected_pathway = "Other, kinases"
        expected_pubchem = "9863776"
        expected_sample_size = "945"
        expected_count = "266"

        # dataset.tables["drug_info"] should be a Polars DataFrame
        drug_df = self.dataset.tables["drug_info"]

        # Basic checks
        self.assertTrue(len(drug_df) > 0)
        self.assertIn("drug_id", drug_df.columns)
        self.assertIn("Name", drug_df.columns)

        # Row lookup
        row = drug_df.filter(pl.col("drug_id") == selected_drug_id)

        self.assertEqual(1, len(row), "Expected exactly one matched drug entry.")

        row = row.to_dicts()[0]

        # Field-level checks
        self.assertEqual(expected_name, row["Name"])
        self.assertEqual(expected_synonyms, row["Synonyms"])
        self.assertEqual(expected_targets, row["Targets"])
        self.assertEqual(expected_pathway, row["Target pathway"])
        self.assertEqual(expected_pubchem, row["PubCHEM"])
        self.assertEqual(expected_sample_size, row["Sample Size"])
        self.assertEqual(expected_count, row["Count"])


if __name__ == "__main__":
    unittest.main(verbosity=2)