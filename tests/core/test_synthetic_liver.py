"""Tests for the synthetic liver survival cohort and its sample dataset."""

import unittest

import torch

from pyhealth.datasets import (
    MAS4_FEATURES,
    MAS6_FEATURES,
    create_synthetic_liver_dataset,
    generate_synthetic_pbcseq,
    get_dataloader,
    write_synthetic_pbcseq_csv,
)


class TestGenerateSyntheticPbcseq(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.visits = generate_synthetic_pbcseq(n_patients=20, seed=0)
        cls.patients = cls.visits.drop_duplicates("id")

    def test_layout(self):
        self.assertEqual(
            list(self.visits.columns), ["id", "day", "futime", "status", *MAS4_FEATURES]
        )
        self.assertEqual(self.patients["id"].nunique(), 20)
        self.assertTrue(set(self.patients["status"]) <= {0, 1, 2})

    def test_deterministic_for_seed(self):
        again = generate_synthetic_pbcseq(n_patients=20, seed=0)
        self.assertTrue(self.visits.equals(again))
        other = generate_synthetic_pbcseq(n_patients=20, seed=1)
        self.assertFalse(self.visits.equals(other))

    def test_tie_free(self):
        futimes = self.patients["futime"]
        self.assertTrue(futimes.is_unique)
        self.assertFalse(set(futimes) & set(self.visits["day"]))

    def test_visits_start_at_zero_and_precede_outcome(self):
        first_day = self.visits.groupby("id")["day"].min()
        self.assertTrue((first_day == 0).all())
        self.assertTrue((self.visits["day"] < self.visits["futime"]).all())

    def test_has_deaths_and_no_missing_values_by_default(self):
        self.assertGreater((self.patients["status"] == 2).sum(), 0)
        self.assertFalse(self.visits[list(MAS4_FEATURES)].isna().any().any())

    def test_missing_rate_only_affects_follow_up_visits(self):
        visits = generate_synthetic_pbcseq(n_patients=20, seed=0, missing_rate=0.5)
        labs = visits[list(MAS4_FEATURES)]
        self.assertTrue(labs.isna().any().any())
        self.assertFalse(labs[visits["day"] == 0].isna().any().any())

    def test_configurable_features(self):
        visits = generate_synthetic_pbcseq(n_patients=5, feature_names=MAS6_FEATURES)
        self.assertEqual(list(visits.columns[4:]), list(MAS6_FEATURES))

    def test_rejects_bad_arguments(self):
        for kwargs in (
            {"n_patients": 0},
            {"feature_names": ()},
            {"baseline_hazard": 0},
            {"missing_rate": 1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_synthetic_pbcseq(**kwargs)


class TestCreateSyntheticLiverDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.visits = generate_synthetic_pbcseq(n_patients=20, seed=0)
        cls.dataset = create_synthetic_liver_dataset(visits=cls.visits)
        loader = get_dataloader(cls.dataset, batch_size=len(cls.dataset), shuffle=False)
        cls.batch = next(iter(loader))

    def test_schema(self):
        self.assertEqual(
            self.dataset.input_schema,
            {
                "features": (
                    "tensor",
                    {"normalization": "zscore", "impute_strategy": "zero"},
                )
            },
        )
        self.assertEqual(
            self.dataset.output_schema, {"survival_interval": "survival_interval"}
        )
        self.assertTrue(self.dataset.input_processors["features"].learns_statistics)

    def test_full_batch_shapes(self):
        n = len(self.dataset)
        self.assertEqual(self.batch["features"].shape, (n, len(MAS4_FEATURES)))
        self.assertEqual(self.batch["survival_interval"].shape, (n, 3))
        self.assertEqual(self.batch["features"].dtype, torch.float32)
        self.assertEqual(len(self.batch["patient_id"]), n)

    def test_features_are_standardized(self):
        features = self.batch["features"]
        torch.testing.assert_close(features.mean(0), torch.zeros(4), atol=1e-5, rtol=0)
        torch.testing.assert_close(
            features.std(0, unbiased=False), torch.ones(4), atol=1e-5, rtol=0
        )

    def test_intervals_valid(self):
        start, stop, event = self.batch["survival_interval"].unbind(-1)
        self.assertTrue(bool((start < stop).all()))
        self.assertTrue(set(event.tolist()) <= {0.0, 1.0})
        n_deaths = int((self.visits.drop_duplicates("id")["status"] == 2).sum())
        self.assertEqual(int(event.sum()), n_deaths)

    def test_six_features(self):
        dataset = create_synthetic_liver_dataset(
            n_patients=5, feature_names=MAS6_FEATURES, seed=2
        )
        self.assertEqual(dataset.input_processors["features"].size(), 6)
        self.assertEqual(dataset[0]["features"].shape, (6,))

    def test_rejects_generator_arguments_with_visit_table(self):
        visits = generate_synthetic_pbcseq(n_patients=3, seed=5)
        with self.assertRaises(ValueError):
            create_synthetic_liver_dataset(visits=visits, missing_rate=0.3)

    def test_uses_given_visit_table(self):
        visits = generate_synthetic_pbcseq(n_patients=3, seed=5)
        dataset = create_synthetic_liver_dataset(visits=visits)
        self.assertEqual(len(dataset), len(visits))
        self.assertEqual(
            sorted({dataset[i]["patient_id"] for i in range(len(dataset))}),
            ["1", "2", "3"],
        )


class TestWriteSyntheticPbcseqCsv(unittest.TestCase):
    def test_writes_every_pbcseq_column(self):
        import shutil
        import tempfile

        import pandas as pd

        from pyhealth.datasets.pbcseq import REQUIRED_COLUMNS

        directory = tempfile.mkdtemp()
        try:
            path = write_synthetic_pbcseq_csv(directory, n_patients=6, seed=1)
            frame = pd.read_csv(path)
            self.assertIn("alk.phos", frame.columns)
            renamed = set(frame.columns) | {"alk_phos"}
            self.assertTrue(set(REQUIRED_COLUMNS) <= renamed)
            self.assertEqual(frame["id"].nunique(), 6)
            per_patient = frame.groupby("id")[["futime", "status", "stage", "sex"]]
            self.assertTrue((per_patient.nunique() == 1).all().all())
            again = pd.read_csv(write_synthetic_pbcseq_csv(directory, 6, seed=1))
            pd.testing.assert_frame_equal(frame, again)
        finally:
            shutil.rmtree(directory)

    def test_rejects_feature_names(self):
        with self.assertRaises(ValueError):
            write_synthetic_pbcseq_csv("unused", feature_names=["bili"])


if __name__ == "__main__":
    unittest.main()
