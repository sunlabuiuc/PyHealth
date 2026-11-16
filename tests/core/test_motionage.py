"""Unit tests for the MotionAge model and the MotionAgeMapping risk-to-age map.

Models are built on a handful of tiny synthetic samples (8 epochs each) in the
NHANESMortalityPrediction schema. No real NHANES data is used.
"""

import math
import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.motionage import MotionAge, MotionAgeMapping
from pyhealth.tasks.nhanes_mortality_prediction import NHANESMortalityPrediction

SEQ_LEN = 8
BATCH = 4


def _samples(covariate_set: str | None) -> list[dict]:
    """Four windows with both labels, some non-wear and a missing covariate."""
    rng = np.random.default_rng(0)
    samples = []
    for i in range(BATCH):
        wear = np.ones(SEQ_LEN, dtype=np.float32)
        wear[SEQ_LEN - i % 3 :] = 0  # last 0, 1, 2, 0 epochs not worn
        intensity = rng.normal(size=(SEQ_LEN, 1)).astype(np.float32)
        intensity[wear == 0] = np.nan
        sample = {
            "patient_id": f"p{i}",
            "intensity": intensity,
            "hour": rng.integers(0, 24, SEQ_LEN),
            "day": rng.integers(0, 7, SEQ_LEN),
            "wear_mask": wear,
            "mortality": i % 2,
        }
        if covariate_set == "clinical":
            waist = math.nan if i == 0 else 90.0 + i
            sample["covariates"] = np.array(
                [40.0 + i, 25.0 + i, waist, 70.0, 120.0 + i], dtype=np.float32
            )
            sample["covariate_codes"] = np.array([1 + i % 2, 1 + i], dtype=np.int64)
        samples.append(sample)
    return samples


def _dataset(covariate_set: str | None = "clinical"):
    task = NHANESMortalityPrediction(seq_len=SEQ_LEN, covariate_set=covariate_set)
    dataset = create_sample_dataset(
        _samples(covariate_set),
        task.input_schema,
        task.output_schema,
        dataset_name="nhanes",
    )
    return dataset, task


class TestMotionAge(unittest.TestCase):
    """Architecture, forward and backward passes."""

    @classmethod
    def setUpClass(cls):
        cls.dataset, cls.task = _dataset("clinical")
        cls.wearable, _ = _dataset(None)
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=BATCH)))
        cls.wearable_batch = next(iter(get_dataloader(cls.wearable, batch_size=BATCH)))

    def model(self, **kwargs) -> MotionAge:
        torch.manual_seed(0)
        return MotionAge(
            self.dataset, category_sizes=self.task.category_sizes, **kwargs
        )

    def test_variants_predict(self):
        """Late fusion, residual fusion and wearable-only return valid output."""
        cases = [
            (self.model(), self.batch),
            (self.model(fusion="residual"), self.batch),
            (MotionAge(self.wearable), self.wearable_batch),
        ]
        for model, batch in cases:
            out = model(**batch)
            self.assertEqual(set(out), {"loss", "y_prob", "y_true", "logit"})
            self.assertEqual(tuple(out["y_prob"].shape), (BATCH, 1))
            self.assertEqual(tuple(out["y_true"].shape), (BATCH, 1))
            self.assertTrue(torch.isfinite(out["loss"]))
            self.assertTrue(((out["y_prob"] > 0) & (out["y_prob"] < 1)).all())

    def test_default_layer_sizes(self):
        """Paper defaults: GRU 11->64 x2, covariates ->64->32, fusion 96->64->1."""
        model = self.model()
        gru = model.gru
        self.assertEqual((gru.input_size, gru.hidden_size, gru.num_layers), (11, 64, 2))
        self.assertAlmostEqual(gru.dropout, 0.117)
        # 5 covariates + 5 missing flags + 2 codes x 4-dim embeddings = 18.
        self.assertEqual(model.covariate_encoder[0].in_features, 18)
        self.assertEqual(model.covariate_encoder[-1].out_features, 32)
        self.assertEqual(model.fusion_mlp[0].in_features, 64 + 32)
        self.assertEqual(model.fusion_mlp[-1].out_features, 1)
        self.assertIsNone(model.classifier)
        residual = self.model(fusion="residual")
        self.assertEqual(residual.classifier.in_features, 64)
        wearable = MotionAge(self.wearable)
        self.assertEqual(wearable.classifier.in_features, 64)
        self.assertFalse(hasattr(wearable, "fusion_mlp"))

    def test_gradients_reach_every_layer(self):
        """Backpropagation updates the GRU, embeddings and both MLPs."""
        model = self.model()
        model(**self.batch)["loss"].backward()
        for name, param in model.named_parameters():
            if name == "_dummy_param":
                continue
            with self.subTest(name=name):
                self.assertIsNotNone(param.grad)
                self.assertGreater(param.grad.abs().sum().item(), 0)

    def test_all_nonwear_window_is_finite(self):
        """A window with no worn epochs pools to zeros, not NaN."""
        model = self.model().eval()
        batch = {**self.batch, "wear_mask": torch.zeros_like(self.batch["wear_mask"])}
        pooled = model.encode_activity(
            batch["intensity"], batch["hour"], batch["day"], batch["wear_mask"]
        )
        self.assertTrue((pooled == 0).all())
        self.assertTrue(torch.isfinite(model(**batch)["logit"]).all())

    def test_pooling_ignores_trailing_nonwear(self):
        """Appending non-worn epochs does not change the activity summary."""
        model = self.model().eval()
        b = self.batch
        pad = 5
        with torch.no_grad():
            base = model.encode_activity(
                b["intensity"], b["hour"], b["day"], b["wear_mask"]
            )
            longer = model.encode_activity(
                torch.cat([b["intensity"], torch.randn(BATCH, pad, 1)], dim=1),
                torch.cat(
                    [b["hour"], torch.zeros(BATCH, pad, dtype=torch.long)], dim=1
                ),
                torch.cat([b["day"], torch.zeros(BATCH, pad, dtype=torch.long)], dim=1),
                torch.cat([b["wear_mask"], torch.zeros(BATCH, pad)], dim=1),
            )
        torch.testing.assert_close(base, longer)

    def test_residual_starts_at_activity_only_logit(self):
        """The residual correction is zero-initialized."""
        model = self.model(fusion="residual").eval()
        b = self.batch
        with torch.no_grad():
            pooled = model.encode_activity(
                b["intensity"], b["hour"], b["day"], b["wear_mask"]
            )
            torch.testing.assert_close(model(**b)["logit"], model.classifier(pooled))

    def test_embed_and_two_dimensional_intensity(self):
        """embed=True returns the pooled summary; [B, T] intensity works."""
        model = self.model().eval()
        out = model(**self.batch, embed=True)
        self.assertEqual(tuple(out["embed"].shape), (BATCH, 64))
        flat = {**self.batch, "intensity": self.batch["intensity"][..., 0]}
        torch.testing.assert_close(model(**flat)["logit"], out["logit"])

    def test_training_lowers_loss(self):
        """A few optimizer steps on one batch reduce the loss."""
        model = self.model(dropout=0.0, fusion_dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        first = model(**self.batch)["loss"].item()
        for _ in range(15):
            optimizer.zero_grad()
            model(**self.batch)["loss"].backward()
            optimizer.step()
        self.assertLess(model(**self.batch)["loss"].item(), first)

    def test_invalid_arguments(self):
        """Missing or inconsistent settings raise ValueError."""
        bad = (
            {},  # covariate_codes present but no category_sizes
            {"category_sizes": [3]},
            {"category_sizes": [3, 14], "fusion": "concat"},
            {"category_sizes": [3, 14], "hidden_dim": 0},
        )
        for kwargs in bad:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                MotionAge(self.dataset, **kwargs)
        task = NHANESMortalityPrediction(seq_len=SEQ_LEN, covariate_set=None)
        schema = {k: v for k, v in task.input_schema.items() if k != "wear_mask"}
        samples = [
            {k: v for k, v in s.items() if k != "wear_mask"} for s in _samples(None)
        ]
        no_mask = create_sample_dataset(samples, schema, task.output_schema)
        with self.assertRaises(ValueError):
            MotionAge(no_mask)


class TestMotionAgeMapping(unittest.TestCase):
    """Fitting and inverting the sex-specific risk-by-age line."""

    @staticmethod
    def population(lines: dict[int, tuple[float, float]], per_age: int = 3):
        """Participants aged 20-85 whose risk lies exactly on each sex's line."""
        ages, sexes, risks = [], [], []
        for sex, (alpha, beta) in lines.items():
            for age in range(20, 86):
                for _ in range(per_age):
                    ages.append(float(age))
                    sexes.append(sex)
                    risks.append(1 / (1 + math.exp(-(alpha + beta * age))))
        return np.array(risks), np.array(ages), np.array(sexes)

    def test_recovers_known_lines(self):
        """Exact per-sex alpha/beta; mapping a risk back gives the age."""
        lines = {1: (-9.0, 0.1), 2: (-10.0, 0.11)}
        risk, age, sex = self.population(lines)
        for weighted in (True, False):
            mapping = MotionAgeMapping(weighted=weighted).fit(risk, age, sex)
            for code, (alpha, beta) in lines.items():
                self.assertAlmostEqual(mapping.params[code][0], alpha, places=6)
                self.assertAlmostEqual(mapping.params[code][1], beta, places=8)
            np.testing.assert_allclose(mapping.transform(risk, sex), age, atol=1e-6)

    def test_uses_median_risk_per_age(self):
        """One outlier per age does not move the fitted line."""
        risk, age, sex = self.population({1: (-9.0, 0.1)}, per_age=1)
        risk = np.r_[risk, risk * 0.5, np.full(risk.size, 0.99)]
        age, sex = np.tile(age, 3), np.tile(sex, 3)
        mapping = MotionAgeMapping().fit(risk, age, sex)
        self.assertAlmostEqual(mapping.params[1][1], 0.1, places=8)

    def test_unseen_sex_and_extreme_risks(self):
        """Unknown sex gives NaN; risks of 0 or 1 are clipped, not infinite."""
        risk, age, sex = self.population({1: (-9.0, 0.1)})
        mapping = MotionAgeMapping().fit(risk, age, sex)
        out = mapping.transform(np.array([0.0, 1.0, 0.5]), np.array([1, 1, 2]))
        self.assertTrue(np.isfinite(out[:2]).all())
        self.assertTrue(np.isnan(out[2]))

    def test_errors(self):
        """Unfitted use, too few people, and risk falling with age raise."""
        risk, age, sex = self.population({1: (-9.0, 0.1)})
        with self.assertRaises(RuntimeError):
            MotionAgeMapping().transform(risk, sex)
        with self.assertRaises(ValueError):
            MotionAgeMapping().fit(risk[:50], age[:50], sex[:50])  # < 100 people
        with self.assertRaises(ValueError):
            MotionAgeMapping().fit(1 - risk, age, sex)  # negative slope
        with self.assertRaises(ValueError):
            MotionAgeMapping(min_participants=1).fit(risk[:3], age[:3], sex[:3])
        with self.assertRaises(ValueError):
            MotionAgeMapping(clip_eps=0.0)


if __name__ == "__main__":
    unittest.main()
