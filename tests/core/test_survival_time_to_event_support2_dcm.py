"""Fast synthetic tests for SurvivalTimeToEventSupport2DCM; no real data.

Contributor: <AUTHOR NAME> (NetID: liu323)
Paper: Deep Cox Mixtures for Survival Regression (Nagpal et al., MLHC 2021),
https://proceedings.mlr.press/v149/nagpal21a.html
"""

import math
import unittest

import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DeepCoxMixtures
from pyhealth.tasks import SurvivalTimeToEventSupport2DCM
from pyhealth.tasks.survival_time_to_event_support2_dcm import (
    SUPPORT2_DCM_CATEGORIES,
    SUPPORT2_DCM_FEATURES,
    SUPPORT2_DCM_NUMERIC,
)


def row(sno, **overrides):
    """One synthetic SUPPORT2 row with every column the task reads."""
    values = {"sno": str(sno)}
    values.update({name: str(10 + j + sno) for j, name in
                   enumerate(SUPPORT2_DCM_NUMERIC)})
    values.update(sex="female", dzgroup="CHF", dzclass="COPD/CHF/Cirrhosis",
                  income="$25-$50k", race="black", ca="no",
                  **{"d.time": str(5 * sno), "death": str(sno % 2)})
    values.update(overrides)
    return values


def patient(rows):
    events = [
        {"patient_id": r["sno"], "event_type": "support2", "timestamp": None,
         **{f"support2/{k}": (v if v != "" else None)
            for k, v in r.items() if k != "sno"}}
        for r in rows
    ]
    return Patient(rows[0]["sno"] if rows else "0", pl.DataFrame(events))


class TestSupport2DCMTask(unittest.TestCase):
    def test_schema_and_feature_names(self):
        task = SurvivalTimeToEventSupport2DCM()
        self.assertEqual(task.input_schema, {"features": "tensor"})
        self.assertEqual(task.output_schema,
                         {"duration": "tensor", "event": "tensor"})
        self.assertEqual(len(SUPPORT2_DCM_NUMERIC), 18)
        self.assertEqual(sum(len(v) for _, v in SUPPORT2_DCM_CATEGORIES), 26)
        self.assertEqual(len(SUPPORT2_DCM_FEATURES), 44)
        self.assertEqual(SUPPORT2_DCM_FEATURES[18:20],
                         ("sex=female", "sex=male"))
        self.assertEqual(len(set(SUPPORT2_DCM_FEATURES)), 44)

    def test_encoding_of_a_patient(self):
        r = row(3)
        (sample,) = SurvivalTimeToEventSupport2DCM()(patient([r]))
        self.assertEqual(sample["patient_id"], "3")
        features = sample["features"]
        self.assertEqual(features[:18],
                         [float(r[name]) for name in SUPPORT2_DCM_NUMERIC])
        hot = {SUPPORT2_DCM_FEATURES[j] for j in range(18, 44)
               if features[j] == 1.0}
        self.assertEqual(hot, {"sex=female", "dzgroup=CHF",
                               "dzclass=COPD/CHF/Cirrhosis", "income=$25-$50k",
                               "race=black", "ca=no"})
        self.assertEqual(sum(features[18:]), 6.0)
        self.assertEqual(sample["duration"], [15.0])
        self.assertEqual(sample["event"], [1.0])
        self.assertEqual(sample["race"], "black")

    def test_missing_values(self):
        r = row(2, alb="", income="", race="")
        (sample,) = SurvivalTimeToEventSupport2DCM()(patient([r]))
        features = sample["features"]
        self.assertTrue(math.isnan(features[SUPPORT2_DCM_FEATURES.index("alb")]))
        income = [j for j, n in enumerate(SUPPORT2_DCM_FEATURES)
                  if n.startswith(("income=", "race="))]
        self.assertEqual([features[j] for j in income], [0.0] * 9)
        self.assertEqual(sample["race"], "missing")
        self.assertEqual(sample["event"], [0.0])
        # encode() also accepts NaN floats for missing values
        direct = SurvivalTimeToEventSupport2DCM.encode(
            {**row(2), "alb": float("nan")}, "2")
        self.assertTrue(math.isnan(direct["features"][8]))

    def test_invalid_records(self):
        task = SurvivalTimeToEventSupport2DCM()
        bad = [dict(age="old"), dict(dzgroup="Flu"), dict(**{"d.time": "0"}),
               dict(**{"d.time": ""}), dict(death="2"), dict(age="inf")]
        for overrides in bad:
            with self.assertRaises(ValueError, msg=str(overrides)):
                task(patient([row(1, **overrides)]))
        with self.assertRaises(ValueError):
            task(patient([row(1), row(2)]))

    def test_samples_train_deep_cox_mixtures(self):
        task = SurvivalTimeToEventSupport2DCM()
        samples = []
        for i in range(1, 7):
            (s,) = task(patient([row(i, bun="" if i % 2 else "9")]))
            samples.append(s)
        # Mean-impute and standardize with "training" statistics, as the
        # task's documentation asks, before handing samples to the model.
        x = torch.tensor([s["features"] for s in samples])
        mean = torch.nanmean(x, dim=0)
        x = torch.where(torch.isnan(x), mean, x)
        x = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
        for s, features in zip(samples, x.tolist()):
            s["features"] = features
        dataset = create_sample_dataset(
            samples=samples, input_schema=task.input_schema,
            output_schema=task.output_schema, dataset_name="support2_dcm",
        )
        self.assertEqual(dataset[0]["race"], "black")
        torch.manual_seed(0)
        model = DeepCoxMixtures(dataset, num_components=2, hidden_dims=(4,),
                                time_grid=[10.0, 20.0])
        self.assertEqual(model.input_dim, 44)
        model.fit(dataset, epochs=1, batch_size=3, seed=0)
        batch = next(iter(get_dataloader(dataset, batch_size=6, shuffle=False)))
        out = model(**batch)
        self.assertEqual(out["y_prob"].shape, (6, 2))
        self.assertTrue(torch.isfinite(out["loss"]))


if __name__ == "__main__":
    unittest.main()
