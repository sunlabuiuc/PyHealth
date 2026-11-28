"""Tests for the vasopressor concept bottleneck port (Wu et al. 2022).

Covers the ``VasopressorOnsetMIMIC3`` task (on four hand-built synthetic
patients) and the models in ``pyhealth.models.concept_bottleneck`` (on
tiny synthetic tensors). No real or demo data is read.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import (
    ConceptBottleneckModel,
    LogisticRegressionWithSummaries,
    LSTMNextStateBaseline,
)
from pyhealth.models.concept_bottleneck import (
    SummaryEncoder,
    apply_topk_mask,
    greedy_select_features,
    top_features_by_weight_mass,
)
from pyhealth.tasks import VasopressorOnsetMIMIC3
from pyhealth.tasks.vasopressor_onset_mimic3 import (
    CHANGING_VARS,
    STATIC_VARS,
    _forward_fill,
    _hourly_series,
    apply_standardizer,
    fit_standardizer,
)

T0 = datetime(2150, 1, 1, 8, 0, 0)
HR, LACTATE, URINE, GCS, TEMP, FIO2 = (
    CHANGING_VARS.index(name)
    for name in ("hr", "lactate", "urine", "GCS", "temp", "fio2")
)


def _event(table, hours=None, **attrs):
    """One synthetic event row in PyHealth's ``table/attribute`` layout."""
    row = {
        "event_type": table,
        "timestamp": None if hours is None else T0 + timedelta(hours=hours),
    }
    row.update({f"{table}/{key}": str(value) for key, value in attrs.items()})
    return row


def _patient(
    patient_id="p1",
    los_hours=48,
    dbsource="metavision",
    vasopressor_hour=None,
    weight=70.0,
    extra=(),
):
    """Builds a synthetic patient with one ICU stay starting at ``T0``."""
    outtime = (T0 + timedelta(hours=los_hours)).strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        _event("patients", gender="F", dob="2050-01-01 00:00:00"),
        _event("admissions", 0, hadm_id=10, admission_type="EMERGENCY"),
        _event(
            "icustays",
            0,
            hadm_id=10,
            icustay_id=100,
            first_careunit="MICU",
            dbsource=dbsource,
            outtime=outtime,
        ),
        _event("chartevents", 0.5, itemid=220045, valuenum=80),
        _event("chartevents", 0.75, itemid=220045, valuenum=100),
        _event("labevents", 2.0, itemid=50813, valuenum=2.5),
        _event("outputevents", 1.2, itemid=226559, value=100),
        _event("outputevents", 1.7, itemid=226559, value=50),
    ]
    if weight is not None:
        rows.append(_event("chartevents", 0.6, itemid=226512, valuenum=weight))
    if vasopressor_hour is not None:
        rows.append(
            _event("inputevents_mv", vasopressor_hour, itemid=221906, icustay_id=100)
        )
    rows.extend(extra)
    frame = pl.from_dicts(
        [{"patient_id": patient_id, **row} for row in rows],
        infer_schema_length=None,
    )
    return Patient(patient_id=patient_id, data_source=frame)


def _tiny_dataset(num_patients=4, changing_dim=2, static_dim=1):
    """Builds a synthetic dataset with (6, C) series per patient."""
    rng = np.random.default_rng(0)
    samples = []
    for i in range(num_patients):
        samples.append(
            {
                "patient_id": f"patient-{i}",
                "values": rng.normal(size=(6, changing_dim)).tolist(),
                "indicators": rng.integers(0, 2, (6, changing_dim)).tolist(),
                "statics": rng.normal(size=(static_dim,)).tolist(),
                "vasopressor_onset": i % 2,
            }
        )
    return create_sample_dataset(
        samples=samples,
        input_schema={
            "values": "tensor",
            "indicators": "tensor",
            "statics": "tensor",
        },
        output_schema={"vasopressor_onset": "binary"},
        dataset_name="test",
    )


class TestVasopressorOnsetTask(unittest.TestCase):
    """Sample processing, labels, features and edge cases of the task."""

    def setUp(self):
        self.task = VasopressorOnsetMIMIC3(normalization=None)

    def test_schema(self):
        """Default schema z-scores values/statics; None leaves them raw."""
        default = VasopressorOnsetMIMIC3()
        self.assertEqual(default.task_name, "VasopressorOnsetMIMIC3")
        self.assertEqual(default.input_schema["values"][1]["normalization"], "zscore")
        self.assertEqual(default.output_schema, {"vasopressor_onset": "binary"})
        self.assertEqual(self.task.input_schema["values"], "tensor")
        with self.assertRaises(ValueError):
            VasopressorOnsetMIMIC3(normalization="minmax")

    def test_sample_shapes(self):
        """One sample per patient with (6, 28), (6, 28) and (10,) inputs."""
        samples = self.task(_patient())
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["patient_id"], "p1")
        self.assertEqual(np.array(sample["values"]).shape, (6, 28))
        self.assertEqual(np.array(sample["indicators"]).shape, (6, 28))
        self.assertEqual(len(sample["statics"]), len(STATIC_VARS))

    def test_label_generation(self):
        """Onset after hour 6 is positive; none is negative."""
        self.assertEqual(self.task(_patient())[0]["vasopressor_onset"], 0)
        positive = self.task(_patient(vasopressor_hour=10))
        self.assertEqual(positive[0]["vasopressor_onset"], 1)
        other_drug = _patient(
            extra=[_event("inputevents_mv", 10, itemid=225158, icustay_id=100)]
        )
        self.assertEqual(self.task(other_drug)[0]["vasopressor_onset"], 0)

    def test_feature_extraction(self):
        """Hourly means, urine sums, carry-forward and indicators."""
        sample = self.task(_patient())[0]
        values = np.array(sample["values"])
        indicators = np.array(sample["indicators"])
        self.assertAlmostEqual(values[0, HR], 90.0)
        self.assertAlmostEqual(values[5, HR], 90.0)  # carried forward
        self.assertEqual(indicators[:, HR].tolist(), [1, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(values[1, URINE], 150.0)
        self.assertTrue(np.isnan(values[1, LACTATE]))
        self.assertAlmostEqual(values[2, LACTATE], 2.5)
        raw = VasopressorOnsetMIMIC3(normalization=None, forward_fill=False)
        self.assertTrue(np.isnan(np.array(raw(_patient())[0]["values"])[1, HR]))

    def test_static_features(self):
        """Statics follow STATIC_VARS; age is capped at 90."""
        statics = dict(zip(STATIC_VARS, self.task(_patient())[0]["statics"]))
        self.assertEqual(statics["Age"], 90.0)
        self.assertEqual(statics["is_F"], 1.0)
        self.assertEqual(statics["is_emergency"], 1.0)
        self.assertEqual(statics["is_urgent"], 0.0)
        self.assertEqual(statics["is_micu"], 1.0)
        self.assertEqual(statics["is_sicu"], 0.0)
        self.assertEqual(statics["weight"], 70.0)
        self.assertEqual(statics["Weight"], 70.0)

    def test_gcs_and_unit_conversions(self):
        """GCS sums its sub-scores; Fahrenheit and FiO2 fractions convert."""
        extra = [
            _event("chartevents", 3.1, itemid=220739, valuenum=4),
            _event("chartevents", 3.2, itemid=223901, valuenum=6),
            _event("chartevents", 3.3, itemid=223900, valuenum=5),
            _event("chartevents", 3.4, itemid=223761, valuenum=98.6),
            _event("chartevents", 3.5, itemid=223835, valuenum=0.5),
        ]
        values = np.array(self.task(_patient(extra=extra))[0]["values"])
        self.assertAlmostEqual(values[3, GCS], 15.0)
        self.assertAlmostEqual(values[3, TEMP], 37.0)
        self.assertAlmostEqual(values[3, FIO2], 50.0)

    def test_cohort_exclusions(self):
        """Edge cases that leave the patient out of the cohort."""
        excluded = {
            "early vasopressor": _patient(vasopressor_hour=2),
            "short stay": _patient(los_hours=5),
            "long stay": _patient(los_hours=700),
            "carevue": _patient(dbsource="carevue"),
            "no weight": _patient(weight=None),
        }
        for name, patient in excluded.items():
            with self.subTest(name):
                self.assertEqual(self.task(patient), [])
        no_stay = Patient(
            patient_id="p0",
            data_source=pl.from_dicts(
                [{"patient_id": "p0", **_event("patients", gender="M")}],
                infer_schema_length=None,
            ),
        )
        self.assertEqual(self.task(no_stay), [])

    def test_cohort_options(self):
        """The cohort rules are arguments and can be relaxed."""
        task = VasopressorOnsetMIMIC3(
            normalization=None, metavision_only=False, require_weight=False
        )
        sample = task(_patient(dbsource="carevue", weight=None))[0]
        self.assertTrue(np.isnan(sample["statics"][-1]))
        narrow = VasopressorOnsetMIMIC3(
            normalization=None, vasopressor_itemids=[222315]
        )
        self.assertEqual(
            narrow(_patient(vasopressor_hour=10))[0]["vasopressor_onset"], 0
        )

    def test_samples_load_with_default_processors(self):
        """Four synthetic patients become a z-scored, NaN-free dataset."""
        task = VasopressorOnsetMIMIC3()
        samples = []
        for i, hour in enumerate([None, 10, None, 20]):
            samples.extend(
                task(_patient(f"p{i}", vasopressor_hour=hour, weight=60.0 + i))
            )
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name="test",
            task_name=task.task_name,
        )
        self.assertEqual(len(dataset), 4)
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        self.assertEqual(batch["values"].shape, (4, 6, 28))
        self.assertEqual(batch["statics"].shape, (4, 10))
        self.assertFalse(torch.isnan(batch["values"]).any())
        self.assertEqual(sorted(batch["vasopressor_onset"].flatten().tolist()),
                         [0.0, 0.0, 1.0, 1.0])


class TestTaskHelpers(unittest.TestCase):
    """Binning, carry-forward and standardization helpers."""

    def test_hourly_series_binning(self):
        """Events land in the right hourly bin with the right mean."""
        events = [
            SimpleNamespace(
                timestamp=T0 + timedelta(hours=1, minutes=m), itemid=211, valuenum=v
            )
            for m, v in ((30, 80.0), (45, 100.0))
        ]
        events.append(
            SimpleNamespace(timestamp=T0 + timedelta(hours=7), itemid=211, valuenum=1)
        )
        values, indicators = _hourly_series(events, {211: 0}, T0, num_vars=1)
        self.assertEqual(values.shape, (6, 1))
        self.assertAlmostEqual(float(values[1, 0]), 90.0)
        self.assertEqual(indicators[:, 0].tolist(), [0, 1, 0, 0, 0, 0])
        self.assertTrue(np.isnan(values[0, 0]))

    def test_forward_fill(self):
        """Values carry forward but never backward."""
        filled = _forward_fill(np.array([[np.nan], [2.0], [np.nan]]))
        self.assertTrue(np.isnan(filled[0, 0]))
        self.assertEqual(filled[2, 0], 2.0)

    def test_standardizers(self):
        """zscore divides by the std, variance by the variance."""
        train = [{"values": [[0.0], [4.0]], "statics": [1.0], "y": 1}]
        zscore = fit_standardizer(train, method="zscore")
        variance = fit_standardizer(train, method="variance")
        self.assertEqual(float(zscore["values"]["scale"][0]), 2.0)
        self.assertEqual(float(variance["values"]["scale"][0]), 4.0)
        self.assertEqual(float(zscore["statics"]["scale"][0]), 1.0)  # no spread
        test = [{"values": [[4.0], [float("nan")]], "statics": [1.0], "y": 0}]
        self.assertEqual(apply_standardizer(test, zscore)[0]["values"], [[1.0], [0.0]])
        self.assertEqual(
            apply_standardizer(test, variance)[0]["values"], [[0.5], [0.0]]
        )
        self.assertEqual(apply_standardizer(test, zscore)[0]["y"], 0)
        with self.assertRaises(ValueError):
            fit_standardizer(train, method="minmax")


class TestSummaryEncoder(unittest.TestCase):
    """Test cases for the SummaryEncoder."""

    def test_output_shape_full_dims(self):
        """Encoder output is (batch, 402) with the original dimensions."""
        encoder = SummaryEncoder()
        values = torch.randn(3, 6, 28)
        indicators = torch.randint(0, 2, (3, 6, 28)).float()
        out = encoder(values, indicators, torch.randn(3, 10))
        self.assertEqual(encoder.encoding_dim, 402)
        self.assertEqual(out.shape, (3, 402))
        self.assertFalse(torch.isnan(out).any())

    def test_summary_values(self):
        """Known inputs give the expected raw row and measurement times."""
        encoder = SummaryEncoder(changing_dim=1, static_dim=1)
        values = torch.arange(6.0).reshape(1, 6, 1)
        indicators = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0]).reshape(1, 6, 1)
        out = encoder(values, indicators, torch.tensor([[7.0]]))[0]
        self.assertEqual(out.shape, (15,))
        self.assertEqual(out[:3].tolist(), [5.0, 0.0, 7.0])  # hour-5 row
        self.assertAlmostEqual(float(out[11]), 1 / 6)  # first measured
        self.assertAlmostEqual(float(out[12]), 5 / 6)  # last measured

    def test_never_measured_variable(self):
        """A never-measured variable gives finite summaries."""
        encoder = SummaryEncoder(changing_dim=2, static_dim=1)
        out = encoder(torch.zeros(2, 6, 2), torch.zeros(2, 6, 2), torch.zeros(2, 1))
        self.assertEqual(out.shape, (2, 29))
        self.assertTrue(torch.isfinite(out).all())

    def test_gradients_flow(self):
        """Loss on the encoding backpropagates to cutoffs and thresholds."""
        encoder = SummaryEncoder(changing_dim=2, static_dim=1)
        out = encoder(torch.randn(2, 6, 2), torch.ones(2, 6, 2), torch.zeros(2, 1))
        out.sum().backward()
        self.assertIsNotNone(encoder.cutoff_times.grad)
        self.assertIsNotNone(encoder.upper_thresholds.grad)

    def test_invalid_arguments(self):
        """Bad cutoff counts and init names are rejected."""
        with self.assertRaises(ValueError):
            SummaryEncoder(changing_dim=2, static_dim=1, num_cutoff_times=3)
        with self.assertRaises(ValueError):
            SummaryEncoder(changing_dim=2, static_dim=1, init_thresholds="x")
        with self.assertRaises(ValueError):
            SummaryEncoder(changing_dim=2, static_dim=1, init_cutoffs="x")


class TestConceptBottleneckModel(unittest.TestCase):
    """Test cases for the ConceptBottleneckModel."""

    def setUp(self):
        torch.manual_seed(0)
        self.dataset = _tiny_dataset()
        self.batch = next(iter(get_dataloader(self.dataset, batch_size=4)))
        self.model = ConceptBottleneckModel(
            dataset=self.dataset,
            changing_dim=2,
            static_dim=1,
            l1_lambda=0.001,
            cos_sim_lambda=0.01,
            pos_weight=2.0,
        )

    def test_forward_keys_and_shapes(self):
        """Forward returns logit/y_prob/concepts/summaries/loss."""
        ret = self.model(**self.batch)
        self.assertEqual(ret["logit"].shape, (4, 1))
        self.assertEqual(ret["y_prob"].shape, (4, 1))
        self.assertEqual(ret["concepts"].shape, (4, 4))
        self.assertEqual(ret["summaries"].shape, (4, 29))
        self.assertEqual(ret["y_true"].shape, (4, 1))
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertTrue(((ret["y_prob"] >= 0) & (ret["y_prob"] <= 1)).all())

    def test_forward_without_label(self):
        """Without the label there is no loss."""
        batch = {k: v for k, v in self.batch.items() if k != "vasopressor_onset"}
        ret = self.model(**batch)
        self.assertNotIn("loss", ret)
        self.assertEqual(ret["logit"].shape, (4, 1))

    def test_loss_backward_and_step(self):
        """Gradients reach every layer and a few steps lower the loss."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        first = self.model(**self.batch)["loss"]
        first.backward()
        self.assertIsNotNone(self.model.bottleneck.weight.grad)
        self.assertIsNotNone(self.model.head.weight.grad)
        self.assertIsNotNone(self.model.encoder.cutoff_times.grad)
        for _ in range(20):
            optimizer.zero_grad()
            loss = self.model(**self.batch)["loss"]
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss), float(first))

    def test_penalties_increase_loss(self):
        """The L1 and cosine penalties add to the plain BCE loss."""
        plain = ConceptBottleneckModel(
            dataset=self.dataset, changing_dim=2, static_dim=1, pos_weight=2.0
        )
        plain.load_state_dict(self.model.state_dict())
        self.assertGreater(
            float(self.model(**self.batch)["loss"]),
            float(plain(**self.batch)["loss"]),
        )

    def test_num_concepts_and_zero_weight(self):
        """k is configurable; zero_weight adds a frozen-at-zero concept."""
        single = ConceptBottleneckModel(
            dataset=self.dataset,
            num_concepts=1,
            cos_sim_lambda=0.01,
            changing_dim=2,
            static_dim=1,
        )
        self.assertEqual(single(**self.batch)["concepts"].shape, (4, 1))
        extra = ConceptBottleneckModel(
            dataset=self.dataset, zero_weight=True, changing_dim=2, static_dim=1
        )
        self.assertEqual(extra(**self.batch)["concepts"].shape, (4, 5))
        self.assertEqual(float(extra.bottleneck.weight[4].abs().sum()), 0.0)

    def test_save_and_load(self):
        """A saved state dict restores identical predictions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "cbm.pt")
            torch.save(self.model.state_dict(), path)
            restored = ConceptBottleneckModel(
                dataset=self.dataset, changing_dim=2, static_dim=1
            )
            restored.load_state_dict(torch.load(path))
        self.assertFalse(os.path.exists(path))
        self.assertTrue(
            torch.equal(
                self.model(**self.batch)["logit"], restored(**self.batch)["logit"]
            )
        )


class TestBaselines(unittest.TestCase):
    """Test cases for the Johnson and LSTM baselines."""

    def setUp(self):
        torch.manual_seed(0)
        self.dataset = _tiny_dataset()
        self.batch = next(iter(get_dataloader(self.dataset, batch_size=4)))

    def test_logistic_regression_with_summaries(self):
        """Johnson baseline returns a (4, 1) logit and a trainable loss."""
        model = LogisticRegressionWithSummaries(
            dataset=self.dataset, changing_dim=2, static_dim=1
        )
        ret = model(**self.batch)
        self.assertEqual(ret["logit"].shape, (4, 1))
        ret["loss"].backward()
        self.assertIsNotNone(model.head.weight.grad)

    def test_lstm_next_state_baseline(self):
        """LSTM baseline returns next-state predictions (4, 5, 2)."""
        model = LSTMNextStateBaseline(
            dataset=self.dataset, changing_dim=2, static_dim=1, lstm_hidden_dim=4
        )
        ret = model(**self.batch)
        self.assertEqual(ret["logit"].shape, (4, 1))
        self.assertEqual(ret["next_state"].shape, (4, 5, 2))
        ret["loss"].backward()
        self.assertIsNotNone(model.lstm.weight_ih_l0.grad)

    def test_lstm_batch_first_changes_predictions(self):
        """batch_first=False (the original's axis swap) mixes patients."""
        kwargs = {
            "dataset": self.dataset,
            "changing_dim": 2,
            "static_dim": 1,
            "lstm_hidden_dim": 4,
        }
        fixed = LSTMNextStateBaseline(**kwargs)
        swapped = LSTMNextStateBaseline(batch_first=False, **kwargs)
        swapped.load_state_dict(fixed.state_dict())
        other = {k: v for k, v in self.batch.items()}
        other["values"] = self.batch["values"].clone()
        other["values"][0] += 1.0  # change the first patient only
        for model, leaks in ((fixed, False), (swapped, True)):
            before = model(**self.batch)["logit"]
            after = model(**other)["logit"]
            changed = not torch.allclose(before[1:], after[1:])
            self.assertEqual(changed, leaks)


class TestGreedySelection(unittest.TestCase):
    """Test cases for the greedy selection helpers."""

    def test_top_features_cover_mass(self):
        """Candidates cover >= 90% of each concept's weight mass."""
        weight = torch.tensor([[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
        candidates = top_features_by_weight_mass(weight, mass=0.90)
        self.assertEqual(candidates, [[0, 1], [1, 2]])
        for row, inds in zip(weight, candidates):
            covered = row.abs()[inds].sum() / row.abs().sum()
            self.assertGreaterEqual(float(covered), 0.90)

    def test_greedy_picks_best_pairs_in_order(self):
        """Greedy selection adds the highest-scoring pair each round."""
        gains = torch.tensor([[0.1, 0.3, 0.0], [0.0, 0.0, 0.2]])

        def score_fn(mask):
            return float((gains * mask).sum())

        selections, aucs = greedy_select_features(
            [[0, 1], [2]], score_fn, budget=5, encoding_dim=3
        )
        self.assertEqual(selections, [(0, 1), (1, 2), (0, 0)])
        self.assertEqual([round(a, 4) for a in aucs], [0.3, 0.5, 0.6])

    def test_apply_topk_mask_zeroes_rest(self):
        """Non-selected weights are zeroed and stay zero in training."""
        dataset = _tiny_dataset()
        model = ConceptBottleneckModel(
            dataset=dataset, changing_dim=2, static_dim=1
        )
        mask = apply_topk_mask(model, [(0, 0), (1, 2)])
        self.assertEqual(mask.shape, (4, 29))
        self.assertEqual(int(mask.sum()), 2)
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model(**batch)["loss"].backward()
        optimizer.step()
        self.assertTrue(bool((model.bottleneck.weight[~mask] == 0).all()))


if __name__ == "__main__":
    unittest.main()
