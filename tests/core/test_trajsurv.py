import importlib.util
import io
import math
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
from pyhealth.metrics import (
    concordance_index,
    cumulative_dynamic_auc,
    ipcw_brier_score,
    survival_metrics_fn,
)
from pyhealth.models import SurvivalMDN, TrajSurv
from pyhealth.models.trajsurv import (
    _CubicPath,
    _natural_cubic_second_derivatives,
    cox_partial_likelihood_loss,
    rank_n_contrast_loss,
)
from pyhealth.tasks import InHospitalSurvivalMIMIC4
from pyhealth.trainer import Trainer

T0 = datetime(2150, 1, 1, 8, 0, 0)  # noqa: DTZ001 - MIMIC times are naive


class TestSurvivalMetrics(unittest.TestCase):
    def test_concordance_known_orderings(self):
        duration, event = [1, 2, 3, 4], [1, 1, 1, 0]
        self.assertEqual(concordance_index(duration, event, [4, 3, 2, 1]), 1.0)
        self.assertEqual(concordance_index(duration, event, [1, 2, 3, 4]), 0.0)
        self.assertEqual(concordance_index(duration, event, [1, 1, 1, 1]), 0.5)
        self.assertTrue(math.isnan(concordance_index([1, 2], [0, 0], [1, 2])))

    @unittest.skipUnless(importlib.util.find_spec("lifelines"), "lifelines missing")
    def test_concordance_matches_lifelines(self):
        from lifelines.utils import concordance_index as lifelines_ci

        rng = np.random.default_rng(0)
        duration = rng.exponential(size=200)
        event = rng.random(200) < 0.6
        risk = -duration + rng.normal(scale=0.5, size=200)
        # lifelines scores "higher predicted time is better": pass -risk
        expected = lifelines_ci(duration, -risk, event)
        self.assertAlmostEqual(concordance_index(duration, event, risk), expected)

    def test_brier_and_auc(self):
        duration, event = [1, 3, 4], [1, 0, 1]
        self.assertAlmostEqual(ipcw_brier_score(duration, event, [0, 1, 1], 2.0), 0)
        self.assertAlmostEqual(
            ipcw_brier_score(duration, event, [1, 0, 0], 2.0), 1.0
        )
        self.assertEqual(
            cumulative_dynamic_auc([1, 2, 5, 6], [1, 1, 0, 1], [4, 3, 2, 1], 3.0), 1.0
        )
        out = survival_metrics_fn(
            duration, event, [3, 2, 1], survival=np.array([[0.0], [1.0], [1.0]]),
            time_grid=[2.0], times=[2.0],
        )
        self.assertEqual(set(out), {"c_index", "auc@2", "brier@2"})

    def test_brier_interpolates_off_grid_horizons(self):
        rng = np.random.default_rng(0)
        duration = rng.exponential(scale=10.0, size=50)
        event = rng.random(50) < 0.7
        scale = rng.uniform(5.0, 20.0, size=(50, 1))
        grid = np.linspace(0.1, 60.0, 120)  # 7 falls between grid points
        survival = np.exp(-grid[None, :] / scale)
        for t in (7.0, 0.05):  # inside the grid, and before its first point
            out = survival_metrics_fn(duration, event, -scale[:, 0],
                                      survival=survival, time_grid=grid, times=[t])
            if t > grid[0]:
                at_t = np.array([np.interp(t, grid, row) for row in survival])
            else:  # linear from S(0) = 1 to the first grid point
                at_t = 1.0 + (t / grid[0]) * (survival[:, 0] - 1.0)
            self.assertAlmostEqual(
                out[f"brier@{t:g}"], ipcw_brier_score(duration, event, at_t, t)
            )
        with self.assertRaises(ValueError):
            survival_metrics_fn(duration, event, -scale[:, 0], survival=survival,
                                time_grid=grid, times=[61.0])
        with self.assertRaisesRegex(ValueError, "time_grid is needed"):
            survival_metrics_fn(duration, event, -scale[:, 0], survival=survival,
                                times=[7.0])

    def test_ipcw_brier_hand_checked(self):
        # Censoring KM: G = 1 before t=2, 0.5 from t=2 (one of two at risk).
        # Record 0: event at 1, weight 1/G(1-) = 1, loss 0.5^2.
        # Record 1: censored at 2, before the horizon, so weight 0.
        # Record 2: still at risk at 2.5, weight 1/G(2.5) = 2, loss 2 * 0.2^2.
        score = ipcw_brier_score([1, 2, 3], [1, 0, 1], [0.5, 0.3, 0.8], 2.5)
        self.assertAlmostEqual(score, (0.25 + 0.0 + 2 * 0.04) / 3)


class TestTrajSurvPieces(unittest.TestCase):
    def test_cox_loss_hand_checked(self):
        loss = cox_partial_likelihood_loss(
            torch.tensor([0.0, 0.0]), torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0])
        )
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        # risks log(1), log(3): the event (risk 0) competes with exp(0) + exp(log 3)
        risk = torch.tensor([0.0, math.log(3.0)])
        loss = cox_partial_likelihood_loss(
            risk, torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0])
        )
        self.assertAlmostEqual(loss.item(), (math.log(4.0) + 0.0) / 2, places=6)
        no_event = cox_partial_likelihood_loss(
            risk.requires_grad_(), torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0])
        )
        self.assertEqual(no_event.item(), 0.0)
        no_event.backward()

    def test_spline_integrates_to_increments(self):
        x = torch.randn(2, 7, 3)
        path = _CubicPath(x)
        # Simpson's rule is exact for the quadratic derivative of a cubic.
        for segment in range(6):
            integral = (
                path.derivative(segment, 0.0)
                + 4 * path.derivative(segment, 0.5)
                + path.derivative(segment, 1.0)
            ) / 6
            torch.testing.assert_close(integral, x[:, segment + 1] - x[:, segment])
        # natural spline: continuous first derivative at the inner knots
        torch.testing.assert_close(path.derivative(0, 1.0), path.derivative(1, 0.0))
        # Ragged lengths: each series is fitted on its own knots only.
        padded = x.clone()
        padded[1, 4:] = padded[1, 3]  # second series has 4 knots, then padding
        ragged = _natural_cubic_second_derivatives(padded, torch.tensor([7, 4]))
        torch.testing.assert_close(ragged[0], _natural_cubic_second_derivatives(x)[0])
        torch.testing.assert_close(
            ragged[1, :4], _natural_cubic_second_derivatives(x[1:, :4])[0]
        )
        self.assertTrue((ragged[1, 4:] == 0).all())
        linear = torch.arange(5.0).view(1, 5, 1)
        self.assertTrue(torch.allclose(_natural_cubic_second_derivatives(linear), 0 * linear))

    def test_rank_n_contrast_prefers_ordered_features(self):
        z = torch.linspace(0, 3, 6).unsqueeze(-1)
        y = torch.arange(6.0).unsqueeze(-1)
        shuffled = y[[0, 3, 1, 5, 2, 4]]  # reversing would keep all distances
        self.assertLess(
            rank_n_contrast_loss(z, y).item(), rank_n_contrast_loss(z, shuffled).item()
        )


def _dense_dataset(n=64, steps=8, seed=0, with_alignment=False):
    """Series whose first feature's level sets the hazard (planted risk)."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        level = rng.normal()
        series = rng.normal(scale=0.3, size=(steps, 3))
        series[:, 0] += level
        duration = rng.exponential(scale=np.exp(-1.5 * level)) + 0.01
        sample = {
            "patient_id": f"p{i}",
            "series": series.tolist(),
            "duration": [float(duration)],
            "event": [float(rng.random() < 0.8)],
        }
        if with_alignment:
            sample["sofa"] = np.repeat([[level]], steps, axis=0).tolist()
        samples.append(sample)
    schema = {"series": "tensor"}
    if with_alignment:
        schema["sofa"] = "tensor"
    return create_sample_dataset(
        samples=samples,
        input_schema=schema,
        output_schema={"duration": "tensor", "event": "tensor"},
        dataset_name="planted",
    )


class TestTrajSurvModel(unittest.TestCase):
    def test_forward_backward_and_outputs(self):
        dataset = _dense_dataset(n=8)
        model = TrajSurv(dataset, feature_key="series", hidden_dim=8)
        # No label processor: the Trainer reports only the loss.
        self.assertEqual(model.label_processors, {})
        self.assertIsNone(model.mode)
        self.assertEqual(model.input_channels, 4)  # time + 3 features, no mask
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        out = model(**batch)
        self.assertEqual(out["risk"].shape, (4,))
        self.assertEqual(out["y_prob"].shape, (4, len(model.time_grid)))
        self.assertEqual(out["y_true"].shape, (4, 2))
        out["loss"].backward()
        self.assertTrue(any(p.grad is not None for p in model.vector_field.parameters()))
        # no labels at inference
        unlabeled = {k: v for k, v in batch.items() if k not in ("duration", "event")}
        self.assertNotIn("loss", model(**unlabeled))

    def test_learns_planted_risk(self):
        torch.manual_seed(0)
        train = _dense_dataset(n=96, seed=1)
        model = TrajSurv(train, feature_key="series", hidden_dim=8,
                         vector_field_width=16, vector_field_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loader = get_dataloader(train, batch_size=32, shuffle=True)
        for _ in range(15):
            for batch in loader:
                optimizer.zero_grad()
                model(**batch)["loss"].backward()
                optimizer.step()
        test = _dense_dataset(n=96, seed=2)
        risks, durations, events = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in get_dataloader(test, batch_size=32):
                out = model(**batch)
                risks.append(out["risk"])
                durations.append(out["y_true"][:, 0])
                events.append(out["y_true"][:, 1])
        c_index = concordance_index(
            torch.cat(durations), torch.cat(events), torch.cat(risks)
        )
        self.assertGreater(c_index, 0.7)

        model.fit_baseline_hazard(loader)
        surv = model(**next(iter(get_dataloader(test, batch_size=8))))["y_prob"]
        self.assertTrue(((surv >= 0) & (surv <= 1)).all())
        self.assertTrue((surv[:, 1:] <= surv[:, :-1] + 1e-6).all())

    def test_alignment_loss_trains_projection(self):
        dataset = _dense_dataset(n=8, with_alignment=True)
        model = TrajSurv(dataset, feature_key="series", alignment_key="sofa",
                         hidden_dim=8, alignment_time_weight=0.1)
        batch = next(iter(get_dataloader(dataset, batch_size=8)))
        with_alignment = model(**batch)["loss"]
        model.alignment_weight = 0.0
        without = model(**batch)["loss"]
        self.assertNotAlmostEqual(with_alignment.item(), without.item())
        model.alignment_weight = 1.0
        model(**batch)["loss"].backward()
        self.assertIsNotNone(model.projection[0].weight.grad)

    def test_breslow_baseline_hand_checked(self):
        samples = [
            {"patient_id": f"p{i}", "series": [[0.0], [1.0]],
             "duration": [d], "event": [e]}
            for i, (d, e) in enumerate([(1.0, 1.0), (2.0, 1.0), (2.0, 0.0), (3.0, 1.0)])
        ]
        dataset = create_sample_dataset(
            samples=samples, input_schema={"series": "tensor"},
            output_schema={"duration": "tensor", "event": "tensor"},
            dataset_name="breslow",
        )
        model = TrajSurv(dataset, feature_key="series", hidden_dim=4)
        with torch.no_grad():  # risk 0 for everyone: Breslow = Nelson-Aalen
            model.cox.weight.zero_()
            model.cox.bias.zero_()
        model.fit_baseline_hazard(get_dataloader(dataset, batch_size=2))
        torch.testing.assert_close(model.baseline_times, torch.tensor([1.0, 2.0, 3.0]))
        # at risk 4, 3, 1 at the event times; one event at each
        expected = torch.tensor([1 / 4, 1 / 4 + 1 / 3, 1 / 4 + 1 / 3 + 1.0])
        torch.testing.assert_close(model.baseline_cumhaz, expected)
        surv = model.survival(torch.zeros(1), torch.tensor([0.5, 2.5]))
        torch.testing.assert_close(surv[0], torch.exp(-torch.tensor([0.0, expected[1]])))

    def test_fitted_baseline_survives_state_dict_round_trip(self):
        dataset = _dense_dataset(n=16)
        model = TrajSurv(dataset, feature_key="series", hidden_dim=8)
        loader = get_dataloader(dataset, batch_size=8)
        unfitted_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.fit_baseline_hazard(loader)
        self.assertGreater(model.baseline_times.numel(), 0)
        batch = next(iter(loader))
        model.eval()
        with torch.no_grad():
            expected = model(**batch)["y_prob"]

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        fresh = TrajSurv(dataset, feature_key="series", hidden_dim=8)
        fresh.load_state_dict(state)  # strict
        fresh.eval()
        with torch.no_grad():
            torch.testing.assert_close(fresh(**batch)["y_prob"], expected)

        # Trainer's checkpoint path (save_ckpt / load_ckpt)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "best.ckpt")
            Trainer(model=model, enable_logging=False).save_ckpt(path)
            other = TrajSurv(dataset, feature_key="series", hidden_dim=8)
            Trainer(model=other, enable_logging=False).load_ckpt(path)
            other.eval()
            with torch.no_grad():
                torch.testing.assert_close(other(**batch)["y_prob"], expected)

        # an unfitted state (e.g. Trainer's in-memory best) loads into a fitted model
        model.load_state_dict(unfitted_state)
        self.assertEqual(model.baseline_times.numel(), 0)

    def test_trainer_trains_trajsurv(self):
        dataset = _dense_dataset(n=16)
        model = TrajSurv(dataset, feature_key="series", hidden_dim=8)
        before = [p.detach().clone() for p in model.parameters()]
        trainer = Trainer(model=model, enable_logging=False, device="cpu")
        trainer.train(get_dataloader(dataset, batch_size=8, shuffle=True), epochs=1)
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        )

    def test_linear_interpolation_and_substeps(self):
        dataset = _dense_dataset(n=8)
        batch = next(iter(get_dataloader(dataset, batch_size=8)))
        risks = {}
        for interpolation, substeps in (("cubic", 1), ("linear", 1), ("linear", 3)):
            torch.manual_seed(0)
            model = TrajSurv(dataset, feature_key="series", hidden_dim=8,
                             interpolation=interpolation, substeps=substeps)
            out = model(**batch)
            self.assertTrue(torch.isfinite(out["loss"]))
            out["loss"].backward()
            risks[interpolation, substeps] = out["risk"].detach()
        # same weights, different paths and step sizes give different risks
        self.assertFalse(torch.allclose(risks["cubic", 1], risks["linear", 1]))
        self.assertFalse(torch.allclose(risks["linear", 1], risks["linear", 3]))
        with self.assertRaises(ValueError):
            TrajSurv(dataset, feature_key="series", interpolation="spline")
        with self.assertRaises(ValueError):
            TrajSurv(dataset, feature_key="series", substeps=0)

    def test_rejects_multi_level_series(self):
        class TwoLevels:
            def schema(self):
                return ((0, "tokens"), (0, "offsets"), (1, "values"), (1, "offsets"))

        fake = SimpleNamespace(input_processors={"x": TwoLevels()})
        with self.assertRaisesRegex(ValueError, "more than one level"):
            TrajSurv._series_roles(fake, "x")

    def test_sequence_v2_series_with_mask_and_ragged_lengths(self):
        features = len(InHospitalSurvivalMIMIC4.DYNAMIC_FEATURES)
        rng = np.random.default_rng(0)
        samples = []
        for i in range(6):
            steps = 36 if i != 1 else 20
            values = rng.normal(size=(steps, features))
            values[rng.random(values.shape) < 0.7] = np.nan
            samples.append({
                "patient_id": f"p{i}",
                "dynamic": {"time": [float(h) for h in range(steps)], "values": values},
                "static": [60.0, 1.0, float("nan"), 70.0],
                "summary": rng.normal(size=features + 4).tolist(),
                "duration": [1.0 + i],
                "event": [float(i % 2)],
            })
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=InHospitalSurvivalMIMIC4.input_schema,
            output_schema=InHospitalSurvivalMIMIC4.output_schema,
            dataset_name="yaib_like",
        )
        model = TrajSurv(dataset, hidden_dim=8, static_key="static")
        self.assertEqual(model.input_channels, 1 + 2 * features)
        batch = next(iter(get_dataloader(dataset, batch_size=3)))
        states, lengths = model.trajectory(**batch)
        self.assertEqual(lengths.tolist(), [36, 20, 36])
        torch.testing.assert_close(states[1, 19], states[1, 35])  # frozen after end
        model(**batch)["loss"].backward()
        # The short series gets the same risk alone as next to longer ones.
        model.eval()
        with torch.no_grad():
            in_batch = model(**batch)["risk"][1]
            alone_batch = list(get_dataloader(dataset, batch_size=1))[1]
            alone = model(**alone_batch)["risk"][0]
        torch.testing.assert_close(in_batch, alone)
        # a series with no records raises instead of reading its neighbour
        value = list(batch["dynamic"])
        offsets = value[model._roles["offsets"]].clone()
        offsets[2] = offsets[1]  # second series now has zero records
        value[model._roles["offsets"]] = offsets
        with self.assertRaisesRegex(ValueError, "at least one record"):
            model._dense(tuple(value))
        # SurvivalMDN reads the flat summary of the same samples
        mdn = SurvivalMDN(dataset, feature_key="summary", hidden_dim=8,
                          num_components=2)
        self.assertTrue(torch.isfinite(mdn(**batch)["loss"]))


# ------------------------------------------------------------------ the task
def _event(event_type, timestamp, **attrs):
    row = {"event_type": event_type, "timestamp": timestamp}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _fmt(time):
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _patient(stays, admissions):
    """stays: (stay_id, hadm_id, intime, hours); admissions: (hadm_id,
    dischtime, expire, deathtime). Each stay gets an hourly heart rate."""
    rows = [_event("patients", None, anchor_age=60, gender="F")]
    for hadm_id, dischtime, expire, deathtime in admissions:
        rows.append(_event(
            "admissions", T0 - timedelta(hours=5), hadm_id=hadm_id,
            dischtime=_fmt(dischtime),
            deathtime=None if deathtime is None else _fmt(deathtime),
            hospital_expire_flag=expire,
        ))
    for stay_id, hadm_id, intime, hours in stays:
        rows.append(_event("icustays", intime, stay_id=stay_id, hadm_id=hadm_id,
                           outtime=_fmt(intime + timedelta(hours=hours))))
        for hour in range(int(hours)):
            rows.append(_event(
                "chartevents", intime + timedelta(hours=hour, minutes=5),
                stay_id=stay_id, hadm_id=hadm_id, itemid=220045,
                valuenum=80 + hour, valueuom="bpm",
            ))
    df = pl.from_dicts(rows, infer_schema_length=None)
    df = df.with_columns(
        pl.lit("p1").alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(c).cast(pl.String) for c in df.columns if "/" in c),
    )
    return Patient(patient_id="p1", data_source=df)


class TestInHospitalSurvivalMIMIC4(unittest.TestCase):
    def setUp(self):
        self.task = InHospitalSurvivalMIMIC4()
        self.hr = self.task.DYNAMIC_FEATURES.index("hr")

    def test_death_after_window_is_an_event(self):
        death = T0 + timedelta(hours=36 + 48)
        patient = _patient([(10, 1, T0, 60)], [(1, death, 1, death)])
        (sample,) = self.task(patient)
        self.assertEqual(sample["event"], [1.0])
        self.assertAlmostEqual(sample["duration"][0], 2.0)  # days after hour 36
        self.assertEqual(sample["dynamic"]["values"].shape,
                         (36, len(self.task.DYNAMIC_FEATURES)))
        self.assertEqual(sample["dynamic"]["values"][35, self.hr], 80 + 35)
        self.assertEqual(len(sample["summary"]),
                         len(self.task.DYNAMIC_FEATURES) + 4)
        self.assertEqual(sample["summary"][self.hr], 80 + 35)  # last value

    def test_discharge_is_censored_and_hours_unit(self):
        discharge = T0 + timedelta(hours=36 + 12)
        patient = _patient([(10, 1, T0, 40)], [(1, discharge, 0, None)])
        (sample,) = InHospitalSurvivalMIMIC4(time_unit="hours")(patient)
        self.assertEqual(sample["event"], [0.0])
        self.assertAlmostEqual(sample["duration"][0], 12.0)

    def test_exclusions_and_one_stay_per_admission(self):
        discharge = T0 + timedelta(days=10)
        short = _patient([(10, 1, T0, 30)], [(1, discharge, 0, None)])
        self.assertEqual(self.task(short), [])
        died_in_window = _patient(
            [(10, 1, T0, 40)], [(1, T0 + timedelta(hours=20), 1, T0 + timedelta(hours=20))]
        )
        self.assertEqual(self.task(died_in_window), [])
        two_stays = _patient(
            [(10, 1, T0, 40), (11, 1, T0 + timedelta(days=3), 40)],
            [(1, discharge, 0, None)],
        )
        self.assertEqual([s["stay_id"] for s in self.task(two_stays)], [10])


if __name__ == "__main__":
    unittest.main()
