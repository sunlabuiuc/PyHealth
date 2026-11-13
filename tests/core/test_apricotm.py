import unittest
from datetime import datetime, timedelta

import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import APRICOTM
from pyhealth.tasks import ICUAcuityMIMIC4

T0 = datetime(2150, 1, 1, 8, 0, 0)  # noqa: DTZ001 - MIMIC times are naive
LABELS = ICUAcuityMIMIC4.LABELS
VITALS = {  # itemid: value, for the six vitals every stay needs
    220045: 80,  # hr
    220210: 18,  # resp
    220050: 120,  # sbp
    220051: 70,  # dbp
    223762: 37,  # temp (Celsius)
    220277: 97,  # o2sat
}


def _event(event_type, timestamp, **attrs):
    row = {"event_type": event_type, "timestamp": timestamp}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _fmt(time):
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _hr(hour, stay_id=10, value=90):
    return _event("chartevents", T0 + timedelta(hours=hour), stay_id=stay_id,
                  hadm_id=1, itemid=220045, valuenum=value, valueuom="bpm")


def _procedure(itemid, start, end, stay_id=10):
    return _event("procedureevents", T0 + timedelta(hours=start), stay_id=stay_id,
                  hadm_id=1, itemid=itemid, endtime=_fmt(T0 + timedelta(hours=end)))


def _input(itemid, start, end, stay_id=10):
    return _event("inputevents", T0 + timedelta(hours=start), stay_id=stay_id,
                  hadm_id=1, itemid=itemid, endtime=_fmt(T0 + timedelta(hours=end)))


def _patient(*, hours=18.0, age=60, deathtime=None, location="HOME",
             vitals=True, hr_hours=(4.5, 8.5, 12.5), extra=(), admission=True):
    """One admission with one ICU stay (stay_id 10) of ``hours`` hours.

    The six vitals are charted at hour 0.5 and a heart rate at ``hr_hours``.
    With ``admission=False`` the admission has no ``admissions`` row.
    """
    outtime = T0 + timedelta(hours=hours)
    rows = [_event("patients", None, anchor_age=age, gender="F")]
    if admission:
        rows.append(
            _event("admissions", T0 - timedelta(hours=5), hadm_id=1,
                   dischtime=_fmt(outtime + timedelta(hours=2)),
                   deathtime=None if deathtime is None else _fmt(deathtime),
                   hospital_expire_flag=int(deathtime is not None),
                   discharge_location="DIED" if deathtime else location)
        )
    rows.append(
        _event("icustays", T0, stay_id=10, hadm_id=1, outtime=_fmt(outtime))
    )
    if vitals:
        for itemid, value in VITALS.items():
            rows.append(
                _event("chartevents", T0 + timedelta(hours=0.5), stay_id=10,
                       hadm_id=1, itemid=itemid, valuenum=value, valueuom="x")
            )
    rows += [_hr(hour) for hour in hr_hours]
    rows += list(extra)
    df = pl.from_dicts(rows, infer_schema_length=None)
    df = df.with_columns(
        pl.lit("p1").alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(c).cast(pl.String) for c in df.columns if "/" in c),
    )
    return Patient(patient_id="p1", data_source=df)


def _positives(sample):
    return {name for name, y in zip(LABELS, sample["acuity"]) if y}


class TestICUAcuityMIMIC4(unittest.TestCase):
    def setUp(self):
        self.task = ICUAcuityMIMIC4()
        # Ventilation in hours 5-10 (intervals 1 and 2) and a vasopressor in
        # hours 13-14 (interval 3) of an 18-hour stay (intervals 0-4).
        self.therapies = [_procedure(225792, 5, 10), _input(221906, 13, 14)]

    def test_labels_of_each_window(self):
        samples = self.task(
            _patient(hr_hours=(4.5, 8.5, 12.5, 15.0), extra=self.therapies)
        )
        self.assertEqual([s["window"] for s in samples], [1, 2, 3, 4])
        self.assertEqual(
            [_positives(s) for s in samples],
            [
                {"unstable", "stable_to_unstable", "mv_onset"},
                {"unstable"},
                {"unstable", "mv_stop", "vp_onset"},
                {"discharge", "vp_stop"},
            ],
        )
        self.assertEqual(len(samples[0]["acuity"]), 12)

    def test_events_are_the_observation_window_triplets(self):
        samples = self.task(_patient(extra=self.therapies))
        first = samples[0]["events"]
        self.assertEqual(len(first["time"]), 6)
        self.assertTrue(all(t == 0.5 for t in first["time"]))
        self.assertEqual(
            sorted(first["tokens"]), ["dbp", "hr", "o2sat", "resp", "sbp", "temp"]
        )
        second = samples[1]["events"]
        self.assertEqual(second["time"], [4.5])
        self.assertEqual(second["tokens"], ["hr"])
        self.assertEqual(second["values"], [[90.0]])
        self.assertEqual(samples[0]["static"][0], 60.0)

    def test_window_without_events_is_skipped(self):
        samples = self.task(_patient(hr_hours=(8.5,), extra=self.therapies))
        self.assertEqual([s["window"] for s in samples], [1, 3])

    def test_icu_death(self):
        death = T0 + timedelta(hours=18)
        samples = self.task(
            _patient(deathtime=death, hr_hours=(4.5, 8.5, 12.5, 15.0))
        )
        self.assertEqual(_positives(samples[-1]), {"deceased"})
        self.assertEqual(_positives(samples[0]), {"stable"})

    def test_death_after_icu_discharge_is_a_discharge(self):
        death = T0 + timedelta(hours=30)
        samples = self.task(_patient(deathtime=death, hr_hours=(4.5, 8.5, 12.5, 15.0)))
        self.assertEqual(_positives(samples[-1]), {"discharge"})

    def test_death_within_tolerance_of_icu_discharge(self):
        hr = (4.5, 8.5, 12.5, 15.0)
        outtime = T0 + timedelta(hours=18)
        within = self.task(
            _patient(deathtime=outtime + timedelta(minutes=50), hr_hours=hr)
        )
        self.assertEqual(_positives(within[-1]), {"deceased"})
        beyond = self.task(
            _patient(deathtime=outtime + timedelta(minutes=70), hr_hours=hr)
        )
        self.assertEqual(_positives(beyond[-1]), {"discharge"})
        strict = ICUAcuityMIMIC4(death_tolerance_hours=0)
        late = strict(_patient(deathtime=outtime + timedelta(minutes=50), hr_hours=hr))
        self.assertEqual(_positives(late[-1]), {"discharge"})

    def test_missing_admission_row_drops_the_last_window(self):
        samples = self.task(
            _patient(admission=False, hr_hours=(4.5, 8.5, 12.5, 15.0),
                     extra=self.therapies)
        )
        self.assertEqual([s["window"] for s in samples], [1, 2, 3])
        self.assertEqual(
            _positives(samples[-1]), {"unstable", "mv_stop", "vp_onset"}
        )

    def test_discharge_home_only(self):
        task = ICUAcuityMIMIC4(discharge_home_only=True)
        hr = (4.5, 8.5, 12.5, 15.0)
        home = task(_patient(location="HOME", hr_hours=hr))
        self.assertEqual(_positives(home[-1]), {"discharge"})
        rehab = task(_patient(location="REHAB", hr_hours=hr))
        self.assertEqual(_positives(rehab[-1]), {"stable"})

    def test_massive_transfusion_only_sets_the_state(self):
        extra = [_procedure(229620, 5, 6)]
        samples = self.task(_patient(extra=extra))
        self.assertEqual(_positives(samples[0]), {"unstable", "stable_to_unstable"})
        task = ICUAcuityMIMIC4(include_massive_transfusion=False)
        self.assertEqual(_positives(task(_patient(extra=extra))[0]), {"stable"})

    def test_point_event_and_therapy_before_admission(self):
        extra = [
            _input(221906, 4.5, 4.5),  # push: interval 1 only
            _procedure(225792, -10, -1),  # before the stay: ignored
            _procedure(225802, -10, 4.5),  # CRRT running at admission
        ]
        samples = self.task(_patient(extra=extra))
        self.assertEqual(
            _positives(samples[0]), {"unstable", "vp_onset"}
        )
        self.assertEqual(
            _positives(samples[1]), {"unstable_to_stable", "stable", "vp_stop",
                                     "crrt_stop"}
        )

    def test_exclusions(self):
        self.assertEqual(self.task(_patient(hours=6)), [])
        self.assertEqual(self.task(_patient(hours=31 * 24)), [])
        self.assertEqual(self.task(_patient(age=17)), [])
        self.assertEqual(self.task(_patient(vitals=False)), [])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            ICUAcuityMIMIC4(window_hours=0)
        with self.assertRaises(ValueError):
            ICUAcuityMIMIC4(window_hours=4, min_stay_hours=6)

    def test_pre_filter_requires_therapy_tables(self):
        df = pl.DataFrame(
            {"event_type": ["chartevents"], "chartevents/itemid": ["220045"],
             "labevents/itemid": [None], "outputevents/itemid": [None],
             "admissions/hadm_id": [None]}
        ).lazy()
        with self.assertRaisesRegex(ValueError, "procedureevents"):
            self.task.pre_filter(df)

    def test_samples_feed_the_model(self):
        patient = _patient(hr_hours=(4.5, 8.5, 12.5, 15.0), extra=self.therapies)
        samples = self.task(patient)
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=ICUAcuityMIMIC4.input_schema,
            output_schema=ICUAcuityMIMIC4.output_schema,
            dataset_name="acuity",
        )
        model = APRICOTM(dataset, static_key="static", embedding_dim=8,
                         hidden_dim=4, state_size=4)
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        out = model(**batch)
        self.assertEqual(out["y_prob"].shape, (4, 12))
        torch.testing.assert_close(out["y_true"], batch["acuity"])


def _toy_dataset(n=6, long_index=None):
    samples = []
    for i in range(n):
        steps = 12 if i == long_index else 2 + i % 3
        samples.append(
            {
                "patient_id": f"p{i}",
                "events": {
                    "time": [0.5 * t for t in range(steps)],
                    "tokens": ["hr" if t % 2 else "sbp" for t in range(steps)],
                    "values": [[80.0 + i + t] for t in range(steps)],
                },
                "static": [50.0 + i, float(i % 2)],
                "acuity": [float(i % 2), float(i % 3 == 0), 0.0, 1.0],
            }
        )
    levels = [{"ordering": "series", "tokens": 1, "values": [1],
               "normalize": {"method": "minmax", "by_tokens": [0]}}]
    return create_sample_dataset(
        samples=samples,
        input_schema={"events": ("sequence_v2", {"levels": levels}),
                      "static": "tensor"},
        output_schema={"acuity": "tensor"},
        dataset_name="toy",
    )


class TestAPRICOTM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dataset = _toy_dataset(long_index=5)
        self.model = APRICOTM(self.dataset, static_key="static", embedding_dim=16,
                              hidden_dim=8, state_size=4, top_k=3)

    def _batch(self, indices):
        return get_dataloader(self.dataset, batch_size=len(indices)).collate_fn(
            [self.dataset[i] for i in indices]
        )

    def test_output_contract(self):
        out = self.model(**self._batch(range(6)))
        self.assertEqual(self.model.mode, "multilabel")
        self.assertEqual(out["logit"].shape, (6, 4))
        self.assertEqual(out["y_true"].shape, (6, 4))
        torch.testing.assert_close(out["y_prob"], torch.sigmoid(out["logit"]))
        out["loss"].backward()
        grads = [p.grad for name, p in self.model.named_parameters()
                 if name != "_dummy_param"]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))

    def test_inference_without_labels(self):
        batch = self._batch(range(3))
        batch.pop("acuity")
        out = self.model(**batch, embed=True)
        self.assertNotIn("loss", out)
        self.assertNotIn("y_true", out)
        self.assertEqual(out["embed"].shape, (3, 16))

    def test_prediction_does_not_depend_on_the_batch(self):
        self.model.eval()
        with torch.no_grad():
            alone = self.model(**self._batch([0]))["logit"]
            batched = self.model(**self._batch([0, 5, 1]))["logit"][:1]
        torch.testing.assert_close(alone, batched, atol=1e-5, rtol=1e-5)

    def test_only_the_last_events_are_read(self):
        model = APRICOTM(self.dataset, static_key="static", embedding_dim=16,
                         hidden_dim=8, state_size=4, top_k=3, max_seq_len=4)
        model.eval()
        batch = self._batch([5])
        tokens, values, time, _ = batch["events"]
        cut = (tokens[-4:], values[-4:], time[-4:], torch.tensor([0, 4]))
        with torch.no_grad():
            full = model(**batch)["logit"]
            last = model(**{**batch, "events": cut})["logit"]
            changed = model(**{**batch, "events": (
                tokens[-5:], values[-5:].clone().fill_(0.9), time[-5:],
                torch.tensor([0, 5]))})["logit"]
        torch.testing.assert_close(full, last)
        self.assertFalse(torch.allclose(full, changed))

    def test_short_sequence_and_no_static(self):
        model = APRICOTM(self.dataset, embedding_dim=8, hidden_dim=4,
                         state_size=4, top_k=10)
        out = model(**self._batch([0, 1]))
        self.assertTrue(torch.isfinite(out["logit"]).all())

    def test_balanced_pos_weight(self):
        batch = self._batch(range(6))
        plain = self.model(**batch)["loss"]
        torch.manual_seed(0)
        weighted = APRICOTM(self.dataset, static_key="static", embedding_dim=16,
                            hidden_dim=8, state_size=4, top_k=3,
                            pos_weight="balanced")
        # "balanced" gives negatives / positives over the dataset, so the last
        # label, positive in every sample, gets weight 0 (only labels with no
        # positive at all fall back to 1).
        torch.testing.assert_close(weighted.label_processor.weight,
                                   torch.tensor([1.0, 2.0, 1.0, 0.0]))
        self.assertIsNone(self.model.label_processor.weight)
        self.assertFalse(torch.allclose(plain, weighted(**batch)["loss"]))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            APRICOTM(self.dataset, feature_key="missing")
        with self.assertRaises(ValueError):
            APRICOTM(self.dataset, feature_key="static")
        with self.assertRaises(ValueError):
            APRICOTM(self.dataset, top_k=0)


if __name__ == "__main__":
    unittest.main()
