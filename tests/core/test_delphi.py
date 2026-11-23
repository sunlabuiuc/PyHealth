"""Tests for Delphi-2M and the MEDS / MIMIC-IV trajectory tasks.

Synthetic patients and a tiny MEDS parquet fixture only. No UK Biobank or
PhysioNet download.
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import MEDSDataset, create_sample_dataset, get_dataloader
from pyhealth.models import Delphi
from pyhealth.models.delphi import DelphiAgeEncoding, _competing_nll
from pyhealth.processors import LabelProcessor
from pyhealth.tasks import (
    DelphiTrajectoryMEDS,
    DelphiTrajectoryMIMIC4,
    DelphiTrajectoryUKB,
)
from pyhealth.trainer import Trainer

torch.set_num_threads(1)

T0 = datetime(2150, 1, 1, 8, 0, 0)


def _event(event_type, timestamp, **attrs):
    row = {"event_type": event_type, "timestamp": timestamp}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _patient(patient_id, rows):
    frame = pl.from_dicts(rows, infer_schema_length=None)
    frame = frame.with_columns(
        pl.lit(patient_id).alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(name).cast(pl.String) for name in frame.columns if "/" in name),
    )
    return Patient(patient_id=patient_id, data_source=frame)


def _trajectory_samples():
    return [
        {
            "patient_id": "p1",
            "codes": ["A"],
            "ages": [0.0],
            "next_code": "DEATH",
            "time_to_next": 12.0,
        },
        {
            "patient_id": "p1",
            "codes": ["A", "B"],
            "ages": [0.0, 10.0],
            "next_code": "A",
            "time_to_next": 4.0,
        },
        {
            "patient_id": "p2",
            "codes": ["B"],
            "ages": [365.25],
            "next_code": "C",
            "time_to_next": 20.0,
        },
        {
            "patient_id": "p2",
            "codes": ["C", "A"],
            "ages": [1.0, 8.0],
            "next_code": "B",
            "time_to_next": 3.0,
        },
        {
            "patient_id": "p3",
            "codes": ["A", "C"],
            "ages": [2.0, 9.0],
            "next_code": "B",
            "time_to_next": 6.0,
        },
        {
            "patient_id": "p3",
            "codes": ["B", "C", "A"],
            "ages": [5.0, 15.0, 40.0],
            "next_code": "DEATH",
            "time_to_next": 30.0,
        },
    ]


def _dataset():
    return create_sample_dataset(
        samples=_trajectory_samples(),
        input_schema={"codes": "sequence", "ages": "tensor"},
        output_schema={"next_code": "multiclass", "time_to_next": "tensor"},
        dataset_name="delphi_synthetic",
        task_name="DelphiTrajectory",
    )


def _model(dataset, **kwargs):
    defaults = {
        "n_embd": 16,
        "n_layer": 2,
        "n_head": 2,
        "dropout": 0.0,
        "mask_ties": True,
    }
    defaults.update(kwargs)
    return Delphi(dataset, **defaults)


class TestDelphiAgeEncoding(unittest.TestCase):
    def test_delphi_age_encoding_shapes_and_values(self):
        encoding = DelphiAgeEncoding(4)
        with torch.no_grad():
            encoding.linear.weight.copy_(torch.eye(4))
        age = torch.tensor([[365.25, 730.5]])
        output = encoding(age)
        self.assertEqual(tuple(output.shape), (1, 2, 4))
        omega = encoding.div_term
        scaled = age.unsqueeze(-1) / 365.25 * omega
        expected = torch.zeros(1, 2, 4)
        expected[..., 0::2] = torch.sin(scaled)
        expected[..., 1::2] = torch.cos(scaled)
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))
        self.assertTrue(torch.allclose(encoding(age.unsqueeze(-1)), expected, atol=1e-5))
        with self.assertRaises(ValueError):
            DelphiAgeEncoding(3)


class TestDelphiModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dataset = _dataset()
        self.model = _model(self.dataset)
        self.vocab = self.dataset.input_processors["codes"].code_vocab

    def _labeled_batch(self, codes, ages, next_codes, waits):
        label_vocab = self.dataset.output_processors["next_code"].label_vocab
        return {
            "codes": torch.tensor(codes, dtype=torch.long),
            "ages": torch.tensor(ages, dtype=torch.float32),
            "next_code": torch.tensor(
                [label_vocab[code] for code in next_codes], dtype=torch.long
            ),
            "time_to_next": torch.tensor(waits, dtype=torch.float32),
        }

    def test_weight_tying_and_label_projection(self):
        code_vocab = self.vocab
        label_vocab = self.dataset.output_processors["next_code"].label_vocab
        extra = [label for label in label_vocab if label not in code_vocab]
        self.assertIn("DEATH", extra)
        self.assertEqual(
            self.model.lm_head.out_features,
            len(code_vocab) + len(extra),
        )
        self.assertEqual(self.model.transformer.wte.padding_idx, 0)
        self.assertIs(self.model.transformer.wte.weight, self.model.lm_head.weight)
        for label, class_index in label_vocab.items():
            mapped = int(self.model.label_to_token_idx[class_index])
            if label in code_vocab:
                self.assertEqual(mapped, code_vocab[label])
            else:
                self.assertGreaterEqual(mapped, len(code_vocab))
        self.assertEqual(len(self.model.label_processors), 1)
        self.assertEqual(self.model.mode, "multiclass")

    def test_delphi_forward_and_backward(self):
        token_a = self.vocab["A"]
        batch = self._labeled_batch(
            [[token_a]],
            [[0.0]],
            ["B"],
            [9.0],
        )
        self.model.train()
        output = self.model(**batch)
        for key in ("loss", "loss_ce", "loss_dt", "y_prob", "y_true", "logit", "dt_pred"):
            self.assertIn(key, output)
            self.assertTrue(torch.isfinite(output[key]).all())
        n_labels = len(self.model.label_processors["next_code"].label_vocab)
        self.assertEqual(tuple(output["logit"].shape), (1, n_labels))
        self.assertTrue(torch.allclose(output["y_prob"].sum(-1), torch.ones(1), atol=1e-5))
        self.assertTrue(torch.allclose(output["loss"], output["loss_ce"] + output["loss_dt"]))
        self.assertGreater(float(output["dt_pred"].detach()), 0.0)
        output["loss"].backward()
        graded = [param.grad for param in self.model.parameters() if param.grad is not None]
        self.assertTrue(graded)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in graded))

        longer = self._labeled_batch(
            [[token_a, self.vocab["B"], 0]],
            [[0.0, 12.0, 0.0]],
            ["C"],
            [6.0],
        )
        again = self.model(**longer)
        self.assertTrue(torch.isfinite(again["loss"]).all())
        self.assertEqual(again["logit"].shape[0], 1)

    def _full_hazard(self, model, batch):
        codes, ages = model._codes_and_ages(batch["codes"], batch["ages"])
        hidden, _ = model._hidden(codes, ages)
        full = model.lm_head(hidden)
        last = (codes != 0).sum(dim=1).clamp(min=1) - 1
        index = torch.arange(codes.size(0), device=codes.device)
        hazard = full[index, last].clone()
        hazard[..., 0] = float("-inf")
        return hazard

    def test_dt_pred_is_mean_wait_without_t_min(self):
        token_a = self.vocab["A"]
        batch = self._labeled_batch([[token_a]], [[0.0]], ["B"], [9.0])
        for t_min in (1.0, 2.5):
            model = _model(self.dataset, t_min=t_min)
            model.eval()
            output = model(**batch)
            hazard = self._full_hazard(model, batch)
            expected = torch.exp(-torch.logsumexp(hazard, dim=-1))
            self.assertTrue(torch.allclose(output["dt_pred"], expected, atol=1e-5))
            self.assertFalse(torch.allclose(output["dt_pred"], expected + t_min, atol=1e-3))
        longer = self._labeled_batch(
            [[token_a, self.vocab["B"]]],
            [[0.0, 12.0]],
            ["C"],
            [6.0],
        )
        model = _model(self.dataset, t_min=1.0, causal_sequence_loss=True)
        model.eval()
        longer_out = model(**longer)
        longer_expected = torch.exp(-torch.logsumexp(self._full_hazard(model, longer), dim=-1))
        self.assertTrue(torch.allclose(longer_out["dt_pred"], longer_expected, atol=1e-5))

    def test_labeled_time_loss_uses_full_vocabulary(self):
        token_a = self.vocab["A"]
        unk = self.vocab["<unk>"]
        label_ids = {int(index) for index in self.model.label_to_token_idx.tolist()}
        self.assertNotIn(unk, label_ids)
        batch = self._labeled_batch([[token_a]], [[0.0]], ["B"], [9.0])

        def boost(_module, _inputs, output):
            boosted = output.clone()
            boosted[..., unk] = 8.0
            return boosted

        handle = self.model.lm_head.register_forward_hook(boost)
        try:
            self.model.eval()
            output = self.model(**batch)
            hazard = self._full_hazard(self.model, batch)
        finally:
            handle.remove()
        expected_nll, _ = _competing_nll(
            hazard, batch["time_to_next"].view(-1), self.model.t_min
        )
        self.assertTrue(torch.allclose(output["loss_dt"], expected_nll.mean(), atol=1e-5))
        expected_dt = torch.exp(-torch.logsumexp(hazard, dim=-1))
        self.assertTrue(torch.allclose(output["dt_pred"], expected_dt, atol=1e-5))
        label_only = torch.exp(-torch.logsumexp(output["logit"], dim=-1))
        gap = (output["dt_pred"] - label_only).abs().detach()
        self.assertGreater(float(gap), 0.05)

    def test_delphi_mask_ties_and_birth_age_zero(self):
        token_a = self.vocab["A"]
        token_b = self.vocab["B"]
        self.model.eval()
        tied = self._labeled_batch(
            [[token_a, token_b, 0]],
            [[0.0, 0.0, 0.0]],
            ["C"],
            [5.0],
        )
        tied_out = self.model(**tied)
        attn = tied_out["attn"].detach()
        self.assertTrue(torch.isfinite(attn).all())
        # Co-occurring age 0.0 cannot see the other visit token.
        self.assertEqual(float(attn[0, 1, 0]), 0.0)
        self.assertAlmostEqual(float(attn[0, 1, 1]), 1.0, places=5)
        # A real birth-age token does not attend to a padded age of 0.0.
        self.assertEqual(float(attn[0, 0, 2]), 0.0)
        self.assertAlmostEqual(float(attn[0, 0, 0]), 1.0, places=5)
        # Padding keeps a diagonal so softmax stays defined.
        self.assertAlmostEqual(float(attn[0, 2, 2]), 1.0, places=5)

        spaced = self._labeled_batch(
            [[token_a, token_b, 0]],
            [[0.0, 30.0, 0.0]],
            ["C"],
            [5.0],
        )
        spaced_attn = self.model(**spaced)["attn"].detach()
        self.assertGreater(float(spaced_attn[0, 1, 0]), 0.0)
        self.assertGreater(float(spaced_attn[0, 1, 1]), 0.0)
        self.assertEqual(float(spaced_attn[0, 1, 2]), 0.0)

        untied = _model(self.dataset, mask_ties=False).eval()
        untied_attn = untied(**tied)["attn"].detach()
        self.assertGreater(float(untied_attn[0, 1, 0]), 0.0)

    def test_single_step_loss_without_prefix(self):
        model = _model(self.dataset, causal_sequence_loss=False)
        token_a = self.vocab["A"]
        batch = self._labeled_batch(
            [[token_a, self.vocab["B"]]],
            [[1.0, 4.0]],
            ["C"],
            [2.0],
        )
        output = model(**batch)
        self.assertTrue(torch.isfinite(output["loss"]).all())

    def test_delphi_generate_trajectory(self):
        token_a = self.vocab["A"]
        token_b = self.vocab["B"]
        codes = torch.tensor([[token_a, token_b], [token_b, token_a]], dtype=torch.long)
        ages = torch.tensor([[10.0, 40.0], [5.0, 25.0]], dtype=torch.float32)
        self.model.eval()
        new_codes, new_ages = self.model.generate_trajectory(
            codes,
            ages,
            max_new_tokens=3,
            max_age=85 * 365.25,
            no_repeat=True,
            top_k=2,
        )
        self.assertEqual(new_codes.shape[0], 2)
        self.assertGreater(new_codes.shape[1], codes.shape[1])
        for row_codes, row_ages in zip(new_codes, new_ages):
            valid = row_codes != 0
            self.assertTrue(torch.all(row_codes[valid] > 0))
            self.assertTrue(torch.all(row_codes[valid] < self.model.lm_head.out_features))
            observed = row_ages[valid]
            self.assertTrue(torch.all(observed[1:] > observed[:-1]))

        stopped, stopped_ages = self.model.generate_trajectory(
            codes[:1],
            ages[:1],
            max_new_tokens=4,
            termination_tokens=[token_a],
            no_repeat=False,
        )
        self.assertEqual(stopped.shape[1], codes.shape[1])
        self.assertTrue(torch.equal(stopped_ages, ages[:1]))

        capped, _ = self.model.generate_trajectory(
            codes[:1],
            ages[:1],
            max_new_tokens=4,
            max_age=10.0,
        )
        self.assertEqual(capped.shape[1], codes.shape[1])

    def test_delphi_with_pyhealth_trainer(self):
        model = _model(self.dataset)
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        trainer = Trainer(
            model=model,
            metrics=["accuracy"],
            device="cpu",
            enable_logging=False,
        )
        trainer.train(train_loader, epochs=1, monitor=None)
        scores = trainer.evaluate(train_loader)
        self.assertIn("accuracy", scores)
        self.assertIn("loss", scores)
        self.assertTrue(scores["loss"] == scores["loss"])  # not NaN
        processors = [
            proc
            for proc in self.dataset.output_processors.values()
            if isinstance(proc, LabelProcessor)
        ]
        self.assertEqual(len(self.dataset.output_processors), 2)
        self.assertEqual(len(processors), 1)


class TestDelphiTrajectoryTasks(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_delphi_trajectory_meds_task(self):
        birth = T0

        def day(n):
            return birth + timedelta(days=n)
        patient = _patient(
            "s1",
            [
                _event("meds", birth, code="MEDS_BIRTH"),
                _event("meds", day(10), code="ICD/A"),
                _event("meds", day(10), code="ICD/B"),
                _event("meds", day(40), code="ICD/C"),
                _event("meds", None, code="STATIC"),
            ],
        )
        samples = DelphiTrajectoryMEDS()(patient)
        self.assertEqual(
            [(s["codes"], s["next_code"], s["time_to_next"]) for s in samples],
            [
                (["ICD/A"], "ICD/B", 0.0),
                (["ICD/A", "ICD/B"], "ICD/C", 30.0),
            ],
        )
        self.assertEqual(samples[0]["ages"], [10.0])
        self.assertNotIn("MEDS_BIRTH", samples[0]["codes"])
        self.assertNotIn("STATIC", [code for s in samples for code in s["codes"]])

        snomed = _patient(
            "s2",
            [
                _event("meds", birth, code="SNOMED/3950001"),
                _event("meds", day(5), code="DX"),
                _event("meds", day(8), code="LAB"),
            ],
        )
        snomed_samples = DelphiTrajectoryMEDS()(snomed)
        self.assertEqual(snomed_samples[0]["codes"], ["DX"])
        self.assertEqual(snomed_samples[0]["ages"], [5.0])
        self.assertEqual(snomed_samples[0]["next_code"], "LAB")
        self.assertAlmostEqual(snomed_samples[0]["time_to_next"], 3.0)

        ehrshot = _patient(
            "s3",
            [
                _event("meds", birth, code="SNOMED/184099003"),
                _event("meds", day(1), code="DX"),
                _event("meds", day(2), code="LAB"),
            ],
        )
        self.assertEqual(DelphiTrajectoryMEDS()(ehrshot)[0]["ages"], [1.0])

        unanchored = _patient(
            "s4",
            [
                _event("meds", birth, code="X"),
                _event("meds", day(5), code="Y"),
                _event("meds", None, code="STATIC"),
            ],
        )
        fallback = DelphiTrajectoryMEDS()(unanchored)
        self.assertEqual(fallback[0]["ages"], [0.0])
        self.assertEqual(fallback[0]["next_code"], "Y")
        self.assertEqual(fallback[0]["time_to_next"], 5.0)
        self.assertEqual(DelphiTrajectoryMEDS(require_birth=True)(unanchored), [])

        capped = DelphiTrajectoryMEDS(max_seq_len=1)(patient)
        self.assertEqual(capped[-1]["codes"], ["ICD/B"])
        self.assertEqual(capped[-1]["next_code"], "ICD/C")
        prefixed = DelphiTrajectoryMEDS(code_prefix="ICD/", exclude_prefixes=("ICD/B",))(
            patient
        )
        self.assertEqual(
            [(s["codes"], s["next_code"]) for s in prefixed],
            [(["ICD/A"], "ICD/C")],
        )
        self.assertEqual(
            DelphiTrajectoryMEDS(min_events=2)(patient)[0]["codes"],
            ["ICD/A", "ICD/B"],
        )
        self.assertEqual(
            DelphiTrajectoryMEDS()(_patient("empty", [_event("notes", T0, code="Z")])),
            [],
        )
        with self.assertRaises(ValueError):
            DelphiTrajectoryMEDS(min_events=0)

        filtered = DelphiTrajectoryMEDS().pre_filter(
            pl.DataFrame({"event_type": ["meds", "other"], "x": [1, 2]}).lazy()
        )
        self.assertEqual(filtered.collect()["event_type"].to_list(), ["meds"])

    def test_meds_set_task_has_one_label_processor(self):
        root = self.temp_dir / "meds"
        (root / "data").mkdir(parents=True)
        birth = datetime(2020, 1, 1)
        frame = pl.DataFrame(
            {
                "subject_id": pl.Series([1, 1, 1, 1], dtype=pl.Int64),
                "time": pl.Series(
                    [
                        birth,
                        birth + timedelta(days=10),
                        birth + timedelta(days=10),
                        birth + timedelta(days=40),
                    ],
                    dtype=pl.Datetime("us"),
                ),
                "code": ["MEDS_BIRTH", "ICD/A", "ICD/B", "ICD/C"],
                "numeric_value": pl.Series([None] * 4, dtype=pl.Float32),
            }
        )
        frame.write_parquet(root / "data" / "0.parquet")
        dataset = MEDSDataset(root=str(root), cache_dir=self.temp_dir / "cache")
        sample_dataset = dataset.set_task(DelphiTrajectoryMEDS(), num_workers=1)
        self.assertGreater(len(sample_dataset), 0)
        self.assertEqual(len(sample_dataset.output_processors), 2)
        labels = [
            processor
            for processor in sample_dataset.output_processors.values()
            if isinstance(processor, LabelProcessor)
        ]
        self.assertEqual(len(labels), 1)
        self.assertIn("next_code", sample_dataset.output_processors)
        self.assertIsInstance(
            sample_dataset.output_processors["next_code"], LabelProcessor
        )
        sample_dataset.close()

    def test_delphi_trajectory_mimic4_task(self):
        early = T0
        later = T0 + timedelta(days=400)
        # dod is midnight on the later discharge day, before that day's
        # admission clock time, so the task must not place death first.
        patient = _patient(
            "p1",
            [
                _event(
                    "patients",
                    None,
                    anchor_age=60,
                    anchor_year=2150,
                    dod="2150-01-01",
                ),
                _event(
                    "admissions",
                    later,
                    hadm_id=2,
                    deathtime=None,
                    dischtime=(later + timedelta(hours=5)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                ),
                _event("admissions", early, hadm_id=1, deathtime=None),
                _event(
                    "diagnoses_icd",
                    later + timedelta(hours=5),
                    hadm_id=2,
                    icd_code="250",
                    seq_num=2,
                ),
                _event(
                    "diagnoses_icd",
                    later + timedelta(hours=5),
                    hadm_id=2,
                    icd_code="401",
                    seq_num=1,
                ),
                _event(
                    "diagnoses_icd",
                    early + timedelta(days=3),
                    hadm_id=1,
                    icd_code="428",
                    seq_num=1,
                ),
            ],
        )
        samples = DelphiTrajectoryMIMIC4()(patient)
        codes = samples[-1]["codes"] + [samples[-1]["next_code"]]
        # The full trajectory is the union of history and the final next code
        # of the longest prefix.
        full = samples[0]["codes"] + [s["next_code"] for s in samples]
        self.assertEqual(full, ["428", "401", "250", "DEATH"])
        self.assertEqual(codes[0], "428")
        # Both diagnoses of hadm 2 share the admission age, so the wait is 0
        # and death, clamped up to that visit, stays tied with them.
        same_visit = next(s for s in samples if s["next_code"] == "250")
        self.assertEqual(same_visit["time_to_next"], 0.0)
        self.assertEqual(samples[-1]["next_code"], "DEATH")
        self.assertEqual(samples[-1]["time_to_next"], 0.0)
        self.assertEqual(samples[-1]["ages"][-1], samples[-1]["ages"][-2])
        expected_age = (
            60 * 365.25
            + (later - datetime(2150, 1, 1)).total_seconds() / 86400.0
        )
        self.assertAlmostEqual(samples[-1]["ages"][-1], expected_age, places=3)

        died = _patient(
            "p2",
            [
                _event("patients", None, anchor_age=40, anchor_year=2150, dod=None),
                _event(
                    "admissions",
                    T0,
                    hadm_id=9,
                    deathtime=(T0 + timedelta(days=3)).strftime(
                        "%Y-%m-%d %H:%M:%S.%f"
                    ),
                ),
                _event(
                    "diagnoses_icd",
                    T0 + timedelta(days=1),
                    hadm_id=9,
                    icd_code="I10",
                    seq_num=1,
                ),
            ],
        )
        died_samples = DelphiTrajectoryMIMIC4()(died)
        self.assertEqual(died_samples[-1]["next_code"], "DEATH")
        self.assertAlmostEqual(died_samples[-1]["time_to_next"], 3.0, places=3)

        alive = _patient(
            "p3",
            [
                _event("patients", None, anchor_age=40, anchor_year=2150, dod=None),
                _event("admissions", T0, hadm_id=3, deathtime=None),
                _event(
                    "diagnoses_icd",
                    T0,
                    hadm_id=3,
                    icd_code="I10",
                    seq_num=1,
                ),
                _event(
                    "diagnoses_icd",
                    T0,
                    hadm_id=3,
                    icd_code="E11",
                    seq_num=2,
                ),
            ],
        )
        alive_samples = DelphiTrajectoryMIMIC4()(alive)
        self.assertEqual(alive_samples[0]["next_code"], "E11")
        self.assertNotIn("DEATH", [s["next_code"] for s in alive_samples])
        self.assertEqual(
            DelphiTrajectoryMIMIC4()(_patient("p4", [_event("notes", T0, code="Z")])),
            [],
        )
        missing_anchor = _patient(
            "p5",
            [
                _event("patients", None, anchor_age=None, anchor_year=2150, dod=None),
                _event("admissions", T0, hadm_id=1, deathtime=None),
                _event(
                    "diagnoses_icd", T0, hadm_id=1, icd_code="I10", seq_num=1
                ),
            ],
        )
        self.assertEqual(DelphiTrajectoryMIMIC4()(missing_anchor), [])
        orphan = _patient(
            "p6",
            [
                _event("patients", None, anchor_age=50, anchor_year=2150, dod=None),
                _event("admissions", T0, hadm_id=1, deathtime=None),
                _event(
                    "diagnoses_icd",
                    T0 + timedelta(days=2),
                    hadm_id=99,
                    icd_code="ORPH",
                    seq_num=1,
                ),
                _event(
                    "diagnoses_icd",
                    T0,
                    hadm_id=1,
                    icd_code=None,
                    seq_num=1,
                ),
                _event(
                    "diagnoses_icd",
                    T0 + timedelta(days=4),
                    hadm_id=1,
                    icd_code="KEEP",
                    seq_num=1,
                ),
            ],
        )
        orphan_samples = DelphiTrajectoryMIMIC4()(orphan)
        self.assertEqual(
            orphan_samples[0]["codes"] + [orphan_samples[0]["next_code"]],
            ["KEEP", "ORPH"],
        )
        self.assertEqual(
            DelphiTrajectoryMIMIC4()(
                _patient(
                    "p7",
                    [
                        _event(
                            "patients",
                            None,
                            anchor_age=50,
                            anchor_year=2150,
                            dod=None,
                        ),
                        _event("admissions", T0, hadm_id=1, deathtime=None),
                    ],
                )
            ),
            [],
        )

        kept = DelphiTrajectoryMIMIC4().pre_filter(
            pl.DataFrame(
                {"event_type": ["patients", "labevents"], "x": [1, 2]}
            ).lazy()
        )
        self.assertEqual(kept.collect()["event_type"].to_list(), ["patients"])

    def test_ukb_task_reads_preextracted_events(self):
        patient = _patient(
            "u1",
            [
                _event("ukb", T0, code="UKB_BIRTH"),
                _event("ukb", T0 + timedelta(days=100), code="C1"),
                _event("ukb", T0 + timedelta(days=140), code="C2"),
            ],
        )
        samples = DelphiTrajectoryUKB()(patient)
        self.assertEqual(samples[0]["codes"], ["C1"])
        self.assertEqual(samples[0]["next_code"], "C2")
        self.assertAlmostEqual(samples[0]["ages"][0], 100.0)
        self.assertEqual(DelphiTrajectoryUKB().task_name, "DelphiTrajectoryUKB")


if __name__ == "__main__":
    unittest.main()
