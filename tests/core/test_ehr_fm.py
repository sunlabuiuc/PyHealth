"""Tests for the EHR foundation-model task and the hf_ehr model wrapper.

Everything runs offline on tiny random backbones and a synthetic vocabulary.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from typing import ClassVar
from unittest.mock import patch

import polars as pl
import torch
from transformers import (
    GPT2Config,
    GPT2Model,
    LlamaConfig,
    LlamaModel,
    MambaConfig,
    MambaModel,
)

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import HFEHRFoundationModel, HFEHRTokenizer
from pyhealth.models.ehr_fm import as_value
from pyhealth.tasks import BenchmarkEHRShot, EHRFoundationModelTask, EventSource
from pyhealth.tasks.base_task import BaseTask
from pyhealth.trainer import Trainer

MIN = -1.7976931348623157e308

# A tiny hf_ehr tokenizer_config.json: codes, numeric ranges, categories.
CONFIG = {
    "timestamp": "2024-01-01",
    "metadata": {},
    "tokens": [
        {"code": "SNOMED/3950001", "type": "code", "tokenization": {}},  # 7
        {"code": "Gender/F", "type": "code", "tokenization": {}},  # 8
        {"code": "LOINC/2236-8", "type": "numerical_range",  # 9
         "tokenization": {"unit": "None", "range_start": MIN, "range_end": 4.0}},
        {"code": "LOINC/2236-8", "type": "numerical_range",  # 10
         "tokenization": {"unit": "None", "range_start": 4.0, "range_end": 9.0}},
        {"code": "SNOMED/228490006", "type": "categorical",  # 11
         "tokenization": {"categories": ["N"]}},
        {"code": "SNOMED/228490006", "type": "code", "tokenization": {}},  # 12
        {"code": "ICD10CM/E11.9", "type": "code", "tokenization": {}},  # 13
    ],
}
T0 = datetime(2010, 5, 1)


def _events(n=3):
    return [(T0 + timedelta(days=i), "ICD10CM/E11.9", None, None)
            for i in range(n)]


def _dataset(samples=None, label="binary"):
    samples = samples or [
        {"patient_id": "p0", "label": 1, "birth_datetime": T0,
         "events": [(T0, "SNOMED/3950001", None, None),
                    (T0, "Gender/F", None, None),
                    (T0 + timedelta(days=9), "LOINC/2236-8", "5.5", "mg/dL"),
                    (T0 + timedelta(days=9), "LOINC/9999-9", "1", None)]},
        {"patient_id": "p1", "label": 0, "birth_datetime": T0,
         "events": [(T0, "SNOMED/3950001", None, None),
                    (T0 + timedelta(days=3), "SNOMED/228490006", "N", None)]},
        {"patient_id": "p2", "label": 1, "birth_datetime": T0,
         "events": _events(5)},
        {"patient_id": "p3", "label": 0, "birth_datetime": T0, "events": []},
    ]
    return create_sample_dataset(
        samples, input_schema={"events": "raw", "birth_datetime": "raw"},
        output_schema={"label": label},
    )


def _gpt2(n_positions=32):
    torch.manual_seed(0)
    return GPT2Model(GPT2Config(vocab_size=len(CONFIG["tokens"]) + 7,
                                n_positions=n_positions, n_embd=16, n_layer=2,
                                n_head=2, resid_pdrop=0.0, embd_pdrop=0.0,
                                attn_pdrop=0.0))


def _model(dataset, mode="frozen", backbone=None, **kwargs):
    kwargs.setdefault("max_length", 32)
    return HFEHRFoundationModel(dataset, mode=mode, backbone=backbone or _gpt2(),
                                tokenizer=CONFIG, **kwargs)


def _batch(dataset, n=4):
    return next(iter(get_dataloader(dataset, batch_size=n, shuffle=False)))


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = HFEHRTokenizer(CONFIG)

    def test_special_tokens_and_token_strings(self):
        self.assertEqual(self.tok.vocab[:7], ["[BOS]", "[EOS]", "[UNK]", "[SEP]",
                                              "[PAD]", "[CLS]", "[MASK]"])
        self.assertEqual(self.tok.pad_token_id, 4)
        # the string format of hf_ehr's README example
        self.assertEqual(self.tok.vocab[9],
                         "LOINC/2236-8 || None || -1.7976931348623157e+308 - 4.0")
        self.assertEqual(self.tok.vocab[11], "SNOMED/228490006 || N")

    def test_numeric_ranges_inclusive_first_match(self):
        self.assertEqual(self.tok.event_to_id("LOINC/2236-8", -3.0), 9)
        self.assertEqual(self.tok.event_to_id("LOINC/2236-8", 4.0), 9)
        self.assertEqual(self.tok.event_to_id("LOINC/2236-8", "5.5"), 10)
        self.assertEqual(self.tok.event_to_id("LOINC/2236-8", 9.0), 10)
        # out of every range: dropped, not the code token
        self.assertIsNone(self.tok.event_to_id("LOINC/2236-8", 9.5))

    def test_categorical_and_code_fallback(self):
        self.assertEqual(self.tok.event_to_id("SNOMED/228490006", "N"), 11)
        self.assertIsNone(self.tok.event_to_id("SNOMED/228490006", "Y"))
        self.assertEqual(self.tok.event_to_id("SNOMED/228490006"), 12)
        # a numeric value on a code without ranges uses the code token
        self.assertEqual(self.tok.event_to_id("ICD10CM/E11.9", 3.0), 13)
        self.assertIsNone(self.tok.event_to_id("LOINC/0000-0"))

    def test_repeated_token_string_keeps_last_id(self):
        tok = HFEHRTokenizer({"tokens": [
            {"code": "A/1", "type": "code", "tokenization": {}},
            {"code": "A/1", "type": "code", "tokenization": {}},
        ]})
        self.assertEqual(tok.event_to_id("A/1"), 8)

    def test_values(self):
        self.assertEqual(as_value(" 7 "), 7.0)
        self.assertEqual(as_value("nan"), "nan")
        self.assertIsNone(as_value(float("nan")))
        self.assertIsNone(as_value(True))

    def test_from_pretrained_reads_local_dir(self):
        import json

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
                json.dump(CONFIG, f)
            self.assertEqual(len(HFEHRTokenizer.from_pretrained(d)), 14)


def _ehrshot_patient():
    """A Patient with EHRSHOT's tables: events, a split and two labels."""
    rows = [
        ("ehrshot", T0, "SNOMED/3950001", None, None, None, None),
        ("ehrshot", T0, "Gender/F", None, None, None, None),
        ("ehrshot", T0 + timedelta(days=10), "LOINC/2236-8", "3.1", "mg/dL",
         None, None),
        ("ehrshot", T0 + timedelta(days=20), "ICD10CM/E11.9", "", None, None, None),
        ("ehrshot", T0 + timedelta(days=30), "LOINC/2236-8", "8", "mg/dL",
         None, None),
        ("splits", None, None, None, None, "train", None),
        ("guo_icu", T0 + timedelta(days=15), None, None, None, None, "1"),
        ("guo_icu", T0 + timedelta(days=25), None, None, None, None, "0"),
    ]
    df = pl.DataFrame(
        {
            "patient_id": ["p0"] * len(rows),
            "event_type": [r[0] for r in rows],
            "timestamp": [r[1] for r in rows],
            "ehrshot/code": [r[2] for r in rows],
            "ehrshot/value": [r[3] for r in rows],
            "ehrshot/unit": [r[4] for r in rows],
            "splits/split": [r[5] for r in rows],
            "guo_icu/value": [r[6] for r in rows],
        },
        schema_overrides={"timestamp": pl.Datetime("ms")},
    )
    return Patient(patient_id="p0", data_source=df)


class TestTask(unittest.TestCase):
    def test_schema(self):
        task = EHRFoundationModelTask(BenchmarkEHRShot("lab_anemia"))
        self.assertEqual(task.input_schema,
                         {"events": "raw", "birth_datetime": "raw"})
        self.assertEqual(task.output_schema, {"label": "multiclass"})
        kept = EHRFoundationModelTask(BenchmarkEHRShot("guo_icu"), keep_inputs=True)
        self.assertIn("feature", kept.input_schema)

    def test_ehrshot_samples_stop_at_label_time(self):
        task = EHRFoundationModelTask(BenchmarkEHRShot("guo_icu"))
        samples = task(_ehrshot_patient())
        self.assertEqual(len(samples), 2)
        first, second = samples
        self.assertEqual(first["label"], "1")
        self.assertEqual(first["split"], "train")
        self.assertNotIn("feature", first)
        self.assertEqual([e[1] for e in first["events"]],
                         ["SNOMED/3950001", "Gender/F", "LOINC/2236-8"])
        self.assertEqual(first["events"][2][2:], ("3.1", "mg/dL"))
        self.assertEqual(len(second["events"]), 4)
        self.assertIsNone(second["events"][3][2])  # "" is missing
        self.assertEqual(first["birth_datetime"], T0)
        # the same cutoff as the wrapped task's codes
        base = BenchmarkEHRShot("guo_icu")(_ehrshot_patient())
        for wrapped, original in zip(samples, base):
            self.assertEqual([e[1] for e in wrapped["events"]], original["feature"])

    def test_max_events_keeps_most_recent(self):
        task = EHRFoundationModelTask(BenchmarkEHRShot("guo_icu"), max_events=2)
        events = task(_ehrshot_patient())[1]["events"]
        self.assertEqual([e[1] for e in events], ["LOINC/2236-8", "ICD10CM/E11.9"])

    def test_other_tasks_need_sources_and_time(self):
        class Other(BaseTask):
            task_name = "other"
            input_schema: ClassVar[dict] = {"codes": "sequence"}
            output_schema: ClassVar[dict] = {"y": "binary"}

            def __call__(self, patient):
                return [{"codes": [], "y": 1, "when": T0 + timedelta(days=12)}]

        with self.assertRaises(ValueError):
            EHRFoundationModelTask(Other())
        with self.assertRaises(ValueError):
            EHRFoundationModelTask(Other(), event_sources={"ehrshot": EventSource("code")})
        task = EHRFoundationModelTask(
            Other(), event_sources={"ehrshot": EventSource("code", vocabulary="X")},
            prediction_time="when", birth_fn=lambda p: T0 - timedelta(days=1))
        sample = task(_ehrshot_patient())[0]
        self.assertEqual([e[1] for e in sample["events"]],
                         ["X/SNOMED/3950001", "X/Gender/F", "X/LOINC/2236-8"])
        self.assertEqual(sample["birth_datetime"], T0 - timedelta(days=1))

    def _other(self, **sample):
        class Other(BaseTask):
            task_name = "other"
            input_schema: ClassVar[dict] = {"codes": "sequence"}
            output_schema: ClassVar[dict] = {"y": "binary"}

            def __call__(self, patient):
                return [{"codes": [], "y": 1, **sample}]

        return Other()

    def test_missing_prediction_time_key_raises(self):
        sources = {"ehrshot": EventSource("code")}
        for sample in ({}, {"when": None}):
            task = EHRFoundationModelTask(self._other(**sample),
                                          event_sources=sources,
                                          prediction_time="when")
            with self.assertRaises(ValueError):
                task(_ehrshot_patient())

    def test_callable_prediction_time(self):
        sources = {"ehrshot": EventSource("code")}
        cutoff = EHRFoundationModelTask(
            self._other(), event_sources=sources,
            prediction_time=lambda patient, sample, i: T0)
        self.assertEqual([e[1] for e in cutoff(_ehrshot_patient())[0]["events"]],
                         ["SNOMED/3950001", "Gender/F"])
        everything = EHRFoundationModelTask(
            self._other(), event_sources=sources,
            prediction_time=lambda patient, sample, i: None)
        full = everything.timeline(_ehrshot_patient())
        self.assertEqual(everything(_ehrshot_patient())[0]["events"], full)


class TestHFEHRFoundationModel(unittest.TestCase):
    def setUp(self):
        self.dataset = _dataset()

    def test_frozen_forward_backward_trains_head_only(self):
        model = _model(self.dataset)
        batch = _batch(self.dataset)
        out = model(**batch)
        self.assertEqual(out["y_prob"].shape, (4, 1))
        self.assertEqual(out["embed"].shape, (4, 16))
        out["loss"].backward()
        self.assertIsNotNone(model.fc.weight.grad)
        names = [n for n, _ in model.named_parameters() if n != "_dummy_param"]
        self.assertEqual(sorted(names), ["fc.bias", "fc.weight"])
        self.assertFalse(any(k.startswith("backbone") for k in model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))

    def test_finetune_trains_backbone(self):
        model = _model(self.dataset, mode="finetune")
        model.train()
        out = model(**_batch(self.dataset))
        out["loss"].backward()
        grads = [p.grad for n, p in model.named_parameters() if n.startswith("backbone.")]
        self.assertTrue(grads and all(g is not None for g in grads))
        self.assertEqual(model._memo, {})

    def test_frozen_matches_finetune_embeddings_in_eval(self):
        frozen = _model(self.dataset, backbone=_gpt2())
        tuned = _model(self.dataset, mode="finetune", backbone=_gpt2())
        tuned.eval()
        batch = _batch(self.dataset)
        with torch.no_grad():
            torch.testing.assert_close(frozen(**batch)["embed"], tuned(**batch)["embed"])

    def test_cache_does_not_change_outputs(self):
        batch = _batch(self.dataset)
        plain = _model(self.dataset, memoize=False)
        expected = plain(**batch)["embed"]
        with tempfile.TemporaryDirectory() as d:
            cached = _model(self.dataset, cache_dir=d, cache_name="tiny-gpt2")
            torch.testing.assert_close(cached(**batch)["embed"], expected)
            # memo hit: the backbone is not called again
            with patch.object(cached, "encode", side_effect=AssertionError):
                torch.testing.assert_close(cached(**batch)["embed"], expected)
            # disk hit from a fresh model
            again = _model(self.dataset, cache_dir=d, cache_name="tiny-gpt2")
            with patch.object(again, "encode", side_effect=AssertionError):
                torch.testing.assert_close(again(**batch)["embed"], expected)
        with self.assertRaises(ValueError):
            _model(self.dataset, cache_dir="x")  # backbone without cache_name

    def test_embedding_does_not_depend_on_batch(self):
        model = _model(self.dataset, memoize=False)
        batch = _batch(self.dataset)
        together = model(**batch)["embed"]
        for i in range(4):
            alone = model(events=[batch["events"][i]],
                          birth_datetime=[batch["birth_datetime"][i]])["embed"]
            torch.testing.assert_close(alone[0], together[i])

    def test_positions_match_hf_ehr_left_padding(self):
        # hf_ehr pads to max_length on the left and passes no position ids
        model = _model(self.dataset, memoize=False)
        timeline = model.tokenize(self.dataset[0]["events"])
        n = len(timeline.ids)
        ids = torch.full((1, 32), 4)
        ids[0, 32 - n:] = torch.tensor(timeline.ids)
        mask = (ids != 4).long()
        with torch.no_grad():
            reference = model.encoder(input_ids=ids, attention_mask=mask
                                      ).last_hidden_state[:, -1]
            torch.testing.assert_close(model.encode([timeline]), reference)

    def test_truncates_to_most_recent_tokens(self):
        model = _model(self.dataset, max_length=4)
        timeline = model.tokenize(_events(10) + [(T0, "Gender/F", None, None)])
        self.assertEqual(timeline.ids, [13, 13, 13, 8])
        special = _model(self.dataset, max_length=5, add_special_tokens=True)
        self.assertEqual(special.tokenize(_events(10)).ids, [5, 0, 13, 13, 1])
        self.assertEqual(model.tokenize([]).ids, [4])  # empty: one [PAD]

    def test_unknown_codes_are_counted(self):
        model = _model(self.dataset)
        model(**_batch(self.dataset))
        report = model.coverage_report()
        self.assertEqual(report["events"], 11)
        self.assertEqual(report["unknown_events"], 1)
        self.assertEqual(report["top_unknown"], [("LOINC/9999-9", 1)])
        self.assertAlmostEqual(report["coverage"], 10 / 11)

    def test_llama_and_mamba_backbones(self):
        torch.manual_seed(0)
        llama = LlamaModel(LlamaConfig(
            vocab_size=14, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=32))
        mamba = MambaModel(MambaConfig(vocab_size=14, hidden_size=16,
                                       state_size=4, num_hidden_layers=1))
        batch = _batch(self.dataset)
        for backbone in (llama, mamba):
            model = _model(self.dataset, backbone=backbone, max_length=16)
            out = model(**batch)
            self.assertEqual(out["embed"].shape, (4, 16))
            self.assertTrue(torch.isfinite(out["embed"]).all())
        self.assertFalse(model._takes_positions)  # mamba pads to max_length

    def test_trainer_fits_and_evaluates(self):
        model = _model(self.dataset)
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        trainer = Trainer(model=model, metrics=["roc_auc"], enable_logging=False)
        trainer.train(train_dataloader=loader, epochs=2)
        self.assertIn("roc_auc", trainer.evaluate(loader))

    def test_multiclass_label(self):
        dataset = _dataset([
            {"patient_id": f"p{i}", "label": i % 3, "birth_datetime": T0,
             "events": _events(i + 1)} for i in range(3)], label="multiclass")
        out = _model(dataset)(**_batch(dataset, 3))
        self.assertEqual(out["y_prob"].shape, (3, 3))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_multilabel_label(self):
        dataset = _dataset([
            {"patient_id": f"p{i}", "label": [str(i % 3), "x"],
             "birth_datetime": T0, "events": _events(i + 1)} for i in range(3)],
            label="multilabel")
        out = _model(dataset)(**_batch(dataset, 3))
        self.assertEqual(out["y_prob"].shape, (3, 4))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_regression_label(self):
        dataset = _dataset([
            {"patient_id": f"p{i}", "label": float(i), "birth_datetime": T0,
             "events": _events(i + 1)} for i in range(3)], label="regression")
        out = _model(dataset)(**_batch(dataset, 3))
        self.assertEqual(out["y_prob"].shape, (3, 1))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_unlabeled_forward(self):
        model = _model(self.dataset)
        out = model(events=[_events(2), []])
        self.assertEqual(out["y_prob"].shape, (2, 1))
        self.assertNotIn("loss", out)
        self.assertNotIn("y_true", out)

    def test_trainer_finetunes_backbone(self):
        model = _model(self.dataset, mode="finetune")
        before = [p.detach().clone() for p in model.backbone.parameters()]
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        trainer = Trainer(model=model, metrics=["roc_auc"], enable_logging=False)
        trainer.train(train_dataloader=loader, epochs=1,
                      optimizer_params={"lr": 1e-2})
        after = list(model.backbone.parameters())
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(after, before)))
        y_true, y_prob, loss = trainer.inference(loader)
        self.assertEqual(len(y_prob), len(self.dataset))

    def test_context_length_from_config(self):
        self.assertEqual(_model(self.dataset, max_length=None).max_length, 32)
        with self.assertRaises(ValueError):
            _model(self.dataset, max_length=64)

    def test_rejects_bad_settings(self):
        with self.assertRaises(ValueError):
            _model(self.dataset, mode="lora")
        text = create_sample_dataset(
            [{"patient_id": "p", "text": "x", "label": 1},
             {"patient_id": "q", "text": "y", "label": 0}],
            input_schema={"text": "text"}, output_schema={"label": "binary"})
        with self.assertRaises(ValueError):
            _model(text)


if __name__ == "__main__":
    unittest.main()
