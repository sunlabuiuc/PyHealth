"""Tests for the LLM EHR encoder (Hegselmann et al.) and its text serialization."""

import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from pyhealth.datasets import MIMIC4EHRDataset, create_sample_dataset, get_dataloader
from pyhealth.models import LLMEHREncoder
from pyhealth.tasks import (
    CodeDescriber,
    EHRTextSerializationTask,
    MortalityPredictionMIMIC4,
    serialize_ehr,
)
from pyhealth.tasks.base_task import BaseTask
from pyhealth.trainer import Trainer

DESCRIPTIONS = {
    "4019": "Unspecified essential hypertension",
    "25000": "Diabetes mellitus without complication",
    "E11.9": "Type 2 diabetes mellitus without complications",
}
OFFLINE = CodeDescriber(lookup=lambda code, vocabularies: DESCRIPTIONS.get(code))


class _CodesTask(BaseTask):
    """A stand-in task: one sample per patient from its stored codes."""

    task_name = "Codes"

    def __init__(self):
        self.input_schema = {"conditions": "sequence", "procedures": "sequence"}
        self.output_schema = {"label": "binary"}

    def __call__(self, patient):
        return [dict(patient.sample)]


class _EHRShotLikeTask(BaseTask):
    task_name = "Feature"

    def __init__(self):
        self.input_schema = {"feature": "sequence"}
        self.output_schema = {"label": "binary"}

    def __call__(self, patient):
        return [dict(patient.sample)]


def _patient(sample, gender="F", age="63"):
    demo = SimpleNamespace(gender=gender, anchor_age=age)
    return SimpleNamespace(
        patient_id="p0", sample=sample,
        get_events=lambda event_type=None, **kw: [demo] if event_type == "patients" else [],
    )


class TestSerialization(unittest.TestCase):
    def test_unique_codes_most_recent_first_with_descriptions(self):
        task = EHRTextSerializationTask(_CodesTask(), describer=OFFLINE)
        patient = _patient({"patient_id": "p0", "label": 1,
                            # "0" is MIMIC's missing code and is dropped
                            "conditions": ["4019", "25000", "4019", "9999", "0"],
                            "procedures": []})
        [sample] = task(patient)
        self.assertEqual(set(sample), {"patient_id", "label", "text"})
        self.assertEqual(sample["text"], (
            "# Electronic Health Record\n\n"
            "## Patient Demographics\n- Gender: F\n- Age: 63\n\n"
            "## Conditions\n"
            "- 9999\n"  # the most recent; no description, so the raw code
            "- Unspecified essential hypertension\n"  # listed once
            "- Diabetes mellitus without complication"
        ))

    def test_nested_visits_truncation_and_kept_inputs(self):
        task = EHRTextSerializationTask(
            _CodesTask(), describer=OFFLINE, max_codes_per_section=1,
            demographics=None, keep_inputs=True,
        )
        sample = {"patient_id": "p0", "label": 0,
                  "conditions": [["4019"], ["25000"]], "procedures": ["P1"]}
        [out] = task(_patient(sample))
        self.assertEqual(out["conditions"], [["4019"], ["25000"]])
        self.assertEqual(out["text"].splitlines(), [
            "# Electronic Health Record", "", "## Conditions",
            "- Diabetes mellitus without complication", "", "## Procedures", "- P1",
        ])
        self.assertEqual(task.input_schema,
                         {"conditions": "sequence", "procedures": "sequence",
                          "text": "text"})
        text = serialize_ehr({"A": [f"entry {i}" for i in range(50)]}, max_chars=60)
        self.assertLessEqual(len(text), 60)
        self.assertTrue(text.endswith(tuple(f"entry {i}" for i in range(50))))

    def test_icd_version_prefix_selects_the_vocabulary(self):
        seen = []

        def lookup(code, vocabularies):
            seen.append((code, vocabularies))

        describe = CodeDescriber(lookup=lookup)
        both = ("ICD10CM", "ICD9CM")
        describe("9_4019", both)
        describe("10_I10", both)
        describe("4019", both)
        self.assertEqual(seen, [("4019", ("ICD9CM",)), ("I10", ("ICD10CM",)),
                                ("4019", both)])

    def test_prefixed_codes_are_grouped_by_vocabulary(self):
        task = EHRTextSerializationTask(_EHRShotLikeTask(), describer=OFFLINE,
                                        demographics=None)
        sample = {"patient_id": "p0", "label": 1, "feature": [
            "ICD10CM/E11.9", "RxNorm/123", "LOINC/2345-7", "SNOMED/1"]}
        text = task(_patient(sample))[0]["text"]
        self.assertIn("## Conditions\n- SNOMED/1\n- Type 2 diabetes", text)
        self.assertIn("## Medications\n- RxNorm/123", text)
        self.assertIn("## Lab tests\n- LOINC/2345-7", text)

    def test_cache_key_is_stable(self):
        def key(task):
            return json.dumps(vars(task), sort_keys=True, default=str)

        a = EHRTextSerializationTask(MortalityPredictionMIMIC4())
        b = EHRTextSerializationTask(MortalityPredictionMIMIC4())
        self.assertEqual(key(a), key(b))
        self.assertNotIn(" at 0x", key(a))
        self.assertNotEqual(key(a), key(EHRTextSerializationTask(
            MortalityPredictionMIMIC4(), max_chars=100)))

    def test_mimic4_demo_end_to_end(self):
        root = Path(__file__).parents[2] / "test-resources" / "core" / "mimic4demo"
        with tempfile.TemporaryDirectory() as tmp:
            dataset = MIMIC4EHRDataset(
                root=str(root), tables=["diagnoses_icd", "procedures_icd",
                                        "prescriptions"], cache_dir=tmp,
            )
            base = MortalityPredictionMIMIC4()
            task = EHRTextSerializationTask(base, describer=OFFLINE)
            patient = next(iter(dataset.iter_patients()))
            expected = base(patient)
            got = task(patient)
            self.assertEqual(len(got), len(expected))
            for old, new in zip(expected, got):
                self.assertEqual(new["mortality"], old["mortality"])
                for code in old["conditions"]:
                    self.assertIn(DESCRIPTIONS.get(code, code), new["text"])


def _tiny_encoder(tmp, padding_side="right"):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "#", "-", "ehr",
             "sepsis", "fracture", "pain", "electronic", "health", "record"]
    path = os.path.join(tmp, "vocab.txt")
    with open(path, "w") as f:
        f.write("\n".join(vocab))
    tokenizer = BertTokenizerFast(vocab_file=path, padding_side=padding_side)
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32,
                        max_position_embeddings=64)
    return BertModel(config).eval(), tokenizer


def _dataset(n=8, mode="binary"):
    samples = [{"patient_id": f"p{i}",
                "text": "# ehr - sepsis pain" if i % 2 else "# ehr - fracture",
                "label": i % 2} for i in range(n)]
    return create_sample_dataset(samples, input_schema={"text": "text"},
                                 output_schema={"label": mode})


class TestLLMEHREncoder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dataset = _dataset()

    def tearDown(self):
        self.tmp.cleanup()

    def test_pooling_matches_the_encoder_output(self):
        texts = ["# ehr - sepsis pain", "fracture"]
        for side in ("right", "left"):
            encoder, tokenizer = _tiny_encoder(self.tmp.name, side)
            batch = tokenizer(texts, padding=True, return_tensors="pt")
            hidden = encoder(**batch).last_hidden_state.detach()
            mask = batch["attention_mask"]
            for pooling in ("last", "mean", "cls"):
                model = LLMEHREncoder(self.dataset, encoder=encoder,
                                      tokenizer=tokenizer, pooling=pooling,
                                      normalize=False)
                got = model.embed(texts)
                for i in range(2):
                    valid = hidden[i][mask[i].bool()]
                    want = {"last": valid[-1], "mean": valid.mean(0),
                            "cls": hidden[i][0]}[pooling]
                    torch.testing.assert_close(got[i], want, msg=f"{side} {pooling}")

    def test_only_the_head_trains_and_is_saved(self):
        encoder, tokenizer = _tiny_encoder(self.tmp.name)
        model = LLMEHREncoder(self.dataset, encoder=encoder, tokenizer=tokenizer)
        batch = next(iter(get_dataloader(self.dataset, batch_size=4)))
        out = model(**batch)
        self.assertEqual(out["y_prob"].shape, (4, 1))
        self.assertEqual(out["embed"].shape, (4, 16))
        out["loss"].backward()
        self.assertIsNotNone(model.fc.weight.grad)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        self.assertEqual({k.split(".")[0] for k in model.state_dict()},
                         {"_dummy_param", "fc"})
        trainer = Trainer(model=model, enable_logging=False)
        trainer.train(train_dataloader=get_dataloader(self.dataset, 4, shuffle=True),
                      epochs=2)
        scores = trainer.evaluate(get_dataloader(self.dataset, 4))
        self.assertIn("roc_auc", scores)

    def test_embeddings_are_cached_in_memory_and_on_disk(self):
        calls = []

        def embed_fn(texts):
            calls.append(list(texts))
            return torch.stack([torch.full((4,), float(len(t))) for t in texts])

        cache = os.path.join(self.tmp.name, "cache")
        model = LLMEHREncoder(self.dataset, embed_fn=embed_fn, cache_dir=cache,
                              model_name="stub")
        calls.clear()
        first = model.embed(["a", "bb", "a"])
        self.assertEqual(calls, [["a", "bb"]])  # each text encoded once
        model.embed(["bb"])
        self.assertEqual(len(calls), 1)  # memoized
        again = LLMEHREncoder(self.dataset, embed_fn=embed_fn, cache_dir=cache,
                              model_name="stub")
        calls.clear()
        torch.testing.assert_close(again.embed(["a", "bb", "a"]), first)
        self.assertEqual(calls, [])  # read from disk
        torch.testing.assert_close(first.norm(dim=-1), torch.ones(3))
        other = LLMEHREncoder(self.dataset, embed_fn=embed_fn, cache_dir=cache,
                              model_name="stub", instruction="Predict death")
        calls.clear()
        other.embed(["a"])
        self.assertEqual(calls, [["a"]])  # other settings, other key

    def test_instruction_and_multiclass(self):
        seen = []

        def embed_fn(texts):
            seen.extend(texts)
            return torch.zeros(len(texts), 4)

        dataset = _dataset(mode="multiclass")
        model = LLMEHREncoder(dataset, embed_fn=embed_fn)
        out = model(text=["x"], label=torch.tensor([1]))
        self.assertEqual(out["y_prob"].shape, (1, 2))
        encoder, tokenizer = _tiny_encoder(self.tmp.name)
        model = LLMEHREncoder(self.dataset, encoder=encoder, tokenizer=tokenizer,
                              instruction="Predict mortality")
        original = tokenizer.__call__
        texts = []
        tokenizer_spy = lambda t, **kw: (texts.extend(t), original(t, **kw))[1]
        model._tokenizer = tokenizer_spy
        model.embed(["sepsis"])
        self.assertEqual(texts, ["Instruct: Predict mortality\nQuery:sepsis"])
        with self.assertRaises(ValueError):
            LLMEHREncoder(self.dataset, embed_fn=embed_fn, pooling="max")
        mixed = create_sample_dataset(
            [{"patient_id": f"p{i}", "text": "x", "codes": ["a"], "label": i % 2}
             for i in range(4)],
            input_schema={"codes": "sequence", "text": "text"},
            output_schema={"label": "binary"})
        model = LLMEHREncoder(mixed, embed_fn=embed_fn)  # other inputs ignored
        self.assertEqual(model.text_key, "text")
        batch = next(iter(get_dataloader(mixed, batch_size=4)))
        self.assertEqual(model(**batch)["y_prob"].shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
