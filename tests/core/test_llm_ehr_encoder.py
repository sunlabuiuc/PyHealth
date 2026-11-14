"""Tests for the LLM EHR encoder (Hegselmann et al.) and its text serialization."""

import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from pyhealth.datasets import MIMIC4EHRDataset, create_sample_dataset, get_dataloader
from pyhealth.models import LLMEHREncoder
from pyhealth.processors import SequenceProcessor
from pyhealth.tasks import (
    CodeDescriber,
    EHRTextSerializationTask,
    InHospitalMortalityMIMIC4,
    MortalityPredictionMIMIC4,
    MortalityPredictionStageNetMIMIC4,
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


def _patient(sample, gender="F", age="63", anchor_year="2150"):
    events = {
        "patients": [SimpleNamespace(gender=gender, anchor_age=age,
                                     anchor_year=anchor_year)],
        # admitted 5 years after the anchor year
        "admissions": [SimpleNamespace(hadm_id="h1",
                                       timestamp=date(2155, 6, 1))],
    }
    return SimpleNamespace(
        patient_id="p0", sample=sample,
        get_events=lambda event_type=None, **kw: events.get(event_type, []),
    )


class TestSerialization(unittest.TestCase):
    def test_unique_codes_most_recent_first_with_descriptions(self):
        task = EHRTextSerializationTask(_CodesTask(), describer=OFFLINE)
        patient = _patient({"patient_id": "p0", "hadm_id": "h1", "label": 1,
                            # "0" is MIMIC's missing code and is dropped
                            "conditions": ["4019", "25000", "4019", "9999", "0"],
                            "procedures": []})
        [sample] = task(patient)
        self.assertEqual(set(sample), {"patient_id", "hadm_id", "label", "text"})
        self.assertEqual(sample["text"], (
            "# Electronic Health Record\n\n"
            # anchor_age 63 in 2150, admitted in 2155: age 68
            "## Patient Demographics\n- Gender: F\n- Age: 68\n\n"
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

    def test_age_is_at_the_sample_admission_or_left_out(self):
        task = EHRTextSerializationTask(_CodesTask(), describer=OFFLINE)

        def header(sample, **kw):
            sample = {"patient_id": "p0", "label": 0, "conditions": ["4019"],
                      "procedures": [], **sample}
            return task(_patient(sample, **kw))[0]["text"].split("\n\n")[1]

        self.assertEqual(header({"visit_id": "h1"}),
                         "## Patient Demographics\n- Gender: F\n- Age: 68")
        for sample, kw in [({}, {}),  # no admission id in the sample
                           ({"hadm_id": "other"}, {}),  # no such admission
                           ({"hadm_id": "h1"}, {"anchor_year": None})]:
            self.assertEqual(header(sample, **kw),
                             "## Patient Demographics\n- Gender: F")

    def test_stagenet_codes_are_serialized(self):
        task = EHRTextSerializationTask(MortalityPredictionStageNetMIMIC4(),
                                        describer=OFFLINE, demographics=None)
        self.assertEqual(task.timed_fields, ["icd_codes"])
        self.assertEqual(list(task.sections), ["icd_codes"])  # labs skipped
        sample = {"patient_id": "p0", "mortality": 0,
                  "icd_codes": ([0.0, 12.5], [["4019"], ["25000", "E11.9"]]),
                  "labs": ([0.0], [[140.0] * 10])}
        text = task.serialize_sample(sample)
        self.assertEqual(text.splitlines()[2:], [
            "## Icd codes",
            "- Type 2 diabetes mellitus without complications",
            "- Diabetes mellitus without complication",
            "- Unspecified essential hypertension",
        ])

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

    def test_icd_version_from_the_source_events(self):
        seen = []

        def lookup(code, vocabularies):
            seen.append((code, vocabularies))

        task = EHRTextSerializationTask(_CodesTask(), demographics=None,
                                        describer=CodeDescriber(lookup=lookup))
        patient = _patient({"patient_id": "p0", "label": 0,
                            "conditions": ["E000", "E119", "E8490"],
                            "procedures": []})
        events = {"diagnoses_icd": [
            SimpleNamespace(icd_code="E000", icd_version="10"),
            SimpleNamespace(icd_code="E119", icd_version="10"),
            SimpleNamespace(icd_code="E8490", icd_version="9"),
        ]}
        get_events = patient.get_events
        patient.get_events = lambda event_type=None, **kw: events.get(
            event_type, get_events(event_type))
        task(patient)
        self.assertEqual(sorted(seen), [("E000", ("ICD10CM",)),
                                        ("E119", ("ICD10CM",)),
                                        ("E8490", ("ICD9CM",))])
        # without a version, an ICD-9 E/V shape tries ICD-9 first
        seen.clear()
        both = ("ICD10CM", "ICD9CM")
        describe = CodeDescriber(lookup=lookup)
        for code in ("E000", "V1582", "E11", "I10"):
            describe(code, both)
        self.assertEqual(seen, [("E000", ("ICD9CM", "ICD10CM")),
                                ("V1582", ("ICD9CM", "ICD10CM")),
                                ("E11", ("ICD9CM", "ICD10CM")),
                                ("I10", both)])

    def test_real_vocabularies(self):
        from pyhealth.medcode import InnerMap

        try:
            InnerMap.load("ICD9CM")
            InnerMap.load("ICD10CM")
        except Exception as e:  # noqa: BLE001 - e.g. offline without a cache
            self.skipTest(f"vocabularies unavailable: {e}")
        describe = CodeDescriber()
        both = ("ICD10CM", "ICD9CM")
        self.assertEqual(describe("4019", both),
                         "Unspecified essential hypertension")
        self.assertEqual(describe("V1582", both),
                         "Personal history of tobacco use")  # ICD-9 V-code
        self.assertEqual(describe("E8490", both), "Home accidents")  # ICD-9 E
        self.assertEqual(describe("E119", both),
                         "Type 2 diabetes mellitus without complications")
        # E000 is a code in both: the version decides
        self.assertEqual(describe("9_E000", both), "External cause status")
        self.assertEqual(describe("10_E000", both),
                         "Congenital iodine-deficiency syndrome, neurological type")

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

    def test_cache_key_tells_lambdas_apart(self):
        def key(describer):
            task = EHRTextSerializationTask(MortalityPredictionMIMIC4(),
                                            describer=describer)
            return json.dumps(vars(task), sort_keys=True, default=str)

        first = lambda code, vocabularies: "first"
        second = lambda code, vocabularies: "second"
        self.assertNotEqual(key(CodeDescriber(lookup=first)),
                            key(CodeDescriber(lookup=second)))
        self.assertEqual(key(CodeDescriber(lookup=first)),
                         key(CodeDescriber(lookup=first)))
        # closures over different data share their code: name them
        tables = [{"4019": "A"}, {"4019": "B"}]
        lookups = [lambda c, v, t=t: t.get(c) for t in tables]
        self.assertNotEqual(key(CodeDescriber(lookup=lookups[0], name="a")),
                            key(CodeDescriber(lookup=lookups[1], name="b")))

    def test_only_code_inputs_are_serialized(self):
        class _MixedTask(_CodesTask):
            def __init__(self):
                self.input_schema = {"conditions": SequenceProcessor,
                                     "labs": "timeseries", "age": "tensor"}
                self.output_schema = {"label": "binary"}

        sample = {"patient_id": "p0", "label": 1, "conditions": ["4019"],
                  "labs": ([1.0, 2.0], [[0.5, float("nan")]]), "age": 63.0}
        with self.assertLogs("pyhealth.tasks.ehr_text_serialization", "WARNING"):
            task = EHRTextSerializationTask(_MixedTask(), describer=OFFLINE,
                                            demographics=None)
        [out] = task(_patient(sample))
        self.assertEqual(set(out), {"patient_id", "label", "text"})
        self.assertEqual(out["text"], "# Electronic Health Record\n\n"
                         "## Conditions\n- Unspecified essential hypertension")
        kept = EHRTextSerializationTask(_MixedTask(), describer=OFFLINE,
                                        keep_inputs=True)
        self.assertEqual(set(kept(_patient(sample))[0]),
                         {"patient_id", "label", "text", "conditions", "labs",
                          "age"})
        with self.assertRaises(ValueError):  # only a lab-value time series
            EHRTextSerializationTask(InHospitalMortalityMIMIC4())
        with self.assertRaises(ValueError):
            EHRTextSerializationTask(_MixedTask(),
                                     sections={"labs": ("Lab tests", ())})

    def test_vocabulary_load_failure_is_not_silent(self):
        from unittest import mock

        with mock.patch("pyhealth.medcode.InnerMap.load",
                        side_effect=OSError("offline")):
            with self.assertRaises(RuntimeError):
                CodeDescriber()("4019", ("ICD9CM",))
            describe = CodeDescriber(raw_fallback=True)
            with self.assertLogs("pyhealth.tasks.ehr_text_serialization",
                                 "WARNING"):
                self.assertIsNone(describe("4019", ("ICD9CM",)))
        self.assertNotEqual(repr(describe), repr(CodeDescriber()))

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
                # age at this admission, from anchor_age and anchor_year
                person = patient.get_events(event_type="patients")[0]
                [admission] = [a for a in patient.get_events(
                    event_type="admissions") if a.hadm_id == old["hadm_id"]]
                age = (int(person.anchor_age) + admission.timestamp.year
                       - int(person.anchor_year))
                self.assertIn(f"- Age: {age}\n", new["text"])


class TestStageNetOnDemo(unittest.TestCase):
    def test_stagenet_task_on_mimic4_demo(self):
        root = Path(__file__).parents[2] / "test-resources" / "core" / "mimic4demo"
        base = MortalityPredictionStageNetMIMIC4()
        task = EHRTextSerializationTask(base, describer=OFFLINE)
        with tempfile.TemporaryDirectory() as tmp:
            dataset = MIMIC4EHRDataset(
                root=str(root), cache_dir=tmp,
                tables=["diagnoses_icd", "procedures_icd", "labevents"],
            )
            checked = 0
            for patient in dataset.iter_patients():
                expected = base(patient)
                if not expected:
                    continue
                [old], [new] = expected, task(patient)
                self.assertEqual(set(new), {"patient_id", "mortality", "text"})
                _times, codes = old["icd_codes"]
                for code in {c for visit in codes for c in visit}:
                    self.assertIn(DESCRIPTIONS.get(code, code), new["text"])
                self.assertNotIn("Lab", new["text"])
                self.assertNotIn("Age", new["text"])  # patient-level sample
                checked += 1
                if checked == 3:
                    break
            self.assertGreater(checked, 0)


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
                            "cls": valid[0]}[pooling]
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

    def test_locked_cache_file_warns_and_keeps_the_embedding(self):
        """Windows: os.replace keeps raising PermissionError on a locked file.
        The cache write must warn, not raise, and the vector is still used."""

        def embed_fn(texts):
            return torch.stack([torch.full((4,), float(len(t))) for t in texts])

        cache = os.path.join(self.tmp.name, "locked-cache")
        model = LLMEHREncoder(self.dataset, embed_fn=embed_fn, cache_dir=cache,
                              cache_name="stub")
        expected = torch.nn.functional.normalize(embed_fn(["a", "bb"]), dim=-1)
        locked = patch("pyhealth.models.utils.os.replace",
                       side_effect=PermissionError("file in use"))
        no_sleep = patch("pyhealth.models.utils.time.sleep")
        with locked, no_sleep, self.assertWarnsRegex(RuntimeWarning,
                                                     "Could not replace"):
            out = model.embed(["a", "bb"])
        torch.testing.assert_close(out, expected)
        self.assertEqual(
            [p for p in Path(cache).rglob("*") if p.is_file()], []
        )  # no temporary files left, and nothing half-written

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_cache_files_are_owner_only(self):
        """Embeddings of patient-derived text are readable by the owner only."""

        def embed_fn(texts):
            return torch.stack([torch.full((4,), float(len(t))) for t in texts])

        cache = os.path.join(self.tmp.name, "private-cache")
        model = LLMEHREncoder(self.dataset, embed_fn=embed_fn, cache_dir=cache,
                              cache_name="stub")
        old = os.umask(0o022)
        try:
            model.embed(["a", "bb"])
        finally:
            os.umask(old)
        files = [p for p in Path(cache).rglob("*") if p.is_file()]
        self.assertEqual(len(files), 2)
        for f in files:
            self.assertEqual(f.stat().st_mode & 0o777, 0o600)

    def test_embeddings_are_cached_in_memory_and_on_disk(self):
        calls = []

        def embed_fn(texts):
            calls.append(list(texts))
            return torch.stack([torch.full((4,), float(len(t))) for t in texts])

        cache = os.path.join(self.tmp.name, "cache")
        model = LLMEHREncoder(self.dataset, embed_fn=embed_fn, cache_dir=cache,
                              cache_name="stub")
        calls.clear()
        first = model.embed(["a", "bb", "a"])
        self.assertEqual(calls, [["a", "bb"]])  # each text encoded once
        model.embed(["bb"])
        self.assertEqual(len(calls), 1)  # memoized
        again = LLMEHREncoder(self.dataset, embed_fn=embed_fn, cache_dir=cache,
                              cache_name="stub")
        calls.clear()
        torch.testing.assert_close(again.embed(["a", "bb", "a"]), first)
        self.assertEqual(calls, [])  # read from disk
        torch.testing.assert_close(first.norm(dim=-1), torch.ones(3))
        lean = LLMEHREncoder(self.dataset, embed_fn=embed_fn, cache_dir=cache,
                             cache_name="stub", memoize=False)
        calls.clear()
        torch.testing.assert_close(lean.embed(["a", "bb"]), first[:2])
        self.assertEqual((calls, lean._memo), ([], {}))  # from disk, not kept
        other = LLMEHREncoder(self.dataset, embed_fn=embed_fn, cache_dir=cache,
                              cache_name="stub", instruction="Predict death")
        calls.clear()
        other.embed(["a"])
        # other settings, other key; the instruction reaches embed_fn too
        self.assertEqual(calls, [["Instruct: Predict death\nQuery:a"]])
        self.assertEqual(
            [p.suffix for p in Path(cache).rglob("*") if p.is_file()],
            [".npy"] * 3,  # atomic writes leave no temporary files
        )
        formatted = LLMEHREncoder(
            self.dataset, embed_fn=embed_fn, cache_dir=cache, cache_name="stub",
            instruction="Predict death",
            instruction_format=lambda task, text: f"{task}: {text}",
        )
        calls.clear()
        formatted.embed(["a"])
        self.assertEqual(calls, [["Predict death: a"]])  # another formatter

    def test_embedders_sharing_a_cache_dir_do_not_collide(self):
        cache = os.path.join(self.tmp.name, "cache")
        ones = lambda texts: torch.ones(len(texts), 4)
        twos = lambda texts: torch.arange(4.0).repeat(len(texts), 1)
        a = LLMEHREncoder(self.dataset, embed_fn=ones, cache_dir=cache,
                          cache_name="ones")
        b = LLMEHREncoder(self.dataset, embed_fn=twos, cache_dir=cache,
                          cache_name="twos")
        a.embed(["x"])
        self.assertFalse(torch.equal(a.embed(["x"]), b.embed(["x"])))
        encoder, tokenizer = _tiny_encoder(self.tmp.name)
        for custom in ({"embed_fn": ones},
                       {"encoder": encoder, "tokenizer": tokenizer}):
            with self.assertRaises(ValueError):  # no identity for the cache
                LLMEHREncoder(self.dataset, cache_dir=cache, **custom)
            LLMEHREncoder(self.dataset, **custom)  # no disk cache: fine

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
