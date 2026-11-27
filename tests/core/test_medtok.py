"""Tests for the MedTok embeddings (pyhealth.models.medtok)."""

import json
import os
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

import pyhealth.models.medtok as medtok_module
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import (
    RNN,
    MedTokEmbedding,
    Transformer,
    apply_medtok,
    convert_medtok_embeddings,
    load_medtok,
    medtok_keys,
)
from pyhealth.models.medtok import iter_medtok_json


class _FakeCrossMap:
    def map(self, code):
        return {"123": ["B01AC06", "N02BA01"], "456": ["B01AC06"]}.get(code, [])


def _vec(value):
    return np.full(256, value, dtype=np.float32)


SAMPLES = [
    {
        "patient_id": "p0",
        "visit_id": "v0",
        "conditions": [["4280", "4019"], ["E8889"]],
        "procedures": [["3893"]],
        "label": 1,
    },
    {
        "patient_id": "p1",
        "visit_id": "v1",
        "conditions": [["4280"]],
        "procedures": [["3893"], ["9904"]],
        "label": 0,
    },
]
SCHEMA = {"conditions": "nested_sequence", "procedures": "nested_sequence"}
MEDTOK = {"428.0": _vec(1.0), "E888.9": _vec(2.0), "38.93": _vec(3.0)}


def _dataset():
    return create_sample_dataset(SAMPLES, SCHEMA, {"label": "binary"})


class TestMedTokKeys(unittest.TestCase):
    def test_icd_codes_get_their_dot(self):
        self.assertEqual(medtok_keys("4280", "ICD9CM"), ["428.0"])
        self.assertEqual(medtok_keys("V5861", "ICD9CM"), ["V58.61"])
        self.assertEqual(medtok_keys("E8889", "ICD9CM"), ["E888.9"])
        self.assertEqual(medtok_keys("428", "ICD9CM"), ["428"])
        self.assertEqual(medtok_keys("428.0", "ICD9CM"), ["428.0"])
        self.assertEqual(medtok_keys("3893", "ICD9PROC"), ["38.93"])
        self.assertEqual(medtok_keys("e119", "ICD10CM"), ["E11.9"])
        self.assertEqual(medtok_keys("0src0j9", "ICD10PCS"), ["0SRC0J9"])
        self.assertEqual(medtok_keys("B01AC06", "ATC"), ["B01AC06"])

    def test_prefix_blank_and_raw(self):
        self.assertEqual(medtok_keys("ICD9CM:4280", "ICD9CM"), ["428.0"])
        self.assertEqual(medtok_keys("  ", "ICD9CM"), [])
        self.assertEqual(medtok_keys("a:b", "RAW"), ["a:b"])
        with self.assertRaises(ValueError):
            medtok_keys("4280", "ICD9")

    def test_ndc_maps_to_atc(self):
        old = medtok_module._NDC_TO_ATC
        medtok_module._NDC_TO_ATC = _FakeCrossMap()
        try:
            self.assertEqual(medtok_keys("123", "NDC"), ["B01AC06", "N02BA01"])
            self.assertEqual(medtok_keys("999", "NDC"), [])
        finally:
            medtok_module._NDC_TO_ATC = old


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = {f"C{i}.{i % 7}": [float(i), i / 3.0] for i in range(50)}
        self.data['we"ird'] = [1, 2]

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, obj, indent=None, name="code2tokens.json"):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w") as f:
            json.dump(obj, f, indent=indent)
        return path

    def test_stream_matches_json_load_at_any_chunk_size(self):
        for indent in (None, 2):
            path = self._write(self.data, indent)
            for chunk in (1, 5, 64, 1 << 20):
                got = dict(iter_medtok_json(path, chunk_size=chunk))
                self.assertEqual(got, self.data)
        got = dict(iter_medtok_json(path, keys={"C3.3", "absent"}))
        self.assertEqual(got, {"C3.3": [3.0, 1.0]})

    def test_stream_rejects_bad_files(self):
        with self.assertRaises(ValueError):
            list(iter_medtok_json(self._write([1, 2])))
        path = os.path.join(self.tmp.name, "cut.json")
        with open(path, "w") as f:
            f.write('{"a": [1, 2], "b": [3,')
        with self.assertRaises(ValueError):
            list(iter_medtok_json(path))
        # Cut between two entries, before the closing brace, and empty.
        with open(path, "w") as f:
            f.write('{"a": [1, 2], "b": [3]')
        with self.assertRaises(ValueError):
            list(iter_medtok_json(path))
        open(path, "w").close()
        with self.assertRaises(ValueError):
            list(iter_medtok_json(path))

    def test_convert_and_load(self):
        emb = {"428.0": [0.5] * 256, "I10": [0.25] * 256}
        src = self._write(emb, 2, "code2embeddings.json")
        npz = os.path.join(self.tmp.name, "medtok.npz")
        self.assertEqual(convert_medtok_embeddings(src, npz), 2)
        for path in (src, npz):
            loaded = load_medtok(path)
            self.assertEqual(sorted(loaded), ["428.0", "I10"])
            np.testing.assert_allclose(loaded["I10"], 0.25)
            self.assertEqual(list(load_medtok(path, keys=["I10"])), ["I10"])
        bad = self._write({"x": [1.0] * 3}, None, "bad.json")
        with self.assertRaises(ValueError):
            convert_medtok_embeddings(bad, npz)
        with self.assertRaises(FileNotFoundError):
            load_medtok(os.path.join(self.tmp.name, "absent.npz"))


VOCAB = {"<pad>": 0, "<unk>": 1, "4280": 2, "E8889": 3, "9999": 4}
SYSTEMS = {"conditions": "ICD9CM", "procedures": "ICD9PROC"}


class TestMedTokEmbedding(unittest.TestCase):
    def test_embedding_mode(self):
        emb = MedTokEmbedding(VOCAB, MEDTOK, "ICD9CM", embedding_dim=8)
        self.assertAlmostEqual(emb.coverage, 2 / 3)
        out = emb(torch.tensor([[2, 3, 4, 1, 0]]))
        self.assertEqual(out.shape, (1, 5, 8))
        torch.testing.assert_close(out[0, 0], emb.proj(torch.ones(256)))
        torch.testing.assert_close(out[0, 1], emb.proj(torch.full((256,), 2.0)))
        # Uncovered codes and <unk> share the learned missing vector.
        torch.testing.assert_close(out[0, 2], emb.proj(emb.missing))
        torch.testing.assert_close(out[0, 3], out[0, 2])
        self.assertTrue(torch.all(out[0, 4] == 0))

    def test_freeze(self):
        frozen = MedTokEmbedding(VOCAB, MEDTOK, "ICD9CM", embedding_dim=8)
        names = {n for n, _ in frozen.named_parameters()}
        self.assertEqual(names, {"missing", "proj.weight", "proj.bias"})
        self.assertIn("table", dict(frozen.named_buffers()))
        tuned = MedTokEmbedding(VOCAB, MEDTOK, "ICD9CM", embedding_dim=8, freeze=False)
        self.assertIn("table", {n for n, _ in tuned.named_parameters()})

    def test_ndc_averages_its_atc_codes(self):
        old = medtok_module._NDC_TO_ATC
        medtok_module._NDC_TO_ATC = _FakeCrossMap()
        try:
            medtok = {"B01AC06": _vec(1.0), "N02BA01": _vec(3.0)}
            vocab = {"<pad>": 0, "123": 1, "456": 2}
            emb = MedTokEmbedding(vocab, medtok, "NDC", embedding_dim=4)
        finally:
            medtok_module._NDC_TO_ATC = old
        torch.testing.assert_close(emb.table[1], torch.full((256,), 2.0))
        torch.testing.assert_close(emb.table[2], torch.full((256,), 1.0))

    def test_tokens_mode(self):
        tokens = {"428.0": np.array([5, 7]), "E888.9": np.array([7, 7, 9])}
        emb = MedTokEmbedding(VOCAB, tokens, "ICD9CM", embedding_dim=6, mode="tokens")
        self.assertEqual(emb.token_ids.shape, (5, 3))
        out = emb(torch.tensor([2, 3, 4, 0]))
        w = emb.tokens.weight
        torch.testing.assert_close(out[0], (w[5] + w[7]) / 2)
        torch.testing.assert_close(out[1], (w[7] + w[7] + w[9]) / 3)
        torch.testing.assert_close(out[2], emb.missing)
        self.assertTrue(torch.all(out[3] == 0))

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            MedTokEmbedding(VOCAB, MEDTOK, "ICD9CM", 8, mode="graph")


class TestApplyMedTok(unittest.TestCase):
    def _train_step(self, model, dataset):
        batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))
        out = model(**batch)
        out["loss"].backward()
        return out

    def test_transformer_trains_with_medtok(self):
        dataset = _dataset()
        model = Transformer(dataset, embedding_dim=16, heads=2)
        coverage = apply_medtok(model, MEDTOK, SYSTEMS)
        self.assertEqual(coverage, {"conditions": 2 / 3, "procedures": 1 / 2})
        layers = model.embedding_model.embedding_layers
        self.assertIsInstance(layers["conditions"], MedTokEmbedding)
        self.assertTrue(torch.isfinite(self._train_step(model, dataset)["loss"]))
        self.assertIsNotNone(layers["conditions"].proj.weight.grad)
        self.assertIsNotNone(layers["procedures"].missing.grad)

    def test_rnn_with_tokens_from_a_file(self):
        dataset = _dataset()
        model = RNN(dataset, embedding_dim=8, hidden_dim=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "code2tokens.json")
            with open(path, "w") as f:
                json.dump({"428.0": [1, 2], "38.93": [3, 4], "999.9": [5, 6]}, f)
            coverage = apply_medtok(
                model, path, {"conditions": "ICD9CM"}, mode="tokens"
            )
        self.assertEqual(coverage, {"conditions": 1 / 3})
        layers = model.embedding_model.embedding_layers
        self.assertIsInstance(layers["procedures"], nn.Embedding)
        self._train_step(model, dataset)
        self.assertIsNotNone(layers["conditions"].tokens.weight.grad)

    def test_errors(self):
        model = Transformer(_dataset(), embedding_dim=8)
        with self.assertRaises(ValueError):
            apply_medtok(model, MEDTOK, {"drugs": "ATC"})
        with self.assertRaises(ValueError):
            apply_medtok(nn.Linear(2, 2), MEDTOK, SYSTEMS)


if __name__ == "__main__":
    unittest.main()
