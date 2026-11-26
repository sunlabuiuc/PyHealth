"""Tests for CLMBR-T (femr 0.2) in plain PyTorch.

Everything runs offline on tiny random configs, a synthetic dictionary and a
synthetic state dict with the published key names.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import msgpack
import torch
from safetensors.torch import save_file

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import CLMBR, CLMBRConfig, CLMBRTokenizer, CLMBRTransformer
from pyhealth.models.clmbr import load_clmbr_weights, local_causal_mask

T0 = datetime(2010, 5, 1, 8, 0)
INF = float("inf")

DICTIONARY = {
    "is_hierarchical": False,
    "age_stats": {"mean": 1.0e9, "std": 5.0e8},
    "vocab": [
        {"type": "code", "code_string": "SNOMED/3950001"},  # 0
        {"type": "code", "code_string": "Gender/F"},  # 1
        {"type": "numeric", "code_string": "LOINC/2236-8",  # 2
         "val_start": -INF, "val_end": 4.0},
        {"type": "numeric", "code_string": "LOINC/2236-8",  # 3
         "val_start": 4.0, "val_end": INF},
        {"type": "text", "code_string": "SNOMED/228490006",  # 4
         "text_string": "N"},
        {"type": "unused", "code_string": "X/1"},  # 5
        {"type": "code", "code_string": "ICD10CM/E11.9"},  # 6
        {"type": "code", "code_string": "LOINC/2236-8"},  # 7
    ],
}


def _config(**kwargs):
    base = {"vocab_size": 8, "hidden_size": 16, "intermediate_size": 24,
            "n_heads": 2, "n_layers": 2, "attention_width": 3}
    return CLMBRConfig(**{**base, **kwargs})


def _timeline(n, start=T0, code="ICD10CM/E11.9"):
    return [(start + timedelta(days=i), code, None, None) for i in range(n)]


def _dataset():
    samples = [
        {"patient_id": "p0", "label": 1, "birth_datetime": T0,
         "events": [(T0, "SNOMED/3950001", None, None),
                    (T0, "Gender/F", None, None),
                    (T0 + timedelta(days=9), "LOINC/2236-8", "5.5", "mg/dL"),
                    (T0 + timedelta(days=9), "LOINC/9999-9", "1", None)]},
        {"patient_id": "p1", "label": 0, "birth_datetime": T0,
         "events": [(T0, "SNOMED/3950001", None, None),
                    (T0 + timedelta(days=3), "SNOMED/228490006", "N", None)]},
        {"patient_id": "p2", "label": 1, "birth_datetime": T0,
         "events": _timeline(6)},
        {"patient_id": "p3", "label": 0, "birth_datetime": T0, "events": []},
    ]
    return create_sample_dataset(
        samples, input_schema={"events": "raw", "birth_datetime": "raw"},
        output_schema={"label": "binary"})


def _model(dataset, mode="frozen", config=None, **kwargs):
    torch.manual_seed(0)
    return CLMBR(dataset, mode=mode, config=config or _config(),
                 dictionary=DICTIONARY, random_init=True, **kwargs)


def _batch(dataset, n=4):
    return next(iter(get_dataloader(dataset, batch_size=n, shuffle=False)))


# ---------------------------------------------------------------------------
# A literal port of femr 0.2.3's packed forward pass (femr/models/transformer.py,
# femr/models/xformers.py and xformers 0.0.23's _materialize_causal_mask),
# used only to check the batched implementation against it.
# ---------------------------------------------------------------------------
def _femr_rotate_every_two_v2(x):
    flat_x = x.reshape(-1, x.shape[-1])
    x1 = flat_x[:, ::2]
    x2 = flat_x[:, 1::2]
    return torch.stack((-x2, x1), axis=-1).reshape(x.shape)


def _femr_fixed_pos_embedding(ages, dim, dtype):
    inv_freq = 1.0 / (10000 ** (torch.linspace(0, 2, steps=dim // 2)))
    inv_freq = inv_freq.reshape(1, 1, dim // 2)
    ages = ages.reshape(ages.shape[0], 1)
    t = inv_freq * ages
    sin, cos = torch.sin(t), torch.cos(t)
    final_shape = (ages.shape[0], 1, dim)
    sin = torch.stack((sin, sin), axis=-1).reshape(final_shape).type(dtype)
    cos = torch.stack((cos, cos), axis=-1).reshape(final_shape).type(dtype)
    return sin, cos


def _femr_apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    return (x * cos) + (_femr_rotate_every_two_v2(x) * sin)


def _xformers_block_local_bias(lengths, width):
    total = sum(lengths)
    bias = torch.full((total, total), float("-inf"))
    start = 0
    for n in lengths:
        mask = torch.tril(torch.ones(n, n))
        mask = torch.triu(mask, diagonal=-width + 1)
        bias[start:start + n, start:start + n] = torch.log(mask)
        start += n
    return bias


def _femr_forward(model, tokens, ages, normed_ages, lengths):
    config = model.config
    x = model.in_norm(model.embed(tokens))
    head_size = config.hidden_size // config.n_heads
    pos_embed = _femr_fixed_pos_embedding(ages, head_size, x.dtype)
    bias = _xformers_block_local_bias(lengths, config.attention_width)
    for layer in model.layers:
        h = layer.norm(x)
        if config.use_normed_ages:
            h[:, -2] = normed_ages.to(dtype=h.dtype)
            h[:, -1] = (normed_ages ** 2).to(dtype=h.dtype)
        transformed = layer.input_proj(h)
        ff = transformed[:, : -config.hidden_size * 3]
        qkv = transformed[:, -config.hidden_size * 3:]
        qkv = qkv.reshape(h.shape[0], 3, config.n_heads, head_size)
        q = _femr_apply_rotary_pos_emb(qkv[:, 0, :, :], pos_embed)
        k = _femr_apply_rotary_pos_emb(qkv[:, 1, :, :], pos_embed)
        v = qkv[:, 2, :, :]
        # ref_attention: per head, softmax(q k^T / sqrt(d) + bias) v
        q, k, v = (t.permute(1, 0, 2) for t in (q, k, v))  # [heads, N, d]
        scores = (q * (1 / head_size ** 0.5)) @ k.transpose(-2, -1) + bias
        attn = (scores.softmax(-1) @ v).permute(1, 0, 2).reshape(h.shape)
        if config.hidden_act == "gelu":
            ff = torch.nn.functional.gelu(ff)
        else:
            x1, x2 = ff.chunk(2, dim=-1)
            ff = torch.nn.functional.silu(x1) * x2
        x = x + layer.output_proj(torch.concatenate((attn, ff), axis=-1))
    return model.out_norm(x)


class TestCLMBRTokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = CLMBRTokenizer(DICTIONARY)

    def test_femr_lookup_rules(self):
        tok = self.tok
        self.assertEqual(tok.event_to_id("LOINC/2236-8", -3.0), 2)
        self.assertEqual(tok.event_to_id("LOINC/2236-8", "4.0"), 3)  # [4, inf)
        self.assertEqual(tok.event_to_id("SNOMED/228490006", "N"), 4)
        self.assertIsNone(tok.event_to_id("SNOMED/228490006", "Y"))
        # a numeric value never falls back to the code entry
        self.assertIsNone(tok.event_to_id("ICD10CM/E11.9", 1.0))
        self.assertEqual(tok.event_to_id("LOINC/2236-8"), 7)
        self.assertIsNone(tok.event_to_id("X/1"))  # "unused" is never produced
        self.assertEqual(tok.event_to_id("ICD10CM/E11.9"), 6)  # ids count "unused"
        self.assertEqual(len(tok), 8)

    def test_same_day_repeats_are_dropped(self):
        events = [(T0, "ICD10CM/E11.9", None, None),
                  (T0 + timedelta(hours=3), "ICD10CM/E11.9", None, None),
                  (T0 + timedelta(days=1), "ICD10CM/E11.9", None, None),
                  (T0 + timedelta(days=1), "Unknown/1", None, None)]
        ids, ages, normed, unknown = self.tok.encode(events, T0 - timedelta(days=2))
        self.assertEqual(ids, [6, 6])
        self.assertEqual(ages, [2.0, 3.0])
        self.assertAlmostEqual(normed[0], (2 * 86400 - 1.0e9) / 5.0e8)
        self.assertEqual(unknown, ["Unknown/1"])

    def test_hierarchical_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            CLMBRTokenizer({**DICTIONARY, "is_hierarchical": True})

    def test_unknown_entry_type_is_rejected(self):
        vocab = [*DICTIONARY["vocab"], {"type": "ontology", "code_string": "Y/1"}]
        with self.assertRaisesRegex(ValueError, "ontology"):
            CLMBRTokenizer({**DICTIONARY, "vocab": vocab})

    def test_age_stats_presence(self):
        self.assertTrue(CLMBRTokenizer(DICTIONARY).has_age_stats)
        bare = {k: v for k, v in DICTIONARY.items() if k != "age_stats"}
        self.assertFalse(CLMBRTokenizer(bare).has_age_stats)


class TestCLMBRTransformer(unittest.TestCase):
    def _check_against_femr(self, config):
        torch.manual_seed(1)
        model = CLMBRTransformer(config).eval()
        lengths = [5, 2, 4]
        tokens = [torch.randint(0, config.vocab_size, (n,)) for n in lengths]
        ages = [torch.cumsum(torch.rand(n) * 300, 0) + 1000 for n in lengths]
        normed = [torch.randn(n) for n in lengths]
        with torch.no_grad():
            packed = _femr_forward(model, torch.cat(tokens), torch.cat(ages),
                                   torch.cat(normed), lengths)
            width = max(lengths)

            def pad(xs, v):
                return torch.stack([
                    torch.cat([x, torch.full((width - len(x),), v, dtype=x.dtype)])
                    for x in xs])

            batched = model(pad(tokens, 0), pad(ages, 0.0), pad(normed, 0.0),
                            torch.tensor(lengths))
        start = 0
        for row, n in enumerate(lengths):
            torch.testing.assert_close(batched[row, :n], packed[start:start + n],
                                       rtol=1e-5, atol=1e-5)
            start += n

    def test_matches_femr_gelu_with_bias(self):
        self._check_against_femr(_config())

    def test_matches_femr_swiglu_normed_ages_no_bias(self):
        self._check_against_femr(_config(hidden_act="swiglu", use_bias=False,
                                         use_normed_ages=True, attention_width=2))

    def test_local_mask_is_banded(self):
        mask = local_causal_mask(torch.tensor([4, 2]), 4, 2)[:, 0].int()
        self.assertEqual(mask[0].tolist(), [[1, 0, 0, 0], [1, 1, 0, 0],
                                            [0, 1, 1, 0], [0, 0, 1, 1]])
        # padding: keys masked out, padding queries see only themselves
        self.assertEqual(mask[1].tolist(), [[1, 0, 0, 0], [1, 1, 0, 0],
                                            [0, 0, 1, 0], [0, 0, 0, 1]])


def _published_state_dict(config, seed=2):
    """Random tensors with femr FEMRModel's key names and shapes."""
    torch.manual_seed(seed)
    reference = CLMBRTransformer(config)
    out = {f"transformer.{k}": torch.randn_like(v)
           for k, v in reference.state_dict().items()}
    out["task_model.final_layer.weight"] = torch.randn(config.vocab_size,
                                                       config.hidden_size)
    out["task_model.final_layer.bias"] = torch.randn(config.vocab_size)
    return out


class TestWeights(unittest.TestCase):
    def test_published_key_names_and_shapes(self):
        config = _config(n_layers=1)
        names = {k: tuple(v.shape) for k, v in _published_state_dict(config).items()}
        h, i = config.hidden_size, config.intermediate_size
        self.assertEqual(names, {
            "transformer.embed.weight": (8, h),
            "transformer.in_norm.weight": (h,),
            "transformer.out_norm.weight": (h,),
            "transformer.layers.0.norm.weight": (h,),
            "transformer.layers.0.input_proj.weight": (3 * h + i, h),
            "transformer.layers.0.input_proj.bias": (3 * h + i,),
            "transformer.layers.0.output_proj.weight": (h, h + i),
            "transformer.layers.0.output_proj.bias": (h,),
            "task_model.final_layer.weight": (8, h),
            "task_model.final_layer.bias": (8,),
        })
        swiglu = _config(n_layers=1, hidden_act="swiglu", use_bias=False)
        names = {k: tuple(v.shape) for k, v in _published_state_dict(swiglu).items()}
        self.assertEqual(names["transformer.layers.0.input_proj.weight"],
                         (3 * h + 2 * i, h))
        self.assertNotIn("transformer.layers.0.input_proj.bias", names)

    def test_loads_a_local_repo(self):
        config = _config()
        published = _published_state_dict(config)
        with tempfile.TemporaryDirectory() as d:
            save_file(published, os.path.join(d, "model.safetensors"))
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"architectures": ["FEMRModel"],
                           "transformer_config": vars(config),
                           "task_config": {"task_type": "clmbr", "task_kwargs":
                                           {"clmbr_vocab_size": 8}}}, f)
            with open(os.path.join(d, "dictionary.msgpack"), "wb") as f:
                msgpack.dump(DICTIONARY, f)
            model = CLMBR(_dataset(), model_name=d)
        self.assertEqual(model.config, config)
        for key, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value, published[f"transformer.{key}"])
        self.assertEqual(model.tokenizer.event_to_id("LOINC/2236-8", 1.0), 2)

    def test_mismatched_weights_are_rejected(self):
        config = _config()
        model = CLMBRTransformer(config)
        published = _published_state_dict(config)
        del published["transformer.layers.1.norm.weight"]
        with self.assertRaisesRegex(ValueError, "missing"):
            load_clmbr_weights(model, published)
        published = _published_state_dict(_config(intermediate_size=32))
        with self.assertRaisesRegex(ValueError, "shape"):
            load_clmbr_weights(model, published)


class TestCLMBR(unittest.TestCase):
    def setUp(self):
        self.dataset = _dataset()

    def test_frozen_forward_backward_trains_head_only(self):
        model = _model(self.dataset)
        out = model(**_batch(self.dataset))
        self.assertEqual(out["y_prob"].shape, (4, 1))
        out["loss"].backward()
        self.assertIsNotNone(model.fc.weight.grad)
        self.assertFalse(any(k.startswith("backbone") for k in model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        # the empty timeline embeds to zeros
        self.assertEqual(out["embed"][3].abs().sum().item(), 0.0)

    def test_finetune_trains_backbone(self):
        model = _model(self.dataset, mode="finetune")
        model.train()
        model(**_batch(self.dataset))["loss"].backward()
        grads = [p.grad for n, p in model.named_parameters()
                 if n.startswith("backbone.")]
        self.assertTrue(grads and all(g is not None for g in grads))

    def test_cache_does_not_change_outputs(self):
        batch = _batch(self.dataset)
        expected = _model(self.dataset, memoize=False)(**batch)["embed"]
        with tempfile.TemporaryDirectory() as d:
            cached = _model(self.dataset, cache_dir=d, cache_name="tiny")
            torch.testing.assert_close(cached(**batch)["embed"], expected)
            with patch.object(cached, "encode", side_effect=AssertionError):
                torch.testing.assert_close(cached(**batch)["embed"], expected)
            again = _model(self.dataset, cache_dir=d, cache_name="tiny")
            with patch.object(again, "encode", side_effect=AssertionError):
                torch.testing.assert_close(again(**batch)["embed"], expected)
        with self.assertRaises(ValueError):
            _model(self.dataset, cache_dir="x")

    def test_embedding_does_not_depend_on_batch(self):
        model = _model(self.dataset, memoize=False)
        batch = _batch(self.dataset)
        together = model(**batch)["embed"]
        for i in range(4):
            alone = model(events=[batch["events"][i]],
                          birth_datetime=[batch["birth_datetime"][i]])["embed"]
            torch.testing.assert_close(alone[0], together[i], rtol=1e-5, atol=1e-5)

    def test_last_token_matches_femr_packed_forward(self):
        model = _model(self.dataset, memoize=False)
        batch = _batch(self.dataset, 3)
        timelines = [model.tokenize(e, b) for e, b in
                     zip(batch["events"], batch["birth_datetime"])]
        with torch.no_grad():
            packed = _femr_forward(
                model.encoder,
                torch.tensor([i for t in timelines for i in t.ids]),
                torch.tensor([a for t in timelines for a in t.ages]),
                torch.tensor([a for t in timelines for a in t.extra]),
                [len(t.ids) for t in timelines])
        ends = torch.cumsum(torch.tensor([len(t.ids) for t in timelines]), 0) - 1
        torch.testing.assert_close(model(**batch)["embed"], packed[ends],
                                   rtol=1e-5, atol=1e-5)

    def test_truncation_keeps_most_recent_tokens_and_ages(self):
        model = _model(self.dataset, max_length=3)
        timeline = model.tokenize(_timeline(6), T0)
        self.assertEqual(timeline.ids, [6, 6, 6])
        self.assertEqual(timeline.ages, [3.0, 4.0, 5.0])

    def test_coverage(self):
        model = _model(self.dataset)
        model(**_batch(self.dataset))
        report = model.coverage_report()
        self.assertEqual((report["events"], report["unknown_events"]), (12, 1))
        self.assertEqual(report["top_unknown"], [("LOINC/9999-9", 1)])

    def test_checkpoint_config_must_set_required_keys(self):
        full = vars(_config())
        self.assertEqual(CLMBRConfig.from_dict(full, strict=True), _config())
        partial = {k: v for k, v in full.items() if k != "n_heads"}
        with self.assertRaisesRegex(ValueError, "n_heads"):
            CLMBRConfig.from_dict({"transformer_config": partial}, strict=True)
        # configs built in code may still rely on the defaults
        self.assertEqual(CLMBRConfig.from_dict(partial).n_heads, 12)

    def test_normed_ages_need_age_stats(self):
        bare = {k: v for k, v in DICTIONARY.items() if k != "age_stats"}
        with self.assertRaisesRegex(ValueError, "age_stats"):
            CLMBR(self.dataset, config=_config(use_normed_ages=True),
                  dictionary=bare, random_init=True)
        CLMBR(self.dataset, config=_config(), dictionary=bare, random_init=True)

    def test_dictionary_larger_than_embeddings_is_rejected(self):
        with self.assertRaises(ValueError):
            _model(self.dataset, config=_config(vocab_size=4))


if __name__ == "__main__":
    unittest.main()
