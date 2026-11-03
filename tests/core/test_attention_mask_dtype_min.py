"""Attention mask fill is fp16-safe and ordinary forwards use fused SDPA.

``-1e9`` is outside the fp16 range, so padding an fp16 forward raised
``value cannot be converted to type ... without overflow``. Ordinary
training forwards now use fused ``scaled_dot_product_attention``; the
explicit path stays behind ``register_hook=True`` for interpretability
and fills with ``finfo(dtype).min``.

A fully-masked query row attends to nothing, which is NaN or garbage
depending on backend and dtype. The fused path lets such rows attend
uniformly and then zeroes their outputs, so an all-padded sample (reachable
once empty modalities become zero-length fields) stays finite.

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/core/test_attention_mask_dtype_min.py -q
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch


class TestAttentionMaskDtypeMin(unittest.TestCase):
    def test_fp16_attention_mask_fill_does_not_overflow(self):
        from pyhealth.models.transformer import Attention

        attn = Attention()
        q = torch.zeros(1, 1, 2, 4, dtype=torch.float16)
        k = torch.zeros(1, 1, 2, 4, dtype=torch.float16)
        v = torch.ones(1, 1, 2, 4, dtype=torch.float16)
        mask = torch.tensor([[[[1, 0], [1, 0]]]], dtype=torch.float16)
        out, weights = attn(q, k, v, mask=mask)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(float(weights[0, 0, 0, 1]), 0.0)

    def test_explicit_path_finite_for_all_padded_row(self):
        from pyhealth.models.transformer import Attention

        attn = Attention()
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        mask = torch.ones(1, 1, 4, 4)
        mask[:, :, 0, :] = 0  # first query row fully masked
        out, weights = attn(q, k, v, mask=mask)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(weights).all())

    def test_raw_minus_1e9_still_overflows_fp16(self):
        scores = torch.zeros(2, 2, dtype=torch.float16)
        with self.assertRaises(RuntimeError):
            scores.masked_fill(torch.tensor([[True, False], [False, True]]), -1e9)

    def test_ordinary_forward_uses_fused_sdpa(self):
        from pyhealth.models import transformer as transformer_module

        torch.manual_seed(0)
        mha = transformer_module.MultiHeadedAttention(h=2, d_model=8).eval()
        q = k = v = torch.zeros(2, 5, 8)
        mask = torch.ones(2, 5)
        real_sdpa = transformer_module.F.scaled_dot_product_attention
        with mock.patch.object(
            transformer_module.F,
            "scaled_dot_product_attention",
            wraps=real_sdpa,
        ) as sdpa:
            mha(q, k, v, mask=mask, register_hook=False)
            self.assertEqual(sdpa.call_count, 1)
        with mock.patch.object(
            transformer_module.F,
            "scaled_dot_product_attention",
            wraps=real_sdpa,
        ) as sdpa:
            mha(q, k, v, mask=mask, register_hook=True)
            sdpa.assert_not_called()

    def test_fused_and_explicit_paths_agree(self):
        from pyhealth.models.transformer import TransformerLayer

        torch.manual_seed(0)
        layer = TransformerLayer(feature_size=128, heads=4, num_layers=2).eval()
        x = torch.randn(4, 32, 128, requires_grad=True)
        mask = torch.ones(4, 32)
        fused, _ = layer(x, mask, register_hook=False)
        explicit, _ = layer(x, mask, register_hook=True)
        self.assertLessEqual(float((fused - explicit).abs().max()), 1e-5)

        mask_pad = torch.cat([torch.ones(4, 20), torch.zeros(4, 12)], dim=1)
        fused_pad, _ = layer(x, mask_pad, register_hook=False)
        explicit_pad, _ = layer(x, mask_pad, register_hook=True)
        # Padded query outputs are discarded downstream (the feed-forward
        # zeroes them), so the paths must agree at valid positions.
        self.assertLessEqual(
            float((fused_pad[:, :20] - explicit_pad[:, :20]).abs().max()), 1e-5
        )

    def test_all_padded_sample_stays_finite(self):
        from pyhealth.models.transformer import MultiHeadedAttention

        torch.manual_seed(0)
        for mask_2d in (True, False):
            mha = MultiHeadedAttention(h=2, d_model=8).eval()
            q = k = v = torch.randn(3, 6, 8)
            valid = torch.ones(3, 6)
            valid[0, :] = 0  # one all-padded sample
            mask = valid if mask_2d else torch.einsum("ab,ac->abc", valid, valid)
            out = mha(q, k, v, mask=mask, register_hook=False)
            self.assertTrue(torch.isfinite(out).all())
            # Padded query rows produce exact zeros in the fused path.
            self.assertTrue(torch.equal(out[0], torch.zeros_like(out[0])))
            self.assertFalse(torch.equal(out[1], torch.zeros_like(out[1])))

    def test_all_padded_sample_layer_forward(self):
        from pyhealth.models.transformer import TransformerLayer

        torch.manual_seed(0)
        layer = TransformerLayer(feature_size=64, heads=4, num_layers=2).eval()
        x = torch.randn(3, 16, 64)
        x[0] = 0.0  # all-padded sample carries no signal
        mask = torch.ones(3, 16)
        mask[0, :] = 0
        fused, _ = layer(x, mask, register_hook=False)
        explicit, _ = layer(x, mask, register_hook=True)
        self.assertTrue(torch.isfinite(fused).all())
        self.assertTrue(torch.isfinite(explicit).all())
        self.assertTrue(torch.equal(fused[0], torch.zeros_like(fused[0])))
