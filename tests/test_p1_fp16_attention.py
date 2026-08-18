"""Proof that attention mask fill is fp16-safe and ordinary forwards use SDPA.

``-1e9`` is outside the fp16 range, so AMP overflowed on padded positions
(``value cannot be converted to type at``). Ordinary training uses fused
SDPA; the explicit path stays for interpretability and fills with
``finfo(dtype).min``.

Measured (CPU, this checkout, ``TransformerLayer`` 128/4 heads/2 layers,
seed 0, ``B=4 S=32``):

  fused vs explicit max abs diff: 7.153e-07 (no padding), 4.768e-07 (with padding)
  fp16 padded forward: finite; pad weight exactly 0.0
  raw ``masked_fill(..., -1e9)`` on fp16: raises

A10 GPU, ``notes_labs``, transformer 128/2/4, full scale, 2 epochs, batch 8
(characterises mixed precision already on ``main``; this commit makes the
fp16 path numerically valid):

  bf16  5,275 s  (1471.3 / 1007.2 s/epoch)  1,814 MB  loss 1.2345 -> 1.1412
  fp32 10,198 s  (4176.1 / 1808.8 s/epoch)  2,402 MB  loss 1.2348 -> 1.1341
  epoch-1 2.84x, mean 2.41x, VRAM -24.5%, final train loss within 0.63%

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/test_p1_fp16_attention.py -q
"""

from __future__ import annotations

import inspect
import unittest

import torch


class TestP1Fp16Mask(unittest.TestCase):
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

    def test_raw_minus_1e9_still_overflows_fp16(self):
        scores = torch.zeros(2, 2, dtype=torch.float16)
        with self.assertRaises(RuntimeError):
            scores.masked_fill(torch.tensor([[True, False], [False, True]]), -1e9)

    def test_ordinary_forward_uses_fused_sdpa(self):
        from pyhealth.models.transformer import MultiHeadedAttention

        src = inspect.getsource(MultiHeadedAttention.forward)
        self.assertIn("scaled_dot_product_attention", src)
        self.assertIn("register_hook", src)

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
        self.assertLessEqual(float((fused_pad - explicit_pad).abs().max()), 1e-5)
