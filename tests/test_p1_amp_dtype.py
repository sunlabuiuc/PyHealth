"""Proof that amp_dtype is validated instead of silently coerced to fp16.

The previous expression was ``bfloat16 if amp_dtype == "bf16" else float16``.
Any other spelling, including ``"bfloat16"``, selected fp16 with no message.
fp16 also constructs a GradScaler, so the silent path changed gradients.

Measured: ``resolve_amp_dtype("bfloat16")`` is bf16; ``"f16"`` raises.

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/test_p1_amp_dtype.py -q
"""

from __future__ import annotations

import unittest

import torch


class TestP1AmpDtype(unittest.TestCase):
    def test_known_spellings_map_to_the_named_dtype(self):
        from pyhealth.trainer import resolve_amp_dtype

        self.assertIs(resolve_amp_dtype("bf16"), torch.bfloat16)
        self.assertIs(resolve_amp_dtype("bfloat16"), torch.bfloat16)
        self.assertIs(resolve_amp_dtype("fp16"), torch.float16)
        self.assertIs(resolve_amp_dtype("float16"), torch.float16)

    def test_unknown_spelling_raises(self):
        from pyhealth.trainer import resolve_amp_dtype

        for bad in ("bfloat_16", "f16", "int8", ""):
            with self.assertRaises(ValueError):
                resolve_amp_dtype(bad)
