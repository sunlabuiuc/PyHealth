"""Proof that unified MLP threads collate pad_mask like the other heads.

Without this, ``--model mlp`` is missing from the six-backbone table, and a
unified MLP would take the ``mask is None`` branch and score padded slots.

Measured: a 3-event + 1-event lab batch collates to pad_mask
``[[True, True, True], [True, False, False]]``, and MLP copies it into
``inputs["labs"]["pad_mask"]``.

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/test_p2_mlp_pad_mask.py -q
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch


class TestP2MlpPadMask(unittest.TestCase):
    def test_mlp_threads_pad_mask_into_unified_inputs(self):
        from pyhealth.datasets.utils import PAD_MASK_SUFFIX, collate_fn_dict_with_padding
        from pyhealth.models.mlp import MLP
        from pyhealth.processors.stagenet_processor import StageNetTensorProcessor

        batch = [
            {"labs": (torch.tensor([6.0, 12.0, 24.0]), torch.ones(3, 2))},
            {"labs": (torch.tensor([6.0]), torch.ones(1, 2))},
        ]
        collated = collate_fn_dict_with_padding(batch)
        self.assertIn(f"labs{PAD_MASK_SUFFIX}", collated)

        host = SimpleNamespace(
            feature_keys=["labs"],
            device="cpu",
            dataset=SimpleNamespace(
                input_processors={"labs": StageNetTensorProcessor()}
            ),
        )
        inputs = MLP._build_unified_inputs(host, collated)
        self.assertEqual(
            inputs["labs"]["pad_mask"].tolist(),
            [[True, True, True], [True, False, False]],
        )
