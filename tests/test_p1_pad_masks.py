"""Proof that batch padding is recorded and skipped by the unified path.

The collator padded short samples with 0.0 and nothing recorded that padding,
so padded slots looked like real measurements at admission time. RNN packed
lengths of 0 also crash once a correct mask exists.

Measured:

  ``collate_temporal`` had zero callers; the dataloader uses
  ``collate_fn_dict_with_padding``. First ``pad_mask`` on the unused collator
  never reached a model.
  Token-budget notes then crashed:
    ``RuntimeError: The size of tensor a (7) must match the size of tensor b (14)``
    (``pad_sequence`` only pads dim 0).
  Reusing event ``pad_mask`` as the BERT token mask then crashed:
    ``RuntimeError: shape '[96, 512]' is invalid for input of size 96``.
  After the collator emits ``{field}__pad_mask``, a 3-event + 1-event note
  batch is ``(2, 3, 4)`` with mask ``[[True, True, True], [True, False, False]]``.
  An all-pad RNN step stays finite (lengths clamped at 1).

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/test_p1_pad_masks.py -q
"""

from __future__ import annotations

import unittest

import torch


class TestP1RnnClamp(unittest.TestCase):
    def test_all_pad_mask_forward_is_finite(self):
        from pyhealth.models.rnn import RNNLayer

        layer = RNNLayer(input_size=4, hidden_size=8, dropout=0.0).eval()
        x = torch.zeros(2, 5, 4)
        mask = torch.zeros(2, 5)
        with torch.no_grad():
            outputs, last = layer(x, mask)
        self.assertEqual(outputs.shape[0], 2)
        self.assertEqual(tuple(last.shape), (2, 8))
        self.assertTrue(torch.isfinite(last).all())


class TestP1BertPadSkip(unittest.TestCase):
    def test_collate_emits_false_pad_mask_on_empty_note_slots(self):
        from pyhealth.datasets.utils import PAD_MASK_SUFFIX, collate_fn_dict_with_padding

        batch = [
            {"notes": (torch.ones(3, 4, dtype=torch.long), torch.ones(3, 4))},
            {"notes": (torch.ones(1, 4, dtype=torch.long), torch.ones(1, 4))},
        ]
        collated = collate_fn_dict_with_padding(batch)
        ids = collated["notes"][0]
        pad = collated[f"notes{PAD_MASK_SUFFIX}"]
        self.assertEqual(tuple(ids.shape), (2, 3, 4))
        self.assertEqual(pad.tolist(), [[True, True, True], [True, False, False]])

    def test_padded_events_sort_last_and_are_zeroed(self):
        from pyhealth.models.embedding.unified import UnifiedMultimodalEmbeddingModel
        from pyhealth.processors.stagenet_processor import StageNetTensorProcessor

        proc = StageNetTensorProcessor()
        proc.fit([{"labs": ([0.0], [[1.0, 2.0]])}], "labs")
        model = UnifiedMultimodalEmbeddingModel(
            processors={"labs": proc},
            embedding_dim=8,
            normalize_content=True,
        )
        model.eval()
        value = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
        time = torch.tensor([[6.0, 0.0]])
        pad_mask = torch.tensor([[True, False]])
        with torch.no_grad():
            out = model({"labs": {"value": value, "time": time, "pad_mask": pad_mask}})
        self.assertEqual(out["mask"].tolist(), [[1.0, 0.0]])
        self.assertAlmostEqual(out["time"][0, 0].item(), 6.0)
        self.assertTrue(torch.allclose(out["sequence"][0, 1], torch.zeros(8), atol=1e-6))

    def test_unified_heads_thread_pad_mask(self):
        import inspect

        from pyhealth.models.bottleneck_transformer import BottleneckTransformer
        from pyhealth.models.ehrmamba import EHRMamba
        from pyhealth.models.jamba_ehr import JambaEHR
        from pyhealth.models.rnn import RNN
        from pyhealth.models.transformer import Transformer

        for cls in (RNN, Transformer, BottleneckTransformer, EHRMamba, JambaEHR):
            src = inspect.getsource(cls._build_unified_inputs)
            self.assertIn("PAD_MASK_SUFFIX", src, msg=cls.__name__)
            self.assertIn("pad_mask", src, msg=cls.__name__)
