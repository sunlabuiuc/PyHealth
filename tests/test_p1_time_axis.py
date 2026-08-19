"""Proof that concatenated stays share one time origin.

The sample is patient-level: every admission up to the first death is one
sequence. Event times were hours from *that stay's* admit, then concatenated,
so stay 2 at +6h sorted with stay 1 at +6h. Collection is still per stay
(admit, admit+window]. Times are hours from the first stay in the sample.

Single-stay patients are unchanged. Time embeddings use geometric
wavelengths from 1 hour to 10 years so a later stay at +9606h does not
alias with +6h.

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/test_p1_time_axis.py -q
"""

from __future__ import annotations

import inspect
import unittest
from datetime import datetime, timedelta

import torch


class TestP1TimeAxis(unittest.TestCase):
    def test_hours_since_does_not_reset_per_stay(self):
        from pyhealth.tasks.multimodal_mimic4 import BaseMultimodalMIMIC4Task

        first = datetime(2180, 5, 6, 8, 0, 0)
        later = first + timedelta(days=400)
        stay1_event = first + timedelta(hours=6)
        stay2_event = later + timedelta(hours=6)

        self.assertEqual(BaseMultimodalMIMIC4Task._hours_since(stay1_event, first), 6.0)
        # Old convention: both events were 6.0 and collided after concat.
        self.assertEqual(BaseMultimodalMIMIC4Task._hours_since(stay2_event, later), 6.0)
        self.assertAlmostEqual(
            BaseMultimodalMIMIC4Task._hours_since(stay2_event, first),
            400 * 24 + 6.0,
        )
        self.assertGreater(
            BaseMultimodalMIMIC4Task._hours_since(stay2_event, first),
            BaseMultimodalMIMIC4Task._hours_since(stay1_event, first),
        )

    def test_collectors_write_hours_from_the_sample_origin(self):
        from pyhealth.tasks.multimodal_mimic4 import (
            BaseMultimodalMIMIC4Task,
            CXRMIMIC4,
            LabsMIMIC4,
            NotesLabsCXRMIMIC4,
            NotesLabsMIMIC4,
        )

        for cls in (LabsMIMIC4, NotesLabsMIMIC4, NotesLabsCXRMIMIC4, CXRMIMIC4):
            src = inspect.getsource(cls.__call__)
            self.assertIn(
                "time_origin = admissions_to_process[0].timestamp",
                src,
                msg=cls.__name__,
            )
            self.assertNotIn(
                "event.timestamp - admission_time",
                src,
                msg=f"{cls.__name__} still resets times per stay",
            )

        labs_src = inspect.getsource(BaseMultimodalMIMIC4Task._collect_labs)
        self.assertIn("time_origin", labs_src)
        self.assertIn("_hours_since", labs_src)
        self.assertNotIn("lab_ts - admission_time", labs_src)

    def test_sinusoid_does_not_wrap_every_thirty_days(self):
        from pyhealth.models.embedding.unified import SinusoidalTimeEmbedding

        emb = SinusoidalTimeEmbedding(dim=32)
        t6 = emb(torch.tensor([6.0]))
        t726 = emb(torch.tensor([6.0 + 720.0]))
        t9606 = emb(torch.tensor([9606.0]))
        self.assertFalse(torch.allclose(t6, t726, atol=1e-5))
        self.assertFalse(torch.allclose(t6, t9606, atol=1e-5))
