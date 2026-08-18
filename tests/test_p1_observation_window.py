"""Proof that every lab/CXR task honours a per-admission observation window.

Four task bodies computed a window from ``window_hours`` and then collected
labs through discharge. For a mortality label that reads the outcome.

Measured on MIMIC-IV ``labs_only``:

  collection through discharge  PR-AUC 0.6204  ROC 0.90
  window honoured               PR-AUC 0.2137  ROC 0.7136

A sweep over 24, 48, and 96 hours produced three identical datasets, because
``window_hours`` was inert. The window also anchored on the patient's first
admission globally, so a later stay received a span that had already closed.

CXR / ``notes_labs_cxr`` still skipped those later stays with
``admission_time >= first_admit + window_hours``. That skip is gone.
``emitted_data_version`` is 2 so caches from version 1 cannot be reused.

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/test_p1_observation_window.py -q
"""

from __future__ import annotations

import inspect
import json
import unittest
import uuid
from datetime import datetime, timedelta


LAB_TASKS = [
    "LabsMIMIC4",
    "ICDLabsMIMIC4",
    "NotesLabsMIMIC4",
    "NotesLabsCXRMIMIC4",
    "CXRMIMIC4",
]


class TestP1ObservationWindow(unittest.TestCase):
    def test_every_task_honours_a_24h_window(self):
        from pyhealth.tasks import multimodal_mimic4 as m

        admit = datetime(2180, 5, 6, 8, 0, 0)
        discharge = admit + timedelta(days=9)
        for name in LAB_TASKS:
            task = getattr(m, name)(window_hours=24)
            end = task._admission_window_end(admit, discharge)
            horizon = (end - admit).total_seconds() / 3600.0
            self.assertAlmostEqual(
                horizon,
                24.0,
                places=2,
                msg=f"{name} collects {horizon:.0f}h past admission",
            )
            self.assertLess(end, discharge)

    def test_window_is_anchored_per_admission(self):
        from pyhealth.tasks import multimodal_mimic4 as m

        first = datetime(2180, 5, 6, 8, 0, 0)
        later = first + timedelta(days=400)
        for name in LAB_TASKS:
            task = getattr(m, name)(window_hours=24)
            self.assertEqual(
                task._admission_window_end(first, first + timedelta(days=9)),
                first + timedelta(hours=24),
            )
            self.assertEqual(
                task._admission_window_end(first, first + timedelta(hours=6)),
                first + timedelta(hours=6),
            )
            end = task._admission_window_end(later, later + timedelta(days=5))
            self.assertEqual(end, later + timedelta(hours=24))
            self.assertGreater(end, later, msg=f"{name} expired before later stay")

    def test_cxr_arms_do_not_skip_later_stays_on_the_first_admit_clock(self):
        from pyhealth.tasks.multimodal_mimic4 import CXRMIMIC4, NotesLabsCXRMIMIC4

        for cls in (NotesLabsCXRMIMIC4, CXRMIMIC4):
            src = inspect.getsource(cls.__call__)
            self.assertNotIn(
                "admission_time >= effective_end",
                src,
                msg=f"{cls.__name__} still drops later stays against first admit + window",
            )

    def test_window_change_invalidates_the_cache(self):
        from pyhealth.tasks import multimodal_mimic4 as m

        task = m.LabsMIMIC4(window_hours=24)
        self.assertIsNotNone(vars(task).get("emitted_data_version"))
        self.assertGreaterEqual(task.emitted_data_version, 2)

        def cache_key(t, drop_version=False):
            v = dict(vars(t))
            if drop_version:
                v.pop("emitted_data_version", None)
            params = json.dumps(
                {
                    **v,
                    "input_schema": t.input_schema,
                    "output_schema": t.output_schema,
                },
                sort_keys=True,
                default=str,
            )
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, params))

        self.assertNotEqual(cache_key(task), cache_key(task, drop_version=True))

    def test_window_none_still_collects_through_discharge(self):
        from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4

        task = LabsMIMIC4(window_hours=None)
        admit = datetime(2180, 5, 6, 8, 0, 0)
        discharge = admit + timedelta(days=9)
        self.assertEqual(task._admission_window_end(admit, discharge), discharge)
