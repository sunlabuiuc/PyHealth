"""Tests for ConformalMetaAnalysisTask.

Uses tiny synthetic patient objects so tests run without requiring
a full dataset.
"""

import unittest
from datetime import datetime
from types import SimpleNamespace


def _make_patient(
    attr_dict: dict,
    patient_id: str = "trial_0",
    event_type: str = "test_table",
):
    """Build a minimal Patient-like object for task testing.

    ``get_events`` accepts an optional ``event_type`` kwarg so tests
    can exercise the task's event_type filter: the fake patient
    returns its single event only when the filter matches or is None.
    """
    event = SimpleNamespace(
        event_type=event_type,
        timestamp=datetime.now(),
        attr_dict=attr_dict,
    )

    def get_events(event_type=None):
        if event_type is None or event_type == event.event_type:
            return [event]
        return []

    patient = SimpleNamespace(
        patient_id=patient_id,
        get_events=get_events,
    )
    return patient


def _make_empty_patient(patient_id: str = "trial_0"):
    """Build a Patient-like object whose get_events returns []."""
    patient = SimpleNamespace(
        patient_id=patient_id,
        get_events=lambda event_type=None: [],
    )
    return patient


class TestConformalMetaAnalysisTask(unittest.TestCase):
    """Tests for ConformalMetaAnalysisTask."""

    def test_basic_extraction(self):
        """Task emits features and true_effect keys."""
        from pyhealth.tasks.conformal_meta_analysis import (
            ConformalMetaAnalysisTask,
        )

        task = ConformalMetaAnalysisTask()
        patient = _make_patient({
            "visit_id": "v1",
            "feat_a": "1.0",
            "feat_b": "2.0",
            "true_effect": "3.5",
        })
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["true_effect"], 3.5)
        self.assertEqual(len(samples[0]["features"]), 2)

    def test_split_filter_excludes_wrong_split(self):
        """split_value filter skips patients not in the requested split."""
        from pyhealth.tasks.conformal_meta_analysis import (
            ConformalMetaAnalysisTask,
        )

        task = ConformalMetaAnalysisTask(
            split_column="split",
            split_value="trusted",
        )
        trusted = _make_patient({
            "visit_id": "v1",
            "feat_a": "1.0",
            "true_effect": "2.0",
            "split": "trusted",
        })
        untrusted = _make_patient({
            "visit_id": "v2",
            "feat_a": "1.0",
            "true_effect": "2.0",
            "split": "untrusted",
        })
        self.assertEqual(len(task(trusted)), 1)
        self.assertEqual(len(task(untrusted)), 0)

    def test_explicit_feature_columns(self):
        """feature_columns parameter selects only requested features."""
        from pyhealth.tasks.conformal_meta_analysis import (
            ConformalMetaAnalysisTask,
        )

        task = ConformalMetaAnalysisTask(
            feature_columns=["feat_a", "feat_c"],
        )
        patient = _make_patient({
            "visit_id": "v1",
            "feat_a": "1.0",
            "feat_b": "2.0",
            "feat_c": "3.0",
            "true_effect": "4.0",
        })
        sample = task(patient)[0]
        self.assertEqual(sample["features"], [1.0, 3.0])

    def test_optional_cma_fields_carried_through(self):
        """observed_effect/variance/prior_mean columns are emitted when set."""
        from pyhealth.tasks.conformal_meta_analysis import (
            ConformalMetaAnalysisTask,
        )

        task = ConformalMetaAnalysisTask(
            observed_column="obs_col",
            variance_column="var_col",
            prior_column="prior_col",
        )
        patient = _make_patient({
            "visit_id": "v1",
            "feat_a": "1.0",
            "true_effect": "2.0",
            "obs_col": "1.9",
            "var_col": "0.1",
            "prior_col": "1.95",
        })
        sample = task(patient)[0]
        self.assertAlmostEqual(sample["observed_effect"], 1.9)
        self.assertAlmostEqual(sample["variance"], 0.1)
        self.assertAlmostEqual(sample["prior_mean"], 1.95)

    def test_missing_target_returns_empty(self):
        """Patient with no target returns an empty list."""
        from pyhealth.tasks.conformal_meta_analysis import (
            ConformalMetaAnalysisTask,
        )

        task = ConformalMetaAnalysisTask()
        patient = _make_patient({"visit_id": "v1", "feat_a": "1.0"})
        self.assertEqual(task(patient), [])

    def test_no_events_returns_empty(self):
        """Patient with zero events returns an empty list."""
        from pyhealth.tasks.conformal_meta_analysis import (
            ConformalMetaAnalysisTask,
        )

        task = ConformalMetaAnalysisTask()
        self.assertEqual(task(_make_empty_patient()), [])

    def test_unparseable_feature_falls_back_to_zero(self):
        """Non-numeric feature values are silently imputed as 0.0.

        This is the documented fallback behavior for the feature
        extraction path in ``__call__``. The target still needs to
        be parseable — a non-numeric target returns ``[]``.
        """
        from pyhealth.tasks.conformal_meta_analysis import (
            ConformalMetaAnalysisTask,
        )

        task = ConformalMetaAnalysisTask(
            feature_columns=["feat_a", "feat_b"],
        )
        patient = _make_patient({
            "visit_id": "v1",
            "feat_a": "1.5",
            "feat_b": "not a number",
            "true_effect": "2.0",
        })
        sample = task(patient)[0]
        self.assertEqual(sample["features"], [1.5, 0.0])

    def test_event_type_filter_matches(self):
        """event_type filter selects patients whose events match."""
        from pyhealth.tasks.conformal_meta_analysis import (
            ConformalMetaAnalysisTask,
        )

        task = ConformalMetaAnalysisTask(event_type="amiodarone_trials")
        matching = _make_patient(
            {"visit_id": "v1", "feat_a": "1.0", "true_effect": "2.0"},
            event_type="amiodarone_trials",
        )
        non_matching = _make_patient(
            {"visit_id": "v2", "feat_a": "1.0", "true_effect": "2.0"},
            event_type="pmlb_meta_analysis",
        )
        self.assertEqual(len(task(matching)), 1)
        self.assertEqual(len(task(non_matching)), 0)


if __name__ == "__main__":
    unittest.main()