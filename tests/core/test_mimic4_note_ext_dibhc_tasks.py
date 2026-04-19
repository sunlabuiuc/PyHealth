# -*- coding: utf-8 -*-
"""Unit tests for MIMIC-IV-Note task definitions (summarization and hallucination detection).

This module contains comprehensive test coverage for the two PyHealth task classes:
- BHCSummarizationTask: Generates patient-friendly summaries from Brief Hospital Course (BHC)
- HallucinationDetectionTask: Detects hallucinations in discharge instructions

All tests use synthetic patient data and mocking to avoid MIMIC data dependencies.
Tests verify correct task initialization, schema properties, and sample generation logic.

Covers Section 4.1 (summarization) and Section 4.7 (hallucination detection)
tasks from:

    Hegselmann et al. "A Data-Centric Approach To Generate Faithful and
    High Quality Patient Summaries with Large Language Models." CHIL 2024.
"""

import unittest
from unittest.mock import Mock

from pyhealth.tasks.mimic4_note_ext_dibhc_tasks import (
    BHCSummarizationTask,
    HallucinationDetectionTask,
)

# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

def _make_mock_event(
    brief_hospital_course: str = "Patient admitted for observation. "
    "Underwent thorough evaluation. Discharged in stable condition.",
    summary: str = "You were admitted to the hospital for evaluation and"
    " monitoring. During your stay, doctors performed comprehensive testing."
    " Your condition improved, and you are now ready to go home.",
    has_hallucination: int = -1,
) -> Mock:
    """Create a mock discharge event with required attributes.

    Args:
        brief_hospital_course (str): Text for the BHC section.
        summary (str): Text for the discharge summary section.
        has_hallucination (int): Hallucination label (-1, 0, or 1).

    Returns:
        Mock object with brief_hospital_course, summary, and has_hallucination attributes.
    """
    event = Mock()
    event.brief_hospital_course = brief_hospital_course
    event.summary = summary
    event.has_hallucination = has_hallucination
    return event


def _make_mock_visit(
    visit_id: str = "visit_001",
    events: list = None,
) -> Mock:
    """Create a mock visit with discharge events.

    Args:
        visit_id (str): Identifier for the visit.
        events (list): List of mock event objects for this visit.

    Returns:
        Mock object representing a visit with discharge events.
    """
    if events is None:
        events = [_make_mock_event()]

    visit = Mock()
    visit.visit_id = visit_id
    visit.get_event_list = Mock(return_value=events)
    return visit


def _make_mock_patient(
    patient_id: str = "patient_001",
    visits: dict = None,
) -> Mock:
    """Create a mock patient with multiple visits.

    Args:
        patient_id (str): Identifier for the patient.
        visits (dict): Dictionary of mock visit objects.

    Returns:
        Mock object representing a patient with visits.
    """
    if visits is None:
        visits = {
            "visit_001": _make_mock_visit(
                visit_id="visit_001",
                events=[_make_mock_event()],
            )
        }

    patient = Mock()
    patient.patient_id = patient_id
    patient.visits = visits
    return patient


# Convenience aliases for existing test data compatibility
def make_event(
    brief_hospital_course: str,
    summary: str,
    has_hallucination: int = -1,
) -> Mock:
    """Build a synthetic discharge note event."""
    return _make_mock_event(
        brief_hospital_course=brief_hospital_course,
        summary=summary,
        has_hallucination=has_hallucination,
    )


def make_visit(visit_id: str, events: list) -> Mock:
    """Build a synthetic visit containing discharge events."""
    return _make_mock_visit(visit_id=visit_id, events=events)


def make_patient(patient_id: str, visits: dict) -> Mock:
    """Build a synthetic patient with a dict of visits."""
    return _make_mock_patient(patient_id=patient_id, visits=visits)


# ========================================================================
# BHCSummarizationTask Tests (Section 4.1)
# ========================================================================

class TestBHCSummarizationTaskInitialization(unittest.TestCase):
    """Test suite for BHCSummarizationTask initialization and schema.

    Verifies that BHCSummarizationTask is properly initialized with correct
    task_name, input_schema, and output_schema attributes.
    """

    def setUp(self):
        print(f"\n{'='*60}")
        print("TEST CLASS: TestBHCSummarizationTaskInitialization")
        print(f"{'='*60}")

    def test_task_name_is_set_correctly(self) -> None:
        """Test that task_name attribute is set to expected value."""
        print("\nTEST: test_task_name_is_set_correctly")
        task = BHCSummarizationTask()
        self.assertEqual(task.task_name, "BHCSummarizationMIMIC4Note")
        print(f"  task_name: {task.task_name}")
        print("  ✓ passed")

    def test_input_schema_has_context_field(self) -> None:
        """Test that input_schema contains 'context' field of type 'str'."""
        print("\nTEST: test_input_schema_has_context_field")
        task = BHCSummarizationTask()
        self.assertIn("context", task.input_schema)
        self.assertEqual(task.input_schema["context"], "str")
        print(f"  input_schema: {task.input_schema}")
        print("  ✓ passed")

    def test_output_schema_has_summary_field(self) -> None:
        """Test that output_schema contains 'summary' field of type 'str'."""
        print("\nTEST: test_output_schema_has_summary_field")
        task = BHCSummarizationTask()
        self.assertIn("summary", task.output_schema)
        self.assertEqual(task.output_schema["summary"], "str")
        print(f"  output_schema: {task.output_schema}")
        print("  ✓ passed")

    def test_input_schema_only_has_context(self) -> None:
        """Test that input_schema contains only 'context' field."""
        print("\nTEST: test_input_schema_only_has_context")
        task = BHCSummarizationTask()
        self.assertEqual(len(task.input_schema), 1)
        print("  ✓ passed")

    def test_output_schema_only_has_summary(self) -> None:
        """Test that output_schema contains only 'summary' field."""
        print("\nTEST: test_output_schema_only_has_summary")
        task = BHCSummarizationTask()
        self.assertEqual(len(task.output_schema), 1)
        print("  ✓ passed")


class TestBHCSummarizationTaskCallMethod(unittest.TestCase):
    """Test suite for BHCSummarizationTask.__call__ method.

    Verifies that the task correctly processes patient objects and generates
    summarization samples with proper structure and content.
    """

    def setUp(self):
        print(f"\n{'='*60}")
        print("TEST CLASS: TestBHCSummarizationTaskCallMethod")
        print(f"{'='*60}")
        self.task = BHCSummarizationTask()

    def test_call_returns_list_of_samples(self) -> None:
        """Test that __call__ returns a list of sample dictionaries."""
        print("\nTEST: test_call_returns_list_of_samples")
        patient = _make_mock_patient()
        result = self.task(patient)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertIsInstance(result[0], dict)
        print(f"  Returned {len(result)} samples")
        print("  ✓ passed")

    def test_sample_contains_required_keys(self) -> None:
        """Test that each sample contains all required keys."""
        print("\nTEST: test_sample_contains_required_keys")
        patient = _make_mock_patient(
            patient_id="patient_123",
            visits={
                "visit_456": _make_mock_visit(
                    visit_id="visit_456",
                    events=[
                        _make_mock_event(
                            brief_hospital_course="Patient was hospitalized.",
                            summary="You were hospitalized.",
                        )
                    ],
                )
            },
        )
        result = self.task(patient)
        required_keys = {"patient_id", "visit_id", "context", "summary"}
        self.assertTrue(required_keys.issubset(result[0].keys()))
        print(f"  Sample keys: {list(result[0].keys())}")
        print("  ✓ passed")

    def test_sample_content_values_are_correct(self) -> None:
        """Test that sample contains correct patient_id, visit_id, and text values."""
        print("\nTEST: test_sample_content_values_are_correct")
        bhc_text = "Patient hospitalized for observation and treatment."
        summary_text = "You were treated during hospitalization."
        patient = _make_mock_patient(
            patient_id="patient_xyz",
            visits={
                "visit_abc": _make_mock_visit(
                    visit_id="visit_abc",
                    events=[
                        _make_mock_event(
                            brief_hospital_course=bhc_text,
                            summary=summary_text,
                        )
                    ],
                )
            },
        )
        result = self.task(patient)
        sample = result[0]
        self.assertEqual(sample["patient_id"], "patient_xyz")
        self.assertEqual(sample["visit_id"], "visit_abc")
        self.assertEqual(sample["context"], bhc_text)
        self.assertEqual(sample["summary"], summary_text)
        print(f"  patient_id: {sample['patient_id']}")
        print(f"  visit_id: {sample['visit_id']}")
        print("  ✓ passed")

    def test_skips_empty_context(self) -> None:
        """Test that samples with empty context are skipped."""
        print("\nTEST: test_skips_empty_context")
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[
                        _make_mock_event(
                            brief_hospital_course="",
                            summary="Valid summary text here.",
                        )
                    ],
                )
            },
        )
        result = self.task(patient)
        self.assertEqual(len(result), 0)
        print("  Empty context event was skipped")
        print("  ✓ passed")

    def test_skips_empty_summary(self) -> None:
        """Test that samples with empty summary are skipped."""
        print("\nTEST: test_skips_empty_summary")
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[
                        _make_mock_event(
                            brief_hospital_course="Valid hospital course.",
                            summary="",
                        )
                    ],
                )
            },
        )
        result = self.task(patient)
        self.assertEqual(len(result), 0)
        print("  Empty summary event was skipped")
        print("  ✓ passed")

    def test_skips_whitespace_only_context(self) -> None:
        """Test that samples with whitespace-only context are skipped."""
        print("\nTEST: test_skips_whitespace_only_context")
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[
                        _make_mock_event(
                            brief_hospital_course="   \n\t  ",
                            summary="Valid summary.",
                        )
                    ],
                )
            },
        )
        result = self.task(patient)
        self.assertEqual(len(result), 0)
        print("  Whitespace-only context event was skipped")
        print("  ✓ passed")

    def test_skips_whitespace_only_summary(self) -> None:
        """Test that samples with whitespace-only summary are skipped."""
        print("\nTEST: test_skips_whitespace_only_summary")
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[
                        _make_mock_event(
                            brief_hospital_course="Valid context.",
                            summary="   \n\t  ",
                        )
                    ],
                )
            },
        )
        result = self.task(patient)
        self.assertEqual(len(result), 0)
        print("  Whitespace-only summary event was skipped")
        print("  ✓ passed")

    def test_strips_leading_trailing_whitespace(self) -> None:
        """Test that leading/trailing whitespace is stripped from context and summary."""
        print("\nTEST: test_strips_leading_trailing_whitespace")
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[
                        _make_mock_event(
                            brief_hospital_course="  \n  BHC text  \n  ",
                            summary="  \n  Summary text  \n  ",
                        )
                    ],
                )
            },
        )
        result = self.task(patient)
        sample = result[0]
        self.assertEqual(sample["context"], "BHC text")
        self.assertEqual(sample["summary"], "Summary text")
        print("  Whitespace was properly stripped")
        print("  ✓ passed")

    def test_handles_multiple_visits(self) -> None:
        """Test that task processes multiple visits for a single patient."""
        print("\nTEST: test_handles_multiple_visits")
        patient = _make_mock_patient(
            patient_id="multi_visit_patient",
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[_make_mock_event()],
                ),
                "visit_002": _make_mock_visit(
                    visit_id="visit_002",
                    events=[_make_mock_event()],
                ),
                "visit_003": _make_mock_visit(
                    visit_id="visit_003",
                    events=[_make_mock_event()],
                ),
            },
        )
        result = self.task(patient)
        self.assertEqual(len(result), 3)
        visit_ids = {sample["visit_id"] for sample in result}
        self.assertEqual(visit_ids, {"visit_001", "visit_002", "visit_003"})
        print(f"  Processed {len(result)} visits")
        print("  ✓ passed")

    def test_handles_multiple_events_per_visit(self) -> None:
        """Test that task processes multiple events within a single visit."""
        print("\nTEST: test_handles_multiple_events_per_visit")
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[
                        _make_mock_event(brief_hospital_course="BHC1", summary="Summary1"),
                        _make_mock_event(brief_hospital_course="BHC2", summary="Summary2"),
                        _make_mock_event(brief_hospital_course="BHC3", summary="Summary3"),
                    ],
                )
            },
        )
        result = self.task(patient)
        self.assertEqual(len(result), 3)
        contexts = {sample["context"] for sample in result}
        self.assertEqual(contexts, {"BHC1", "BHC2", "BHC3"})
        print(f"  Processed {len(result)} events")
        print("  ✓ passed")

    def test_handles_missing_attributes(self) -> None:
        """Test that task handles events with missing BHC or summary attributes."""
        print("\nTEST: test_handles_missing_attributes")
        event_no_bhc = Mock()
        event_no_bhc.brief_hospital_course = ""  # Explicitly set to empty
        event_no_bhc.summary = "Has summary but no BHC"
        # Simulate missing brief_hospital_course by setting it to empty

        event_no_summary = Mock()
        event_no_summary.brief_hospital_course = "Has BHC but no summary"
        event_no_summary.summary = ""  # Explicitly set to empty
        # Simulate missing summary by setting it to empty

        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[event_no_bhc, event_no_summary],
                )
            },
        )
        result = self.task(patient)
        # Both should be skipped due to empty attributes
        self.assertEqual(len(result), 0)
        print("  Events with empty attributes were skipped")
        print("  ✓ passed")


# ========================================================================
# HallucinationDetectionTask Tests (Section 4.7)
# ========================================================================

class TestHallucinationDetectionTaskInitialization(unittest.TestCase):
    """Test suite for HallucinationDetectionTask initialization and schema.

    Verifies that HallucinationDetectionTask is properly initialized with correct
    task_name, input_schema, output_schema, and default_label attributes.
    """

    def setUp(self):
        print(f"\n{'='*60}")
        print("TEST CLASS: TestHallucinationDetectionTaskInitialization")
        print(f"{'='*60}")

    def test_task_name_is_set_correctly(self) -> None:
        """Test that task_name attribute is set to expected value."""
        print("\nTEST: test_task_name_is_set_correctly")
        task = HallucinationDetectionTask()
        self.assertEqual(task.task_name, "HallucinationDetectionMIMIC4Note")
        print(f"  task_name: {task.task_name}")
        print("  ✓ passed")

    def test_input_schema_has_context_and_summary(self) -> None:
        """Test that input_schema contains 'context' and 'summary' fields."""
        print("\nTEST: test_input_schema_has_context_and_summary")
        task = HallucinationDetectionTask()
        self.assertIn("context", task.input_schema)
        self.assertIn("summary", task.input_schema)
        self.assertEqual(task.input_schema["context"], "str")
        self.assertEqual(task.input_schema["summary"], "str")
        print(f"  input_schema: {task.input_schema}")
        print("  ✓ passed")

    def test_output_schema_has_label_field(self) -> None:
        """Test that output_schema contains 'label' field of type 'binary'."""
        print("\nTEST: test_output_schema_has_label_field")
        task = HallucinationDetectionTask()
        self.assertIn("label", task.output_schema)
        self.assertEqual(task.output_schema["label"], "binary")
        print(f"  output_schema: {task.output_schema}")
        print("  ✓ passed")

    def test_default_label_is_minus_one_by_default(self) -> None:
        """Test that default_label is -1 when not specified."""
        print("\nTEST: test_default_label_is_minus_one_by_default")
        task = HallucinationDetectionTask()
        self.assertEqual(task.default_label, -1)
        print(f"  default_label: {task.default_label}")
        print("  ✓ passed")

    def test_default_label_can_be_customized(self) -> None:
        """Test that default_label can be set via constructor."""
        print("\nTEST: test_default_label_can_be_customized")
        task = HallucinationDetectionTask(default_label=0)
        self.assertEqual(task.default_label, 0)
        print(f"  custom default_label: {task.default_label}")
        print("  ✓ passed")

    def test_custom_default_labels(self) -> None:
        """Test various custom default_label values."""
        print("\nTEST: test_custom_default_labels")
        for label in [-1, 0, 1]:
            task = HallucinationDetectionTask(default_label=label)
            self.assertEqual(task.default_label, label)
        print("  All custom labels set correctly")
        print("  ✓ passed")


class TestHallucinationDetectionTaskCallMethod(unittest.TestCase):
    """Test suite for HallucinationDetectionTask.__call__ method.

    Verifies that the task correctly processes patient objects and generates
    hallucination detection samples with proper structure and label values.
    """

    def setUp(self):
        print(f"\n{'='*60}")
        print("TEST CLASS: TestHallucinationDetectionTaskCallMethod")
        print(f"{'='*60}")
        self.task = HallucinationDetectionTask()

    def test_call_returns_list_of_samples(self) -> None:
        """Test that __call__ returns a list of sample dictionaries."""
        print("\nTEST: test_call_returns_list_of_samples")
        patient = _make_mock_patient()
        result = self.task(patient)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertIsInstance(result[0], dict)
        print(f"  Returned {len(result)} samples")
        print("  ✓ passed")

    def test_sample_contains_required_keys(self) -> None:
        """Test that each sample contains all required keys."""
        print("\nTEST: test_sample_contains_required_keys")
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[_make_mock_event()],
                )
            },
        )
        result = self.task(patient)
        required_keys = {"patient_id", "visit_id", "context", "summary", "label"}
        self.assertTrue(required_keys.issubset(result[0].keys()))
        print(f"  Sample keys: {list(result[0].keys())}")
        print("  ✓ passed")

    def test_sample_content_values_are_correct(self) -> None:
        """Test that sample contains correct patient_id, visit_id, and text values."""
        print("\nTEST: test_sample_content_values_are_correct")
        bhc_text = "Patient hospitalized for treatment."
        summary_text = "You were treated in the hospital."
        patient = _make_mock_patient(
            patient_id="patient_xyz",
            visits={
                "visit_abc": _make_mock_visit(
                    visit_id="visit_abc",
                    events=[
                        _make_mock_event(
                            brief_hospital_course=bhc_text,
                            summary=summary_text,
                            has_hallucination=-1,
                        )
                    ],
                )
            },
        )
        result = self.task(patient)
        sample = result[0]
        self.assertEqual(sample["patient_id"], "patient_xyz")
        self.assertEqual(sample["visit_id"], "visit_abc")
        self.assertEqual(sample["context"], bhc_text)
        self.assertEqual(sample["summary"], summary_text)
        print(f"  patient_id: {sample['patient_id']}")
        print(f"  visit_id: {sample['visit_id']}")
        print("  ✓ passed")

    def test_label_zero_for_no_hallucination(self) -> None:
        """Test that label is 0 when has_hallucination is 0."""
        print("\nTEST: test_label_zero_for_no_hallucination")
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[_make_mock_event(has_hallucination=0)],
                )
            },
        )
        result = self.task(patient)
        self.assertEqual(result[0]["label"], 0)
        print("  Label correctly set to 0 for faithful summary")
        print("  ✓ passed")

    def test_label_one_for_hallucination_present(self) -> None:
        """Test that label is 1 when has_hallucination is 1."""
        print("\nTEST: test_label_one_for_hallucination_present")
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[_make_mock_event(has_hallucination=1)],
                )
            },
        )
        result = self.task(patient)
        self.assertEqual(result[0]["label"], 1)
        print("  Label correctly set to 1 for hallucination present")
        print("  ✓ passed")

    def test_label_default_when_annotation_unavailable(self) -> None:
        """Test that label uses default_label when annotation is unavailable."""
        print("\nTEST: test_label_default_when_annotation_unavailable")
        task = HallucinationDetectionTask(default_label=-1)
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[_make_mock_event(has_hallucination=-1)],
                )
            },
        )
        result = task(patient)
        self.assertEqual(result[0]["label"], -1)
        print("  Label correctly set to default_label (-1)")
        print("  ✓ passed")

    def test_custom_default_label_zero(self) -> None:
        """Test that custom default_label=0 is used when annotation unavailable."""
        print("\nTEST: test_custom_default_label_zero")
        task = HallucinationDetectionTask(default_label=0)
        
        # Use a simple class to avoid Mock auto-creating undefined attributes
        class SimpleEvent:
            def __init__(self):
                self.brief_hospital_course = "Patient was treated."
                self.summary = "You were treated."
                # has_hallucination is intentionally not set
        
        event = SimpleEvent()
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[event],
                )
            },
        )
        result = task(patient)
        self.assertEqual(result[0]["label"], 0)
        print("  Custom default_label (0) correctly applied")
        print("  ✓ passed")

    def test_skips_empty_context(self) -> None:
        """Test that samples with empty context are skipped."""
        print("\nTEST: test_skips_empty_context")
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[
                        _make_mock_event(
                            brief_hospital_course="",
                            summary="Valid summary.",
                        )
                    ],
                )
            },
        )
        result = self.task(patient)
        self.assertEqual(len(result), 0)
        print("  Empty context event was skipped")
        print("  ✓ passed")

    def test_skips_empty_summary(self) -> None:
        """Test that samples with empty summary are skipped."""
        print("\nTEST: test_skips_empty_summary")
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[
                        _make_mock_event(
                            brief_hospital_course="Valid context.",
                            summary="",
                        )
                    ],
                )
            },
        )
        result = self.task(patient)
        self.assertEqual(len(result), 0)
        print("  Empty summary event was skipped")
        print("  ✓ passed")

    def test_skips_both_empty_context_and_summary(self) -> None:
        """Test that samples with both empty context and summary are skipped."""
        print("\nTEST: test_skips_both_empty_context_and_summary")
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[
                        _make_mock_event(
                            brief_hospital_course="",
                            summary="",
                        )
                    ],
                )
            },
        )
        result = self.task(patient)
        self.assertEqual(len(result), 0)
        print("  Event with both empty context and summary was skipped")
        print("  ✓ passed")

    def test_strips_leading_trailing_whitespace(self) -> None:
        """Test that leading/trailing whitespace is stripped."""
        print("\nTEST: test_strips_leading_trailing_whitespace")
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[
                        _make_mock_event(
                            brief_hospital_course="  \n  BHC text  \n  ",
                            summary="  \n  Summary text  \n  ",
                        )
                    ],
                )
            },
        )
        result = self.task(patient)
        sample = result[0]
        self.assertEqual(sample["context"], "BHC text")
        self.assertEqual(sample["summary"], "Summary text")
        print("  Whitespace was properly stripped")
        print("  ✓ passed")

    def test_handles_multiple_visits(self) -> None:
        """Test that task processes multiple visits for a single patient."""
        print("\nTEST: test_handles_multiple_visits")
        patient = _make_mock_patient(
            patient_id="multi_visit_patient",
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[_make_mock_event(has_hallucination=0)],
                ),
                "visit_002": _make_mock_visit(
                    visit_id="visit_002",
                    events=[_make_mock_event(has_hallucination=1)],
                ),
                "visit_003": _make_mock_visit(
                    visit_id="visit_003",
                    events=[_make_mock_event(has_hallucination=-1)],
                ),
            },
        )
        result = self.task(patient)
        self.assertEqual(len(result), 3)
        labels = {sample["label"] for sample in result}
        self.assertEqual(labels, {0, 1, -1})
        visit_ids = {sample["visit_id"] for sample in result}
        self.assertEqual(visit_ids, {"visit_001", "visit_002", "visit_003"})
        print(f"  Processed {len(result)} visits with correct labels")
        print("  ✓ passed")

    def test_handles_multiple_events_per_visit(self) -> None:
        """Test that task processes multiple events within a single visit."""
        print("\nTEST: test_handles_multiple_events_per_visit")
        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[
                        _make_mock_event(
                            brief_hospital_course="BHC1",
                            summary="Summary1",
                            has_hallucination=0,
                        ),
                        _make_mock_event(
                            brief_hospital_course="BHC2",
                            summary="Summary2",
                            has_hallucination=1,
                        ),
                        _make_mock_event(
                            brief_hospital_course="BHC3",
                            summary="Summary3",
                            has_hallucination=-1,
                        ),
                    ],
                )
            },
        )
        result = self.task(patient)
        self.assertEqual(len(result), 3)
        labels = {sample["label"] for sample in result}
        self.assertEqual(labels, {0, 1, -1})
        print(f"  Processed {len(result)} events with correct labels")
        print("  ✓ passed")

    def test_handles_missing_attributes(self) -> None:
        """Test that task handles events with missing attributes gracefully."""
        print("\nTEST: test_handles_missing_attributes")
        event_no_bhc = Mock()
        event_no_bhc.brief_hospital_course = ""  # Explicitly set to empty
        event_no_bhc.summary = "Has summary but no BHC"
        event_no_bhc.has_hallucination = -1

        event_no_summary = Mock()
        event_no_summary.brief_hospital_course = "Has BHC but no summary"
        event_no_summary.summary = ""  # Explicitly set to empty
        event_no_summary.has_hallucination = -1

        patient = _make_mock_patient(
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[event_no_bhc, event_no_summary],
                )
            },
        )
        result = self.task(patient)
        self.assertEqual(len(result), 0)
        print("  Events with empty attributes were skipped")
        print("  ✓ passed")


# ========================================================================
# Integration Tests
# ========================================================================

class TestTaskIntegration(unittest.TestCase):
    """Integration tests for both summarization and hallucination detection tasks.

    Tests interactions between the two task classes and ensures they work
    correctly with the same patient data.
    """

    def setUp(self):
        print(f"\n{'='*60}")
        print("TEST CLASS: TestTaskIntegration")
        print(f"{'='*60}")

    def test_both_tasks_process_same_patient(self) -> None:
        """Test that both tasks can process the same patient object."""
        print("\nTEST: test_both_tasks_process_same_patient")
        summ_task = BHCSummarizationTask()
        halluc_task = HallucinationDetectionTask()

        patient = _make_mock_patient(
            patient_id="patient_001",
            visits={
                "visit_001": _make_mock_visit(
                    visit_id="visit_001",
                    events=[
                        _make_mock_event(
                            brief_hospital_course="Admitted for chest pain.",
                            summary="Treated for chest pain.",
                            has_hallucination=0,
                        )
                    ],
                )
            },
        )

        summ_samples = summ_task(patient)
        halluc_samples = halluc_task(patient)

        self.assertEqual(len(summ_samples), 1)
        self.assertEqual(len(halluc_samples), 1)
        self.assertEqual(summ_samples[0]["patient_id"], halluc_samples[0]["patient_id"])
        print("  Both tasks successfully processed the same patient")
        print("  ✓ passed")

    def test_different_tasks_have_different_schemas(self) -> None:
        """Test that the two tasks have different input/output schemas."""
        print("\nTEST: test_different_tasks_have_different_schemas")
        summ_task = BHCSummarizationTask()
        halluc_task = HallucinationDetectionTask()

        # Summarization task has only context as input
        self.assertEqual(len(summ_task.input_schema), 1)
        self.assertIn("context", summ_task.input_schema)

        # Hallucination task has context and summary as inputs
        self.assertEqual(len(halluc_task.input_schema), 2)
        self.assertIn("context", halluc_task.input_schema)
        self.assertIn("summary", halluc_task.input_schema)

        # Summarization output is summary (str)
        self.assertEqual(summ_task.output_schema["summary"], "str")

        # Hallucination output is label (binary)
        self.assertEqual(halluc_task.output_schema["label"], "binary")

        print("  Schemas correctly differ between tasks")
        print("  ✓ passed")


if __name__ == "__main__":
    unittest.main()
