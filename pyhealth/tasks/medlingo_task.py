"""
MedLingo abbreviation expansion task for PyHealth 2.0.

Defines AbbreviationExpansionMedLingo, a BaseTask subclass that converts
each MedLingo patient (abbreviation) into a sample dict with:
    - question : contextualised fill-in-the-blank prompt
    - answer   : expected expansion string (output label)
"""

from typing import Dict, List, Type, Union

from pyhealth.tasks import BaseTask


class AbbreviationExpansionMedLingo(BaseTask):
    """Task for medical abbreviation expansion on the MedLingo dataset.

    Each abbreviation entry is converted into one sample containing the
    contextualised question and the expected answer. This task is compatible
    with text-based models and LLM evaluation pipelines.

    Attributes:
        task_name (str): Identifier for this task.
        input_schema (Dict): Maps input field names to their types.
        output_schema (Dict): Maps output field names to their types.

    Examples:
        >>> from pyhealth.datasets import MedLingoDataset
        >>> from pyhealth.tasks.medlingo_task import AbbreviationExpansionMedLingo
        >>> ds = MedLingoDataset(root="test-resources/MedLingo")
        >>> samples = ds.set_task(AbbreviationExpansionMedLingo())
        >>> samples[0]
        {'patient_id': 'PRN', 'visit_id': 'PRN', 'question': '...', 'answer': '...'}
    """

    task_name: str = "abbreviation_expansion_medlingo"
    input_schema: Dict[str, Union[str, Type]] = {"question": "text"}
    output_schema: Dict[str, Union[str, Type]] = {"answer": "text"}

    def __call__(self, patient) -> List[Dict]:
        """Convert a Patient object into a list of sample dicts.

        In PyHealth 2.0, ``patient.data_source`` is a Polars DataFrame
        containing all events for this patient. Column names are prefixed
        with the table name by ``load_table`` (e.g. ``questions/question``).

        Args:
            patient: A ``Patient`` object from ``MedLingoDataset``.

        Returns:
            List[Dict]: One sample per row, each with keys
            ``patient_id``, ``visit_id``, ``question``, and ``answer``.
        """
        samples: List[Dict] = []

        df = patient.data_source
        if df is None or len(df) == 0:
            return samples

        question_col = "questions/question"
        answer_col = "questions/answer"

        if question_col not in df.columns or answer_col not in df.columns:
            return samples

        for row in df.iter_rows(named=True):
            question = str(row.get(question_col, "") or "").strip()
            answer = str(row.get(answer_col, "") or "").strip()

            if not question or not answer:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": patient.patient_id,
                    "question": question,
                    "answer": answer,
                }
            )

        return samples


# Keep a functional alias for backward compatibility
def abbreviation_expansion_medlingo_fn(patient) -> List[Dict]:
    """Functional wrapper around AbbreviationExpansionMedLingo.__call__.

    Useful for quick ad-hoc use without instantiating the task class.
    Note: use AbbreviationExpansionMedLingo() with set_task() for the
    full PyHealth 2.0 pipeline.
    """
    return AbbreviationExpansionMedLingo()(patient)