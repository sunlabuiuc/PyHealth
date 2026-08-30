"""EHR sequence-generation tasks for PyHealth generative models.

This is the shared task for every generator in
:mod:`pyhealth.models.generators` (HALO, MedGAN, CorGAN, PromptEHR, ...). It
extracts, for each patient, the ordered list of visits where each visit is the
list of medical codes recorded in that admission. The single input feature
``visits`` is processed by :class:`~pyhealth.processors.NestedSequenceProcessor`;
there is no prediction label, so ``output_schema`` is empty.

:class:`EHRGeneration` holds all the extraction logic; dataset-specific
subclasses only declare which event type and code attribute to read.

Evaluating generated data
-------------------------
The privacy/utility metrics in :mod:`pyhealth.metrics.generative` (``utils.py``,
``privacy.py``, ``utility.py`` -- exposed through ``evaluate_synthetic_ehr``)
consume **long-form** dataframes: one row per ``(patient, visit, code)`` with
columns ``id`` / ``time`` / ``visit_codes`` / ``labels``. ``id`` is the patient
identifier, ``time`` the (integer) visit index, ``visit_codes`` a single code
string, and ``labels`` a patient-level binary label (reduced via ``max`` over
the patient's rows).

Both the real task samples and a generator's ``generate()`` output use the same
``{"visits": [[code, ...], ...]}`` record shape, so
:func:`to_evaluation_dataframe` converts either into that long-form table. A
processed ``SampleDataset`` can be turned back into records with
:func:`decode_dataset`. Subjects are renumbered sequentially (0, 1, 2, ...) in
the ``id`` column -- synthetic patients do not correspond to real ones, so any
original ``patient_id`` is ignored.

.. code-block:: python

    from pyhealth.tasks.generate_ehr import decode_dataset, to_evaluation_dataframe
    from pyhealth.metrics.generative import evaluate_synthetic_ehr

    # Real train/test EHR come from the processed SampleDataset(s):
    train_df = to_evaluation_dataframe(decode_dataset(train_dataset))
    test_df = to_evaluation_dataframe(decode_dataset(test_dataset))

    # Synthetic EHR comes straight from the trained generator (HALO, GPT2, ...):
    synthetic = model.generate(num_samples=len(train_dataset))
    syn_df = to_evaluation_dataframe(synthetic)

    # Privacy metrics need no labels:
    results = evaluate_synthetic_ehr(train_df, test_df, syn_df, metrics="privacy")

The **utility** metrics (machine-learning efficacy, next-visit prediction)
additionally require a meaningful binary ``labels`` column. Since this task is
unconditional (no labels), pass a ``label_fn`` to derive one per patient -- e.g.
``label_fn=lambda r: any("250" in c for v in r["visits"] for c in v)`` for a
diabetes flag -- and the same ``label_fn`` must be applied to the real and
synthetic frames. With no label available, restrict to ``metrics="privacy"``.

Note:
    The MLE component currently hard-codes the downstream task to
    next-visit prediction, which is degenerate for bag-of-codes
    generators (MedGAN, CorGAN) that emit a single aggregate visit per
    patient. A future revision will let callers plug in static-label
    tasks (e.g. mortality, readmission, "ever diagnosed with X") so MLE
    is meaningful for both sequential (HALO, GPT2, PromptEHR) and
    bag-of-codes generators. Until then, restrict bag-of-codes
    evaluation to ``metrics="privacy"`` plus the prevalence metrics.
"""

import logging
from typing import Callable, Dict, List, Optional, Type, Union

from pyhealth.data.data import Patient
from pyhealth.processors import NestedSequenceProcessor

from .base_task import BaseTask

logger = logging.getLogger(__name__)


class EHRGeneration(BaseTask):
    """Generic per-visit code-sequence task for unconditional EHR generators.

    Builds one sample per qualifying patient: the ordered list of visits, each
    visit being the list of codes (read from ``code_attr`` on ``event_type``
    events) recorded in that admission. Patients with fewer than ``min_visits``
    qualifying visits are skipped.

    Subclass and override the class attributes for a specific dataset, or set
    them on an instance. The defaults read MIMIC-III ICD-9 diagnosis codes.

    Args:
        task_name: Name of the task.
        input_schema: ``{"visits": NestedSequenceProcessor}``.
        output_schema: empty (generative task, no labels).
        event_type: Event type to pull per admission. Default
            ``"diagnoses_icd"``.
        code_attr: Event attribute holding the code string. Default
            ``"icd9_code"``.
        min_visits: Minimum qualifying visits to keep a patient. Default 2.
    """

    task_name: str = "ehr_generation"
    input_schema: Dict[str, Union[str, Type]] = {"visits": NestedSequenceProcessor}
    output_schema: Dict[str, Union[str, Type]] = {}

    event_type: str = "diagnoses_icd"
    code_attr: str = "icd9_code"
    min_visits: int = 2

    def __call__(self, patient: Patient) -> List[Dict]:
        """Extract the per-visit code sequence for a patient."""
        visits: List[List[str]] = []
        admissions = patient.get_events(event_type="admissions")
        for admission in admissions:
            events = patient.get_events(
                event_type=self.event_type,
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            codes = [
                getattr(event, self.code_attr)
                for event in events
                if getattr(event, self.code_attr, None)
            ]
            if codes:
                visits.append(codes)

        if len(visits) < self.min_visits:
            return []

        return [{"patient_id": patient.patient_id, "visits": visits}]


class EHRGenerationMIMIC3(EHRGeneration):
    """EHR generation task for MIMIC-III (ICD-9 diagnosis codes).

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import EHRGenerationMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd"],
        ... )
        >>> samples = dataset.set_task(EHRGenerationMIMIC3())
    """

    task_name: str = "ehr_generation_mimic3"
    event_type: str = "diagnoses_icd"
    code_attr: str = "icd9_code"


class EHRGenerationMIMIC4(EHRGeneration):
    """EHR generation task for MIMIC-IV (ICD diagnosis codes).

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import EHRGenerationMIMIC4
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimiciv/2.2/",
        ...     ehr_tables=["patients", "admissions", "diagnoses_icd"],
        ... )
        >>> samples = dataset.set_task(EHRGenerationMIMIC4())
    """

    task_name: str = "ehr_generation_mimic4"
    event_type: str = "diagnoses_icd"
    code_attr: str = "icd_code"


# ----------------------------------------------------------------------------
# Conversion helpers for pyhealth.metrics.generative.evaluate_synthetic_ehr
# ----------------------------------------------------------------------------
def to_evaluation_dataframe(
    records,
    label_fn: Optional[Callable[[Dict], int]] = None,
    subject_col: str = "id",
    visit_col: str = "time",
    code_col: str = "visit_codes",
    label_col: str = "labels",
):
    """Flatten EHR-generation records into the long-form evaluation dataframe.

    Produces the one-row-per-``(patient, visit, code)`` table consumed by
    :func:`pyhealth.metrics.generative.evaluate_synthetic_ehr` (and the
    ``utils.py`` / ``privacy.py`` / ``utility.py`` functions beneath it).

    Subjects are numbered **sequentially** (0, 1, 2, ...) in ``subject_col``;
    any ``"patient_id"`` on the records is ignored, since synthetic patients do
    not correspond to real ones.

    Args:
        records: Iterable of ``{"visits": [[code, ...], ...]}`` dicts. Both the
            :class:`EHRGeneration` task output and a generator's ``generate()``
            output have this shape.
        label_fn: Optional callable mapping a record to a binary patient label
            (0/1) used by the utility metrics. Defaults to all-zeros.
        subject_col: Output patient-id column. Default ``"id"``.
        visit_col: Output visit-index column. Default ``"time"``.
        code_col: Output single-code column. Default ``"visit_codes"``.
        label_col: Output binary-label column. Default ``"labels"``.

    Returns:
        ``pandas.DataFrame`` with columns
        ``[subject_col, visit_col, code_col, label_col]``.
    """
    import pandas as pd

    rows = []
    for subject_id, record in enumerate(records):
        label = 0 if label_fn is None else int(label_fn(record))
        for visit_idx, visit in enumerate(record["visits"]):
            for code in visit:
                rows.append(
                    {
                        subject_col: subject_id,
                        visit_col: visit_idx,
                        code_col: code,
                        label_col: label,
                    }
                )
    return pd.DataFrame(
        rows, columns=[subject_col, visit_col, code_col, label_col]
    )


def decode_dataset(sample_dataset, feature_key: str = "visits") -> List[Dict]:
    """Decode a processed EHRGeneration ``SampleDataset`` back into records.

    Inverts the :class:`~pyhealth.processors.NestedSequenceProcessor` encoding
    using its vocabulary (skipping ``<pad>``/``<unk>``), yielding one
    ``{"visits": [[code_str, ...], ...]}`` record per sample. Use this to build
    the real train/test frames that ``evaluate_synthetic_ehr`` compares against.

    Args:
        sample_dataset: A ``SampleDataset`` produced by :class:`EHRGeneration`.
        feature_key: Input feature key holding the nested code sequence.
            Default ``"visits"``.

    Returns:
        List of ``{"visits": [[code_str, ...], ...]}`` records.
    """
    processor = sample_dataset.input_processors[feature_key]
    index_to_code = {idx: code for code, idx in processor.code_vocab.items()}

    records: List[Dict] = []
    for i in range(len(sample_dataset)):
        sample = sample_dataset[i]
        visits: List[List[str]] = []
        for row in sample[feature_key].tolist():
            codes = [
                index_to_code[int(idx)]
                for idx in row
                if index_to_code.get(int(idx)) not in (None, "<pad>", "<unk>")
            ]
            if codes:
                visits.append(codes)
        records.append({"visits": visits})
    return records
