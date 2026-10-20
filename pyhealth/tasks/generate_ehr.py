"""EHR sequence-generation tasks for PyHealth generative models.

These back every generator in :mod:`pyhealth.models.generators`. Each one
extracts, for each patient, the codes recorded across their admissions. There
is no prediction label, so ``output_schema`` is empty.

The classes are flat and independent -- one per (model family, dataset) -- with
the dataset named in the class, because the extraction is MIMIC-shaped: it
assumes an ``admissions`` event type and a ``hadm_id`` linking codes to an
admission. eICU, OMOP and MEDS do not look like that, so a task for those
datasets belongs alongside these rather than inheriting from them.

=============================  ==========================  ==================
Task                           Encoding                    Models
=============================  ==========================  ==================
``EHRGenerationMIMIC3/4``      per-visit multi-hot rows    HALO
``EHRSequenceGenerationMIMIC3/4``  per-visit code indices  GPT2, PromptEHR
``EHRCodeSetGenerationMIMIC3/4``   one code set per patient  MedGAN, CorGAN
=============================  ==========================  ==================

Match the task to the model. Handing a model the wrong encoding does not raise:
the numbers still have the right shape and dtype, so training runs and produces
a confidently wrong result. That is why these are separate classes rather than
one task with a flag.

``event_type`` / ``code_attr`` are class attributes naming the dataset's coding
columns; override them on an instance to read a different table.

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
from collections.abc import Callable
from typing import ClassVar

from pyhealth.data.data import Patient
from pyhealth.processors import (
    MultiHotProcessor,
    NestedMultiHotProcessor,
    NestedSequenceProcessor,
)

from .base_task import BaseTask

logger = logging.getLogger(__name__)


def _mimic_visits(
    patient: Patient, event_type: str, code_attr: str
) -> list[list[str]]:
    """Ordered per-admission code lists for a MIMIC-style patient.

    Deliberately MIMIC-specific and deliberately not a base-class method: it
    assumes an ``admissions`` event type and a ``hadm_id`` linking codes to an
    admission. eICU, OMOP and MEDS do not look like this, so a task for those
    writes its own extraction rather than inheriting one that would silently
    return nothing.

    Args:
        patient: Patient to read.
        event_type: Event type carrying the codes (e.g. ``"diagnoses_icd"``).
        code_attr: Attribute on those events holding the code string.

    Returns:
        One list of codes per admission, admissions with no codes dropped.
    """
    visits: list[list[str]] = []
    for admission in patient.get_events(event_type="admissions"):
        events = patient.get_events(
            event_type=event_type,
            filters=[("hadm_id", "==", admission.hadm_id)],
        )
        codes = [
            getattr(event, code_attr)
            for event in events
            if getattr(event, code_attr, None)
        ]
        if codes:
            visits.append(codes)
    return visits


class EHRGenerationMIMIC3(BaseTask):
    """Per-visit ICD-9 code sets from MIMIC-III as multi-hot rows. For HALO.

    HALO's transformer consumes a multi-hot vector per context position, so
    this hands it exactly that and nothing is repacked on the way in.

    Patients with fewer than ``min_visits`` coded admissions are skipped.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import EHRGenerationMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4", tables=["diagnoses_icd"]
        ... )
        >>> samples = dataset.set_task(EHRGenerationMIMIC3())
        >>> samples[0]["visits"].shape  # (num_visits, vocab_size)
        torch.Size([3, 512])
    """

    task_name: str = "ehr_generation_mimic3"
    input_schema: ClassVar[dict[str, str | type]] = {
        "visits": NestedMultiHotProcessor
    }
    output_schema: ClassVar[dict[str, str | type]] = {}

    event_type: str = "diagnoses_icd"
    code_attr: str = "icd9_code"
    min_visits: int = 2

    def __call__(self, patient: Patient) -> list[dict]:
        """Extract the per-visit code sequence for a patient."""
        visits = _mimic_visits(patient, self.event_type, self.code_attr)
        if len(visits) < self.min_visits:
            return []
        return [{"patient_id": patient.patient_id, "visits": visits}]


class EHRGenerationMIMIC4(BaseTask):
    """Per-visit ICD code sets from MIMIC-IV as multi-hot rows. For HALO.

    MIMIC-IV's diagnosis codes live on ``icd_code`` rather than MIMIC-III's
    ``icd9_code``; otherwise identical to :class:`EHRGenerationMIMIC3`.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import EHRGenerationMIMIC4
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimiciv/2.2/",
        ...     ehr_tables=["patients", "admissions", "diagnoses_icd"],
        ... )
        >>> samples = dataset.set_task(EHRGenerationMIMIC4())
        >>> samples[0]["visits"].shape  # (num_visits, vocab_size)
        torch.Size([3, 512])
    """

    task_name: str = "ehr_generation_mimic4"
    input_schema: ClassVar[dict[str, str | type]] = {
        "visits": NestedMultiHotProcessor
    }
    output_schema: ClassVar[dict[str, str | type]] = {}

    event_type: str = "diagnoses_icd"
    code_attr: str = "icd_code"
    min_visits: int = 2

    def __call__(self, patient: Patient) -> list[dict]:
        """Extract the per-visit code sequence for a patient."""
        visits = _mimic_visits(patient, self.event_type, self.code_attr)
        if len(visits) < self.min_visits:
            return []
        return [{"patient_id": patient.patient_id, "visits": visits}]


class EHRSequenceGenerationMIMIC3(BaseTask):
    """Per-visit ICD-9 code indices from MIMIC-III. For GPT2 and PromptEHR.

    Both are token-sequence models: they flatten each visit into a stream of
    code ids, so indices are what they want. Handing them the multi-hot form
    means encoding a code set and decoding it straight back.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import EHRSequenceGenerationMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4", tables=["diagnoses_icd"]
        ... )
        >>> samples = dataset.set_task(EHRSequenceGenerationMIMIC3())
        >>> samples[0]["visits"].shape  # (num_visits, max_codes_per_visit)
        torch.Size([3, 12])
    """

    task_name: str = "ehr_sequence_generation_mimic3"
    input_schema: ClassVar[dict[str, str | type]] = {
        "visits": NestedSequenceProcessor
    }
    output_schema: ClassVar[dict[str, str | type]] = {}

    event_type: str = "diagnoses_icd"
    code_attr: str = "icd9_code"
    min_visits: int = 2

    def __call__(self, patient: Patient) -> list[dict]:
        """Extract the per-visit code sequence for a patient."""
        visits = _mimic_visits(patient, self.event_type, self.code_attr)
        if len(visits) < self.min_visits:
            return []
        return [{"patient_id": patient.patient_id, "visits": visits}]


class EHRSequenceGenerationMIMIC4(BaseTask):
    """Per-visit ICD code indices from MIMIC-IV. For GPT2 and PromptEHR.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import EHRSequenceGenerationMIMIC4
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimiciv/2.2/",
        ...     ehr_tables=["patients", "admissions", "diagnoses_icd"],
        ... )
        >>> samples = dataset.set_task(EHRSequenceGenerationMIMIC4())
        >>> samples[0]["visits"].shape  # (num_visits, max_codes_per_visit)
        torch.Size([3, 12])
    """

    task_name: str = "ehr_sequence_generation_mimic4"
    input_schema: ClassVar[dict[str, str | type]] = {
        "visits": NestedSequenceProcessor
    }
    output_schema: ClassVar[dict[str, str | type]] = {}

    event_type: str = "diagnoses_icd"
    code_attr: str = "icd_code"
    min_visits: int = 2

    def __call__(self, patient: Patient) -> list[dict]:
        """Extract the per-visit code sequence for a patient."""
        visits = _mimic_visits(patient, self.event_type, self.code_attr)
        if len(visits) < self.min_visits:
            return []
        return [{"patient_id": patient.patient_id, "visits": visits}]


class EHRCodeSetGenerationMIMIC3(BaseTask):
    """One pooled ICD-9 code set per MIMIC-III patient. For MedGAN and CorGAN.

    Bag-of-codes generators emit a single aggregate vector per patient, so the
    visit axis is collapsed here rather than inside the model. ``min_visits``
    still counts real admissions, before the codes are pooled.

    Note:
        With the visit axis gone, the next-visit utility metric in
        :mod:`pyhealth.metrics.generative` is not meaningful for these models;
        see this module's header.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import EHRCodeSetGenerationMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4", tables=["diagnoses_icd"]
        ... )
        >>> samples = dataset.set_task(EHRCodeSetGenerationMIMIC3())
        >>> samples[0]["visits"].shape  # (vocab_size,)
        torch.Size([512])
    """

    task_name: str = "ehr_codeset_generation_mimic3"
    input_schema: ClassVar[dict[str, str | type]] = {"visits": MultiHotProcessor}
    output_schema: ClassVar[dict[str, str | type]] = {}

    event_type: str = "diagnoses_icd"
    code_attr: str = "icd9_code"
    min_visits: int = 2

    def __call__(self, patient: Patient) -> list[dict]:
        """Pool every visit's codes into one per-patient set."""
        visits = _mimic_visits(patient, self.event_type, self.code_attr)
        if len(visits) < self.min_visits:
            return []
        codes = sorted({code for visit in visits for code in visit})
        return [{"patient_id": patient.patient_id, "visits": codes}]


class EHRCodeSetGenerationMIMIC4(BaseTask):
    """One pooled ICD code set per MIMIC-IV patient. For MedGAN and CorGAN.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import EHRCodeSetGenerationMIMIC4
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimiciv/2.2/",
        ...     ehr_tables=["patients", "admissions", "diagnoses_icd"],
        ... )
        >>> samples = dataset.set_task(EHRCodeSetGenerationMIMIC4())
        >>> samples[0]["visits"].shape  # (vocab_size,)
        torch.Size([512])
    """

    task_name: str = "ehr_codeset_generation_mimic4"
    input_schema: ClassVar[dict[str, str | type]] = {"visits": MultiHotProcessor}
    output_schema: ClassVar[dict[str, str | type]] = {}

    event_type: str = "diagnoses_icd"
    code_attr: str = "icd_code"
    min_visits: int = 2

    def __call__(self, patient: Patient) -> list[dict]:
        """Pool every visit's codes into one per-patient set."""
        visits = _mimic_visits(patient, self.event_type, self.code_attr)
        if len(visits) < self.min_visits:
            return []
        codes = sorted({code for visit in visits for code in visit})
        return [{"patient_id": patient.patient_id, "visits": codes}]


# ----------------------------------------------------------------------------
# Conversion helpers for pyhealth.metrics.generative.evaluate_synthetic_ehr
# ----------------------------------------------------------------------------
def to_evaluation_dataframe(
    records,
    label_fn: Callable[[dict], int] | None = None,
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
            generation tasks' output and a generator's ``generate()``
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

    Examples:
        >>> from pyhealth.tasks.generate_ehr import to_evaluation_dataframe
        >>> records = [{"visits": [["4019", "25000"], ["4019"]]}]
        >>> to_evaluation_dataframe(records)
           id  time visit_codes  labels
        0   0     0        4019       0
        1   0     0       25000       0
        2   0     1        4019       0
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


def decode_dataset(sample_dataset, feature_key: str = "visits") -> list[dict]:
    """Decode a processed multi-hot ``SampleDataset`` back into code records.

    Inverts the :class:`~pyhealth.processors.NestedMultiHotProcessor` encoding
    using its vocabulary (skipping ``<pad>``/``<unk>``), yielding one
    ``{"visits": [[code_str, ...], ...]}`` record per sample. Use this to build
    the real train/test frames that ``evaluate_synthetic_ehr`` compares against.

    Codes come back in vocabulary order, not the order they were charted in,
    and repeats collapse -- the multi-hot form records presence, not sequence
    or count within a visit.

    Args:
        sample_dataset: A ``SampleDataset`` (or a split of one) produced by
            :class:`EHRGenerationMIMIC3` / :class:`EHRGenerationMIMIC4`.
        feature_key: Input feature key holding the nested code sequence.
            Default ``"visits"``.

    Returns:
        List of ``{"visits": [[code_str, ...], ...]}`` records.

    Raises:
        TypeError: If ``feature_key`` is not backed by a
            :class:`~pyhealth.processors.NestedMultiHotProcessor`.

    Examples:
        >>> from pyhealth.tasks.generate_ehr import decode_dataset
        >>> records = decode_dataset(samples)
        >>> records[0]["visits"][0]
        ['4019', '25000']
    """
    # split_by_patient hands back a torch Subset, which carries no processors
    # of its own -- decoding a split is the common case, so resolve through it.
    source = getattr(sample_dataset, "dataset", sample_dataset)
    processor = source.input_processors[feature_key]
    if not isinstance(processor, NestedMultiHotProcessor):
        raise TypeError(
            f"decode_dataset inverts the multi-hot encoding, but '{feature_key}' "
            f"is a {type(processor).__name__}. Use one of the multi-hot tasks\n            (EHRGenerationMIMIC3/4), or "
            "read the codes off the index tensor directly."
        )
    index_to_code = {idx: code for code, idx in processor.code_vocab.items()}

    records: list[dict] = []
    for i in range(len(sample_dataset)):
        sample = sample_dataset[i]
        visits: list[list[str]] = []
        # Each row is a multi-hot vector over the vocabulary, so the codes
        # present are its nonzero columns -- the values are all 1.0.
        for row in sample[feature_key]:
            codes = [
                index_to_code[int(col)]
                for col in row.nonzero(as_tuple=True)[0].tolist()
                if index_to_code.get(int(col)) not in (None, "<pad>", "<unk>")
            ]
            if codes:
                visits.append(codes)
        records.append({"visits": visits})
    return records
