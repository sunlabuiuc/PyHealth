"""Each generator family gets the encoding it actually consumes.

Extraction is shared across the EHR-generation tasks, the encoding is not:
HALO reads multi-hot rows, GPT2/PromptEHR read code indices, MedGAN/CorGAN read
one pooled set per patient. Pairing a model with the wrong task does not raise
-- the shapes and dtypes stay valid -- so these tests pin the pairing itself,
and check that real codes survive the round trip rather than collapsing to
<unk>.
"""

import unittest
from typing import ClassVar

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import GPT2, PromptEHR
from pyhealth.processors import (
    MultiHotProcessor,
    NestedMultiHotProcessor,
    NestedSequenceProcessor,
)
from pyhealth.tasks import (
    EHRCodeSetGenerationMIMIC3,
    EHRCodeSetGenerationMIMIC4,
    EHRGenerationMIMIC3,
    EHRGenerationMIMIC4,
    EHRSequenceGenerationMIMIC3,
    EHRSequenceGenerationMIMIC4,
)
from pyhealth.tasks.base_task import BaseTask

SAMPLES = [
    {"patient_id": "p0", "visits": [["A05B", "A05C"], ["A11D"], ["C129"]]},
    {"patient_id": "p1", "visits": [["A05B"], ["A04A", "B035"]]},
    {"patient_id": "p2", "visits": [["C129", "A11D"], ["A05C"], ["A04A"]]},
    {"patient_id": "p3", "visits": [["B035"], ["A05B", "C129"]]},
]


def _dataset(schema_key, name):
    return create_sample_dataset(
        samples=SAMPLES,
        input_schema={"visits": schema_key},
        output_schema={},
        dataset_name=name,
    )


class _Event:
    def __init__(self, hadm_id, code=None):
        self.hadm_id = hadm_id
        self.icd9_code = code


class _Patient:
    """Minimal stand-in for pyhealth.data.Patient: admissions plus coded events."""

    def __init__(self, patient_id, visits):
        self.patient_id = patient_id
        self._admissions = [_Event(f"h{i}") for i in range(len(visits))]
        self._codes = [
            _Event(f"h{i}", code)
            for i, codes in enumerate(visits)
            for code in codes
        ]

    def get_events(self, event_type, filters=None):
        if event_type == "admissions":
            return self._admissions
        hadm = filters[0][2]
        return [e for e in self._codes if e.hadm_id == hadm]


class TestExtraction(unittest.TestCase):
    """Extraction, and the pooling the code-set tasks add."""

    VISITS: ClassVar[list] = [["A05B", "A05C"], ["A11D"], ["A05B"]]

    def test_per_visit_tasks_keep_visit_structure(self):
        patient = _Patient("p0", self.VISITS)
        samples = EHRGenerationMIMIC3()(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["visits"], self.VISITS)
        # Same extraction regardless of encoding -- only input_schema differs.
        self.assertEqual(EHRSequenceGenerationMIMIC3()(patient)[0]["visits"],
                         self.VISITS)

    def test_codeset_task_pools_and_dedupes(self):
        samples = EHRCodeSetGenerationMIMIC3()(_Patient("p0", self.VISITS))
        self.assertEqual(len(samples), 1)
        # One flat set: A05B appears in two visits and survives once.
        self.assertEqual(samples[0]["visits"], ["A05B", "A05C", "A11D"])

    def test_min_visits_counts_real_visits_before_pooling(self):
        """Pooling must not let a 1-visit patient past a min_visits=2 filter."""
        one_visit = _Patient("p1", [["A05B", "A05C", "A11D"]])
        self.assertEqual(EHRCodeSetGenerationMIMIC3()(one_visit), [])
        self.assertEqual(EHRGenerationMIMIC3()(one_visit), [])

    def test_codeless_admissions_are_dropped(self):
        patient = _Patient("p2", [["A05B"], [], ["A11D"]])
        self.assertEqual(EHRGenerationMIMIC3()(patient)[0]["visits"],
                         [["A05B"], ["A11D"]])


class TestTaskEncodings(unittest.TestCase):
    """Each task declares the processor its models consume."""

    def test_each_family_gets_its_own_encoding(self):
        for task in (EHRGenerationMIMIC3, EHRGenerationMIMIC4):
            self.assertIs(task.input_schema["visits"], NestedMultiHotProcessor)
        for task in (EHRSequenceGenerationMIMIC3, EHRSequenceGenerationMIMIC4):
            self.assertIs(task.input_schema["visits"], NestedSequenceProcessor)
        for task in (EHRCodeSetGenerationMIMIC3, EHRCodeSetGenerationMIMIC4):
            self.assertIs(task.input_schema["visits"], MultiHotProcessor)

    def test_tasks_are_flat(self):
        """No task inherits from another: the MIMIC extraction is not a base.

        A parent class would invite subclassing it for eICU/OMOP, where there
        is no ``admissions`` event type and no ``hadm_id`` -- which would
        return no samples rather than fail.
        """
        tasks = [
            EHRGenerationMIMIC3, EHRGenerationMIMIC4,
            EHRSequenceGenerationMIMIC3, EHRSequenceGenerationMIMIC4,
            EHRCodeSetGenerationMIMIC3, EHRCodeSetGenerationMIMIC4,
        ]
        for task in tasks:
            with self.subTest(task=task.__name__):
                self.assertEqual(task.__bases__, (BaseTask,))
                self.assertIn("MIMIC", task.__name__)

    def test_task_names_are_unique(self):
        tasks = [
            EHRGenerationMIMIC3, EHRGenerationMIMIC4,
            EHRSequenceGenerationMIMIC3, EHRSequenceGenerationMIMIC4,
            EHRCodeSetGenerationMIMIC3, EHRCodeSetGenerationMIMIC4,
        ]
        names = [t.task_name for t in tasks]
        self.assertEqual(len(names), len(set(names)))

    def test_mimic4_reads_the_mimic4_code_column(self):
        self.assertEqual(EHRGenerationMIMIC3.code_attr, "icd9_code")
        self.assertEqual(EHRGenerationMIMIC4.code_attr, "icd_code")


class TestVisitCodeIds(unittest.TestCase):
    """NestedSequenceProcessor inverts its own rows for the token generators."""

    def test_matches_the_multihot_columns(self):
        """Both encodings of the same visit name the same codes.

        NestedMultiHotProcessor has no visit_code_ids -- nothing consumes one --
        so its codes are read here the way decode_dataset reads them, as the
        row's nonzero columns.
        """
        multihot = _dataset("nested_multihot", "vci_mh")
        indexed = _dataset("nested_sequence", "vci_ix")
        ix_proc = indexed.input_processors["visits"]

        # Same samples, same traversal order, so the vocabularies must match --
        # that is what makes the per-visit comparison below meaningful.
        self.assertEqual(multihot.input_processors["visits"].code_vocab,
                         ix_proc.code_vocab)

        for i in range(len(SAMPLES)):
            mh_row = multihot[i]["visits"]
            ix_row = indexed[i]["visits"]
            for visit in range(mh_row.shape[0]):
                # Multi-hot columns come out in vocabulary order, the index form
                # in charted order, so compare as sets.
                self.assertEqual(
                    set(mh_row[visit].nonzero(as_tuple=True)[0].tolist()),
                    set(ix_proc.visit_code_ids(ix_row[visit])),
                )

    def test_padding_is_dropped_not_read_as_a_code(self):
        processor = _dataset("nested_sequence", "vci_pad").input_processors["visits"]
        row = processor.process([["A05B", "A05C"]])[0]
        ids = processor.visit_code_ids(row)
        self.assertEqual(len(ids), 2)
        self.assertNotIn(processor.PAD, ids)
        self.assertNotIn(processor.UNK, ids)


class TestTokenGeneratorsOnIndices(unittest.TestCase):
    """GPT2 and PromptEHR serialise real codes from their own encoding."""

    MODELS: ClassVar[list] = [
        (GPT2, {"embed_dim": 16, "n_heads": 2, "n_layers": 2, "max_len": 64}),
        (PromptEHR, {"embed_dim": 16, "n_heads": 2, "n_layers": 2, "max_len": 64,
                     "prompt_length": 4}),
    ]

    def _streams(self, model, visits):
        if hasattr(model, "_serialize"):
            return model._serialize(visits)
        input_ids, _, _ = model._encode_visits(visits)
        return [row.tolist() for row in input_ids]

    def test_codes_survive_serialisation(self):
        for cls, kwargs in self.MODELS:
            with self.subTest(model=cls.__name__):
                dataset = _dataset("nested_sequence", f"gen_{cls.__name__}")
                model = cls(dataset=dataset, batch_size=2, epochs=1, **kwargs)
                # Patients have different visit counts, so let the dataloader
                # pad the visit dimension rather than stacking raw samples.
                batch = next(iter(get_dataloader(dataset, batch_size=2)))
                visits = batch["visits"]
                streams = self._streams(model, visits)

                code_ids = [
                    t
                    for stream in streams
                    for t in stream
                    if t < model.code_vocab_size and t != 0
                ]
                self.assertTrue(code_ids, "no code tokens were emitted at all")
                self.assertGreater(
                    len(set(code_ids) - {model.visits_processor.UNK}),
                    1,
                    f"{cls.__name__} emitted only <unk>: the visit row was read "
                    "as raw values instead of via visit_code_ids",
                )


if __name__ == "__main__":
    unittest.main()
