import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import MIMIC4FHIR
from pyhealth.processors.cehr_processor import PAD_TOKEN
from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

from tests.core.test_fhir_ndjson_fixtures import (
    run_task,
    write_two_class_ndjson,
)


class TestMPFClinicalPredictionTask(unittest.TestCase):
    """Verifies the task emits boundary-marker strings at the expected
    positions in its raw output. Vocab → integer-id mapping is the
    ``CehrProcessor``'s job and is exercised separately via the standard
    ``SampleBuilder.fit`` pipeline.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.mkdtemp()
        write_two_class_ndjson(Path(cls._tmp))
        # One shared build for the whole class; tests reuse it via run_task
        # (run_task just applies the task to cached patients — no rebuild).
        cls.ds = MIMIC4FHIR(
            root=cls._tmp, glob_pattern="*.ndjson", cache_dir=cls._tmp
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_max_len_validation(self) -> None:
        with self.assertRaises(ValueError):
            MPFClinicalPredictionTask(max_len=1, use_mpf=True)

    def test_mpf_sets_boundary_tokens(self) -> None:
        task = MPFClinicalPredictionTask(max_len=32, use_mpf=True)
        samples = run_task(self.ds, task)
        self.assertGreater(len(samples), 0)
        keys = samples[0]["concept_ids"]
        first = next(i for i, x in enumerate(keys) if x != PAD_TOKEN)
        last_nz = next(
            i for i in range(len(keys) - 1, -1, -1) if keys[i] != PAD_TOKEN
        )
        self.assertEqual(keys[first], "<mor>")
        self.assertEqual(keys[last_nz], "<reg>")
        self.assertEqual(keys[-1], "<reg>")

    def test_no_mpf_uses_cls_reg(self) -> None:
        task = MPFClinicalPredictionTask(max_len=32, use_mpf=False)
        samples = run_task(self.ds, task)
        keys = samples[0]["concept_ids"]
        first = next(i for i, x in enumerate(keys) if x != PAD_TOKEN)
        last_nz = next(
            i for i in range(len(keys) - 1, -1, -1) if keys[i] != PAD_TOKEN
        )
        self.assertEqual(keys[first], "<cls>")
        self.assertEqual(keys[last_nz], "<reg>")
        self.assertEqual(keys[-1], "<reg>")

    def test_schema_keys(self) -> None:
        task = MPFClinicalPredictionTask(max_len=16, use_mpf=True)
        samples = run_task(self.ds, task)
        for k in task.input_schema:
            self.assertIn(k, samples[0])
        self.assertIn("label", samples[0])

    def test_max_len_two_keeps_boundary_tokens(self) -> None:
        """At ``max_len=2`` the sequence is exactly ``[<mor>, <reg>]``."""

        task = MPFClinicalPredictionTask(max_len=2, use_mpf=True)
        samples = run_task(self.ds, task)
        for s in samples:
            keys = s["concept_ids"]
            self.assertEqual(len(keys), 2)
            self.assertEqual(keys[0], "<mor>")
            self.assertEqual(keys[1], "<reg>")

    def test_fixed_length_alignment(self) -> None:
        """All six per-event lists must be the same length (max_len)."""

        task = MPFClinicalPredictionTask(max_len=24, use_mpf=True)
        samples = run_task(self.ds, task)
        for s in samples:
            self.assertEqual(len(s["concept_ids"]), 24)
            self.assertEqual(len(s["token_type_ids"]), 24)
            self.assertEqual(len(s["time_stamps"]), 24)
            self.assertEqual(len(s["ages"]), 24)
            self.assertEqual(len(s["visit_orders"]), 24)
            self.assertEqual(len(s["visit_segments"]), 24)


if __name__ == "__main__":
    unittest.main()
