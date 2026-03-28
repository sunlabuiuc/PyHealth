import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import MIMIC4FHIRDataset
from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

from tests.core.mimic4_fhir_ndjson_fixtures import write_two_class_ndjson


class TestMPFClinicalPredictionTask(unittest.TestCase):
    def _two_patient_ds(self) -> MIMIC4FHIRDataset:
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda p=tmp: shutil.rmtree(p, ignore_errors=True))
        write_two_class_ndjson(Path(tmp))
        return MIMIC4FHIRDataset(
            root=tmp, glob_pattern="*.ndjson", cache_dir=tmp
        )

    def test_max_len_validation(self) -> None:
        with self.assertRaises(ValueError):
            MPFClinicalPredictionTask(max_len=1, use_mpf=True)

    def test_mpf_sets_boundary_tokens(self) -> None:
        task = MPFClinicalPredictionTask(max_len=32, use_mpf=True)
        ds = self._two_patient_ds()
        samples = ds.gather_samples(task)
        vocab = ds.vocab
        self.assertGreater(len(samples), 0)
        s0 = samples[0]
        mor = vocab["<mor>"]
        reg = vocab["<reg>"]
        pad_id = vocab.pad_id
        ids = s0["concept_ids"]
        first = next(i for i, x in enumerate(ids) if x != pad_id)
        last_nz = next(i for i in range(len(ids) - 1, -1, -1) if ids[i] != pad_id)
        self.assertEqual(ids[first], mor)
        self.assertEqual(ids[last_nz], reg)
        self.assertEqual(ids[-1], reg)

    def test_no_mpf_uses_cls_reg(self) -> None:
        task = MPFClinicalPredictionTask(max_len=32, use_mpf=False)
        ds = self._two_patient_ds()
        samples = ds.gather_samples(task)
        vocab = ds.vocab
        s0 = samples[0]
        cls_id = vocab["<cls>"]
        reg_id = vocab["<reg>"]
        pad_id = vocab.pad_id
        ids = s0["concept_ids"]
        first = next(i for i, x in enumerate(ids) if x != pad_id)
        last_nz = next(i for i in range(len(ids) - 1, -1, -1) if ids[i] != pad_id)
        self.assertEqual(ids[first], cls_id)
        self.assertEqual(ids[last_nz], reg_id)
        self.assertEqual(ids[-1], reg_id)

    def test_schema_keys(self) -> None:
        task = MPFClinicalPredictionTask(max_len=16, use_mpf=True)
        ds = self._two_patient_ds()
        samples = ds.gather_samples(task)
        for k in task.input_schema:
            self.assertIn(k, samples[0])
        self.assertIn("label", samples[0])

    def test_max_len_two_keeps_boundary_tokens(self) -> None:
        """``clinical_cap=0`` must yield ``[<mor>, <reg>]`` left-padded, not truncated."""

        task = MPFClinicalPredictionTask(max_len=2, use_mpf=True)
        ds = self._two_patient_ds()
        samples = ds.gather_samples(task)
        vocab = ds.vocab
        mor = vocab["<mor>"]
        reg = vocab["<reg>"]
        pad_id = vocab.pad_id
        for s in samples:
            ids = s["concept_ids"]
            first = next(i for i, x in enumerate(ids) if x != pad_id)
            last_nz = next(
                i for i in range(len(ids) - 1, -1, -1) if ids[i] != pad_id
            )
            self.assertEqual(ids[first], mor)
            self.assertEqual(ids[last_nz], reg)
            self.assertEqual(ids[-1], reg)


if __name__ == "__main__":
    unittest.main()
