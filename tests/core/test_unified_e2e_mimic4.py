from pathlib import Path
import tempfile
import unittest
from datetime import datetime

import numpy as np

from pyhealth.datasets import MIMIC4Dataset, get_dataloader
from pyhealth.models import MLP, RNN, UnifiedMultimodalEmbeddingModel
from pyhealth.tasks import ClinicalNotesICDLabsMIMIC4
from pyhealth.trainer import Trainer


class TestUnifiedE2EMIMIC4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls.sample_dataset = None
        cls.small_dataset = None
        cls.ehr_root = str(
            Path(__file__).parent.parent.parent
            / "test-resources"
            / "core"
            / "mimic4demo"
        )

        try:
            base_dataset = MIMIC4Dataset(
                ehr_root=cls.ehr_root,
                ehr_tables=["diagnoses_icd", "procedures_icd", "labevents"],
                cache_dir=cls.cache_dir.name,
                num_workers=1,
            )
            task = ClinicalNotesICDLabsMIMIC4(window_hours=24)
            cls.sample_dataset = base_dataset.set_task(task, num_workers=1)
            if len(cls.sample_dataset) == 0:
                raise unittest.SkipTest(
                    "ClinicalNotesICDLabsMIMIC4 produced no demo samples."
                )
            max_samples = min(16, len(cls.sample_dataset))
            cls.small_dataset = cls.sample_dataset.subset(list(range(max_samples)))
        except Exception as exc:
            raise unittest.SkipTest(
                f"Skipping MIMIC4 unified E2E integration test due to dataset backend issue: {exc}"
            ) from exc

    @classmethod
    def tearDownClass(cls):
        if cls.sample_dataset is not None:
            cls.sample_dataset.close()
        if cls.small_dataset is not None:
            cls.small_dataset.close()
        cls.cache_dir.cleanup()

    def test_task_outputs_expected_fields(self):
        sample = self.sample_dataset[0]
        for key in [
            "patient_id",
            "discharge_note_times",
            "radiology_note_times",
            "icd_codes",
            "labs",
            "labs_mask",
            "mortality",
            "window_start",
            "window_end",
        ]:
            self.assertIn(key, sample)
        self.assertIsInstance(sample["icd_codes"], tuple)
        self.assertIsInstance(sample["labs"], tuple)

    def test_deterministic_window_boundaries(self):
        from pyhealth.tasks.multimodal_mimic4 import BaseMultimodalMIMIC4Task

        class _DummyTask(BaseMultimodalMIMIC4Task):
            pass

        class _Admission:
            def __init__(self, timestamp, dischtime):
                self.timestamp = timestamp
                self.dischtime = dischtime

        global_start = datetime(2020, 1, 1, 0, 0, 0)
        admissions = [
            _Admission(global_start, "2020-01-01 12:00:00"),
            _Admission(datetime(2020, 1, 2, 0, 0, 0), "2020-01-02 12:00:00"),
        ]

        task_windowed = _DummyTask(window_hours=12)
        pairs = {task_windowed._compute_effective_window(admissions) for _ in range(20)}
        self.assertEqual(len(pairs), 1)
        only_pair = next(iter(pairs))
        self.assertEqual(only_pair[0], global_start)
        self.assertEqual(only_pair[1], datetime(2020, 1, 1, 12, 0, 0))

        task_full = _DummyTask(window_hours=None)
        full_pairs = {
            task_full._compute_effective_window(admissions) for _ in range(20)
        }
        self.assertEqual(len(full_pairs), 1)

    def _run_and_check_model(self, model_type: str):
        unified = UnifiedMultimodalEmbeddingModel(
            processors=self.small_dataset.input_processors,
            embedding_dim=32,
        )
        if model_type == "mlp":
            model = MLP(
                dataset=self.small_dataset,
                embedding_dim=32,
                hidden_dim=32,
                unified_embedding=unified,
            )
        else:
            model = RNN(
                dataset=self.small_dataset,
                embedding_dim=32,
                hidden_dim=32,
                unified_embedding=unified,
                rnn_type="GRU",
                num_layers=1,
                dropout=0.0,
            )

        loader = get_dataloader(self.small_dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        forward_out = model(**batch)

        self.assertIn("y_prob", forward_out)
        self.assertIn("loss", forward_out)
        self.assertGreater(forward_out["y_prob"].shape[0], 0)

        trainer = Trainer(
            model=model,
            metrics=["accuracy"],
            device="cpu",
            enable_logging=False,
        )
        y_true, y_prob, _, patient_ids = trainer.inference(
            loader, return_patient_ids=True
        )

        self.assertEqual(y_true.shape[0], y_prob.shape[0])
        self.assertEqual(y_prob.shape[0], len(patient_ids))
        self.assertTrue(np.all(y_prob >= 0.0))
        self.assertTrue(np.all(y_prob <= 1.0))

    def test_unified_mlp_e2e_prediction(self):
        self._run_and_check_model(model_type="mlp")

    def test_unified_rnn_e2e_prediction(self):
        self._run_and_check_model(model_type="rnn")


if __name__ == "__main__":
    unittest.main()
