import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import EHRMambaCEHR
from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask


def _tiny_samples(seq: int = 16) -> tuple:
    from pyhealth.datasets.mimic4_fhir import ConceptVocab, ensure_special_tokens

    task = MPFClinicalPredictionTask(max_len=seq, use_mpf=True)
    task.vocab = ConceptVocab()
    sp = ensure_special_tokens(task.vocab)
    mid = task.vocab.add_token("test|filler")
    samples = []
    for lab in (0, 1):
        samples.append(
            {
                "patient_id": f"p{lab}",
                "visit_id": f"v{lab}",
                "concept_ids": [sp["<mor>"]] + [mid] * (seq - 2) + [sp["<reg>"]],
                "token_type_ids": [0] * seq,
                "time_stamps": [0.0] * seq,
                "ages": [50.0] * seq,
                "visit_orders": [0] * seq,
                "visit_segments": [0] * seq,
                "label": lab,
            }
        )
    return samples, task


class TestEHRMambaCEHR(unittest.TestCase):
    def test_end_to_end_fhir_pipeline(self) -> None:
        from pyhealth.datasets import (
            build_fhir_sample_dataset_from_lines,
            create_sample_dataset,
            get_dataloader,
            synthetic_ndjson_lines_two_class,
        )

        task = MPFClinicalPredictionTask(max_len=32, use_mpf=True)
        _, _, samples = build_fhir_sample_dataset_from_lines(
            synthetic_ndjson_lines_two_class(), task
        )
        ds = create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name="fhir_test",
        )
        vocab_size = max(max(s["concept_ids"]) for s in samples) + 1
        model = EHRMambaCEHR(
            dataset=ds,
            vocab_size=vocab_size,
            embedding_dim=64,
            num_layers=1,
        )
        batch = next(iter(get_dataloader(ds, batch_size=2, shuffle=False)))
        out = model(**batch)
        self.assertIn("loss", out)
        out["loss"].backward()

    def test_forward_backward(self) -> None:
        samples, task = _tiny_samples()
        ds = create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
        )
        vocab_size = max(max(s["concept_ids"]) for s in samples) + 1
        model = EHRMambaCEHR(
            dataset=ds,
            vocab_size=vocab_size,
            embedding_dim=64,
            num_layers=1,
            state_size=8,
        )
        batch = next(iter(get_dataloader(ds, batch_size=2, shuffle=False)))
        out = model(**batch)
        self.assertEqual(out["logit"].shape[0], 2)
        out["loss"].backward()

    def test_eval_mode(self) -> None:
        samples, task = _tiny_samples()
        ds = create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
        )
        vocab_size = max(max(s["concept_ids"]) for s in samples) + 1
        model = EHRMambaCEHR(dataset=ds, vocab_size=vocab_size, embedding_dim=32, num_layers=1)
        model.eval()
        with torch.no_grad():
            batch = next(iter(get_dataloader(ds, batch_size=2, shuffle=False)))
            out = model(**batch)
        self.assertIn("y_prob", out)


if __name__ == "__main__":
    unittest.main()
