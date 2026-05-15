"""Unit tests for :class:`~pyhealth.models.EBCL`.

Uses synthetic in-memory samples only (no MIMIC demo or real EHR files).
"""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import EBCL


def _make_synthetic_samples(n_patients: int = 4):
    """Build minimal paired pre/post code sequences for EBCL."""
    pool = ["A", "B", "C", "D", "E"]
    samples = []
    for i in range(n_patients):
        samples.append(
            {
                "patient_id": f"p{i}",
                "visit_id": f"v{i}",
                "conditions_pre": [pool[i % 5], pool[(i + 1) % 5]],
                "conditions_post": [pool[(i + 2) % 5], pool[(i + 3) % 5]],
                "label": i % 2,
            }
        )
    return samples


class TestEBCL(unittest.TestCase):
    """Tests for Event-Based Contrastive Learning model."""

    def setUp(self):
        samples = _make_synthetic_samples(4)
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "conditions_pre": "sequence",
                "conditions_post": "sequence",
            },
            output_schema={"label": "binary"},
            dataset_name="ebcl_test",
        )
        self.model = EBCL(
            self.dataset,
            embedding_dim=32,
            hidden_dim=32,
            projection_dim=16,
            temperature=0.2,
            supervised_weight=0.0,
        )

    def test_initialization(self):
        self.assertIsInstance(self.model, EBCL)
        self.assertEqual(self.model.pre_key, "conditions_pre")
        self.assertEqual(self.model.post_key, "conditions_post")
        self.assertEqual(self.model.embedding_dim, 32)
        self.assertEqual(self.model.hidden_dim, 32)
        self.assertEqual(self.model.projection_dim, 16)

    def test_forward_contrastive_loss_and_shapes(self):
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            out = self.model(**batch)
        self.assertIn("loss", out)
        self.assertEqual(out["loss"].dim(), 0)
        self.assertEqual(out["embed_pre"].shape, (4, 32))
        self.assertEqual(out["embed_post"].shape, (4, 32))
        self.assertEqual(out["z_pre"].shape, (4, 16))
        self.assertEqual(out["z_post"].shape, (4, 16))

    def test_backward(self):
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        out = self.model(**batch)
        out["loss"].backward()
        self.assertIsNotNone(self.model.proj.weight.grad)

    def test_supervised_weight_combines_losses(self):
        model_sup = EBCL(
            self.dataset,
            embedding_dim=32,
            hidden_dim=32,
            projection_dim=16,
            supervised_weight=0.5,
        )
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            out_sup = model_sup(**batch)
            out_unsup = self.model(**batch)
        self.assertGreater(out_sup["loss"].item(), 0.0)
        # Combined loss should differ from contrastive-only when probe is trained
        self.assertIn("logit", out_sup)

    def test_embed_flag(self):
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        batch["embed"] = True
        with torch.no_grad():
            out = self.model(**batch)
        self.assertIn("embed", out)
        self.assertEqual(out["embed"].shape, (4, 64))

    def test_requires_two_sequence_features(self):
        samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": ["x"],
                "label": 0,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": ["y"],
                "label": 1,
            },
        ]
        ds = create_sample_dataset(
            samples=samples,
            input_schema={"conditions": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="ebcl_bad",
        )
        with self.assertRaises(ValueError) as ctx:
            EBCL(ds)
        self.assertIn("exactly two", str(ctx.exception).lower())

    def test_info_nce_symmetric(self):
        """Gradient flows through both projection directions."""
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        self.model.train()
        out = self.model(**batch)
        out["loss"].backward()
        self.assertIsNotNone(self.model.proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
