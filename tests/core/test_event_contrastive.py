import random
import tempfile
import unittest
from typing import Any, Dict, List

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.event_contrastive import EventContrastiveModel


class TestEventContrastiveModel(unittest.TestCase):
    """Unit tests for EventContrastiveModel using small synthetic data."""

    def setUp(self) -> None:
        """Create a tiny synthetic dataset and initialize the model."""
        random.seed(0)
        torch.manual_seed(0)

        self._tmp_dir = tempfile.TemporaryDirectory()
        self._dataset_name = "test_event_contrastive"

        self.samples: List[Dict[str, Any]] = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "x": torch.randn(20, 8).tolist(),
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "x": torch.randn(20, 8).tolist(),
            },
        ]

        self.input_schema = {"x": "tensor"}
        self.output_schema: Dict[str, Any] = {}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name=self._dataset_name,
        )

        self.model = EventContrastiveModel(
            dataset=self.dataset,
            input_dim=8,
            hidden_dim=16,
            projection_dim=12,
            temperature=0.2,
        )

    def tearDown(self) -> None:
        """Clean up temporary resources."""
        self._tmp_dir.cleanup()

    def test_model_initialization(self) -> None:
        """Model instantiation and basic attributes."""
        self.assertIsInstance(self.model, EventContrastiveModel)
        self.assertEqual(self.model.input_dim, 8)
        self.assertEqual(self.model.hidden_dim, 16)
        self.assertAlmostEqual(self.model.temperature, 0.2)
        self.assertIsNotNone(self.model.encoder)
        self.assertIsNotNone(self.model.projection_head)

    def test_split_events_counts_and_shapes(self) -> None:
        """split_events should create fixed-size windows and drop remainders."""
        x = torch.randn(2, 20, 8)
        events = self.model.split_events(x, window_size=5)

        self.assertEqual(len(events), 4)
        for event in events:
            self.assertEqual(tuple(event.shape), (2, 5, 8))

        x2 = torch.randn(2, 12, 8)
        events2 = self.model.split_events(x2, window_size=5)

        self.assertEqual(len(events2), 2)
        for event in events2:
            self.assertEqual(tuple(event.shape), (2, 5, 8))

    def test_encode_event_shape_and_normalization(self) -> None:
        """encode_event should return normalized embeddings with correct shape."""
        event = torch.randn(3, 5, 8)
        emb = self.model.encode_event(event)

        self.assertEqual(tuple(emb.shape), (3, 12))

        norms = torch.linalg.norm(emb, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_forward_output_shapes(self) -> None:
        """forward should return a list of embeddings with expected shapes."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        x = batch["x"]

        with torch.no_grad():
            embeddings = self.model(x=x)

        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 2)

        for emb in embeddings:
            self.assertEqual(tuple(emb.shape), (2, 12))

    def test_compute_loss_raises_with_single_event(self) -> None:
        """compute_loss should raise if fewer than 2 events are provided."""
        with self.assertRaises(ValueError):
            _ = self.model.compute_loss([torch.randn(2, 12)])

    def test_backward_computes_gradients(self) -> None:
        """Loss backward pass should produce gradients for model parameters."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        x = batch["x"]

        embeddings = self.model(x=x)
        loss = self.model.compute_loss(embeddings)

        self.assertEqual(loss.dim(), 0)
        loss.backward()

        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()