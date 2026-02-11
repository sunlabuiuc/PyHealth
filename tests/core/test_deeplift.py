import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret.methods import DeepLift
from pyhealth.interpret.methods.base_interpreter import BaseInterpreter
from pyhealth.models import MLP, StageNet


class TestDeepLiftMLP(unittest.TestCase):
    """Test cases for DeepLIFT with MLP model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": [1.0, 2.0, 3.5, 4],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": [5.0, 2.0, 3.5, 4],
                "label": 1,
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "conditions": ["cond-55", "cond-12"],
                "procedures": [2.0, 3.0, 1.5, 5],
                "label": 1,
            },
        ]

        self.input_schema = {
            "conditions": "sequence",
            "procedures": "tensor",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_deeplift",
        )

        self.model = MLP(
            dataset=self.dataset,
            embedding_dim=32,
            hidden_dim=32,
            n_layers=2,
        )
        self.model.eval()

        self.test_loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)

    def test_initialization(self):
        """Test that DeepLift initializes correctly."""
        dl = DeepLift(self.model)
        self.assertIsInstance(dl, DeepLift)
        self.assertIsInstance(dl, BaseInterpreter)
        self.assertEqual(dl.model, self.model)

    def test_basic_attribution(self):
        """Test basic attribution computation with default settings."""
        dl = DeepLift(self.model)
        data_batch = next(iter(self.test_loader))

        attributions = dl.attribute(**data_batch)

        self.assertIn("conditions", attributions)
        self.assertIn("procedures", attributions)

        self.assertEqual(
            attributions["conditions"].shape, data_batch["conditions"].shape
        )
        self.assertEqual(
            attributions["procedures"].shape, data_batch["procedures"].shape
        )

        self.assertIsInstance(attributions["conditions"], torch.Tensor)
        self.assertIsInstance(attributions["procedures"], torch.Tensor)

    def test_attribution_with_target_class(self):
        """Test attribution computation with specific target class."""
        dl = DeepLift(self.model)
        data_batch = next(iter(self.test_loader))

        attr_class_0 = dl.attribute(**data_batch, target_class_idx=0)
        attr_class_1 = dl.attribute(**data_batch, target_class_idx=1)

        # Attributions should differ for different classes
        self.assertFalse(
            torch.allclose(attr_class_0["conditions"], attr_class_1["conditions"])
        )

    def test_attribution_values_are_finite(self):
        """Test that attribution values are finite (no NaN or Inf)."""
        dl = DeepLift(self.model)
        data_batch = next(iter(self.test_loader))

        attributions = dl.attribute(**data_batch)

        self.assertTrue(torch.isfinite(attributions["conditions"]).all())
        self.assertTrue(torch.isfinite(attributions["procedures"]).all())

    def test_state_reset_between_calls(self):
        """Multiple DeepLIFT calls should not leak activation state."""
        dl = DeepLift(self.model)

        test_loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)
        batches = list(test_loader)

        first_attr = dl.attribute(**batches[0])
        second_attr = dl.attribute(**batches[1])

        # Both should succeed and produce valid output
        self.assertIn("conditions", first_attr)
        self.assertIn("conditions", second_attr)
        self.assertTrue(torch.isfinite(first_attr["conditions"]).all())
        self.assertTrue(torch.isfinite(second_attr["conditions"]).all())

    def test_callable_interface_delegates_to_attribute(self):
        """DeepLIFT instances should be callable via BaseInterpreter.__call__."""
        dl = DeepLift(self.model)
        data_batch = next(iter(self.test_loader))

        from_attribute = dl.attribute(**data_batch)
        from_call = dl(**data_batch)

        torch.testing.assert_close(
            from_call["conditions"], from_attribute["conditions"]
        )
        torch.testing.assert_close(
            from_call["procedures"], from_attribute["procedures"]
        )

    def test_multiple_samples(self):
        """Test attribution on batch with multiple samples."""
        dl = DeepLift(self.model)

        test_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(test_loader))

        attributions = dl.attribute(**data_batch)

        self.assertEqual(attributions["conditions"].shape[0], 2)
        self.assertEqual(attributions["procedures"].shape[0], 2)


class TestDeepLiftStageNet(unittest.TestCase):
    """Test cases for DeepLIFT with StageNet model.

    StageNet supports the new interpretability API (forward_from_embedding,
    get_embedding_model, processor schemas). DeepLIFT operates in embedding
    space so discrete codes are handled correctly.
    """

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ([0.0, 2.0, 1.3], ["cond-33", "cond-86", "cond-80"]),
                "procedures": (None, [[1.0, 2.0, 3.5], [5.0, 2.0, 3.5], [2.0, 3.0, 1.5]]),
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ([0.0, 2.0], ["cond-33", "cond-86"]),
                "procedures": (None, [[1.0, 2.0, 3.5], [5.0, 2.0, 3.5]]),
                "label": 1,
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "conditions": ([0.0, 1.5], ["cond-86", "cond-80"]),
                "procedures": (None, [[2.0, 3.0, 1.5], [5.0, 2.0, 3.5]]),
                "label": 1,
            },
        ]

        self.input_schema = {
            "conditions": "stagenet",
            "procedures": "stagenet_tensor",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_deeplift_stagenet",
        )

        self.model = StageNet(
            dataset=self.dataset,
            embedding_dim=32,
            chunk_size=32,
        )
        self.model.eval()

        self.test_loader = get_dataloader(
            self.dataset, batch_size=1, shuffle=False
        )

    def test_basic_attribution(self):
        """Test basic DeepLIFT attribution with StageNet."""
        dl = DeepLift(self.model)
        data_batch = next(iter(self.test_loader))

        attributions = dl.attribute(**data_batch)

        self.assertIn("conditions", attributions)
        self.assertIn("procedures", attributions)

    def test_attribution_shapes_match_input(self):
        """Test that attribution shapes match input shapes."""
        dl = DeepLift(self.model)
        data_batch = next(iter(self.test_loader))

        attributions = dl.attribute(**data_batch)

        # Check shape matches the value tensor in the input tuple
        cond_schema = self.model.dataset.input_processors["conditions"].schema()
        cond_value = data_batch["conditions"][cond_schema.index("value")]
        self.assertEqual(attributions["conditions"].shape, cond_value.shape)

    def test_attribution_with_target_class(self):
        """Test DeepLIFT with specific target class for StageNet."""
        dl = DeepLift(self.model)
        data_batch = next(iter(self.test_loader))

        attr_0 = dl.attribute(**data_batch, target_class_idx=0)
        attr_1 = dl.attribute(**data_batch, target_class_idx=1)

        self.assertIn("conditions", attr_0)
        self.assertIn("conditions", attr_1)

    def test_attribution_values_are_finite(self):
        """Test that StageNet DeepLIFT values are finite."""
        dl = DeepLift(self.model)
        data_batch = next(iter(self.test_loader))

        attributions = dl.attribute(**data_batch)

        for key in attributions:
            self.assertTrue(
                torch.isfinite(attributions[key]).all(),
                f"Non-finite values in {key} attributions",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
