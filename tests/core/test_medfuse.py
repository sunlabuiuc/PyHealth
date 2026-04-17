import unittest

import torch
import torch.nn as nn

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MedFuse, MedFuseLayer

try:
    import torchvision  # noqa: F401

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


@unittest.skipUnless(HAS_TORCHVISION, "torchvision is required for MedFuse tests")
class TestMedFuse(unittest.TestCase):
    """Test cases for the MedFuse model."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.binary_dataset = cls._create_binary_dataset()
        cls.multilabel_dataset = cls._create_multilabel_dataset()

    @staticmethod
    def _create_binary_dataset():
        generator = torch.Generator().manual_seed(7)
        samples = []
        for index in range(4):
            samples.append(
                {
                    "patient_id": f"patient-{index}",
                    "visit_id": f"visit-{index}",
                    "ehr": torch.randn(5, 10, generator=generator).tolist(),
                    "cxr": torch.randn(3, 32, 32, generator=generator).tolist(),
                    "label": index % 2,
                }
            )

        return create_sample_dataset(
            samples=samples,
            input_schema={"ehr": "tensor", "cxr": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="medfuse_binary_test",
        )

    @staticmethod
    def _create_multilabel_dataset():
        generator = torch.Generator().manual_seed(17)
        labels = [[0], [1], [0, 1], []]
        samples = []
        for index, label in enumerate(labels):
            samples.append(
                {
                    "patient_id": f"patient-multi-{index}",
                    "visit_id": f"visit-multi-{index}",
                    "ehr": torch.randn(5, 10, generator=generator).tolist(),
                    "cxr": torch.randn(3, 32, 32, generator=generator).tolist(),
                    "label": label,
                }
            )

        return create_sample_dataset(
            samples=samples,
            input_schema={"ehr": "tensor", "cxr": "tensor"},
            output_schema={"label": "multilabel"},
            dataset_name="medfuse_multilabel_test",
        )

    @staticmethod
    def _build_model(dataset):
        model = MedFuse(
            dataset=dataset,
            ehr_feature_key="ehr",
            cxr_feature_key="cxr",
            cxr_mask_key="cxr_mask",
            ehr_hidden_dim=8,
            ehr_num_layers=1,
            cxr_backbone="resnet18",
            cxr_pretrained=False,
            fusion_hidden_dim=16,
            projection_dim=8,
            dropout=0.0,
        )
        # Speed up unit tests: keep MedFuse API/logic but replace heavyweight
        # image encoder compute with a tiny synthetic module.
        projection_in_features = model.layer.projection.in_features
        model.layer.cxr_encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(8, projection_in_features),
        )
        return model

    @staticmethod
    def _next_batch(dataset):
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        return next(iter(loader))

    def test_instantiation(self):
        """Model can be created with valid args."""
        model = self._build_model(self.binary_dataset)
        self.assertIsInstance(model, MedFuse)
        self.assertIsInstance(model.layer, MedFuseLayer)
        self.assertEqual(model.ehr_input_dim, 10)

    def test_forward_pass_both_modalities(self):
        """Forward pass works with both EHR and CXR input."""
        model = self._build_model(self.binary_dataset)
        batch = self._next_batch(self.binary_dataset)

        with torch.no_grad():
            output = model(**batch)

        self.assertIn("loss", output)
        self.assertIn("y_prob", output)
        self.assertIn("y_true", output)
        self.assertIn("logit", output)

    def test_forward_pass_ehr_only(self):
        """Forward pass works with EHR only (missing CXR)."""
        model = self._build_model(self.binary_dataset)
        batch = self._next_batch(self.binary_dataset)

        with torch.no_grad():
            output = model(ehr=batch["ehr"], label=batch["label"])

        self.assertIn("loss", output)
        self.assertIn("y_prob", output)
        self.assertEqual(output["logit"].shape, (2, 1))

    def test_output_shape(self):
        """Output tensor has correct shape [batch_size, num_labels]."""
        model = self._build_model(self.binary_dataset)
        batch = self._next_batch(self.binary_dataset)

        with torch.no_grad():
            output = model(**batch)

        self.assertEqual(output["logit"].shape, (2, 1))
        self.assertEqual(output["y_prob"].shape, (2, 1))

    def test_gradient_computation(self):
        """Gradients flow through the entire network."""
        model = self._build_model(self.binary_dataset)
        batch = self._next_batch(self.binary_dataset)

        output = model(**batch)
        loss = output["loss"]
        loss.backward()

        for name, parameter in model.named_parameters():
            if name == "_dummy_param":
                continue
            if parameter.requires_grad:
                if parameter.grad is None:
                    print(f"Gradient missing for: {name}")
                self.assertIsNotNone(parameter.grad)

    def test_missing_modality_robustness(self):
        """Model produces valid output for mixed modality availability."""
        model = self._build_model(self.binary_dataset)
        batch = self._next_batch(self.binary_dataset)
        cxr_mask = torch.tensor([1, 0], dtype=torch.long)

        with torch.no_grad():
            output = model(
                ehr=batch["ehr"],
                cxr=batch["cxr"],
                cxr_mask=cxr_mask,
                label=batch["label"],
            )

        self.assertEqual(output["logit"].shape, (2, 1))
        self.assertFalse(torch.isnan(output["y_prob"]).any().item())

    def test_binary_vs_multilabel_mode(self):
        """Model works in both binary and multilabel modes."""
        binary_model = self._build_model(self.binary_dataset)
        binary_batch = self._next_batch(self.binary_dataset)

        with torch.no_grad():
            binary_output = binary_model(**binary_batch)

        self.assertEqual(binary_output["logit"].shape[1], 1)

        multilabel_model = self._build_model(self.multilabel_dataset)
        multilabel_batch = self._next_batch(self.multilabel_dataset)

        with torch.no_grad():
            multilabel_output = multilabel_model(**multilabel_batch)

        self.assertEqual(multilabel_output["logit"].shape[1], 2)
        self.assertTrue(torch.all(multilabel_output["y_prob"] >= 0).item())
        self.assertTrue(torch.all(multilabel_output["y_prob"] <= 1).item())


if __name__ == "__main__":
    unittest.main()
