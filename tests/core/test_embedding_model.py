import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.embedding import EmbeddingModel


class TestEmbeddingModelSequence(unittest.TestCase):
    """Test EmbeddingModel with sequence processor inputs."""

    def setUp(self):
        """Set up test data and model with sequence inputs."""
        torch.manual_seed(42)
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "diagnoses": ["A", "B", "C"],
                "procedures": ["X", "Y"],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "diagnoses": ["D", "E"],
                "procedures": ["Y"],
                "label": 0,
            },
        ]

        self.input_schema = {
            "diagnoses": "sequence",
            "procedures": "sequence",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = EmbeddingModel(
            dataset=self.dataset, embedding_dim=32,
        )

    def test_initialization(self):
        """Test that the EmbeddingModel initializes correctly."""
        self.assertIsInstance(self.model, EmbeddingModel)
        self.assertEqual(self.model.embedding_dim, 32)
        self.assertIn("diagnoses", self.model.embedding_layers)
        self.assertIn("procedures", self.model.embedding_layers)

    def test_embedding_layers_are_correct_type(self):
        """Test that sequence inputs use nn.Embedding layers."""
        self.assertIsInstance(
            self.model.embedding_layers["diagnoses"], torch.nn.Embedding,
        )
        self.assertIsInstance(
            self.model.embedding_layers["procedures"], torch.nn.Embedding,
        )

    def test_forward_output_shapes(self):
        """Test that forward pass produces correct output shapes."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(loader))

        inputs = {}
        masks = {}
        for key in ["diagnoses", "procedures"]:
            feature = data_batch[key]
            if isinstance(feature, torch.Tensor):
                feature = (feature,)
            schema = self.dataset.input_processors[key].schema()
            inputs[key] = feature[schema.index("value")]
            if "mask" in schema:
                masks[key] = feature[schema.index("mask")]

        with torch.no_grad():
            embedded = self.model(inputs, masks=masks)

        self.assertIn("diagnoses", embedded)
        self.assertIn("procedures", embedded)
        self.assertEqual(embedded["diagnoses"].shape[-1], 32)
        self.assertEqual(embedded["procedures"].shape[-1], 32)
        self.assertEqual(embedded["diagnoses"].shape[0], 2)

    def test_forward_with_output_mask(self):
        """Test that forward pass returns masks when requested."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(loader))

        inputs = {}
        masks = {}
        for key in ["diagnoses", "procedures"]:
            feature = data_batch[key]
            if isinstance(feature, torch.Tensor):
                feature = (feature,)
            schema = self.dataset.input_processors[key].schema()
            inputs[key] = feature[schema.index("value")]
            if "mask" in schema:
                masks[key] = feature[schema.index("mask")]

        with torch.no_grad():
            embedded, out_masks = self.model(
                inputs, masks=masks, output_mask=True,
            )

        self.assertIsInstance(embedded, dict)
        self.assertIsInstance(out_masks, dict)
        self.assertIn("diagnoses", out_masks)
        self.assertIn("procedures", out_masks)

    def test_gradients_flow(self):
        """Test that gradients flow through the embedding layers."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(loader))

        inputs = {}
        masks = {}
        for key in ["diagnoses", "procedures"]:
            feature = data_batch[key]
            if isinstance(feature, torch.Tensor):
                feature = (feature,)
            schema = self.dataset.input_processors[key].schema()
            inputs[key] = feature[schema.index("value")]
            if "mask" in schema:
                masks[key] = feature[schema.index("mask")]

        embedded = self.model(inputs, masks=masks)
        loss = sum(v.sum() for v in embedded.values())
        loss.backward()

        has_gradient = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_gradient)


class TestEmbeddingModelTensor(unittest.TestCase):
    """Test EmbeddingModel with tensor processor inputs."""

    def setUp(self):
        """Set up test data and model with tensor inputs."""
        torch.manual_seed(42)
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "labs": [1.0, 2.0, 3.0],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "labs": [4.0, 5.0, 6.0],
                "label": 0,
            },
        ]

        self.input_schema = {"labs": "tensor"}
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = EmbeddingModel(
            dataset=self.dataset, embedding_dim=16,
        )

    def test_initialization(self):
        """Test that tensor inputs use nn.Linear layers."""
        self.assertIn("labs", self.model.embedding_layers)
        self.assertIsInstance(
            self.model.embedding_layers["labs"], torch.nn.Linear,
        )

    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shapes."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(loader))

        feature = data_batch["labs"]
        if isinstance(feature, torch.Tensor):
            feature = (feature,)
        schema = self.dataset.input_processors["labs"].schema()
        inputs = {"labs": feature[schema.index("value")]}

        with torch.no_grad():
            embedded = self.model(inputs)

        self.assertIn("labs", embedded)
        self.assertEqual(embedded["labs"].shape[-1], 16)
        self.assertEqual(embedded["labs"].shape[0], 2)


class TestEmbeddingModelMultiHot(unittest.TestCase):
    """Test EmbeddingModel with multi_hot processor inputs."""

    def setUp(self):
        """Set up test data and model with multi-hot inputs."""
        torch.manual_seed(42)
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "demographics": ["asian", "male"],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "demographics": ["white", "female"],
                "label": 0,
            },
        ]

        self.input_schema = {"demographics": "multi_hot"}
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = EmbeddingModel(
            dataset=self.dataset, embedding_dim=16,
        )

    def test_initialization(self):
        """Test that multi-hot inputs use nn.Linear layers."""
        self.assertIn("demographics", self.model.embedding_layers)
        self.assertIsInstance(
            self.model.embedding_layers["demographics"], torch.nn.Linear,
        )

    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shapes."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(loader))

        feature = data_batch["demographics"]
        if isinstance(feature, torch.Tensor):
            feature = (feature,)
        schema = self.dataset.input_processors["demographics"].schema()
        inputs = {"demographics": feature[schema.index("value")]}

        with torch.no_grad():
            embedded = self.model(inputs)

        self.assertIn("demographics", embedded)
        self.assertEqual(embedded["demographics"].shape[-1], 16)
        self.assertEqual(embedded["demographics"].shape[0], 2)


class TestEmbeddingModelNestedSequence(unittest.TestCase):
    """Test EmbeddingModel with nested_sequence processor inputs."""

    def setUp(self):
        """Set up test data and model with nested sequence inputs."""
        torch.manual_seed(42)
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": [["A", "B"], ["C", "D", "E"]],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": [["F"], ["G", "H"]],
                "label": 0,
            },
        ]

        self.input_schema = {"conditions": "nested_sequence"}
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = EmbeddingModel(
            dataset=self.dataset, embedding_dim=16,
        )

    def test_initialization(self):
        """Test that nested sequence inputs use nn.Embedding layers."""
        self.assertIn("conditions", self.model.embedding_layers)
        self.assertIsInstance(
            self.model.embedding_layers["conditions"], torch.nn.Embedding,
        )

    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shapes."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(loader))

        feature = data_batch["conditions"]
        if isinstance(feature, torch.Tensor):
            feature = (feature,)
        schema = self.dataset.input_processors["conditions"].schema()
        inputs = {"conditions": feature[schema.index("value")]}
        masks = {}
        if "mask" in schema:
            masks["conditions"] = feature[schema.index("mask")]

        with torch.no_grad():
            embedded = self.model(inputs, masks=masks)

        self.assertIn("conditions", embedded)
        self.assertEqual(embedded["conditions"].shape[-1], 16)
        self.assertEqual(embedded["conditions"].shape[0], 2)


class TestEmbeddingModelMixedInputs(unittest.TestCase):
    """Test EmbeddingModel with mixed processor types."""

    def setUp(self):
        """Set up test data and model with mixed input types."""
        torch.manual_seed(42)
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "diagnoses": ["A", "B", "C"],
                "labs": [1.0, 2.0, 3.0],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "diagnoses": ["D", "E"],
                "labs": [4.0, 5.0, 6.0],
                "label": 0,
            },
        ]

        self.input_schema = {
            "diagnoses": "sequence",
            "labs": "tensor",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = EmbeddingModel(
            dataset=self.dataset, embedding_dim=32,
        )

    def test_mixed_initialization(self):
        """Test that mixed inputs use appropriate layer types."""
        self.assertIsInstance(
            self.model.embedding_layers["diagnoses"], torch.nn.Embedding,
        )
        self.assertIsInstance(
            self.model.embedding_layers["labs"], torch.nn.Linear,
        )

    def test_mixed_forward(self):
        """Test forward pass with mixed input types."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(loader))

        inputs = {}
        masks = {}
        for key in ["diagnoses", "labs"]:
            feature = data_batch[key]
            if isinstance(feature, torch.Tensor):
                feature = (feature,)
            schema = self.dataset.input_processors[key].schema()
            inputs[key] = feature[schema.index("value")]
            if "mask" in schema:
                masks[key] = feature[schema.index("mask")]

        with torch.no_grad():
            embedded = self.model(inputs, masks=masks)

        self.assertEqual(embedded["diagnoses"].shape[-1], 32)
        self.assertEqual(embedded["labs"].shape[-1], 32)

    def test_custom_embedding_dim(self):
        """Test EmbeddingModel with a custom embedding dimension."""
        model = EmbeddingModel(dataset=self.dataset, embedding_dim=64)
        self.assertEqual(model.embedding_dim, 64)

        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(loader))

        inputs = {}
        masks = {}
        for key in ["diagnoses", "labs"]:
            feature = data_batch[key]
            if isinstance(feature, torch.Tensor):
                feature = (feature,)
            schema = self.dataset.input_processors[key].schema()
            inputs[key] = feature[schema.index("value")]
            if "mask" in schema:
                masks[key] = feature[schema.index("mask")]

        with torch.no_grad():
            embedded = model(inputs, masks=masks)

        self.assertEqual(embedded["diagnoses"].shape[-1], 64)
        self.assertEqual(embedded["labs"].shape[-1], 64)


if __name__ == "__main__":
    unittest.main()
