import unittest
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import StageNet


class TestStageNet(unittest.TestCase):
    """Test cases for the StageNet model with StageNetProcessor."""

    def setUp(self):
        """Set up test data and model."""
        # Create samples with different StageNet input patterns
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                # Case 1: Flat code sequence with time intervals
                "codes": ([0.0, 2.0, 1.3], ["505800458", "50580045810", "50580045811"]),
                # Case 2: Nested code sequence with time intervals
                "procedures": (
                    [0.0, 1.5],
                    [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],
                ),
                # Case 3: Numeric feature vectors without time
                "lab_values": (None, [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]]),
                "label": 1,
            },
            {
                "patient_id": "patient-0",
                "visit_id": "visit-1",
                "codes": (
                    [0.0, 2.0, 1.3, 1.0, 2.0],
                    [
                        "55154191800",
                        "551541928",
                        "55154192800",
                        "705182798",
                        "70518279800",
                    ],
                ),
                "procedures": ([0.0], [["A04A", "B035", "C129"]]),
                "lab_values": (
                    None,
                    [
                        [1.4, 3.2, 3.5],
                        [4.1, 5.9, 1.7],
                        [4.5, 5.9, 1.7],
                    ],
                ),
                "label": 0,
            },
        ]

        # Define input and output schemas
        self.input_schema = {
            "codes": "stagenet",
            "procedures": "stagenet",
            "lab_values": "stagenet_tensor",  # numeric features
        }
        self.output_schema = {"label": "binary"}

        # Create dataset
        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_stagenet",
        )

        # Create model
        self.model = StageNet(dataset=self.dataset, chunk_size=2, levels=3)

    def test_model_initialization(self):
        """Test that the StageNet model initializes correctly."""
        self.assertIsInstance(self.model, StageNet)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.chunk_size, 2)
        self.assertEqual(self.model.levels, 3)
        self.assertEqual(len(self.model.feature_keys), 3)
        self.assertIn("codes", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertIn("lab_values", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_processor_output_format(self):
        """Test that processors produce (time, values) tuples."""
        # Get a sample from the dataset
        sample = self.dataset[0]

        # Check that each feature is a tuple with time and values
        for key in ["codes", "procedures", "lab_values"]:
            self.assertIsInstance(sample[key], tuple)
            self.assertEqual(len(sample[key]), 2)
            time, values = sample[key]
            # Values should always be a tensor
            self.assertIsInstance(values, torch.Tensor)
            # Time can be None or a tensor
            if time is not None:
                self.assertIsInstance(time, torch.Tensor)

    def test_dataloader_batching(self):
        """Test dataloader batching of temporal tuples."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        # Check structure
        self.assertIn("codes", data_batch)
        self.assertIn("procedures", data_batch)
        self.assertIn("lab_values", data_batch)
        self.assertIn("label", data_batch)

        # Check that batched features are still tuples
        for key in ["codes", "procedures", "lab_values"]:
            feature = data_batch[key]
            self.assertIsInstance(feature, tuple)
            self.assertEqual(len(feature), 2)
            time, values = feature
            self.assertIsInstance(values, torch.Tensor)
            self.assertEqual(values.shape[0], 2)  # batch size

            # Check time tensor
            if time is not None:
                self.assertIsInstance(time, torch.Tensor)
                self.assertEqual(time.shape[0], 2)  # batch size

    def test_tensor_shapes(self):
        """Test that tensor shapes are correct after batching."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        # Flat codes: [batch, seq_len]
        time, codes_values = data_batch["codes"]
        self.assertEqual(len(codes_values.shape), 2)
        self.assertEqual(codes_values.shape[0], 2)  # batch size
        self.assertEqual(time.shape[0], 2)  # batch size
        self.assertEqual(len(time.shape), 2)  # [batch, seq_len]

        # Nested codes (procedures): [batch, seq_len, max_inner_len]
        time, procedures_values = data_batch["procedures"]
        self.assertEqual(len(procedures_values.shape), 3)
        self.assertEqual(procedures_values.shape[0], 2)  # batch size
        self.assertEqual(time.shape[0], 2)  # batch size

        # Nested numerics (lab_values): [batch, seq_len, feature_dim]
        time, lab_values = data_batch["lab_values"]
        self.assertEqual(len(lab_values.shape), 3)
        self.assertEqual(lab_values.shape[0], 2)  # batch size
        self.assertEqual(lab_values.shape[2], 3)  # feature_dim
        self.assertIsNone(time)  # No time for lab_values

    def test_model_forward(self):
        """Test that the StageNet model forward pass works."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        # Debug: Print feature types and shapes
        print("\n=== Debug: test_model_forward ===")
        for key in ["codes", "procedures", "lab_values"]:
            if key in data_batch:
                time, values = data_batch[key]
                print(f"{key}:")
                print(f"  value dtype: {values.dtype}")
                print(f"  value shape: {values.shape}")
                if time is not None:
                    print(f"  time dtype: {time.dtype}")
                    print(f"  time shape: {time.shape}")

        # Forward pass
        with torch.no_grad():
            ret = self.model(**data_batch)

        # Check output structure
        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        # Check tensor shapes
        self.assertEqual(ret["y_prob"].shape[0], 2)  # batch size
        self.assertEqual(ret["y_true"].shape[0], 2)  # batch size
        self.assertEqual(ret["logit"].shape[0], 2)  # batch size

        # Check that loss is a scalar
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the StageNet model backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        # Forward pass
        ret = self.model(**data_batch)

        # Backward pass
        ret["loss"].backward()

        # Check that at least one parameter has gradients
        has_gradient = False
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradient = True
                break
        self.assertTrue(
            has_gradient, "No parameters have gradients after backward pass"
        )

    def test_time_handling_with_none(self):
        """Test StageNet with time=None (uniform intervals)."""
        # Create samples with all time=None
        samples_no_time = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "codes": (None, ["code1", "code2", "code3"]),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "codes": (None, ["code4", "code5"]),
                "label": 0,
            },
        ]

        dataset_no_time = SampleDataset(
            samples=samples_no_time,
            input_schema={"codes": "stagenet"},
            output_schema={"label": "binary"},
            dataset_name="test_stagenet_no_time",
        )

        model_no_time = StageNet(dataset=dataset_no_time, chunk_size=2, levels=2)

        # Test forward pass
        train_loader = get_dataloader(dataset_no_time, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        # Verify time is None (first element of tuple)
        time, values = data_batch["codes"]
        self.assertIsNone(time)

        # Forward pass should work
        with torch.no_grad():
            ret = model_no_time(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_nested_code_padding(self):
        """Test that nested codes are padded correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        time, value_tensor = data_batch["procedures"]

        # Shape should be [batch=2, seq_len, max_inner_len]
        self.assertEqual(len(value_tensor.shape), 3)
        batch_size, seq_len, max_inner_len = value_tensor.shape

        # First sample has [3 codes, 2 codes] -> max_inner_len = 3
        # Second sample has [3 codes] -> padded to [3 codes]
        self.assertEqual(max_inner_len, 3)

        # Padding value should be 0 (after embedding lookup)
        # Check that shorter sequences are padded
        # Second visit has 2 codes, position [0, 1, 2] is padding
        self.assertEqual(value_tensor[0, 1, 2].item(), 0)

    def test_custom_hyperparameters(self):
        """Test StageNet model with custom hyperparameters."""
        model = StageNet(
            dataset=self.dataset,
            embedding_dim=64,
            chunk_size=3,
            levels=2,
            dropout=0.2,
        )

        self.assertEqual(model.embedding_dim, 64)
        # hidden_dim = chunk_size * levels
        self.assertEqual(model.chunk_size * model.levels, 6)
        self.assertEqual(model.chunk_size, 3)
        self.assertEqual(model.levels, 2)

        # Test forward pass
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_single_feature_input(self):
        """Test StageNet with only a single feature."""
        samples_single = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "codes": ([0.0, 1.0], ["code1", "code2"]),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "codes": ([0.0, 0.5, 1.5], ["code3", "code4", "code5"]),
                "label": 0,
            },
        ]

        dataset_single = SampleDataset(
            samples=samples_single,
            input_schema={"codes": "stagenet"},
            output_schema={"label": "binary"},
            dataset_name="test_stagenet_single",
        )

        model_single = StageNet(dataset=dataset_single, chunk_size=2, levels=2)

        # Test forward pass
        train_loader = get_dataloader(dataset_single, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model_single(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertEqual(ret["y_prob"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
