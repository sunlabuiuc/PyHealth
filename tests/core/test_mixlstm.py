import os
import tempfile
import shutil
import unittest
import torch
import torch.nn as nn
from pyhealth.models import MixLSTM, BaselineLSTM

class DummySchemaInfo:
    """Mocks the dimension attribute of PyHealth's input schema.
    
    Args:
        dim (int): The feature dimension size.
    """
    def __init__(self, dim: int) -> None:
        self.dim = dim

class DummyProcessor:
    """Mocks PyHealth's output processor to return the expected output size.
    
    Args:
        size (int): The output dimension size.
    """
    def __init__(self, size: int) -> None:
        self._size = size
    def size(self) -> int:
        """Returns the expected output size."""
        return self._size

class DummyDataset:
    """Mocks a PyHealth SampleDataset for binary mortality prediction.
    
    Args:
        input_dim (int, optional): The input feature dimension. Defaults to 76.
        output_size (int, optional): The expected output size. Defaults to 1.
    """
    def __init__(self, input_dim: int = 76, output_size: int = 1) -> None:
        self.input_schema = {"physio_features": DummySchemaInfo(dim=input_dim)}
        self.output_schema = {"mortality": "binary"} # PyHealth mode resolution
        self.output_processors = {"mortality": DummyProcessor(size=output_size)}

class TestMixLSTM(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the synthetic data and model before each test."""
        # Hyperparameters
        self.batch_size = 2
        self.seq_len = 48
        self.input_dim = 76
        self.hidden_size = 128
        self.num_experts = 2
        
        # 1. Mock Dataset
        self.dataset = DummyDataset(input_dim=self.input_dim)
        
        # 2. Instantiate Model
        self.model = MixLSTM(
            dataset=self.dataset,
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            max_seq_len=self.seq_len
        )

        # 3. Instantiate Baseline Model
        self.baseline_model = BaselineLSTM(
            dataset=self.dataset,
            hidden_size=self.hidden_size,
            max_seq_len=self.seq_len
        )
        
        # 4. Create Synthetic Data
        # Input features: (batch_size, seq_len, input_dim)
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        # Binary labels: (batch_size, 1)
        self.y_true = torch.randint(0, 2, (self.batch_size, 1)).float()
        
        # Formulate kwargs exactly as PyHealth's Trainer would pass them
        self.kwargs = {
            "physio_features": self.x,
            "mortality": self.y_true
        }

    def test_instantiation(self) -> None:
        """Verify the model initializes correctly and inherits from nn.Module."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.input_size, self.input_dim)
        self.assertEqual(self.model.output_size, 1)
        self.assertEqual(self.model.mode, "binary")

    def test_forward_pass_and_shapes(self) -> None:
        """Verify the forward pass runs without error and returns correct shapes."""
        # Run forward pass
        out = self.model(**self.kwargs)
        
        # Check required PyHealth dictionary keys
        self.assertIn("logit", out)
        self.assertIn("y_prob", out)
        self.assertIn("loss", out)
        
        # Check shapes
        # Logits and probabilities should be (batch_size, output_size)
        expected_shape = (self.batch_size, 1)
        self.assertEqual(out["logit"].shape, expected_shape)
        self.assertEqual(out["y_prob"].shape, expected_shape)
        
        # Loss should be a 0-dimensional scalar tensor
        self.assertEqual(out["loss"].dim(), 0)

    def test_gradient_computation(self) -> None:
        """Verify that gradients can be computed and flow back through the model."""
        # Forward pass
        out = self.model(**self.kwargs)
        loss = out["loss"]
        
        # Ensure no gradients exist yet
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNone(param.grad)
        
        # Backward pass
        loss.backward()
        
        # Verify gradients are populated and are not entirely zero
        has_gradients = False
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name == '_dummy_param':
                    continue  # Skip the dummy parameter used for device inference

                self.assertIsNotNone(
                    param.grad, 
                    f"Gradient missing for parameter: {name}"
                )
                # Check if at least some gradient values are non-zero
                if torch.sum(torch.abs(param.grad)) > 0:
                    has_gradients = True
                    
        self.assertTrue(has_gradients, "Gradients are all zero, check the computational graph.")
    
    def test_baseline_instantiation(self) -> None:
        """Verify the BaselineLSTM initializes correctly."""
        self.assertIsInstance(self.baseline_model, nn.Module)
        self.assertEqual(self.baseline_model.input_size, self.input_dim)
        self.assertEqual(self.baseline_model.output_size, 1)

    def test_baseline_forward_pass_and_shapes(self) -> None:
        """Verify the forward pass runs for BaselineLSTM."""
        out = self.baseline_model(**self.kwargs)
        
        self.assertIn("logit", out)
        self.assertIn("y_prob", out)
        self.assertIn("loss", out)
        
        expected_shape = (self.batch_size, 1)
        self.assertEqual(out["logit"].shape, expected_shape)
        self.assertEqual(out["y_prob"].shape, expected_shape)
        self.assertEqual(out["loss"].dim(), 0)

    def test_baseline_gradient_computation(self) -> None:
        """Verify that gradients flow back through the BaselineLSTM."""
        out = self.baseline_model(**self.kwargs)
        loss = out["loss"]
        
        loss.backward()
        
        has_gradients = False
        for name, param in self.baseline_model.named_parameters():
            if param.requires_grad:
                if name == '_dummy_param':
                    continue
                self.assertIsNotNone(param.grad, f"Gradient missing for parameter: {name}")
                if torch.sum(torch.abs(param.grad)) > 0:
                    has_gradients = True
                    
        self.assertTrue(has_gradients, "BaselineLSTM gradients are all zero.")
    

    def test_model_save_and_load(self) -> None:
        """Test model saving and loading using a temporary directory with proper cleanup."""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Set up the save path
            save_path = os.path.join(temp_dir, "test_mixlstm.pth")
            
            # Save the model state dictionary
            torch.save(self.model.state_dict(), save_path)
            self.assertTrue(os.path.exists(save_path), "Model weight file was not created.")
            
            # Instantiate a new model to receive the loaded weights
            new_model = MixLSTM(
                dataset=self.dataset,
                input_size=self.input_dim,
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                max_seq_len=self.seq_len
            )
            new_model.load_state_dict(torch.load(save_path))
            
            # Verify parameters were successfully loaded (check if weights match)
            original_param = next(self.model.parameters())
            loaded_param = next(new_model.parameters())
            self.assertTrue(torch.equal(original_param, loaded_param))
            
        finally:
            # Proper cleanup: unconditionally remove the temp directory and its contents
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    unittest.main()