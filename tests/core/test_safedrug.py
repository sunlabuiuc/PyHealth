import unittest
import torch

from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import SafeDrug
from pyhealth.tasks import DrugRecommendationMIMIC3


class TestSafeDrug(unittest.TestCase):
    """Test cases for the SafeDrug model."""

    @classmethod
    def setUpClass(cls):
        """Set up test data and model for all tests."""
        try:
            cls.base_dataset = MIMIC3Dataset(
                root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
                tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
                dev=True,
            )
            task = DrugRecommendationMIMIC3()
            cls.dataset = cls.base_dataset.set_task(task)
            
            if len(cls.dataset.samples) == 0:
                raise unittest.SkipTest(
                    "No samples generated from MIMIC3 dataset"
                )
        except Exception as e:
            raise unittest.SkipTest(
                f"MIMIC3 dataset not available for testing: {e}. "
                "Please ensure MIMIC3 dataset is accessible."
            )

    def setUp(self):
        """Set up model for each test."""
        self.model = SafeDrug(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the SafeDrug model initializes correctly."""
        self.assertIsInstance(self.model, SafeDrug)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.hidden_dim, 128)
        self.assertEqual(self.model.num_layers, 1)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "drugs")
        self.assertEqual(self.model.mode, "multilabel")

        self.assertIsNotNone(self.model.safedrug)
        self.assertIsNotNone(self.model.cond_rnn)
        self.assertIsNotNone(self.model.proc_rnn)
        self.assertIsNotNone(self.model.query)

    def test_model_forward(self):
        """Test that the SafeDrug model forward pass works correctly."""
        train_loader = get_dataloader(
            self.dataset, batch_size=2, shuffle=True
        )
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)

        batch_size = ret["y_prob"].shape[0]
        self.assertEqual(ret["y_true"].shape[0], batch_size)
        self.assertEqual(ret["y_prob"].shape[1], self.model.label_size)
        self.assertEqual(ret["y_true"].shape[1], self.model.label_size)

        self.assertEqual(ret["loss"].dim(), 0)
        
        self.assertTrue(
            torch.all(ret["y_prob"] >= 0) and torch.all(ret["y_prob"] <= 1),
            "y_prob should be in range [0, 1]"
        )

    def test_model_backward(self):
        """Test that the SafeDrug model backward pass works correctly."""
        train_loader = get_dataloader(
            self.dataset, batch_size=2, shuffle=True
        )
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)

        ret["loss"].backward()

        has_gradient = False
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradient = True
                break
        self.assertTrue(
            has_gradient, "No parameters have gradients after backward pass"
        )

    def test_custom_hyperparameters(self):
        """Test SafeDrug model with custom hyperparameters."""
        model = SafeDrug(
            dataset=self.dataset,
            embedding_dim=64,
            hidden_dim=32,
            num_layers=2,
            dropout=0.3,
        )

        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.hidden_dim, 32)
        self.assertEqual(model.num_layers, 2)
        self.assertEqual(model.dropout, 0.3)

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        
        self.assertEqual(ret["y_prob"].shape[1], model.label_size)
        self.assertEqual(ret["y_true"].shape[1], model.label_size)

    def test_forward_input_format(self):
        """Test that forward pass accepts correct input format.

        note:
        - conditions: tensor of shape [batch, visits, codes_per_visit] (already processed)
        - procedures: tensor of shape [batch, visits, codes_per_visit] (already processed)
        - drugs: tensor of shape [batch, num_labels] (already processed)
        """
        train_loader = get_dataloader(
            self.dataset, batch_size=2, shuffle=True
        )
        data_batch = next(iter(train_loader))

        self.assertIn("conditions", data_batch)
        self.assertIn("procedures", data_batch)
        self.assertIn("drugs", data_batch)

        conditions = data_batch["conditions"]
        procedures = data_batch["procedures"]
        drugs = data_batch["drugs"]

        self.assertIsInstance(conditions, torch.Tensor)
        self.assertIsInstance(procedures, torch.Tensor)
        self.assertIsInstance(drugs, torch.Tensor)

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_model_device(self):
        """Test that model can be moved to different devices."""
        self.assertEqual(next(self.model.parameters()).device.type, "cpu")

        if torch.cuda.is_available():
            model_cuda = SafeDrug(dataset=self.dataset)
            model_cuda = model_cuda.to("cuda")
            self.assertEqual(
                next(model_cuda.parameters()).device.type, "cuda"
            )


if __name__ == "__main__":
    unittest.main()

