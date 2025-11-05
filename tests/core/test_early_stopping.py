import unittest

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import MLP
from pyhealth.trainer import Trainer


class TestEarlyStopping(unittest.TestCase):
    """Test cases for early stopping with patience in Trainer."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": f"patient-{i}",
                "visit_id": f"visit-{i}",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": [1.0, 2.0, 3.5, 4],
                "label": i % 2,
            }
            for i in range(100)
        ]

        # Define input and output schemas
        self.input_schema = {
            "conditions": "sequence",
            "procedures": "tensor",
        }
        self.output_schema = {"label": "binary"}

        # Split into train and val
        self.train_dataset = SampleDataset(
            samples=self.samples[:80],
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="train",
        )

        self.val_dataset = SampleDataset(
            samples=self.samples[80:],
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="val",
        )

        # Create model
        self.model = MLP(dataset=self.train_dataset)

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers when patience is exceeded."""
        trainer = Trainer(model=self.model)

        # Mock the evaluate method to return progressively worse metrics
        epoch_count = [0]

        def mock_evaluate(dataloader):
            epoch_count[0] += 1
            # Loss increases (gets worse) each epoch
            return {"loss": 1.0 + epoch_count[0] * 0.1, "accuracy": 0.5}

        trainer.evaluate = mock_evaluate

        # Create dataloaders
        train_loader = get_dataloader(self.train_dataset,
                                      batch_size=16,
                                      shuffle=True)
        val_loader = get_dataloader(self.val_dataset,
                                    batch_size=16,
                                    shuffle=False)

        # Train with patience=3, should stop before 50 epochs
        max_epochs = 50
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=max_epochs,
            monitor="loss",
            monitor_criterion="min",
            patience=3,
            load_best_model_at_last=False,
        )

        # Should stop at epoch 4 (initial + 3 patience epochs)
        self.assertLessEqual(epoch_count[0], 5)

    def test_early_stopping_no_trigger(self):
        """Test that training continues when metric improves."""
        trainer = Trainer(model=self.model)

        # Mock the evaluate method to return improving metrics
        epoch_count = [0]

        def mock_evaluate(dataloader):
            epoch_count[0] += 1
            # Loss decreases (improves) each epoch
            return {
                "loss": 1.0 - epoch_count[0] * 0.01,
                "accuracy": 0.5 + epoch_count[0] * 0.01,
            }

        trainer.evaluate = mock_evaluate

        # Create dataloaders
        train_loader = get_dataloader(self.train_dataset,
                                      batch_size=16,
                                      shuffle=True)
        val_loader = get_dataloader(self.val_dataset,
                                    batch_size=16,
                                    shuffle=False)

        # Train with patience=3, should complete all epochs
        max_epochs = 50
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=max_epochs,
            monitor="loss",
            monitor_criterion="min",
            patience=3,
            load_best_model_at_last=False,
        )

        # Should complete all epochs
        self.assertEqual(epoch_count[0], max_epochs)

    def test_no_early_stopping_without_patience(self):
        """Test that patience=None disables early stopping."""
        trainer = Trainer(model=self.model)

        # Mock the evaluate method to return constant bad metrics
        epoch_count = [0]

        def mock_evaluate(dataloader):
            epoch_count[0] += 1
            return {"loss": 1.0, "accuracy": 0.5}

        trainer.evaluate = mock_evaluate

        # Create dataloaders
        train_loader = get_dataloader(self.train_dataset,
                                      batch_size=16,
                                      shuffle=True)
        val_loader = get_dataloader(self.val_dataset,
                                    batch_size=16,
                                    shuffle=False)

        # Train without patience
        max_epochs = 50
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=max_epochs,
            monitor="loss",
            monitor_criterion="min",
            patience=None,
            load_best_model_at_last=False,
        )

        # Should complete all epochs
        self.assertEqual(epoch_count[0], max_epochs)


if __name__ == "__main__":
    unittest.main()
