import unittest
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import MLP


class TestProcessorTransfer(unittest.TestCase):
    """Test cases for transferring pre-fitted processors between datasets."""

    def setUp(self):
        """Set up train and test data."""
        # Training samples
        self.train_samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": [1.0, 2.0, 3.5, 4.0],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": [5.0, 2.0, 3.5, 4.0],
                "label": 1,
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "conditions": ["cond-12", "cond-50"],
                "procedures": [1.5, 3.0, 2.5, 5.0],
                "label": 0,
            },
        ]

        # Test samples with potentially new codes and different distributions
        self.test_samples = [
            {
                "patient_id": "patient-3",
                "visit_id": "visit-3",
                "conditions": ["cond-33", "cond-99"],  # cond-99 is new
                "procedures": [2.0, 3.0, 4.5, 6.0],
                "label": 1,
            },
            {
                "patient_id": "patient-4",
                "visit_id": "visit-4",
                "conditions": ["cond-80", "cond-12"],
                "procedures": [3.0, 1.0, 2.0, 7.0],
                "label": 0,
            },
        ]

        # Define input and output schemas
        self.input_schema = {
            "conditions": "sequence",  # sequence of condition codes
            "procedures": "tensor",  # tensor of procedure values
        }
        self.output_schema = {"label": "binary"}  # binary classification

    def test_basic_processor_transfer(self):
        """Test basic processor transfer from train to test dataset."""
        # Create training dataset (processors will be fitted)
        train_dataset = SampleDataset(
            samples=self.train_samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="train",
        )

        # Create test dataset with transferred processors
        test_dataset = SampleDataset(
            samples=self.test_samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
            input_processors=train_dataset.input_processors,
            output_processors=train_dataset.output_processors,
        )

        # Verify processors are the same objects
        self.assertIs(
            train_dataset.input_processors["conditions"],
            test_dataset.input_processors["conditions"],
        )
        self.assertIs(
            train_dataset.output_processors["label"],
            test_dataset.output_processors["label"],
        )

        # Verify datasets work correctly
        self.assertEqual(len(train_dataset), 3)
        self.assertEqual(len(test_dataset), 2)

    def test_processor_vocabulary_consistency(self):
        """Test that transferred processors maintain vocabulary consistency."""
        # Create training dataset
        train_dataset = SampleDataset(
            samples=self.train_samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="train",
        )

        # Get vocabulary from train processor
        # (use code_vocab for SequenceProcessor)
        train_vocab = train_dataset.input_processors["conditions"].code_vocab

        # Create test dataset with transferred processors
        test_dataset = SampleDataset(
            samples=self.test_samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
            input_processors=train_dataset.input_processors,
            output_processors=train_dataset.output_processors,
        )

        # Get vocabulary from test processor
        # (use code_vocab for SequenceProcessor)
        test_vocab = test_dataset.input_processors["conditions"].code_vocab

        # Vocabularies should be identical (same object)
        self.assertIs(train_vocab, test_vocab)

        # Check that vocabulary contains training codes
        self.assertIn("cond-33", train_vocab)
        self.assertIn("cond-86", train_vocab)
        self.assertIn("cond-80", train_vocab)
        self.assertIn("cond-12", train_vocab)
        self.assertIn("cond-50", train_vocab)

        # Note: cond-99 from test set WILL be added to vocabulary
        # during processing because SequenceProcessor adds new codes
        # on the fly in process(). The key is that the processor
        # object itself is shared
        self.assertIn("cond-99", train_vocab)

    def test_model_training_with_transferred_processors(self):
        """Test end-to-end training and inference with processor transfer."""
        # Create training dataset
        train_dataset = SampleDataset(
            samples=self.train_samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="train",
        )

        # Create test dataset with transferred processors
        test_dataset = SampleDataset(
            samples=self.test_samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
            input_processors=train_dataset.input_processors,
            output_processors=train_dataset.output_processors,
        )

        # Create model
        model = MLP(dataset=train_dataset, embedding_dim=64, hidden_dim=32)

        # Test training
        train_loader = get_dataloader(train_dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = model(**data_batch)
        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

        # Test inference on test set
        test_loader = get_dataloader(test_dataset, batch_size=2, shuffle=False)
        test_batch = next(iter(test_loader))

        with torch.no_grad():
            test_ret = model(**test_batch)

        self.assertIn("y_prob", test_ret)
        self.assertEqual(test_ret["y_prob"].shape[0], 2)

    def test_without_processor_transfer(self):
        """Test that without transfer, each dataset fits its own processors."""
        # Create training dataset
        train_dataset = SampleDataset(
            samples=self.train_samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="train",
        )

        # Create test dataset WITHOUT transferred processors
        test_dataset = SampleDataset(
            samples=self.test_samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        # Processors should be different objects
        self.assertIsNot(
            train_dataset.input_processors["conditions"],
            test_dataset.input_processors["conditions"],
        )

        # Vocabularies should be different
        train_vocab = train_dataset.input_processors["conditions"].code_vocab
        test_vocab = test_dataset.input_processors["conditions"].code_vocab

        self.assertIsNot(train_vocab, test_vocab)

        # Test vocab should contain cond-99, train vocab should not
        self.assertNotIn("cond-99", train_vocab)
        self.assertIn("cond-99", test_vocab)

    def test_partial_processor_transfer(self):
        """Test transferring only input processors, not output processors."""
        # Create training dataset
        train_dataset = SampleDataset(
            samples=self.train_samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="train",
        )

        # Create test dataset with only input processors transferred
        test_dataset = SampleDataset(
            samples=self.test_samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
            input_processors=train_dataset.input_processors,
            # output_processors NOT transferred
        )

        # Input processors should be the same
        self.assertIs(
            train_dataset.input_processors["conditions"],
            test_dataset.input_processors["conditions"],
        )

        # Output processors should be different
        self.assertIsNot(
            train_dataset.output_processors["label"],
            test_dataset.output_processors["label"],
        )

    def test_empty_processor_dict_transfer(self):
        """Test passing empty processor dictionaries."""
        # Create dataset with empty processor dicts (should fit new ones)
        dataset = SampleDataset(
            samples=self.train_samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="train",
            input_processors={},
            output_processors={},
        )

        # Should have fitted processors
        self.assertIn("conditions", dataset.input_processors)
        self.assertIn("procedures", dataset.input_processors)
        self.assertIn("label", dataset.output_processors)

    def test_cross_validation_scenario(self):
        """Test processor transfer in a cross-validation scenario."""
        import copy

        # Simulate 3-fold CV with all samples having both labels
        # Use deep copies to avoid mutating shared samples
        fold1 = [
            copy.deepcopy(self.train_samples[0]),
            copy.deepcopy(self.train_samples[1]),
        ]  # has 0 and 1
        fold2 = [
            copy.deepcopy(self.train_samples[1]),
            copy.deepcopy(self.train_samples[2]),
        ]  # has 1 and 0

        # Fold 1 as train
        train_dataset = SampleDataset(
            samples=fold1,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="fold1_train",
        )

        # Fold 2 as validation with transferred processors
        val_dataset = SampleDataset(
            samples=fold2,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="fold1_val",
            input_processors=train_dataset.input_processors,
            output_processors=train_dataset.output_processors,
        )

        # Verify they share processors
        self.assertIs(
            train_dataset.input_processors["conditions"],
            val_dataset.input_processors["conditions"],
        )

        # Both datasets should be valid
        self.assertEqual(len(train_dataset), 2)
        self.assertEqual(len(val_dataset), 2)

    def test_multimodal_processor_transfer(self):
        """Test processor transfer with multiple input modalities."""
        # Samples with more modalities
        multi_train = [
            {
                "patient_id": "p-0",
                "visit_id": "v-0",
                "conditions": ["c1", "c2"],
                "procedures": [1.0, 2.0],
                "medications": ["m1", "m2", "m3"],
                "age": 45.5,
                "label": 0,
            },
            {
                "patient_id": "p-1",
                "visit_id": "v-1",
                "conditions": ["c2", "c3"],
                "procedures": [3.0, 4.0],
                "medications": ["m2", "m4"],
                "age": 60.2,
                "label": 1,
            },
        ]

        multi_test = [
            {
                "patient_id": "p-2",
                "visit_id": "v-2",
                "conditions": ["c1", "c4"],  # c4 is new
                "procedures": [2.0, 3.0],
                "medications": ["m1", "m5"],  # m5 is new
                "age": 55.0,
                "label": 0,
            },
        ]

        multi_input_schema = {
            "conditions": "sequence",
            "procedures": "tensor",
            "medications": "sequence",
            "age": "tensor",
        }

        # Create train dataset
        train_dataset = SampleDataset(
            samples=multi_train,
            input_schema=multi_input_schema,
            output_schema=self.output_schema,
            dataset_name="multi_train",
        )

        # Create test dataset with transferred processors
        test_dataset = SampleDataset(
            samples=multi_test,
            input_schema=multi_input_schema,
            output_schema=self.output_schema,
            dataset_name="multi_test",
            input_processors=train_dataset.input_processors,
            output_processors=train_dataset.output_processors,
        )

        # Verify all input processors are shared
        for key in multi_input_schema.keys():
            self.assertIs(
                train_dataset.input_processors[key],
                test_dataset.input_processors[key],
            )

        # Check medication vocabulary
        med_vocab = train_dataset.input_processors["medications"].code_vocab
        self.assertIn("m1", med_vocab)
        self.assertIn("m2", med_vocab)
        self.assertIn("m3", med_vocab)
        self.assertIn("m4", med_vocab)
        # m5 from test will be added during processing since
        # SequenceProcessor adds new codes on the fly
        self.assertIn("m5", med_vocab)

    def test_backward_compatibility(self):
        """Test that existing code without processor transfer still works."""
        # Old-style usage without any processor parameters
        dataset1 = SampleDataset(
            samples=self.train_samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
        )

        dataset2 = SampleDataset(
            samples=self.test_samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
        )

        # Both should work independently
        self.assertEqual(len(dataset1), 3)
        self.assertEqual(len(dataset2), 2)

        # Both should have processors
        self.assertIn("conditions", dataset1.input_processors)
        self.assertIn("conditions", dataset2.input_processors)

        # But processors should be different
        self.assertIsNot(
            dataset1.input_processors["conditions"],
            dataset2.input_processors["conditions"],
        )


class TestProcessorTransferEdgeCases(unittest.TestCase):
    """Test edge cases for processor transfer functionality."""

    def test_none_processors_explicitly(self):
        """Test explicitly passing None for processors."""
        samples = [
            {
                "patient_id": "p-0",
                "visit_id": "v-0",
                "conditions": ["c1", "c2"],
                "label": 0,
            },
            {
                "patient_id": "p-1",
                "visit_id": "v-1",
                "conditions": ["c3", "c4"],
                "label": 1,
            },
        ]

        input_schema = {"conditions": "sequence"}
        output_schema = {"label": "binary"}

        # Explicitly pass None
        dataset = SampleDataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            input_processors=None,
            output_processors=None,
        )

        # Should fit new processors
        self.assertIn("conditions", dataset.input_processors)
        self.assertIn("label", dataset.output_processors)

    def test_mixed_transfer_and_schema(self):
        """Test using different schemas with transferred processors."""
        train_samples = [
            {
                "patient_id": "p-0",
                "visit_id": "v-0",
                "conditions": ["c1", "c2"],
                "procedures": [1.0, 2.0],
                "label": 0,
            },
            {
                "patient_id": "p-1",
                "visit_id": "v-1",
                "conditions": ["c1", "c3"],
                "procedures": [2.0, 3.0],
                "label": 1,
            },
        ]

        test_samples = [
            {
                "patient_id": "p-2",
                "visit_id": "v-2",
                "conditions": ["c1", "c3"],
                "procedures": [2.0, 3.0],
                "label": 1,
            },
        ]

        input_schema = {"conditions": "sequence", "procedures": "tensor"}
        output_schema = {"label": "binary"}

        # Create train dataset
        train_dataset = SampleDataset(
            samples=train_samples,
            input_schema=input_schema,
            output_schema=output_schema,
        )

        # Create test dataset with same schema
        test_dataset = SampleDataset(
            samples=test_samples,
            input_schema=input_schema,
            output_schema=output_schema,
            input_processors=train_dataset.input_processors,
            output_processors=train_dataset.output_processors,
        )

        # Should work correctly
        self.assertEqual(len(test_dataset), 1)
        self.assertIs(
            train_dataset.input_processors["conditions"],
            test_dataset.input_processors["conditions"],
        )


if __name__ == "__main__":
    unittest.main()
