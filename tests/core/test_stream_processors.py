import unittest
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.processors import (
    BinaryLabelProcessor,
    MultiClassLabelProcessor,
    MultiLabelProcessor,
    StageNetProcessor,
    StageNetTensorProcessor,
    NestedSequenceProcessor,
    NestedFloatsProcessor,
)


class TestStreamProcessors(unittest.TestCase):
    """Test cases for streaming fit functionality across all processors."""

    def test_binary_label_processor_streaming(self):
        """Test BinaryLabelProcessor with streaming mode."""
        # Create batches of samples
        batch1 = [
            {"label": 0},
            {"label": 1},
            {"label": 0},
        ]
        batch2 = [
            {"label": 1},
            {"label": 0},
        ]
        batch3 = [
            {"label": 1},
        ]

        # Non-streaming mode (baseline)
        processor_baseline = BinaryLabelProcessor()
        all_samples = batch1 + batch2 + batch3
        processor_baseline.fit(all_samples, "label", stream=False)

        # Streaming mode
        processor_streaming = BinaryLabelProcessor()
        processor_streaming.fit(batch1, "label", stream=True)
        processor_streaming.fit(batch2, "label", stream=True)
        processor_streaming.fit(batch3, "label", stream=True)
        processor_streaming.finalize_fit()

        # Verify vocabs are identical
        self.assertEqual(
            processor_baseline.label_vocab,
            processor_streaming.label_vocab,
        )

        # Test processing works correctly
        self.assertTrue(
            torch.equal(
                processor_baseline.process(0),
                processor_streaming.process(0),
            )
        )
        self.assertTrue(
            torch.equal(
                processor_baseline.process(1),
                processor_streaming.process(1),
            )
        )

    def test_multiclass_label_processor_streaming(self):
        """Test MultiClassLabelProcessor with streaming mode."""
        batch1 = [
            {"label": "class_a"},
            {"label": "class_b"},
        ]
        batch2 = [
            {"label": "class_c"},
            {"label": "class_a"},
        ]
        batch3 = [
            {"label": "class_d"},
            {"label": "class_b"},
        ]

        # Non-streaming mode
        processor_baseline = MultiClassLabelProcessor()
        all_samples = batch1 + batch2 + batch3
        processor_baseline.fit(all_samples, "label", stream=False)

        # Streaming mode
        processor_streaming = MultiClassLabelProcessor()
        processor_streaming.fit(batch1, "label", stream=True)
        processor_streaming.fit(batch2, "label", stream=True)
        processor_streaming.fit(batch3, "label", stream=True)
        processor_streaming.finalize_fit()

        # Verify vocabs are identical
        self.assertEqual(
            processor_baseline.label_vocab,
            processor_streaming.label_vocab,
        )
        self.assertEqual(processor_baseline.size(), 4)
        self.assertEqual(processor_streaming.size(), 4)

        # Test processing
        for label in ["class_a", "class_b", "class_c", "class_d"]:
            self.assertTrue(
                torch.equal(
                    processor_baseline.process(label),
                    processor_streaming.process(label),
                )
            )

    def test_multilabel_processor_streaming(self):
        """Test MultiLabelProcessor with streaming mode."""
        batch1 = [
            {"tags": ["tag1", "tag2"]},
            {"tags": ["tag2", "tag3"]},
        ]
        batch2 = [
            {"tags": ["tag1", "tag4"]},
            {"tags": ["tag3", "tag5"]},
        ]
        batch3 = [
            {"tags": ["tag2", "tag5", "tag6"]},
        ]

        # Non-streaming mode
        processor_baseline = MultiLabelProcessor()
        all_samples = batch1 + batch2 + batch3
        processor_baseline.fit(all_samples, "tags", stream=False)

        # Streaming mode
        processor_streaming = MultiLabelProcessor()
        processor_streaming.fit(batch1, "tags", stream=True)
        processor_streaming.fit(batch2, "tags", stream=True)
        processor_streaming.fit(batch3, "tags", stream=True)
        processor_streaming.finalize_fit()

        # Verify vocabs are identical
        self.assertEqual(
            processor_baseline.label_vocab,
            processor_streaming.label_vocab,
        )
        self.assertEqual(processor_baseline.size(), 6)
        self.assertEqual(processor_streaming.size(), 6)

        # Test processing
        test_tags = ["tag1", "tag3", "tag5"]
        result_baseline = processor_baseline.process(test_tags)
        result_streaming = processor_streaming.process(test_tags)
        self.assertTrue(torch.equal(result_baseline, result_streaming))

    def test_stagenet_processor_streaming(self):
        """Test StageNetProcessor with streaming mode."""
        # Flat codes
        batch1 = [
            {"codes": ([0.0, 1.0], ["code1", "code2"])},
            {"codes": ([0.0, 1.5], ["code2", "code3"])},
        ]
        batch2 = [
            {"codes": ([0.0], ["code4"])},
            {"codes": ([0.0, 0.5, 1.0], ["code1", "code5", "code6"])},
        ]

        # Non-streaming mode
        processor_baseline = StageNetProcessor()
        all_samples = batch1 + batch2
        processor_baseline.fit(all_samples, "codes", stream=False)

        # Streaming mode
        processor_streaming = StageNetProcessor()
        processor_streaming.fit(batch1, "codes", stream=True)
        processor_streaming.fit(batch2, "codes", stream=True)
        processor_streaming.finalize_fit()

        # Verify vocabs are identical
        self.assertEqual(
            processor_baseline.code_vocab,
            processor_streaming.code_vocab,
        )
        self.assertEqual(processor_baseline._is_nested, False)
        self.assertEqual(processor_streaming._is_nested, False)

        # Test processing
        test_data = ([0.0, 1.0], ["code1", "code2"])
        result_baseline = processor_baseline.process(test_data)
        result_streaming = processor_streaming.process(test_data)

        # Results are tuples of (time, values)
        time_b, values_b = result_baseline
        time_s, values_s = result_streaming

        if time_b is not None and time_s is not None:
            self.assertTrue(torch.equal(time_b, time_s))
        self.assertTrue(torch.equal(values_b, values_s))

    def test_stagenet_processor_nested_streaming(self):
        """Test StageNetProcessor with nested codes in streaming mode."""
        # Nested codes
        batch1 = [
            {"procs": ([0.0], [["A01", "A02"], ["B01"]])},
            {"procs": ([0.0, 1.0], [["A03"], ["B02", "C01"]])},
        ]
        batch2 = [
            {"procs": ([0.0, 1.0], [["A01", "A04"], ["C02"]])},
            {"procs": ([0.0], [["D01", "D02", "D03"]])},
        ]

        # Non-streaming mode
        processor_baseline = StageNetProcessor()
        all_samples = batch1 + batch2
        processor_baseline.fit(all_samples, "procs", stream=False)

        # Streaming mode
        processor_streaming = StageNetProcessor()
        processor_streaming.fit(batch1, "procs", stream=True)
        processor_streaming.fit(batch2, "procs", stream=True)
        processor_streaming.finalize_fit()

        # Verify vocabs and structure
        self.assertEqual(
            processor_baseline.code_vocab,
            processor_streaming.code_vocab,
        )
        self.assertEqual(processor_baseline._is_nested, True)
        self.assertEqual(processor_streaming._is_nested, True)
        self.assertEqual(
            processor_baseline._max_nested_len,
            processor_streaming._max_nested_len,
        )
        # Max nested length should be 3 (from ["D01", "D02", "D03"])
        self.assertEqual(processor_baseline._max_nested_len, 3)

    def test_stagenet_tensor_processor_streaming(self):
        """Test StageNetTensorProcessor with streaming mode."""
        batch1 = [
            {"values": (None, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            {"values": (None, [[7.0, 8.0, 9.0]])},
        ]
        batch2 = [
            {"values": ([0.0, 1.0], [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])},
        ]

        # Non-streaming mode
        processor_baseline = StageNetTensorProcessor()
        all_samples = batch1 + batch2
        processor_baseline.fit(all_samples, "values", stream=False)

        # Streaming mode
        processor_streaming = StageNetTensorProcessor()
        processor_streaming.fit(batch1, "values", stream=True)
        processor_streaming.fit(batch2, "values", stream=True)
        processor_streaming.finalize_fit()

        # Verify structure detection
        self.assertEqual(processor_baseline._is_nested, True)
        self.assertEqual(processor_streaming._is_nested, True)
        self.assertEqual(processor_baseline._size, 3)
        self.assertEqual(processor_streaming._size, 3)

    def test_nested_sequence_processor_streaming(self):
        """Test NestedSequenceProcessor with streaming mode."""
        batch1 = [
            {"codes": [["A", "B"], ["C", "D", "E"]]},
            {"codes": [["F"]]},
        ]
        batch2 = [
            {"codes": [["A", "G"], ["H", "I"]]},
            {"codes": [["B", "C"], ["J", "K", "L", "M"]]},
        ]

        # Non-streaming mode
        processor_baseline = NestedSequenceProcessor()
        all_samples = batch1 + batch2
        processor_baseline.fit(all_samples, "codes", stream=False)

        # Streaming mode
        processor_streaming = NestedSequenceProcessor()
        processor_streaming.fit(batch1, "codes", stream=True)
        processor_streaming.fit(batch2, "codes", stream=True)
        processor_streaming.finalize_fit()

        # Verify vocabs are identical
        self.assertEqual(
            processor_baseline.code_vocab,
            processor_streaming.code_vocab,
        )
        # Max inner length should be 4 (from ["J", "K", "L", "M"])
        self.assertEqual(processor_baseline._max_inner_len, 4)
        self.assertEqual(processor_streaming._max_inner_len, 4)

        # Test processing
        test_data = [["A", "B"], ["C"]]
        result_baseline = processor_baseline.process(test_data)
        result_streaming = processor_streaming.process(test_data)
        self.assertTrue(torch.equal(result_baseline, result_streaming))

    def test_nested_floats_processor_streaming(self):
        """Test NestedFloatsProcessor with streaming mode."""
        batch1 = [
            {"values": [[1.0, 2.0], [3.0, 4.0, 5.0]]},
            {"values": [[6.0]]},
        ]
        batch2 = [
            {"values": [[7.0, 8.0], [9.0, 10.0]]},
            {"values": [[11.0, 12.0, 13.0, 14.0]]},
        ]

        # Non-streaming mode
        processor_baseline = NestedFloatsProcessor()
        all_samples = batch1 + batch2
        processor_baseline.fit(all_samples, "values", stream=False)

        # Streaming mode
        processor_streaming = NestedFloatsProcessor()
        processor_streaming.fit(batch1, "values", stream=True)
        processor_streaming.fit(batch2, "values", stream=True)
        processor_streaming.finalize_fit()

        # Max inner length should be 4 (from [11.0, 12.0, 13.0, 14.0])
        self.assertEqual(processor_baseline._max_inner_len, 4)
        self.assertEqual(processor_streaming._max_inner_len, 4)

        # Test processing
        test_data = [[1.0, 2.0], [3.0]]
        result_baseline = processor_baseline.process(test_data)
        result_streaming = processor_streaming.process(test_data)
        self.assertTrue(torch.equal(result_baseline, result_streaming))

    def test_streaming_with_empty_batches(self):
        """Test streaming mode handles empty batches gracefully."""
        batch1 = [{"label": 0}, {"label": 1}]
        batch2 = []  # Empty batch
        batch3 = [{"label": 0}]

        processor = BinaryLabelProcessor()
        processor.fit(batch1, "label", stream=True)
        processor.fit(batch2, "label", stream=True)  # Should not crash
        processor.fit(batch3, "label", stream=True)
        processor.finalize_fit()

        self.assertEqual(len(processor.label_vocab), 2)

    def test_streaming_integration_with_dataset_nonstreaming(self):
        """Test dataset creation with non-streaming processor fit."""
        # Create a small dataset (< 100k samples, non-streaming mode)
        samples = []
        for i in range(50):
            samples.append(
                {
                    "patient_id": f"patient-{i}",
                    "visit_id": f"visit-{i}",
                    "conditions": ["cond-33", "cond-86", "cond-80"],
                    "procedures": [1.0, 2.0, 3.5, 4.0],
                    "label": i % 2,
                }
            )

        input_schema = {
            "conditions": "sequence",
            "procedures": "tensor",
        }
        output_schema = {"label": "binary"}

        # Create dataset - should use non-streaming fit (< 100k samples)
        dataset = SampleDataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_nonstreaming_integration",
        )

        # Verify dataset created successfully
        self.assertEqual(len(dataset), 50)
        self.assertIn("conditions", dataset[0])
        self.assertIn("procedures", dataset[0])
        self.assertIn("label", dataset[0])

        # Test dataloader works
        train_loader = get_dataloader(dataset, batch_size=10, shuffle=False)
        data_batch = next(iter(train_loader))

        self.assertEqual(data_batch["label"].shape[0], 10)
        self.assertIsInstance(data_batch["conditions"], torch.Tensor)
        self.assertIsInstance(data_batch["procedures"], torch.Tensor)

    def test_manual_streaming_vs_nonstreaming_dataset(self):
        """Test that streaming and non-streaming produce identical results."""
        # Create mock samples similar to test_mlp.py
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86"],
                "label": 1,
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "conditions": ["cond-80", "cond-12", "cond-99"],
                "label": 0,
            },
            {
                "patient_id": "patient-3",
                "visit_id": "visit-3",
                "conditions": ["cond-86", "cond-99", "cond-33"],
                "label": 1,
            },
        ]

        input_schema = {"conditions": "sequence"}
        output_schema = {"label": "binary"}

        # Create two separate datasets to ensure clean state
        # Dataset 1: Normal non-streaming mode
        dataset_normal = SampleDataset(
            samples=samples.copy(),
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_normal_vs_stream",
        )

        # Get the fitted label processor from normal dataset
        label_proc_normal = dataset_normal.output_processors["label"]

        # Manually create and fit a streaming version with fresh processor
        from pyhealth.processors import BinaryLabelProcessor

        label_proc_streaming = BinaryLabelProcessor()

        # Debug: Print object IDs to verify they're different instances
        print(f"\nNormal processor id: {id(label_proc_normal)}")
        print(f"Streaming processor id: {id(label_proc_streaming)}")
        print(f"Initial _all_labels: {label_proc_streaming._all_labels}")

        # Debug: Check initial state
        self.assertEqual(len(label_proc_streaming._all_labels), 0)

        # Simulate streaming fit in batches
        batch1 = samples[:2]
        batch2 = samples[2:]

        label_proc_streaming.fit(batch1, "label", stream=True)
        print(f"After batch1, _all_labels: {label_proc_streaming._all_labels}")
        # After first batch, should have 2 labels (0 and 1)
        self.assertEqual(len(label_proc_streaming._all_labels), 2)

        label_proc_streaming.fit(batch2, "label", stream=True)
        print(f"After batch2, _all_labels: {label_proc_streaming._all_labels}")
        # After second batch, still should have 2 labels (0 and 1)
        self.assertEqual(len(label_proc_streaming._all_labels), 2)

        label_proc_streaming.finalize_fit()

        # Verify vocabs match
        self.assertEqual(
            label_proc_normal.label_vocab,
            label_proc_streaming.label_vocab,
        )

        # Test that both process values the same way
        for label in [0, 1]:
            result_normal = label_proc_normal.process(label)
            result_streaming = label_proc_streaming.process(label)
            self.assertTrue(torch.equal(result_normal, result_streaming))

    def test_stagenet_integration_with_dataset(self):
        """Test StageNet processors with SampleDataset."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "codes": ([0.0, 2.0], ["code1", "code2"]),
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "codes": ([0.0, 1.0, 2.0], ["code2", "code3", "code4"]),
                "label": 1,
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "codes": ([0.0], ["code1"]),
                "label": 0,
            },
        ]

        input_schema = {"codes": "stagenet"}
        output_schema = {"label": "binary"}

        # Create dataset
        dataset = SampleDataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_stagenet_integration",
        )

        # Verify dataset created successfully
        self.assertEqual(len(dataset), 3)

        # Check sample structure
        sample = dataset[0]
        self.assertIn("codes", sample)
        self.assertIn("label", sample)

        # codes should be tuple (time, values)
        self.assertIsInstance(sample["codes"], tuple)
        time, values = sample["codes"]
        self.assertIsInstance(values, torch.Tensor)

        # Test dataloader works
        train_loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        self.assertEqual(data_batch["label"].shape[0], 2)
        self.assertIsInstance(data_batch["codes"], tuple)

    def test_binary_label_validation_streaming(self):
        """Test that binary label validation works in streaming mode."""
        # Create batches that will result in 3 unique labels (invalid)
        batch1 = [{"label": 0}, {"label": 1}]
        batch2 = [{"label": 2}]  # Third label - should fail

        processor = BinaryLabelProcessor()
        processor.fit(batch1, "label", stream=True)
        processor.fit(batch2, "label", stream=True)

        # Should raise ValueError during finalize_fit
        with self.assertRaises(ValueError) as context:
            processor.finalize_fit()

        self.assertIn("Expected 2 unique labels, got 3", str(context.exception))

    def test_streaming_preserves_backward_compatibility(self):
        """Test that default stream=False maintains backward compatibility."""
        samples = [
            {"label": 0},
            {"label": 1},
            {"label": 0},
            {"label": 1},
        ]

        # Old API (implicit stream=False)
        processor_old = BinaryLabelProcessor()
        processor_old.fit(samples, "label")

        # New API (explicit stream=False)
        processor_new = BinaryLabelProcessor()
        processor_new.fit(samples, "label", stream=False)

        # Should be identical
        self.assertEqual(processor_old.label_vocab, processor_new.label_vocab)

        # Test processing works the same
        self.assertTrue(
            torch.equal(
                processor_old.process(0),
                processor_new.process(0),
            )
        )

    def test_processors_output_tensor_types_streaming(self):
        """Test that streaming processors produce correct tensor types and shapes."""

        # Test BinaryLabelProcessor
        binary_samples = [{"label": 0}, {"label": 1}]
        binary_proc = BinaryLabelProcessor()
        binary_proc.fit(binary_samples, "label", stream=True)
        binary_proc.finalize_fit()

        result = binary_proc.process(1)
        self.assertIsInstance(result, torch.Tensor, "BinaryLabel should return tensor")
        self.assertEqual(result.dtype, torch.float32, "BinaryLabel should be float32")
        self.assertEqual(result.shape, torch.Size([1]), "BinaryLabel should be [1]")
        self.assertEqual(result.item(), 1.0, "BinaryLabel value should match")

        # Test MultiClassLabelProcessor
        multiclass_samples = [
            {"label": "class_a"},
            {"label": "class_b"},
            {"label": "class_c"},
        ]
        multiclass_proc = MultiClassLabelProcessor()
        multiclass_proc.fit(multiclass_samples, "label", stream=True)
        multiclass_proc.finalize_fit()

        result = multiclass_proc.process("class_b")
        self.assertIsInstance(result, torch.Tensor, "MultiClass should return tensor")
        self.assertEqual(result.dtype, torch.long, "MultiClass should be long tensor")
        self.assertEqual(result.shape, torch.Size([]), "MultiClass should be scalar")

        # Test MultiLabelProcessor
        multilabel_samples = [
            {"tags": ["tag1", "tag2"]},
            {"tags": ["tag2", "tag3", "tag4"]},
        ]
        multilabel_proc = MultiLabelProcessor()
        multilabel_proc.fit(multilabel_samples, "tags", stream=True)
        multilabel_proc.finalize_fit()

        result = multilabel_proc.process(["tag1", "tag3"])
        self.assertIsInstance(result, torch.Tensor, "MultiLabel should return tensor")
        self.assertEqual(result.dtype, torch.float, "MultiLabel should be float tensor")
        self.assertEqual(result.shape[0], 4, "MultiLabel should have size of vocab")
        # Should be one-hot encoded
        self.assertEqual(
            result.sum().item(), 2.0, "MultiLabel should have 2 active labels"
        )

        # Test StageNetProcessor (flat)
        stagenet_samples = [
            {"codes": ([0.0, 1.0], ["code1", "code2"])},
            {"codes": ([0.0, 2.0], ["code3", "code4"])},
        ]
        stagenet_proc = StageNetProcessor()
        stagenet_proc.fit(stagenet_samples, "codes", stream=True)
        stagenet_proc.finalize_fit()

        time, values = stagenet_proc.process(([0.0, 1.0], ["code1", "code2"]))
        self.assertIsInstance(time, torch.Tensor, "StageNet time should be tensor")
        self.assertIsInstance(values, torch.Tensor, "StageNet values should be tensor")
        self.assertEqual(time.dtype, torch.float, "StageNet time should be float")
        self.assertEqual(values.dtype, torch.long, "StageNet values should be long")
        self.assertEqual(
            time.shape, torch.Size([2]), "StageNet time should match input"
        )
        self.assertEqual(
            values.shape, torch.Size([2]), "StageNet values should match input"
        )

        # Test StageNetTensorProcessor
        stagenet_tensor_samples = [
            {"values": (None, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            {"values": (None, [[7.0, 8.0, 9.0]])},
        ]
        stagenet_tensor_proc = StageNetTensorProcessor()
        stagenet_tensor_proc.fit(stagenet_tensor_samples, "values", stream=True)
        stagenet_tensor_proc.finalize_fit()

        time, values = stagenet_tensor_proc.process((None, [[1.0, 2.0, 3.0]]))
        self.assertIsNone(time, "StageNetTensor time can be None")
        self.assertIsInstance(
            values, torch.Tensor, "StageNetTensor values should be tensor"
        )
        self.assertEqual(values.dtype, torch.float, "StageNetTensor should be float")
        self.assertEqual(
            values.shape,
            torch.Size([1, 3]),
            "StageNetTensor shape should be [visits, features]",
        )

        # Test NestedSequenceProcessor
        nested_seq_samples = [
            {"codes": [["A", "B"], ["C", "D", "E"]]},
            {"codes": [["F", "G"]]},
        ]
        nested_seq_proc = NestedSequenceProcessor()
        nested_seq_proc.fit(nested_seq_samples, "codes", stream=True)
        nested_seq_proc.finalize_fit()

        result = nested_seq_proc.process([["A", "C"], ["B"]])
        self.assertIsInstance(
            result, torch.Tensor, "NestedSequence should return tensor"
        )
        self.assertEqual(
            result.dtype, torch.long, "NestedSequence should be long tensor"
        )
        self.assertEqual(
            len(result.shape), 2, "NestedSequence should be 2D [visits, codes]"
        )

        # Test NestedFloatsProcessor
        nested_floats_samples = [
            {"values": [[1.0, 2.0], [3.0, 4.0, 5.0]]},
            {"values": [[6.0]]},
        ]
        nested_floats_proc = NestedFloatsProcessor()
        nested_floats_proc.fit(nested_floats_samples, "values", stream=True)
        nested_floats_proc.finalize_fit()

        result = nested_floats_proc.process([[1.0, 2.0], [3.0]])
        self.assertIsInstance(result, torch.Tensor, "NestedFloats should return tensor")
        self.assertEqual(
            result.dtype, torch.float, "NestedFloats should be float tensor"
        )
        self.assertEqual(
            len(result.shape), 2, "NestedFloats should be 2D [visits, values]"
        )

    def test_processors_output_tensor_types_nonstreaming(self):
        """Test that non-streaming processors produce correct tensor types (baseline)."""

        # Test BinaryLabelProcessor
        binary_samples = [{"label": 0}, {"label": 1}]
        binary_proc = BinaryLabelProcessor()
        binary_proc.fit(binary_samples, "label", stream=False)

        result = binary_proc.process(1)
        self.assertIsInstance(result, torch.Tensor, "BinaryLabel should return tensor")
        self.assertEqual(result.dtype, torch.float32, "BinaryLabel should be float32")

        # Test MultiClassLabelProcessor
        multiclass_samples = [{"label": "class_a"}, {"label": "class_b"}]
        multiclass_proc = MultiClassLabelProcessor()
        multiclass_proc.fit(multiclass_samples, "label", stream=False)

        result = multiclass_proc.process("class_b")
        self.assertIsInstance(result, torch.Tensor, "MultiClass should return tensor")
        self.assertEqual(result.dtype, torch.long, "MultiClass should be long tensor")

        # Test MultiLabelProcessor
        multilabel_samples = [{"tags": ["tag1", "tag2"]}, {"tags": ["tag3"]}]
        multilabel_proc = MultiLabelProcessor()
        multilabel_proc.fit(multilabel_samples, "tags", stream=False)

        result = multilabel_proc.process(["tag1", "tag3"])
        self.assertIsInstance(result, torch.Tensor, "MultiLabel should return tensor")
        self.assertEqual(result.dtype, torch.float, "MultiLabel should be float tensor")

    def test_streaming_vs_nonstreaming_tensor_equality(self):
        """Test that streaming and non-streaming produce identical tensors."""

        # BinaryLabelProcessor
        samples_binary = [{"label": 0}, {"label": 1}, {"label": 0}]

        proc_stream = BinaryLabelProcessor()
        proc_stream.fit(samples_binary[:2], "label", stream=True)
        proc_stream.fit(samples_binary[2:], "label", stream=True)
        proc_stream.finalize_fit()

        proc_normal = BinaryLabelProcessor()
        proc_normal.fit(samples_binary, "label", stream=False)

        for label in [0, 1]:
            result_stream = proc_stream.process(label)
            result_normal = proc_normal.process(label)
            self.assertTrue(
                torch.equal(result_stream, result_normal),
                f"Tensors should be identical for label {label}",
            )
            self.assertEqual(
                result_stream.dtype, result_normal.dtype, "Tensor dtypes should match"
            )
            self.assertEqual(
                result_stream.shape, result_normal.shape, "Tensor shapes should match"
            )

        # MultiLabelProcessor
        samples_multilabel = [
            {"tags": ["tag1", "tag2"]},
            {"tags": ["tag2", "tag3"]},
            {"tags": ["tag4"]},
        ]

        proc_multi_stream = MultiLabelProcessor()
        proc_multi_stream.fit(samples_multilabel[:2], "tags", stream=True)
        proc_multi_stream.fit(samples_multilabel[2:], "tags", stream=True)
        proc_multi_stream.finalize_fit()

        proc_multi_normal = MultiLabelProcessor()
        proc_multi_normal.fit(samples_multilabel, "tags", stream=False)

        test_tags = ["tag1", "tag3"]
        result_stream = proc_multi_stream.process(test_tags)
        result_normal = proc_multi_normal.process(test_tags)

        self.assertTrue(
            torch.equal(result_stream, result_normal),
            "MultiLabel tensors should be identical",
        )
        self.assertEqual(result_stream.dtype, result_normal.dtype)
        self.assertEqual(result_stream.shape, result_normal.shape)


if __name__ == "__main__":
    unittest.main()
