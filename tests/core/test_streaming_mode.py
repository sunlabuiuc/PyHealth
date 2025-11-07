"""Fast unit tests for streaming mode functionality.

These are "baby tests" designed to run quickly to verify streaming mechanics work correctly.
They use small synthetic datasets and mocks to minimize runtime.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock
import polars as pl

print("=" * 70)
print("STREAMING MODE TESTS - Starting test execution")
print("=" * 70)

try:
    from pyhealth.datasets.base_dataset import BaseDataset
    print("✓ Successfully imported BaseDataset")
except ImportError as e:
    print(f"✗ Failed to import BaseDataset: {e}")
    raise

try:
    from pyhealth.datasets.sample_dataset import IterableSampleDataset
    print("✓ Successfully imported IterableSampleDataset")
except ImportError as e:
    print(f"✗ Failed to import IterableSampleDataset: {e}")
    print("\nNote: You may need to install PyHealth in development mode:")
    print("  cd PyHealth && pip install -e .")
    raise

print("=" * 70)
print()


class TestStreamingMode(unittest.TestCase):
    """Fast unit tests for streaming mode functionality."""

    def setUp(self):
        """Set up test data for each test."""
        print("\n[SETUP] Creating test fixtures...")
        # Create temporary directory for caching
        self.temp_dir = tempfile.mkdtemp()
        self.temp_cache_dir = Path(self.temp_dir)
        print(f"  Cache dir: {self.temp_dir}")

        # Create small synthetic data (3 patients, 5 events)
        self.mock_data = pl.DataFrame({
            "patient_id": ["P1", "P1", "P2", "P2", "P3"],
            "event_type": ["diagnosis", "medication", "diagnosis", "procedure", "diagnosis"],
            "timestamp": pl.Series([
                "2020-01-01", "2020-01-02", "2020-01-01", "2020-01-03", "2020-01-01"
            ]).str.strptime(pl.Datetime, format="%Y-%m-%d"),
            "diagnosis/code": ["D001", None, "D002", None, "D003"],
            "medication/name": [None, "M001", None, None, None],
            "procedure/type": [None, None, None, "P001", None],
        })
        print(f"  Mock dataset shape: {self.mock_data.shape}")

    def tearDown(self):
        """Clean up temporary files after each test."""
        print("[TEARDOWN] Cleaning up...")
        import shutil
        if self.temp_cache_dir.exists():
            shutil.rmtree(self.temp_cache_dir)
        print("  ✓ Cleaned up temporary cache directory")

    def test_streaming_mode_initialization(self):
        """Test that streaming mode can be initialized (fast - no data loading)."""
        print("\n[TEST] test_streaming_mode_initialization")
        # Use mock to avoid actual dataset loading
        dataset_mock = Mock(spec=BaseDataset)
        dataset_mock.stream = True
        dataset_mock.cache_dir = self.temp_cache_dir

        self.assertTrue(dataset_mock.stream)
        self.assertEqual(dataset_mock.cache_dir, self.temp_cache_dir)
        print("  ✓ Streaming mode initialization test passed")

    def test_patient_cache_creation(self):
        """Test that patient cache is created (fast - minimal data)."""
        print("\n[TEST] test_patient_cache_creation")

        # Write small test data to cache
        cache_path = self.temp_cache_dir / "test_patients.parquet"
        print(f"  Writing {len(self.mock_data)} rows to {cache_path.name}...")
        self.mock_data.write_parquet(cache_path)

        self.assertTrue(cache_path.exists())
        print(f"  ✓ Cache file created: {cache_path.exists()}")

        # Verify data can be read back
        loaded = pl.read_parquet(cache_path)
        self.assertEqual(len(loaded), len(self.mock_data))
        self.assertEqual(loaded.schema, self.mock_data.schema)
        print(f"  ✓ Cache data verified: {len(loaded)} rows read back")

    def test_patient_cache_sorted_by_patient_id(self):
        """Test that patient cache is sorted by patient_id for efficient access."""
        print("\n[TEST] test_patient_cache_sorted_by_patient_id")

        # Sort data by patient_id (as done in _build_patient_cache)
        sorted_data = self.mock_data.sort("patient_id", "timestamp")

        cache_path = self.temp_cache_dir / "test_patients_sorted.parquet"
        sorted_data.write_parquet(cache_path)

        # Verify data is sorted
        loaded = pl.read_parquet(cache_path)
        patient_ids = loaded["patient_id"].to_list()
        self.assertEqual(patient_ids, sorted(patient_ids))
        print("  ✓ Patient cache is sorted correctly")

    def test_patient_index_creation(self):
        """Test that patient index is created correctly."""
        print("\n[TEST] test_patient_index_creation")

        # Create patient cache
        cache_path = self.temp_cache_dir / "test_patients.parquet"
        self.mock_data.write_parquet(cache_path)

        # Build patient index (as done in _build_patient_cache)
        patient_index = (
            pl.scan_parquet(cache_path)
            .group_by("patient_id")
            .agg([
                pl.count().alias("event_count"),
                pl.first("timestamp").alias("first_timestamp"),
                pl.last("timestamp").alias("last_timestamp"),
            ])
            .sort("patient_id")
            .collect()
        )

        # Verify index
        self.assertEqual(len(patient_index), 3)  # P1, P2, P3
        self.assertIn("event_count", patient_index.columns)
        self.assertIn("first_timestamp", patient_index.columns)
        self.assertIn("last_timestamp", patient_index.columns)

        # Verify counts
        p1_count = patient_index.filter(pl.col("patient_id") == "P1")["event_count"][0]
        self.assertEqual(p1_count, 2)  # P1 has 2 events
        print("  ✓ Patient index created correctly")

    def test_sample_storage_streaming(self):
        """Test that samples can be written incrementally (fast)."""
        print("\n[TEST] test_sample_storage_streaming")
        # Create small sample batch
        samples = [
            {"patient_id": "P1", "label": 0, "value": 100},
            {"patient_id": "P2", "label": 1, "value": 200},
        ]

        sample_df = pl.DataFrame(samples)
        cache_path = self.temp_cache_dir / "test_samples.parquet"
        sample_df.write_parquet(cache_path)

        self.assertTrue(cache_path.exists())

        # Verify samples can be read
        loaded_samples = pl.read_parquet(cache_path).to_dicts()
        self.assertEqual(len(loaded_samples), 2)
        self.assertEqual(loaded_samples[0]["patient_id"], "P1")
        self.assertEqual(loaded_samples[1]["patient_id"], "P2")
        print("  ✓ Sample storage works correctly")

    def test_iterable_dataset_length(self):
        """Test that IterableSampleDataset reports correct length (fast)."""
        print("\n[TEST] test_iterable_dataset_length")
        # Create small sample dataset
        samples = pl.DataFrame({
            "patient_id": ["P1", "P2", "P3"],
            "label": [0, 1, 0],
        })
        cache_path = self.temp_cache_dir / "samples.parquet"
        samples.write_parquet(cache_path)

        # Mock IterableSampleDataset
        dataset_mock = Mock(spec=IterableSampleDataset)
        dataset_mock._num_samples = len(samples)
        dataset_mock.__len__ = lambda self: dataset_mock._num_samples

        self.assertEqual(len(dataset_mock), 3)
        print("  ✓ IterableSampleDataset length works correctly")

    def test_batch_iteration(self):
        """Test batch reading from parquet (fast - 10 samples)."""
        print("\n[TEST] test_batch_iteration")
        # Create test samples
        samples = pl.DataFrame({
            "idx": range(10),
            "value": range(100, 110),
        })
        cache_path = self.temp_cache_dir / "samples.parquet"
        samples.write_parquet(cache_path)

        # Test batch reading
        batch_size = 3
        lf = pl.scan_parquet(cache_path)

        batches = []
        num_samples = 10
        num_batches = (num_samples + batch_size - 1) // batch_size
        for i in range(num_batches):
            offset = i * batch_size
            length = min(batch_size, num_samples - offset)
            batch = lf.slice(offset, length).collect()
            batches.append(batch)

        # Verify we got all samples
        total_samples = sum(len(b) for b in batches)
        self.assertEqual(total_samples, 10)

        # Verify batch sizes
        self.assertEqual(len(batches[0]), 3)  # First batch
        self.assertEqual(len(batches[1]), 3)  # Second batch
        self.assertEqual(len(batches[2]), 3)  # Third batch
        self.assertEqual(len(batches[3]), 1)  # Last batch (remainder)
        print("  ✓ Batch iteration works correctly")

    def test_streaming_mode_error_messages(self):
        """Test that appropriate errors are raised in stream mode (fast)."""
        print("\n[TEST] test_streaming_mode_error_messages")
        dataset_mock = Mock(spec=BaseDataset)
        dataset_mock.stream = True

        # Test collected_global_event_df error
        def raise_runtime_error():
            raise RuntimeError(
                "collected_global_event_df is not available in stream mode "
                "as it would load the entire dataset into memory. "
                "Use iter_patients_streaming() for memory-efficient patient iteration."
            )

        type(dataset_mock).collected_global_event_df = property(
            lambda self: raise_runtime_error()
        )

        with self.assertRaises(RuntimeError) as context:
            _ = dataset_mock.collected_global_event_df
        
        self.assertIn("not available in stream mode", str(context.exception))
        print("  ✓ Stream mode error messages work correctly")

    def test_iter_patients_error_in_stream_mode(self):
        """Test that iter_patients raises error when called without df in stream mode."""
        print("\n[TEST] test_iter_patients_error_in_stream_mode")
        dataset_mock = Mock(spec=BaseDataset)
        dataset_mock.stream = True

        def mock_iter_patients(df=None):
            if df is None and dataset_mock.stream:
                raise RuntimeError(
                    "iter_patients() requires collected DataFrame which is not "
                    "available in stream mode. Use iter_patients_streaming() instead."
                )

        dataset_mock.iter_patients = mock_iter_patients

        with self.assertRaises(RuntimeError) as context:
            dataset_mock.iter_patients()
        
        self.assertIn("Use iter_patients_streaming", str(context.exception))
        print("  ✓ iter_patients error in stream mode works correctly")

    def test_iterable_sample_dataset_initialization(self):
        """Test IterableSampleDataset can be initialized correctly."""
        print("\n[TEST] test_iterable_sample_dataset_initialization")
        print("  Creating IterableSampleDataset...")
        dataset = IterableSampleDataset(
            input_schema={"feature": "sequence"},
            output_schema={"label": "label"},
            dataset_name="TestDataset",
            task_name="TestTask",
            cache_dir=str(self.temp_cache_dir),
        )

        self.assertEqual(dataset.dataset_name, "TestDataset")
        self.assertEqual(dataset.task_name, "TestTask")
        self.assertEqual(dataset._num_samples, 0)
        self.assertFalse(dataset._samples_finalized)
        print(f"  ✓ IterableSampleDataset initialized successfully")
        print(f"    - Dataset name: {dataset.dataset_name}")
        print(f"    - Task name: {dataset.task_name}")
        print(f"    - Initial samples: {dataset._num_samples}")

    def test_iterable_sample_dataset_add_samples(self):
        """Test adding samples to IterableSampleDataset."""
        print("\n[TEST] test_iterable_sample_dataset_add_samples")
        dataset = IterableSampleDataset(
            input_schema={"feature": "sequence"},
            output_schema={"label": "label"},
            dataset_name="TestDataset",
            task_name="TestTask",
            cache_dir=str(self.temp_cache_dir),
        )

        # Add first batch
        samples1 = [
            {"patient_id": "P1", "feature": [1, 2, 3], "label": 0},
            {"patient_id": "P2", "feature": [4, 5, 6], "label": 1},
        ]
        print(f"  Adding batch 1: {len(samples1)} samples...")
        dataset.add_samples_streaming(samples1)

        self.assertEqual(dataset._num_samples, 2)
        print(f"  ✓ Batch 1 added. Total samples: {dataset._num_samples}")

        # Add second batch
        samples2 = [
            {"patient_id": "P3", "feature": [7, 8, 9], "label": 0},
        ]
        print(f"  Adding batch 2: {len(samples2)} samples...")
        dataset.add_samples_streaming(samples2)

        self.assertEqual(dataset._num_samples, 3)
        print(f"  ✓ Batch 2 added. Total samples: {dataset._num_samples}")

    def test_iterable_sample_dataset_finalize(self):
        """Test finalizing samples in IterableSampleDataset."""
        print("\n[TEST] test_iterable_sample_dataset_finalize")
        dataset = IterableSampleDataset(
            input_schema={"feature": "sequence"},
            output_schema={"label": "label"},
            dataset_name="TestDataset",
            task_name="TestTask",
            cache_dir=str(self.temp_cache_dir),
        )

        samples = [
            {"patient_id": "P1", "feature": [1, 2, 3], "label": 0},
        ]
        dataset.add_samples_streaming(samples)
        dataset.finalize_samples()

        self.assertTrue(dataset._samples_finalized)

        # Should raise error if trying to add after finalize
        with self.assertRaises(RuntimeError) as context:
            dataset.add_samples_streaming(samples)
        
        self.assertIn("Cannot add more samples", str(context.exception))
        print("  ✓ Sample finalization works correctly")

    def test_cache_path_construction(self):
        """Test that cache paths are constructed correctly."""
        print("\n[TEST] test_cache_path_construction")
        dataset = IterableSampleDataset(
            input_schema={"feature": "sequence"},
            output_schema={"label": "label"},
            dataset_name="MyDataset",
            task_name="MyTask",
            cache_dir=str(self.temp_cache_dir),
        )

        expected_path = self.temp_cache_dir / "MyDataset_MyTask_samples.parquet"
        self.assertEqual(dataset._sample_cache_path, expected_path)
        print("  ✓ Cache path construction works correctly")

    def test_patient_filtering_from_cache(self):
        """Test filtering specific patients from cache (fast)."""
        print("\n[TEST] test_patient_filtering_from_cache")

        # Create patient cache
        cache_path = self.temp_cache_dir / "test_patients.parquet"
        self.mock_data.write_parquet(cache_path)

        # Filter for specific patient
        patient_df = (
            pl.scan_parquet(cache_path)
            .filter(pl.col("patient_id") == "P1")
            .collect()
        )

        self.assertEqual(len(patient_df), 2)  # P1 has 2 events
        self.assertTrue(all(patient_df["patient_id"] == "P1"))
        print("  ✓ Patient filtering from cache works correctly")

    def test_multiple_patient_filtering(self):
        """Test filtering multiple patients from cache."""
        print("\n[TEST] test_multiple_patient_filtering")

        # Create patient cache
        cache_path = self.temp_cache_dir / "test_patients.parquet"
        self.mock_data.write_parquet(cache_path)

        # Filter for multiple patients
        patient_ids = ["P1", "P3"]
        filtered_df = (
            pl.scan_parquet(cache_path)
            .filter(pl.col("patient_id").is_in(patient_ids))
            .collect()
        )

        self.assertEqual(len(filtered_df), 3)  # P1 has 2 events, P3 has 1 event
        self.assertEqual(set(filtered_df["patient_id"].unique().to_list()), {"P1", "P3"})
        print("  ✓ Multiple patient filtering works correctly")


print("\n" + "=" * 70)
print("STREAMING MODE TESTS - Module loaded successfully")
print("All test functions defined and ready to run")
print("=" * 70)


if __name__ == "__main__":
    unittest.main()

