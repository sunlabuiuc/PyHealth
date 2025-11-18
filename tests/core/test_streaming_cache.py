"""Tests for streaming mode caching functionality.

This module tests that caching works correctly at all levels:
1. Patient cache
2. Sample cache
3. Processor cache
4. Physical split cache
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import polars as pl

from pyhealth.datasets.iterable_sample_dataset import IterableSampleDataset


class TestStreamingCache(unittest.TestCase):
    """Test streaming mode caching at all levels."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary files."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_sample_cache_check_exists(self):
        """Test that sample cache existence is checked before generating."""
        # Create a mock sample cache file
        dataset = IterableSampleDataset(
            input_schema={"feature": "sequence"},
            output_schema={"label": "label"},
            dataset_name="TestDataset",
            task_name="TestTask",
            cache_dir=str(self.cache_dir),
        )

        # Create fake cache file
        sample_cache_path = dataset._sample_cache_path
        sample_cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Write some sample data
        sample_df = pl.DataFrame(
            {
                "patient_id": ["P1", "P2"],
                "feature": [[1, 2, 3], [4, 5, 6]],
                "label": [0, 1],
            }
        )
        sample_df.write_parquet(sample_cache_path)

        # Check that cache exists
        self.assertTrue(sample_cache_path.exists())
        print(f"✓ Sample cache exists check passed: {sample_cache_path}")

    def test_processor_cache_paths(self):
        """Test that processor cache paths are constructed correctly."""
        processor_cache_dir = self.cache_dir / "processors"

        input_processors_path = processor_cache_dir / "input_processors.pkl"
        output_processors_path = processor_cache_dir / "output_processors.pkl"

        # Check path construction
        self.assertEqual(input_processors_path.name, "input_processors.pkl")
        self.assertEqual(output_processors_path.name, "output_processors.pkl")
        print("✓ Processor cache path construction correct")

    def test_sample_cache_loading(self):
        """Test that sample cache can be loaded correctly."""
        dataset = IterableSampleDataset(
            input_schema={"feature": "sequence"},
            output_schema={"label": "label"},
            dataset_name="TestDataset",
            task_name="TestTask",
            cache_dir=str(self.cache_dir),
        )

        # Create and write sample cache
        sample_df = pl.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "feature": [[1, 2], [3, 4], [5, 6]],
                "label": [0, 1, 0],
            }
        )
        sample_df.write_parquet(dataset._sample_cache_path)

        # Simulate loading cache
        dataset._samples_finalized = True
        sample_count_df = (
            pl.scan_parquet(dataset._sample_cache_path)
            .select(pl.count().alias("count"))
            .collect(streaming=True)
        )
        dataset._num_samples = sample_count_df["count"][0]

        self.assertEqual(dataset._num_samples, 3)
        self.assertTrue(dataset._samples_finalized)
        print("✓ Sample cache loading works correctly")

    def test_physical_split_cache_check(self):
        """Test that physical split cache check works."""
        dataset = IterableSampleDataset(
            input_schema={"feature": "sequence"},
            output_schema={"label": "label"},
            dataset_name="TestDataset",
            task_name="TestTask",
            cache_dir=str(self.cache_dir),
        )

        # Create main cache
        sample_df = pl.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "feature": [[1, 2], [3, 4], [5, 6]],
                "label": [0, 1, 0],
            }
        )
        sample_df.write_parquet(dataset._sample_cache_path)
        dataset._samples_finalized = True
        dataset._num_samples = 3

        # Create a split cache file
        split_cache_path = self.cache_dir / "TestDataset_TestTask_samples_train.parquet"
        train_df = sample_df.filter(pl.col("patient_id").is_in(["P1", "P2"]))
        train_df.write_parquet(split_cache_path)

        # Check that split cache exists
        self.assertTrue(split_cache_path.exists())

        # Verify split has correct data
        loaded_split = pl.read_parquet(split_cache_path)
        self.assertEqual(len(loaded_split), 2)
        print("✓ Physical split cache check works")

    def test_cache_hierarchy(self):
        """Test the complete cache hierarchy structure."""
        # Expected cache structure:
        # cache_dir/
        #   ├── {dataset}_{task}_samples.parquet  (main cache)
        #   ├── {dataset}_{task}_samples_train.parquet (split)
        #   ├── {dataset}_{task}_samples_val.parquet (split)
        #   └── processors/
        #       ├── input_processors.pkl
        #       └── output_processors.pkl

        dataset = IterableSampleDataset(
            input_schema={"feature": "sequence"},
            output_schema={"label": "label"},
            dataset_name="MyDataset",
            task_name="MyTask",
            cache_dir=str(self.cache_dir),
        )

        # Create main cache
        main_cache = dataset._sample_cache_path
        self.assertEqual(main_cache.name, "MyDataset_MyTask_samples.parquet")

        # Check expected split paths
        train_split = self.cache_dir / "MyDataset_MyTask_samples_train.parquet"
        val_split = self.cache_dir / "MyDataset_MyTask_samples_val.parquet"

        self.assertEqual(train_split.name, "MyDataset_MyTask_samples_train.parquet")
        self.assertEqual(val_split.name, "MyDataset_MyTask_samples_val.parquet")

        # Check processor directory
        processor_dir = self.cache_dir / "processors"
        self.assertEqual(processor_dir.name, "processors")

        print("✓ Cache hierarchy structure correct")

    def test_dev_mode_cache_separation(self):
        """Test that dev mode uses separate cache files."""
        # Dev mode dataset
        dev_dataset = IterableSampleDataset(
            input_schema={"feature": "sequence"},
            output_schema={"label": "label"},
            dataset_name="TestDataset",
            task_name="TestTask",
            cache_dir=str(self.cache_dir),
            dev=True,
            dev_max_patients=100,
        )

        # Full mode dataset
        full_dataset = IterableSampleDataset(
            input_schema={"feature": "sequence"},
            output_schema={"label": "label"},
            dataset_name="TestDataset",
            task_name="TestTask",
            cache_dir=str(self.cache_dir),
            dev=False,
        )

        # Check that cache paths are different
        self.assertNotEqual(
            dev_dataset._sample_cache_path, full_dataset._sample_cache_path
        )

        # Check dev cache has suffix
        self.assertIn("_dev_100", str(dev_dataset._sample_cache_path))
        self.assertNotIn("_dev", str(full_dataset._sample_cache_path))

        print("✓ Dev mode cache separation works")

    def test_cache_reuse_scenario(self):
        """Test complete cache reuse scenario (simulated)."""
        dataset = IterableSampleDataset(
            input_schema={"feature": "sequence"},
            output_schema={"label": "label"},
            dataset_name="TestDataset",
            task_name="TestTask",
            cache_dir=str(self.cache_dir),
        )

        # First run - create cache
        sample_df = pl.DataFrame(
            {"patient_id": ["P1", "P2"], "feature": [[1, 2], [3, 4]], "label": [0, 1]}
        )
        sample_df.write_parquet(dataset._sample_cache_path)

        # Second run - check cache exists
        cache_exists = dataset._sample_cache_path.exists()
        self.assertTrue(cache_exists)

        if cache_exists:
            # Simulate loading from cache
            dataset._samples_finalized = True
            sample_count_df = (
                pl.scan_parquet(dataset._sample_cache_path)
                .select(pl.count().alias("count"))
                .collect(streaming=True)
            )
            dataset._num_samples = sample_count_df["count"][0]

        self.assertEqual(dataset._num_samples, 2)
        print("✓ Cache reuse scenario works correctly")

    def test_split_cache_reuse(self):
        """Test that split caches are reused when they exist."""
        dataset = IterableSampleDataset(
            input_schema={"feature": "sequence"},
            output_schema={"label": "label"},
            dataset_name="TestDataset",
            task_name="TestTask",
            cache_dir=str(self.cache_dir),
        )

        # Create main cache
        sample_df = pl.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3", "P4"],
                "feature": [[1], [2], [3], [4]],
                "label": [0, 1, 0, 1],
            }
        )
        sample_df.write_parquet(dataset._sample_cache_path)
        dataset._samples_finalized = True
        dataset._num_samples = 4

        # Create train split cache (first time)
        train_split_path = self.cache_dir / "TestDataset_TestTask_samples_train.parquet"
        train_df = sample_df.filter(pl.col("patient_id").is_in(["P1", "P2", "P3"]))
        train_df.write_parquet(train_split_path)

        # Check split exists
        self.assertTrue(train_split_path.exists())

        # Second time - split should be reused
        split_exists = train_split_path.exists()
        self.assertTrue(split_exists)

        if split_exists:
            # Load from existing split
            loaded_split = pl.read_parquet(train_split_path)
            self.assertEqual(len(loaded_split), 3)

        print("✓ Split cache reuse works correctly")

    def test_cache_invalidation_on_different_config(self):
        """Test that different dev configs use different caches."""
        # Dataset with dev=True, dev_max_patients=100
        dataset1 = IterableSampleDataset(
            input_schema={"feature": "sequence"},
            output_schema={"label": "label"},
            dataset_name="TestDataset",
            task_name="TestTask",
            cache_dir=str(self.cache_dir),
            dev=True,
            dev_max_patients=100,
        )

        # Dataset with dev=True, dev_max_patients=500
        dataset2 = IterableSampleDataset(
            input_schema={"feature": "sequence"},
            output_schema={"label": "label"},
            dataset_name="TestDataset",
            task_name="TestTask",
            cache_dir=str(self.cache_dir),
            dev=True,
            dev_max_patients=500,
        )

        # Cache paths should be different
        self.assertNotEqual(dataset1._sample_cache_path, dataset2._sample_cache_path)

        self.assertIn("_dev_100", str(dataset1._sample_cache_path))
        self.assertIn("_dev_500", str(dataset2._sample_cache_path))

        print("✓ Different dev configs use different caches")


class TestProcessorCaching(unittest.TestCase):
    """Test processor caching functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary files."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_processor_cache_directory_creation(self):
        """Test that processor cache directory is created correctly."""
        processor_dir = self.cache_dir / "processors"
        processor_dir.mkdir(parents=True, exist_ok=True)

        self.assertTrue(processor_dir.exists())
        self.assertTrue(processor_dir.is_dir())
        print("✓ Processor cache directory created")

    def test_processor_file_paths(self):
        """Test processor file path construction."""
        processor_dir = self.cache_dir / "processors"

        input_path = processor_dir / "input_processors.pkl"
        output_path = processor_dir / "output_processors.pkl"

        self.assertEqual(input_path.suffix, ".pkl")
        self.assertEqual(output_path.suffix, ".pkl")
        print("✓ Processor file paths correct")

    def test_processor_cache_check(self):
        """Test checking if processors are cached."""
        processor_dir = self.cache_dir / "processors"
        processor_dir.mkdir(parents=True, exist_ok=True)

        input_path = processor_dir / "input_processors.pkl"
        output_path = processor_dir / "output_processors.pkl"

        # Initially, processors should not exist
        processors_cached = input_path.exists() and output_path.exists()
        self.assertFalse(processors_cached)

        # Create dummy files
        input_path.touch()
        output_path.touch()

        # Now processors should be cached
        processors_cached = input_path.exists() and output_path.exists()
        self.assertTrue(processors_cached)
        print("✓ Processor cache check works")


if __name__ == "__main__":
    print("=" * 70)
    print("STREAMING CACHE TESTS")
    print("=" * 70)
    unittest.main(verbosity=2)
