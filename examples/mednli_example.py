#!/usr/bin/env python3
"""
Simplified MedNLI Example for Testing Implementation
====================================================

This script tests the basic functionality of the MedNLI dataset implementation
without requiring complex dependencies like transformers or torch.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our implementation
from pyhealth.datasets import MedNLIDataset
from pyhealth.tasks import MedNLITask


def test_mednli_dataset(data_dir):
    """Test the MedNLI dataset implementation.

    Args:
        data_dir: Directory containing the MedNLI data files.
    """
    # Load dataset
    logger.info(f"Loading MedNLI dataset from {data_dir}")
    try:
        dataset = MedNLIDataset(root=data_dir)

        # Display dataset statistics
        logger.info("Dataset statistics:")
        dataset.stat()

        # Set task and get samples
        logger.info("Setting up MedNLI task")
        task = MedNLITask()
        sample_dataset = dataset.set_task(task)

        # Display sample information
        logger.info(f"Generated {len(sample_dataset)} samples")
        if sample_dataset:
            logger.info("Example sample:")
            for key, value in sample_dataset[0].items():
                # Truncate long text fields
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                logger.info(f"  {key}: {value}")

        logger.info("✅ MedNLI implementation test successful!")
        return True
    except Exception as e:
        logger.error(f"❌ Error testing MedNLI implementation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to test the MedNLI implementation."""
    # Default data directory
    data_dir = "data/mednli"

    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} not found: {data_dir}")
        logger.info(
            "Please create the directory and add train.jsonl, dev.jsonl, and test.jsonl files"
        )
        return False

    # Test the implementation
    return test_mednli_dataset(data_dir)


if __name__ == "__main__":
    main()
