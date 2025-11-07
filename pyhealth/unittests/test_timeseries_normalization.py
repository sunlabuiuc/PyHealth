#!/usr/bin/env python3
"""
Quick test of the enhanced TimeseriesProcessor with normalization
"""
import numpy as np
from datetime import datetime, timedelta
from pyhealth.processors.timeseries_processor import TimeseriesProcessor

# Create sample timeseries data
def create_sample_data():
    # Sample 1: Blood pressure readings
    timestamps1 = [
        datetime(2023, 1, 1, 8, 0),
        datetime(2023, 1, 1, 12, 0),
        datetime(2023, 1, 1, 18, 0),
        datetime(2023, 1, 2, 8, 0),
    ]
    values1 = np.array([
        [120, 80],    # systolic, diastolic
        [130, 85],
        [125, 82],
        [135, 88]
    ])
    
    # Sample 2: Different patient
    timestamps2 = [
        datetime(2023, 1, 1, 6, 0),
        datetime(2023, 1, 1, 14, 0),
        datetime(2023, 1, 1, 22, 0),
    ]
    values2 = np.array([
        [140, 90],
        [145, 95],
        [138, 87]
    ])
    
    return [
        {"vitals": (timestamps1, values1)},
        {"vitals": (timestamps2, values2)}
    ]

def test_normalization():
    print("Testing TimeseriesProcessor with normalization...")
    
    # Test data
    samples = create_sample_data()
    
    # Test 1: Without normalization (baseline)
    processor_no_norm = TimeseriesProcessor(
        sampling_rate=timedelta(hours=4),
        impute_strategy="forward_fill",
        normalize=False
    )
    
    print("\n1. Testing without normalization:")
    result1 = processor_no_norm.process(samples[0]["vitals"])
    print(f"   Sample 1 shape: {result1.shape}")
    print(f"   Sample 1 values:\n{result1}")
    
    # Test 2: With Z-score normalization
    processor_zscore = TimeseriesProcessor(
        sampling_rate=timedelta(hours=4),
        impute_strategy="forward_fill",
        normalize=True,
        norm_method="z_score",
        norm_axis="global"
    )
    
    print("\n2. Testing with Z-score normalization:")
    # Fit on training data
    processor_zscore.fit(samples, "vitals")
    print(f"   Fitted mean: {processor_zscore.mean_}")
    print(f"   Fitted std: {processor_zscore.std_}")
    
    # Process samples
    result2 = processor_zscore.process(samples[0]["vitals"])
    print(f"   Normalized sample 1:\n{result2}")
    
    # Test 3: Per-feature normalization
    processor_per_feature = TimeseriesProcessor(
        sampling_rate=timedelta(hours=4),
        impute_strategy="forward_fill",
        normalize=True,
        norm_method="z_score", 
        norm_axis="per_feature"
    )
    
    print("\n3. Testing per-feature normalization:")
    processor_per_feature.fit(samples, "vitals")
    print(f"   Per-feature means: {processor_per_feature.mean_}")
    print(f"   Per-feature stds: {processor_per_feature.std_}")
    
    result3 = processor_per_feature.process(samples[0]["vitals"])
    print(f"   Per-feature normalized sample 1:\n{result3}")
    
    print("\n✓ All tests completed successfully!")

if __name__ == "__main__":
    test_normalization()