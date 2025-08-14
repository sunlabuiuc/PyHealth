#!/usr/bin/env python3
"""
Advanced test of TimeseriesProcessor normalization methods
"""
import numpy as np
from datetime import datetime, timedelta
from pyhealth.processors.timeseries_processor import TimeseriesProcessor

def create_diverse_data():
    """Create more diverse test data"""
    # Sample 1: Patient with high blood pressure
    timestamps1 = [
        datetime(2023, 1, 1, 8, 0),
        datetime(2023, 1, 1, 12, 0), 
        datetime(2023, 1, 1, 16, 0),
        datetime(2023, 1, 1, 20, 0),
    ]
    values1 = np.array([
        [160, 100],  # Very high BP
        [155, 98],
        [162, 102],
        [158, 99]
    ])
    
    # Sample 2: Patient with normal blood pressure
    timestamps2 = [
        datetime(2023, 1, 1, 9, 0),
        datetime(2023, 1, 1, 13, 0),
        datetime(2023, 1, 1, 17, 0),
    ]
    values2 = np.array([
        [120, 80],   # Normal BP
        [118, 78],
        [122, 82]
    ])
    
    # Sample 3: Patient with low blood pressure
    timestamps3 = [
        datetime(2023, 1, 1, 10, 0),
        datetime(2023, 1, 1, 14, 0),
    ]
    values3 = np.array([
        [90, 60],    # Low BP
        [95, 65]
    ])
    
    return [
        {"vitals": (timestamps1, values1)},
        {"vitals": (timestamps2, values2)},
        {"vitals": (timestamps3, values3)}
    ]

def test_normalization_methods():
    print("Testing different normalization methods...")
    
    samples = create_diverse_data()
    
    # Test Min-Max normalization
    print("\n1. Min-Max Normalization (Global):")
    processor_minmax = TimeseriesProcessor(
        sampling_rate=timedelta(hours=2),
        impute_strategy="forward_fill",
        normalize=True,
        norm_method="min_max",
        norm_axis="global"
    )
    
    processor_minmax.fit(samples, "vitals")
    print(f"   Global min: {processor_minmax.min:.2f}")
    print(f"   Global max: {processor_minmax.max:.2f}")
    
    result_minmax = processor_minmax.process(samples[0]["vitals"])
    print(f"   Min-max normalized (should be in [0,1]):\n{result_minmax}")
    print(f"   Range: [{result_minmax.min():.3f}, {result_minmax.max():.3f}]")
    
    # Test Robust normalization
    print("\n2. Robust Normalization (Per-feature):")
    processor_robust = TimeseriesProcessor(
        sampling_rate=timedelta(hours=2),
        impute_strategy="forward_fill", 
        normalize=True,
        norm_method="robust",
        norm_axis="per_feature"
    )
    
    processor_robust.fit(samples, "vitals")
    print(f"   Per-feature medians: {processor_robust.median}")
    print(f"   Per-feature MADs: {processor_robust.mad_}")
    
    result_robust = processor_robust.process(samples[0]["vitals"])
    print(f"   Robust normalized:\n{result_robust}")
    
    # Test that statistics come from training set only
    print("\n3. Training vs Test Set Statistics:")
    train_samples = samples[:2]  # First 2 samples as "training"
    test_sample = samples[2]     # Last sample as "test"
    
    processor_train = TimeseriesProcessor(
        normalize=True,
        norm_method="z_score",
        norm_axis="per_feature"
    )
    
    # Fit only on training data
    processor_train.fit(train_samples, "vitals")
    train_mean = processor_train.mean.copy()
    train_std = processor_train.std.copy()
    
    print(f"   Training set statistics:")
    print(f"     Mean: {train_mean}")
    print(f"     Std:  {train_std}")
    
    # Process test sample with training statistics
    test_normalized = processor_train.process(test_sample["vitals"])
    print(f"   Test sample (low BP) normalized with training stats:")
    print(f"     {test_normalized}")
    print(f"     -> Should show negative values (below training mean)")
    
    print("\n✓ All advanced normalization tests completed!")

if __name__ == "__main__":
    test_normalization_methods()