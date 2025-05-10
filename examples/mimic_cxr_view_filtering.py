"""
Simple demonstration of MIMIC-CXR dataset with view filtering [PA, AP, Lateral].
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
from pyhealth.datasets import MIMICCXRDataset

# Path to test data
TEST_DATA_PATH = os.path.expanduser("~/test-mimic-cxr")

def main():
    """Demonstrate MIMIC-CXR dataset with view filtering."""
    print("MIMIC-CXR Dataset with View Filtering Demonstration")
    print("==================================================")
    
    # Step 1: Load the dataset with all views
    print("\nLoading dataset with all views...")
    dataset = MIMICCXRDataset(root=TEST_DATA_PATH)
    
    # Directly read the metadata file to analyze it
    metadata_file = os.path.join(TEST_DATA_PATH, "mimic_cxr-metadata-pyhealth.csv")
    if os.path.exists(metadata_file):
        df = pd.read_csv(metadata_file)
        print(f"Total records: {len(df)}")
        
        # Show view distribution
        view_counts = df['view_position'].value_counts().to_dict()
        print(f"View distribution: {view_counts}")
        
        # List available views
        available_views = df['view_position'].unique()
        print(f"Available views: {list(available_views)}")
        
        # Step 2: Load dataset with filtered views (if multiple views exist)
        if len(available_views) > 1:
            test_view = available_views[0]
            print(f"\nLoading dataset with only {test_view} views...")
            
            filtered_dataset = MIMICCXRDataset(
                root=TEST_DATA_PATH,
                views=[test_view]
            )
            
            # Read the filtered metadata file
            filtered_metadata_file = os.path.join(TEST_DATA_PATH, f"mimic_cxr-metadata-pyhealth-{test_view}.csv")
            if os.path.exists(filtered_metadata_file):
                filtered_df = pd.read_csv(filtered_metadata_file)
                print(f"Filtered records: {len(filtered_df)}")
                
                # Verify filtering worked
                filtered_view_counts = filtered_df['view_position'].value_counts().to_dict()
                print(f"Filtered view distribution: {filtered_view_counts}")
            else:
                print(f"Filtered metadata file not found: {filtered_metadata_file}")
                print("This suggests the filtered dataset was created but the metadata file has a different name.")
                
                # List files in directory to find the filtered metadata
                metadata_files = [f for f in os.listdir(TEST_DATA_PATH) if f.startswith("mimic_cxr-metadata")]
                print(f"Available metadata files: {metadata_files}")
        
        # Step 3: Display a sample image if available
        if len(df) > 0:
            try:
                sample_path = df.iloc[0]['path']
                print(f"\nDisplaying sample image from: {sample_path}")
                
                # Load and display DICOM
                dcm = pydicom.dcmread(sample_path)
                pixel_array = dcm.pixel_array
                
                plt.figure(figsize=(8, 8))
                plt.imshow(pixel_array, cmap='gray')
                plt.title(f"View Position: {dcm.ViewPosition if hasattr(dcm, 'ViewPosition') else 'Unknown'}")
                plt.axis('off')
                plt.savefig("mimic_cxr_sample.png")
                plt.close()
                
                print(f"Image saved as 'mimic_cxr_sample.png'")
                print(f"Patient ID: {dcm.PatientID if hasattr(dcm, 'PatientID') else 'Unknown'}")
                print(f"View Position: {dcm.ViewPosition if hasattr(dcm, 'ViewPosition') else 'Unknown'}")
            except Exception as e:
                print(f"Error displaying sample image: {e}")
    else:
        print(f"Metadata file not found: {metadata_file}")

if __name__ == "__main__":
    main()