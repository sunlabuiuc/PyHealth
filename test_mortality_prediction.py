import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path if needed
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.mortality_prediction import MortalityPredictionMIMIC4,  MultimodalMortalityPredictionMIMIC4
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets.utils import get_dataloader


def test_mortality_prediction_mimic4():
    """
    Test case for mortality prediction using the new MIMIC4 dataset format.
    """
    # Define paths
    mimic_iv_root = "/srv/local/data/jw3/physionet.org/files/MIMIC-IV/2.0"
    mimic_note_root = "/srv/local/data/jw3/physionet.org/files/mimic-iv-note/2.2/note"
    mimic_cxr_root = "/srv/local/data/jw3/physionet.org/files/MIMIC-CXR"
    
    logger.info("Initializing MIMIC4Dataset with multiple data sources (dev mode)...")
    
    # Initialize dataset with relevant tables for mortality prediction
    # Enable dev mode to limit memory usage
    dataset = MIMIC4Dataset(
        ehr_root=mimic_iv_root,
        notes_root=mimic_note_root,
        ehr_tables=[
            "patients",           # Demographics
            "admissions",         # Admission/discharge info
            "diagnoses_icd",      # Diagnoses codes
            "procedures_icd",     # Procedure codes
            "prescriptions"       # Medications
        ],
        note_tables=[
            "discharge"           # Discharge notes for enriched prediction
        ],
        dev=True  # Enable dev mode to limit to 1000 patients
    )
    
    logger.info(f"Dataset initialized with {len(dataset.unique_patient_ids)} patients")
    
    # Create mortality prediction task
    mortality_task = MortalityPredictionMIMIC4()
    
    # Set up mortality prediction task
    logger.info("Setting up mortality prediction task...")
    sample_dataset = dataset.set_task(mortality_task)
    
    # Display task statistics
    n_samples = len(sample_dataset)
    logger.info(f"Created {n_samples} task samples")
    
    if n_samples > 0:
        # Get class distribution for mortality
        mortality_counts = {}
        for sample in sample_dataset:
            label = sample["mortality"]
            mortality_counts[label] = mortality_counts.get(label, 0) + 1
        
        logger.info("Mortality label distribution:")
        for label, count in mortality_counts.items():
            logger.info(f"  - Mortality {label}: {count} samples ({count/n_samples*100:.2f}%)")
        
        # Create train/val/test splits
        logger.info("Creating dataset splits...")
        train_dataset, val_dataset, test_dataset = split_by_patient(
            sample_dataset, 
            ratios=[0.7, 0.1, 0.2], 
            seed=42
        )
        
        logger.info(f"Train set: {len(train_dataset)} samples")
        logger.info(f"Validation set: {len(val_dataset)} samples")
        logger.info(f"Test set: {len(test_dataset)} samples")
        
        # Create dataloaders with smaller batch size for dev mode
        batch_size = 16  # Smaller batch size for dev mode
        logger.info(f"Creating dataloaders with batch size {batch_size}...")
        train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = get_dataloader(val_dataset, batch_size=batch_size)
        test_loader = get_dataloader(test_dataset, batch_size=batch_size)
        
        # Examine a sample to verify task setup
        logger.info("Examining sample data for verification...")
        
        sample_idx = 1
        sample = sample_dataset[sample_idx]
        logger.info(f"Sample {sample_idx}:")
        logger.info(f"  - Patient ID: {sample['patient_id']}")
        logger.info(f"  - Visit ID: {sample['visit_id']}")
        logger.info(f"  - Mortality label: {sample['mortality']}")
        logger.info(f"  - Number of conditions: {len(sample['conditions'])}")
        logger.info(f"  - Number of procedures: {len(sample['procedures'])}")
        logger.info(f"  - Number of drugs: {len(sample['drugs'])}")
        
        # Check data batch format
        logger.info("Checking dataloader batch format...")
        for batch_idx, batch in enumerate(train_loader):
            logger.info(f"Batch {batch_idx} keys: {list(batch.keys())}")
            logger.info(f"Batch size: {len(batch['patient_id'])}")
            logger.info(f"Conditions shape (first sample): {len(batch['conditions'])} sequences")
            logger.info(f"Mortality labels shape: {len(batch['mortality'])}")
            break  # Just check the first batch
    else:
        logger.warning("No samples created. Check task implementation and dataset content.")
    
    logger.info("Mortality prediction MIMIC4 test completed!")


def test_multimodal_mortality_prediction_mimic4():
    """
    Test case for multimodal mortality prediction using MIMIC4 dataset 
    with EHR, clinical notes, and chest X-ray data.
    """
    # Define paths
    mimic_iv_root = "/srv/local/data/jw3/physionet.org/files/MIMIC-IV/2.0"
    mimic_note_root = "/srv/local/data/jw3/physionet.org/files/mimic-iv-note/2.2/note"
    mimic_cxr_root = "/srv/local/data/jw3/physionet.org/files/MIMIC-CXR"
    
    logger.info("Initializing MIMIC4Dataset with multiple data sources (dev mode)...")
    
    # Initialize dataset with comprehensive tables
    dataset = MIMIC4Dataset(
        ehr_root=mimic_iv_root,
        notes_root=mimic_note_root,
        cxr_root=mimic_cxr_root,
        ehr_tables=[
            "patients",           # Demographics
            "admissions",         # Admission/discharge info
            "diagnoses_icd",      # Diagnoses codes
            "procedures_icd",     # Procedure codes
            "prescriptions"       # Medications
        ],
        note_tables=[
            "discharge",          # Discharge notes
            "radiology"           # Radiology notes
        ],
        cxr_tables=[
            "xrays_metadata",     # X-ray metadata
            "xrays_negbio"        # Negbio labels for X-rays
        ],
        dev=True  # Enable dev mode to limit to 1000 patients
    )
    
    logger.info(f"Dataset initialized with {len(dataset.unique_patient_ids)} patients")
    
    # Create multimodal mortality prediction task
    multimodal_mortality_task = MultimodalMortalityPredictionMIMIC4()
    
    # Set up multimodal mortality prediction task
    logger.info("Setting up multimodal mortality prediction task...")
    sample_dataset = dataset.set_task(multimodal_mortality_task)
    
    # Display task statistics
    n_samples = len(sample_dataset)
    logger.info(f"Created {n_samples} task samples")
    
    if n_samples > 0:
        # Get class distribution for mortality
        mortality_counts = {}
        for sample in sample_dataset:
            label = sample["mortality"]
            mortality_counts[label] = mortality_counts.get(label, 0) + 1
        
        # Create train/val/test splits
        logger.info("Creating dataset splits...")
        train_dataset, val_dataset, test_dataset = split_by_patient(
            sample_dataset, 
            ratios=[0.7, 0.1, 0.2], 
            seed=42
        )
        
        logger.info(f"Train set: {len(train_dataset)} samples")
        logger.info(f"Validation set: {len(val_dataset)} samples")
        logger.info(f"Test set: {len(test_dataset)} samples")
        
        # Create dataloaders with smaller batch size for dev mode
        batch_size = 16  # Smaller batch size for dev mode
        logger.info(f"Creating dataloaders with batch size {batch_size}...")
        train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = get_dataloader(val_dataset, batch_size=batch_size)
        test_loader = get_dataloader(test_dataset, batch_size=batch_size)
        
        # Examine a sample to verify task setup
        logger.info("Examining sample data for verification...")
        
        sample_idx = 1
        sample = sample_dataset[sample_idx]
        logger.info(f"Sample {sample_idx}:")
        logger.info(f"  - Patient ID: {sample['patient_id']}")
        logger.info(f"  - Visit ID: {sample.get('visit_id', 'N/A')}")
        logger.info(f"  - Mortality label: {sample['mortality']}")
        logger.info(f"  - Number of conditions: {len(sample.get('conditions', []))}")
        logger.info(f"  - Number of procedures: {len(sample.get('procedures', []))}")
        logger.info(f"  - Number of drugs: {len(sample.get('drugs', []))}")
        logger.info(f"  - Discharge note length: {len(sample.get('discharge', '')) if sample.get('discharge') else 'N/A'}")
        logger.info(f"  - Radiology note length: {len(sample.get('radiology', '')) if sample.get('radiology') else 'N/A'}")
        logger.info(f"  - X-ray NegBio features: {sample.get('xrays_negbio')}")
        
        # Check data batch format
        logger.info("Checking dataloader batch format...")
        for batch_idx, batch in enumerate(train_loader):
            logger.info(f"Batch {batch_idx} keys: {list(batch.keys())}")
            logger.info(f"Batch size: {len(batch['patient_id'])}")
            logger.info(f"Conditions shape: {len(batch.get('conditions', []))}")
            logger.info(f"Mortality labels shape: {len(batch.get('mortality', []))}")
            
            # Additional logging for multimodal features
            if 'discharge' in batch:
                logger.info(f"Discharge notes shape: {len(batch['discharge'])}")
            if 'radiology' in batch:
                logger.info(f"Radiology notes shape: {len(batch['radiology'])}")
            if 'xrays_negbio' in batch:
                logger.info(f"X-ray NegBio features: {len(batch['xrays_negbio'])}")
            
            break  # Just check the first batch
    else:
        logger.warning("No samples created. Check task implementation and dataset content.")
    
    logger.info("Multimodal mortality prediction MIMIC4 test completed!")


if __name__ == "__main__":
    test_multimodal_mortality_prediction_mimic4()
    # test_mortality_prediction_mimic4()