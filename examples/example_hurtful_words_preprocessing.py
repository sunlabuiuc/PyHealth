# ------------------------------------------------------------------------------
# Example Script: MIMIC3Dataset + HurtfulWordsPreprocessingTask
#
# Name:
#   Zilal Eiz Al Din && Payel Chakraborty
#
# NetID:
#   zelalae2 && payelc2
#
# Purpose:
#   This script demonstrates how to:
#   - Load the MIMIC3Dataset (MIMIC-III noteevents) 
#   - Apply the HurtfulWordsPreprocessing to mask the clinical notes
#   - Iterate through the notes to preprocess them and mask them
#   - Print first sample
#
# Paper Reference:
#   Hurtful Words: Quantifying Biases in Clinical Contextual Word Embeddings
#   https://arxiv.org/abs/2003.11515
#
# Usage:
#   This is a simple script for manual inspection and demonstration.
#   To run:
#       $ python example_hurtful_words_preprocessing.py
#
# Note:
#   This script assumes you have access to MIMIC-III CSVs at the specified root path.
# ------------------------------------------------------------------------------
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import HurtfulWordsPreprocessingTask

# ------------------------------------------------------------------------------
# STEP 1: Load the MIMIC3Dataset
# ------------------------------------------------------------------------------

# Update this path to point to your local MIMIC-III CSV files
PATH_TO_CSVs='/srv/local/data/physionet.org/files/mimiciii/1.4'

# Initialize the dataset
mimic3_base = MIMIC3Dataset(root=f"{PATH_TO_CSVs}", tables=["noteevents"])

# Print basic dataset statistics
print(mimic3_base.stats())

# ------------------------------------------------------------------------------
# STEP 2: Set the task
# ------------------------------------------------------------------------------k

task = HurtfulWordsPreprocessingTask()
sample_dataset = mimic3_base.set_task(task)

# ------------------------------------------------------------------------------
# STEP 3: Iterate through the samples and display results
# ------------------------------------------------------------------------------
for sample in sample_dataset.samples:
    patient_id = sample["patient_id"][0]  # Get the patient ID from the tuple
    masked_text = sample["masked_text"]   # Get the masked text
    print(f"Patient ID: {patient_id}")
    print(f"Masked Text:\n{masked_text[:300]}...")  # Print only first 300 characters
    print("-" * 80)