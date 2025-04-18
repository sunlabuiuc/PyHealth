# ------------------------------------------------------------------------------
# Example Script: HurtfulWordsDataset + MaskedBiasScoreTask
#
# Name:
#   Zilal Eiz Al Din
#
# NetID:
#   zelalae2
#
# Purpose:
#   This script demonstrates how to:
#   - Load the HurtfulWordsDataset (MIMIC-III noteevents) 
#   - Apply the MaskedBiasScoreTask to compute gender bias scores
#   - Iterate through the generated samples
#   - Print patient information and bias-related log-probabilities
#
# Paper Reference:
#   Hurtful Words: Quantifying Biases in Clinical Contextual Word Embeddings
#   https://arxiv.org/abs/2003.11515
#
# Usage:
#   This is a simple script for manual inspection and demonstration.
#   To run:
#       $ python example_hurtfulwords_task.py
#
# Note:
#   This script assumes you have access to MIMIC-III CSVs at the specified root path.
# ------------------------------------------------------------------------------
from pyhealth.datasets import HurtfulWordsDataset
from pyhealth.tasks import MaskedBiasScoreTask

# ------------------------------------------------------------------------------
# STEP 1: Load the HurtfulWordsDataset
# ------------------------------------------------------------------------------

# Update this path to point to your local MIMIC-III CSV files
root='/srv/local/data/physionet.org/files/mimiciii/1.4'

# Initialize the dataset
dataset = HurtfulWordsDataset(root=f"{root}", tables=["noteevents"])

# Print basic dataset statistics
print(dataset.stats())

# ------------------------------------------------------------------------------
# STEP 2: Set the MaskedBiasScoreTask
# ------------------------------------------------------------------------------k

# Initialize the bias evaluation t
task = MaskedBiasScoreTask()

# Apply the task to the dataset
sample_dataset = dataset.set_task(task)

# ------------------------------------------------------------------------------
# STEP 3: Iterate through the samples and display results
# ------------------------------------------------------------------------------
for sample in sample_dataset.samples:
    patient_id = sample["patient_id"][0]  # Get the patient ID from the tuple
    male_log_prob = round(sample["male_log_prob"].item(), 4)
    female_log_prob = round(sample["female_log_prob"].item(), 4)
    bias_score = round(sample["bias_score"].item(), 4)
    masked_text = sample["masked_text"]   # Get the masked text
    print(f"Patient ID: {patient_id}")
    print(f"LogP(M): {male_log_prob}")
    print(f"LogP(F): {female_log_prob}")
    print(f"Bias Score: {bias_score}")
    print(f"Masked Text:\n{masked_text[:300]}...")  # Print only first 300 characters
    print("-" * 80)