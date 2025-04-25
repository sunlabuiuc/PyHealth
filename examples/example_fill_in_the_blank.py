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
#   - Initialize the model
#   - Print results
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
from pyhealth.models  import FillInTheBlank

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
# STEP 2: Apply preprocessing task
# ------------------------------------------------------------------------------k
task = HurtfulWordsPreprocessingTask()
sample_dataset = mimic3_base.set_task(task)

# ------------------------------------------------------------------------------
# STEP 3: Initialize model 
# ------------------------------------------------------------------------------
model = FillInTheBlank(sample_dataset)

# Step 4: Use the first real masked_text from the dataset
first_masked_text = sample_dataset.samples[0]["masked_text"]
print("Using first masked text:", first_masked_text)

# Step 5: Run model on this text
batch = {"masked_text": [first_masked_text]}
log_probs = model(batch)

# Step 6: Display results
he_logp, she_logp = log_probs[0].tolist()
print("LogP(he), LogP(she):", round(he_logp, 4), round(she_logp, 4))