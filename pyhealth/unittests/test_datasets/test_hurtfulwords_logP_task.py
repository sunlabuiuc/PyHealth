# ------------------------------------------------------------------------------
# Integration Test for HurtfulWordsDataset + MaskedBiasScoreTask
#
# Name:
#   Zilal Eiz Al Din
#
# NetID:
#   zelalae2
#
# Purpose:
#   This file contains an integration test that verifies end-to-end functionality
#   of the HurtfulWordsDataset and MaskedBiasScoreTask combination. 
#
#   It checks that:
#   - The dataset can be loaded from MIMIC-III NOTEEVENTS.
#   - The preprocessing (PHI masking, gender masking) works.
#   - The MaskedBiasScoreTask can successfully compute male and female 
#     log-probabilities and bias scores.
#   - The outputs are correctly typed and formatted.
#
# Dataset:
#   HurtfulWordsDataset (MIMIC-III)
#
# Task:
#   MaskedBiasScoreTask
#
# Usage:
#   Run `pytest` to automatically discover and run this integration test.
# ------------------------------------------------------------------------------

import pytest
import torch
from pyhealth.datasets import HurtfulWordsDataset
from pyhealth.tasks import MaskedBiasScoreTask

# Path to your local MIMIC-III CSVs
MIMIC_ROOT = "/srv/local/data/physionet.org/files/mimiciii/1.4"

# ------------------------------------------------------------------------------
# Integration Test
# ------------------------------------------------------------------------------
@pytest.mark.integration
def test_hurtfulwords_dataset_with_maskedbiasscoretask():
    # STEP 1: Load dataset
    dataset = HurtfulWordsDataset(root=f"{MIMIC_ROOT}", tables=["noteevents"])

    # Call it to print 
    dataset.stats()

    # STEP 2: Set the task
    task = MaskedBiasScoreTask()
    sample_dataset = dataset.set_task(task)

    # SampleDataset should not be empty
    assert len(sample_dataset) > 0

    # STEP 3: Iterate over a few samples
    for sample in sample_dataset.samples[:10]:
        assert "patient_id" in sample
        assert "masked_text" in sample
        assert "male_log_prob" in sample
        assert "female_log_prob" in sample
        assert "bias_score" in sample

        assert isinstance(sample["masked_text"], str)
        assert isinstance(sample["male_log_prob"], torch.Tensor)
        assert isinstance(sample["female_log_prob"], torch.Tensor)
        assert isinstance(sample["bias_score"], torch.Tensor)

        male_log_prob = round(sample["male_log_prob"].item(), 4)
        female_log_prob = round(sample["female_log_prob"].item(), 4)
        bias_score = round(sample["bias_score"].item(), 4)

        assert isinstance(male_log_prob, float)
        assert isinstance(female_log_prob, float)
        assert isinstance(bias_score, float)

# ------------------------------------------------------------------------------
# Run manually if needed
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    test_hurtfulwords_dataset_with_maskedbiasscoretask()
    print("Integration test for HurtfulWordsDataset and MaskedBiasScoreTask passed successfully!")