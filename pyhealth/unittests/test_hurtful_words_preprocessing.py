# ------------------------------------------------------------------------------
# Unit Tests for HurtfulWordsDataset
#
# Names:
#   Zilal Eiz Al Din && Payel Chakraborty
#
# NetIDs:
#   zelalae2  && payelc2
#
# Purpose:
#   This file contains unit tests for verifying the preprocessing logic of 
#   the HurtfulWordsPreprocessingTask, including:
#   - PHI replacement
#   - Gendered word masking
#   - Clinical note formatting cleanup
#
# Dataset:
#   from pyhealth.datasets import MIMIC3Dataset
#
# Usage:
#   Run `pytest` to automatically discover and run these tests.
# ------------------------------------------------------------------------------

import pytest
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import HurtfulWordsPreprocessingTask


# STEP 1: Load dataset
# Update the root path to where MIMIC-III CSVs are stored
PATH_TO_CSVs ='/content/drive/MyDrive/Data/physionet.org/files/mimiciii/1.4'
mimic3_base = MIMIC3Dataset(root=f"{PATH_TO_CSVs}", tables=["noteevents"])

# STEP 2: set task
task = HurtfulWordsPreprocessingTask()

# ------------------------------------------------------------------------------
# Test 1: PHI Replacement
# ------------------------------------------------------------------------------

def test_phi_replacement():
    test_note = "Patient admitted to [**Hospital**] on [**Date**]. Treated by [**Name**]."
    masked_note = task._replace_phi(test_note)
    assert "PHIHOSPITALPHI" in masked_note
    assert "PHINAMEPHI" in masked_note
# ------------------------------------------------------------------------------
# Test 2: Gendered Word Masking
# ------------------------------------------------------------------------------

def test_mask_gendered_terms():
    test_note = "The patient is a male and he was treated."
    masked_note = task._mask_gendered_terms(test_note)
    assert "[MASK]" in masked_note
    assert "male" not in masked_note.lower()

# ------------------------------------------------------------------------------
# Test 3: Clinical Note Formatting
# ------------------------------------------------------------------------------

def test_clean_note_format():
    dirty_note = "1. This is a line\n2. Another line\n--- end ---"
    clean_note = task._clean_note_format(dirty_note)
    assert "\n" not in clean_note
    assert "---" not in clean_note
    assert clean_note.startswith("This is a line")

# ------------------------------------------------------------------------------
# Run tests directly (if needed)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    test_phi_replacement()
    test_mask_gendered_terms()
    test_clean_note_format()
    print("All HurtfulWordsPreprocessing tests passed successfully!")