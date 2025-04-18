# ------------------------------------------------------------------------------
# Unit Tests for HurtfulWordsDataset
#
# Name:
#   Zilal Eiz Al Din
#
# NetID:
#   zelalae2
#
# Purpose:
#   This file contains unit tests for verifying the preprocessing logic of 
#   the HurtfulWordsDataset, including:
#   - PHI replacement
#   - Gendered word masking
#   - Clinical note formatting cleanup
#
# Dataset:
#   HurtfulWordsDataset from pyhealth.datasets
#
# Usage:
#   Run `pytest` to automatically discover and run these tests.
# ------------------------------------------------------------------------------

import pytest
from pyhealth.datasets import HurtfulWordsDataset

# STEP 1: Load dataset
# Update the root path to where MIMIC-III CSVs are stored
root='/content/drive/MyDrive/Data/physionet.org/files/mimiciii/1.4'
dataset = HurtfulWordsDataset(root=f"{root}", tables=["noteevents"])

# ------------------------------------------------------------------------------
# Test 1: PHI Replacement
# ------------------------------------------------------------------------------

def test_phi_replacement():
    test_note = "Patient admitted to [**Hospital**] on [**Date**]. Treated by [**Name**]."
    masked_note = dataset._replace_phi(test_note)
    assert "PHIHOSPITALPHI" in masked_note
    assert "PHINAMEPHI" in masked_note
# ------------------------------------------------------------------------------
# Test 2: Gendered Word Masking
# ------------------------------------------------------------------------------

def test_gendered_word_masking():
    test_note = "The patient is a male and he was treated."
    masked_note = dataset._mask_gendered_terms(test_note)
    assert "[MASK]" in masked_note
    assert "male" not in masked_note.lower()

# ------------------------------------------------------------------------------
# Test 3: Clinical Note Formatting
# ------------------------------------------------------------------------------

def test_clean_note_format():
    dirty_note = "1. This is a line\n2. Another line\n--- end ---"
    clean_note = dataset._clean_note_format(dirty_note)
    assert "\n" not in clean_note
    assert "---" not in clean_note
    assert clean_note.startswith("This is a line")

# ------------------------------------------------------------------------------
# Run tests directly (if needed)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    test_phi_replacement()
    test_gendered_word_masking()
    test_clean_note_format()
    print("All HurtfulWordsDataset tests passed successfully!")