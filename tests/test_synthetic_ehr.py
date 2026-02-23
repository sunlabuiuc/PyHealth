"""
Unit tests for synthetic EHR generation functionality.

These tests verify the utility functions and data conversions work correctly.
"""

import unittest
import pandas as pd
import sys
import os

# Add pyhealth to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyhealth.synthetic_ehr_utils.synthetic_ehr_utils import (
    tabular_to_sequences,
    sequences_to_tabular,
    nested_codes_to_sequences,
    sequences_to_nested_codes,
    create_flattened_representation,
    VISIT_DELIM,
)


class TestSyntheticEHRUtils(unittest.TestCase):
    """Test utility functions for synthetic EHR generation."""

    def setUp(self):
        """Set up test data."""
        # Create sample EHR DataFrame
        self.sample_df = pd.DataFrame({
            'SUBJECT_ID': [1, 1, 1, 1, 2, 2, 2],
            'HADM_ID': [100, 100, 200, 200, 300, 300, 400],
            'ICD9_CODE': ['410', '250', '410', '401', '250', '401', '430']
        })

        # Expected sequences
        self.expected_sequences = [
            f'410 250 {VISIT_DELIM} 410 401',
            f'250 401 {VISIT_DELIM} 430'
        ]

        # Nested codes structure
        self.nested_codes = [
            [['410', '250'], ['410', '401']],
            [['250', '401'], ['430']]
        ]

    def test_tabular_to_sequences(self):
        """Test converting tabular data to sequences."""
        sequences = tabular_to_sequences(self.sample_df)

        self.assertEqual(len(sequences), 2)
        self.assertEqual(sequences[0], self.expected_sequences[0])
        self.assertEqual(sequences[1], self.expected_sequences[1])

    def test_sequences_to_tabular(self):
        """Test converting sequences back to tabular."""
        df = sequences_to_tabular(self.expected_sequences)

        # Check structure
        self.assertIn('SUBJECT_ID', df.columns)
        self.assertIn('HADM_ID', df.columns)
        self.assertIn('ICD9_CODE', df.columns)

        # Check counts
        patient_0 = df[df['SUBJECT_ID'] == 0]
        patient_1 = df[df['SUBJECT_ID'] == 1]

        self.assertEqual(len(patient_0), 4)  # 2 + 2 codes
        self.assertEqual(len(patient_1), 3)  # 2 + 1 codes

        # Check codes present
        codes_0 = set(patient_0['ICD9_CODE'].values)
        self.assertIn('410', codes_0)
        self.assertIn('250', codes_0)
        self.assertIn('401', codes_0)

    def test_nested_codes_to_sequences(self):
        """Test converting nested codes to sequences."""
        sequences = nested_codes_to_sequences(self.nested_codes)

        self.assertEqual(len(sequences), 2)
        self.assertEqual(sequences[0], self.expected_sequences[0])
        self.assertEqual(sequences[1], self.expected_sequences[1])

    def test_sequences_to_nested_codes(self):
        """Test converting sequences to nested codes."""
        nested = sequences_to_nested_codes(self.expected_sequences)

        self.assertEqual(len(nested), 2)
        self.assertEqual(len(nested[0]), 2)  # 2 visits for patient 0
        self.assertEqual(len(nested[1]), 2)  # 2 visits for patient 1

        # Check codes
        self.assertEqual(nested[0][0], ['410', '250'])
        self.assertEqual(nested[0][1], ['410', '401'])
        self.assertEqual(nested[1][0], ['250', '401'])
        self.assertEqual(nested[1][1], ['430'])

    def test_create_flattened_representation(self):
        """Test creating flattened patient-level representation."""
        flattened = create_flattened_representation(self.sample_df)

        # Check shape
        self.assertEqual(len(flattened), 2)  # 2 patients

        # Check columns (should have all unique codes)
        unique_codes = self.sample_df['ICD9_CODE'].unique()
        for code in unique_codes:
            self.assertIn(code, flattened.columns)

        # Check counts
        # Patient 0 (SUBJECT_ID=1): 410 appears twice, 250 once, 401 once
        # Patient 1 (SUBJECT_ID=2): 250 once, 401 once, 430 once

        # Note: The exact row indices might differ, so we check the values exist
        self.assertIn(2, flattened['410'].values)  # Patient 0 has 2x 410
        self.assertIn(1, flattened['430'].values)  # Patient 1 has 1x 430

    def test_roundtrip_conversion(self):
        """Test roundtrip: tabular -> sequence -> tabular."""
        # Original -> sequences
        sequences = tabular_to_sequences(self.sample_df)

        # Sequences -> tabular
        df_reconstructed = sequences_to_tabular(sequences)

        # Check that code counts are preserved (order might differ)
        original_counts = self.sample_df['ICD9_CODE'].value_counts().to_dict()
        reconstructed_counts = df_reconstructed['ICD9_CODE'].value_counts().to_dict()

        self.assertEqual(original_counts, reconstructed_counts)

    def test_empty_sequences(self):
        """Test handling of empty sequences."""
        empty_sequences = ['', '']
        df = sequences_to_tabular(empty_sequences)

        # Should return empty DataFrame with correct columns
        self.assertEqual(len(df), 0)
        self.assertIn('SUBJECT_ID', df.columns)
        self.assertIn('HADM_ID', df.columns)
        self.assertIn('ICD9_CODE', df.columns)

    def test_single_visit_patient(self):
        """Test patient with only one visit."""
        single_visit_df = pd.DataFrame({
            'SUBJECT_ID': [1, 1],
            'HADM_ID': [100, 100],
            'ICD9_CODE': ['410', '250']
        })

        sequences = tabular_to_sequences(single_visit_df)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0], '410 250')  # No delimiter for single visit

    def test_nested_to_sequences_roundtrip(self):
        """Test roundtrip: nested -> sequences -> nested."""
        # Nested -> sequences
        sequences = nested_codes_to_sequences(self.nested_codes)

        # Sequences -> nested
        nested_reconstructed = sequences_to_nested_codes(sequences)

        # Should match original
        self.assertEqual(self.nested_codes, nested_reconstructed)


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and edge cases."""

    def test_special_characters_in_codes(self):
        """Test handling of special characters in medical codes."""
        df = pd.DataFrame({
            'SUBJECT_ID': [1, 1],
            'HADM_ID': [100, 100],
            'ICD9_CODE': ['410.01', '250.00']
        })

        sequences = tabular_to_sequences(df)
        df_reconstructed = sequences_to_tabular(sequences)

        # Check codes preserved
        self.assertIn('410.01', df_reconstructed['ICD9_CODE'].values)
        self.assertIn('250.00', df_reconstructed['ICD9_CODE'].values)

    def test_multiple_patients_multiple_visits(self):
        """Test with realistic multi-patient, multi-visit scenario."""
        df = pd.DataFrame({
            'SUBJECT_ID': [1, 1, 1, 2, 2, 3, 3, 3, 3],
            'HADM_ID': [100, 100, 200, 300, 400, 500, 500, 600, 600],
            'ICD9_CODE': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        })

        sequences = tabular_to_sequences(df)

        # Should have 3 patients
        self.assertEqual(len(sequences), 3)

        # Patient 0: 2 visits
        self.assertIn(VISIT_DELIM, sequences[0])

        # Patient 1: 2 visits
        self.assertIn(VISIT_DELIM, sequences[1])

        # Patient 2: 2 visits
        self.assertIn(VISIT_DELIM, sequences[2])


if __name__ == '__main__':
    # Run tests
    unittest.main()
