"""
Lalith Devireddy, Trisha Rayan
NetID: lalithd, trayan2
Paper Title: Racial Disparities and Mistrust in End-of-Life Care
Paper Link: https://proceedings.mlr.press/v85/boag18a.html
Description: Test suite for trust metrics module.
"""

import unittest
import pandas as pd
import numpy as np
from trust_metrics import (
    TrustMetrics, 
    extract_chartevents_features,
    extract_trust_indicators_from_notes,
    get_noncompliance_labels,
    get_autopsy_labels
)

class TestTrustMetrics(unittest.TestCase):
    """
    Test cases for the TrustMetrics class and related functions.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create test features
        self.num_patients = 10
        self.features = pd.DataFrame({
            'pain': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'agitated': [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
            'restraint': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            'education_barrier': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            'family_meeting': [0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
        })
        
        # Create test labels
        self.noncompliance_labels = pd.Series([0, 1, 1, 0, 0, 1, 1, 0, 0, 1])
        self.autopsy_labels = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        
        # Create test notes
        self.notes = [
            "Patient is cooperative and following instructions.",
            "Patient refused medication and is non-compliant with treatment plan.",
            "Patient is agitated and non-compliant. Restrained for safety.",
            "Discussed autopsy with family. They agreed to procedure.",
            "Patient is calm and pain-free. Following treatment plan.",
            "Patient is frustrated and mistrusting of healthcare system.",
            "Patient left AMA despite medical advice.",
            "Family requested autopsy to determine cause of death.",
            "Patient showed good compliance with therapy regimen.",
            "Patient expressed mistrust in doctors and treatment plan."
        ]
        
        # Initialize trust metrics
        self.trust_metrics = TrustMetrics(regularization_strength=0.1)
        
    def test_fit_noncompliance_metric(self):
        """
        Test fitting noncompliance metric.
        """
        self.trust_metrics.fit_noncompliance_metric(self.features, self.noncompliance_labels)
        self.assertIsNotNone(self.trust_metrics.noncompliance_model)
        self.assertEqual(len(self.trust_metrics.feature_names), self.features.shape[1])
        
    def test_fit_autopsy_metric(self):
        """
        Test fitting autopsy metric.
        """
        self.trust_metrics.fit_autopsy_metric(self.features, self.autopsy_labels)
        self.assertIsNotNone(self.trust_metrics.autopsy_model)
        self.assertEqual(len(self.trust_metrics.feature_names), self.features.shape[1])
        
    def test_calculate_noncompliance_mistrust(self):
        """
        Test calculating noncompliance-derived mistrust.
        """
        self.trust_metrics.fit_noncompliance_metric(self.features, self.noncompliance_labels)
        scores = self.trust_metrics.calculate_noncompliance_mistrust(self.features)
        self.assertEqual(len(scores), self.num_patients)
        self.assertTrue(all(0 <= score <= 1 for score in scores))
        
    def test_calculate_autopsy_mistrust(self):
        """
        Test calculating autopsy-derived mistrust.
        """
        self.trust_metrics.fit_autopsy_metric(self.features, self.autopsy_labels)
        scores = self.trust_metrics.calculate_autopsy_mistrust(self.features)
        self.assertEqual(len(scores), self.num_patients)
        self.assertTrue(all(0 <= score <= 1 for score in scores))
        
    def test_calculate_sentiment_mistrust(self):
        """
        Test calculating sentiment-based mistrust.
        """
        scores = self.trust_metrics.calculate_sentiment_mistrust(self.notes)
        self.assertEqual(len(scores), self.num_patients)
        
    def test_combine_metrics(self):
        """
        Test combining multiple trust metrics.
        """
        self.trust_metrics.fit_noncompliance_metric(self.features, self.noncompliance_labels)
        self.trust_metrics.fit_autopsy_metric(self.features, self.autopsy_labels)
        
        noncompliance_scores = self.trust_metrics.calculate_noncompliance_mistrust(self.features)
        autopsy_scores = self.trust_metrics.calculate_autopsy_mistrust(self.features)
        sentiment_scores = self.trust_metrics.calculate_sentiment_mistrust(self.notes)
        
        # Test with default weights
        combined_scores = self.trust_metrics.combine_metrics(
            noncompliance_scores, autopsy_scores, sentiment_scores
        )
        self.assertEqual(len(combined_scores), self.num_patients)
        
        # Test with custom weights
        custom_weights = [0.5, 0.3, 0.2]
        custom_combined_scores = self.trust_metrics.combine_metrics(
            noncompliance_scores, autopsy_scores, sentiment_scores,
            weights=custom_weights
        )
        self.assertEqual(len(custom_combined_scores), self.num_patients)
        
    def test_get_important_features(self):
        """
        Test getting important features.
        """
        self.trust_metrics.fit_noncompliance_metric(self.features, self.noncompliance_labels)
        features = self.trust_metrics.get_important_features('noncompliance')
        self.assertIn('top_positive', features)
        self.assertIn('top_negative', features)
        
    def test_find_noncompliance_keywords(self):
        """
        Test finding noncompliance keywords in notes.
        """
        keyword_to_notes = self.trust_metrics.find_noncompliance_keywords(self.notes)
        self.assertIn('non-compliant', keyword_to_notes)
        self.assertIn('AMA', keyword_to_notes)
        
    def test_find_autopsy_keywords(self):
        """
        Test finding autopsy keywords in notes.
        """
        keyword_to_notes = self.trust_metrics.find_autopsy_keywords(self.notes)
        self.assertIn('autopsy', keyword_to_notes)
        
    def test_normalize_scores(self):
        """
        Test normalizing scores.
        """
        scores = np.array([1, 2, 3, 4, 5])
        normalized = self.trust_metrics.normalize_scores(scores)
        self.assertAlmostEqual(np.mean(normalized), 0, places=10)
        self.assertAlmostEqual(np.std(normalized), 1, places=10)


class TestHelperFunctions(unittest.TestCase):
    """
    Test cases for helper functions.
    """
    
    def test_extract_chartevents_features(self):
        """
        Test extracting features from chartevents.
        """
        # Create mock chartevents data
        chartevents_df = pd.DataFrame({
            'SUBJECT_ID': [1, 1, 2, 2, 3],
            'ITEMID': ['pain', 'agitated', 'pain', 'restraint device', 'education barrier'],
            'VALUE': ['yes', 'yes', 'no', 'wrist', 'language']
        })
        
        features_df = extract_chartevents_features(chartevents_df)
        self.assertEqual(len(features_df), 3)  
        
    def test_extract_trust_indicators_from_notes(self):
        """
        Test extracting trust indicators from notes.
        """
        # Create mock notes data
        notes_df = pd.DataFrame({
            'SUBJECT_ID': [1, 1, 2, 2, 3],
            'TEXT': [
                "Patient is cooperative.",
                "Patient is following instructions.",
                "Patient refused medication and is non-compliant.",
                "Patient with mistrust of healthcare system.",
                "Family requested autopsy."
            ]
        })
        
        indicators = extract_trust_indicators_from_notes(notes_df)
        self.assertEqual(len(indicators), 3)
        self.assertIn(2, indicators)
        self.assertIn('noncompliance', indicators[2])
        self.assertIn(3, indicators)
        self.assertIn('autopsy', indicators[3])
        
    def test_get_noncompliance_labels(self):
        """
        Test generating noncompliance labels from notes.
        """
        # Create mock notes data
        notes_df = pd.DataFrame({
            'SUBJECT_ID': [1, 2, 3],
            'TEXT': [
                "Patient is cooperative.",
                "Patient refused medication and is non-compliant.",
                "Patient with mistrust of healthcare system."
            ]
        })
        
        labels = get_noncompliance_labels(notes_df)
        self.assertEqual(len(labels), 3)
        self.assertEqual(labels[1], 0)  # No noncompliance keywords
        self.assertEqual(labels[2], 1)  # Has noncompliance keywords
        
    def test_get_autopsy_labels(self):
        """
        Test generating autopsy labels from notes.
        """
        # Create mock notes data
        notes_df = pd.DataFrame({
            'SUBJECT_ID': [1, 2, 3],
            'TEXT': [
                "Patient is cooperative.",
                "Patient refused medication and is non-compliant.",
                "Family requested autopsy to determine cause of death."
            ]
        })
        
        labels = get_autopsy_labels(notes_df)
        self.assertEqual(len(labels), 3)
        self.assertEqual(labels[1], 0)  # No autopsy keywords
        self.assertEqual(labels[3], 1)  # Has autopsy keywords


if __name__ == "__main__":
    unittest.main()