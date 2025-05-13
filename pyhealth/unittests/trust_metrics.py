"""
Lalith Devireddy, Trisha Rayan
NetID: lalithd, trayan2
Paper Title: Racial Disparities and Mistrust in End-of-Life Care
Paper Link: https://proceedings.mlr.press/v85/boag18a.html
Description: Implementation of trust metrics for healthcare data analysis based on
            the "Racial Disparities and Mistrust in End-of-Life Care" paper.
            This module provides functions to calculate trust metrics from
            clinical notes and structured data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import re
from typing import Dict, List, Tuple, Union, Optional, Any

class TrustMetrics:
    """
    Class implementing trust metrics from the paper 
    "Racial Disparities and Mistrust in End-of-Life Care" (Boag et al.)
    
    This class provides methods to:
    1. Compute noncompliance-derived mistrust 
    2. Compute autopsy-derived mistrust
    3. Calculate sentiment-based mistrust
    4. Combine trust metrics for prediction tasks
    """
    
    def __init__(self, regularization_strength: float = 0.1):
        """
        Initialize trust metrics calculator with configurable parameters.
        
        Args:
            regularization_strength: L1 regularization strength for logistic regression models
        """
        self.regularization_strength = regularization_strength
        self.noncompliance_model = None
        self.autopsy_model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
    def fit_noncompliance_metric(self, 
                               features: pd.DataFrame, 
                               noncompliance_labels: pd.Series) -> None:
        """
        Trains a logistic regression model to predict noncompliance from chartevents features.
        
        Args:
            features: DataFrame containing interpersonal features from chartevents
            noncompliance_labels: Series with binary labels for patient noncompliance
                                  (1 = noncompliant, 0 = compliant)
        
        Returns:
            None, but updates the internal noncompliance_model
        """
        self.feature_names = features.columns
        X = self.scaler.fit_transform(features)
        
        # Create L1-regularized logistic regression model as in the original paper
        self.noncompliance_model = LogisticRegression(
            penalty='l1', 
            C=1/self.regularization_strength,
            solver='liblinear',  # Required for L1 penalty
            max_iter=1000,
            random_state=42
        )
        
        self.noncompliance_model.fit(X, noncompliance_labels)
        
    def fit_autopsy_metric(self, 
                         features: pd.DataFrame, 
                         autopsy_labels: pd.Series) -> None:
        """
        Trains a logistic regression model to predict autopsy from chartevents features.
        
        Args:
            features: DataFrame containing interpersonal features from chartevents
            autopsy_labels: Series with binary labels for patient autopsy
                           (1 = autopsy performed, 0 = no autopsy)
        
        Returns:
            None, but updates the internal autopsy_model
        """
        if self.feature_names is None:
            self.feature_names = features.columns
            
        X = self.scaler.fit_transform(features)
        
        # Create L1-regularized logistic regression model as in the original paper
        self.autopsy_model = LogisticRegression(
            penalty='l1', 
            C=1/self.regularization_strength,
            solver='liblinear',  # Required for L1 penalty
            max_iter=1000,
            random_state=42
        )
        
        self.autopsy_model.fit(X, autopsy_labels)
    
    def calculate_noncompliance_mistrust(self, features: pd.DataFrame) -> np.ndarray:
        """
        Calculates noncompliance-derived mistrust scores for new patients.
        
        Args:
            features: DataFrame containing interpersonal features from chartevents
            
        Returns:
            Array of mistrust scores based on noncompliance model
        """
        if self.noncompliance_model is None:
            raise ValueError("Noncompliance model has not been trained. Call fit_noncompliance_metric first.")
            
        X = self.scaler.transform(features)
        # Return probability of noncompliance as trust score
        return self.noncompliance_model.predict_proba(X)[:, 1]
    
    def calculate_autopsy_mistrust(self, features: pd.DataFrame) -> np.ndarray:
        """
        Calculates autopsy-derived mistrust scores for new patients.
        
        Args:
            features: DataFrame containing interpersonal features from chartevents
            
        Returns:
            Array of mistrust scores based on autopsy model
        """
        if self.autopsy_model is None:
            raise ValueError("Autopsy model has not been trained. Call fit_autopsy_metric first.")
            
        X = self.scaler.transform(features)
        # Return probability of autopsy as trust score  
        return self.autopsy_model.predict_proba(X)[:, 1]
    
    def calculate_sentiment_mistrust(self, notes: List[str], 
                                   sentiment_analyzer=None) -> np.ndarray:
        """
        Calculates sentiment-based mistrust scores from clinical notes.
        Higher negative sentiment corresponds to higher mistrust.
        
        Args:
            notes: List of clinical notes as strings
            sentiment_analyzer: Optional sentiment analyzer function or object.
                              If None, uses a simple rule-based approach.
                              
        Returns:
            Array of sentiment-based mistrust scores
        """
        if sentiment_analyzer is None:
            # Simple rule-based sentiment analysis
            # In a real implementation, use a more sophisticated approach
            # such as VADER or another sentiment analysis library
            
            # Define lists of positive and negative words based on the paper's approach
            negative_words = [
                'mistrust', 'mistrusting', 'distrust', 'distrusting', 
                'refuse', 'refused', 'refusal', 'deny', 'denies', 'denied',
                'non-compliant', 'noncompliant', 'non compliant',
                'agitated', 'agitation', 'angry', 'upset', 'frustrated'
            ]
            
            positive_words = [
                'trust', 'trusting', 'comply', 'compliant', 'compliance',
                'cooperative', 'agreeable', 'calm', 'satisfied'
            ]
            
            # Calculate sentiment scores for each note
            scores = []
            for note in notes:
                note = note.lower()
                negative_count = sum(note.count(word) for word in negative_words)
                positive_count = sum(note.count(word) for word in positive_words)
                
                # Compute sentiment score: negative - positive
                # More negative words = higher mistrust
                scores.append(negative_count - positive_count)
                
            return np.array(scores)
        else:
            # Use provided sentiment analyzer
            return np.array([sentiment_analyzer(note) for note in notes])
    
    def get_important_features(self, model_type: str = 'noncompliance', 
                             top_n: int = 5) -> Dict[str, float]:
        """
        Returns the most important features for the specified model.
        
        Args:
            model_type: Either 'noncompliance' or 'autopsy'
            top_n: Number of top positive and negative features to return
            
        Returns:
            Dictionary with top positive and negative features and their weights
        """
        if model_type == 'noncompliance':
            if self.noncompliance_model is None:
                raise ValueError("Noncompliance model has not been trained.")
            model = self.noncompliance_model
        elif model_type == 'autopsy':
            if self.autopsy_model is None:
                raise ValueError("Autopsy model has not been trained.")
            model = self.autopsy_model
        else:
            raise ValueError("model_type must be either 'noncompliance' or 'autopsy'")
            
        # Get feature coefficients
        coefficients = model.coef_[0]
        feature_importance = list(zip(self.feature_names, coefficients))
        
        # Sort by absolute coefficient value
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get top positive and negative features
        top_positive = [(name, coef) for name, coef in feature_importance if coef > 0][:top_n]
        top_negative = [(name, coef) for name, coef in feature_importance if coef < 0][:top_n]
        
        return {
            'top_positive': dict(top_positive),
            'top_negative': dict(top_negative)
        }
    
    def find_noncompliance_keywords(self, notes: List[str]) -> Dict[str, List[int]]:
        """
        Identifies notes that contain noncompliance language.
        
        Args:
            notes: List of clinical notes
            
        Returns:
            Dictionary mapping keywords to the indices of notes containing them
        """
        # Keywords associated with noncompliance as per the paper
        noncompliance_keywords = [
            'non-compliant', 'noncompliant', 'non compliant',
            'refused', 'refuses', 'refusal', 'decline', 'declined',
            'against medical advice', 'AMA', 'left AMA',
            'denied', 'denies', 'reluctant', 'unwilling'
        ]
        
        # Find notes containing each keyword
        keyword_to_notes = {}
        for keyword in noncompliance_keywords:
            indices = []
            for i, note in enumerate(notes):
                if re.search(r'\b' + re.escape(keyword) + r'\b', note, re.IGNORECASE):
                    indices.append(i)
            if indices:
                keyword_to_notes[keyword] = indices
                
        return keyword_to_notes
    
    def find_autopsy_keywords(self, notes: List[str]) -> Dict[str, List[int]]:
        """
        Identifies notes that contain autopsy-related language.
        
        Args:
            notes: List of clinical notes
            
        Returns:
            Dictionary mapping keywords to the indices of notes containing them
        """
        # Keywords associated with autopsy as per the paper
        autopsy_keywords = [
            'autopsy', 'post-mortem', 'post mortem', 'postmortem', 
            'cause of death', 'medical examiner'
        ]
        
        # Find notes containing each keyword
        keyword_to_notes = {}
        for keyword in autopsy_keywords:
            indices = []
            for i, note in enumerate(notes):
                if re.search(r'\b' + re.escape(keyword) + r'\b', note, re.IGNORECASE):
                    indices.append(i)
            if indices:
                keyword_to_notes[keyword] = indices
                
        return keyword_to_notes
    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalizes scores to have zero mean and unit variance.
        
        Args:
            scores: Array of mistrust scores
            
        Returns:
            Normalized scores
        """
        return (scores - np.mean(scores)) / np.std(scores)
    
    def combine_metrics(self, 
                      noncompliance_scores: np.ndarray, 
                      autopsy_scores: np.ndarray, 
                      sentiment_scores: np.ndarray,
                      weights: List[float] = None) -> np.ndarray:
        """
        Combines multiple trust metrics into a single score using weighted average.
        
        Args:
            noncompliance_scores: Array of noncompliance-derived mistrust scores
            autopsy_scores: Array of autopsy-derived mistrust scores
            sentiment_scores: Array of sentiment-based mistrust scores
            weights: List of weights for each metric (default: equal weights)
            
        Returns:
            Combined mistrust scores
        """
        # Normalize all scores
        norm_noncompliance = self.normalize_scores(noncompliance_scores)
        norm_autopsy = self.normalize_scores(autopsy_scores)
        norm_sentiment = self.normalize_scores(sentiment_scores)
        
        # Create weight vector (default: equal weights)
        if weights is None:
            weights = [1/3, 1/3, 1/3]
        else:
            # Normalize weights to sum to 1
            weights = np.array(weights) / sum(weights)
            
        # Combine scores using weighted average
        combined_scores = (
            weights[0] * norm_noncompliance + 
            weights[1] * norm_autopsy + 
            weights[2] * norm_sentiment
        )
        
        return combined_scores


def extract_chartevents_features(chartevents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract binary features from chartevents table for trust metric calculation.
    
    Helper function to process MIMIC-III chartevents data into features for trust metrics.
    
    Args:
        chartevents_df: DataFrame containing chartevents data
        
    Returns:
        DataFrame with binary features for each patient
    """
    # Define categories of chartevents based on the paper
    feature_categories = {
        'pain_management': [
            'pain', 'pain level', 'pain present', 'pain assess method',
            'pain level acceptable', 'currently experiencing pain'
        ],
        'agitation': [
            'riker-sas scale', 'richmond-ras scale', 'agitated'
        ],
        'restraints': [
            'restraint device', 'restraint type', 'wrist restraints',
            'violent restraints ordered', 'non-violent restraints',
            'reason for restraint', 'reapplied restraints'
        ],
        'education': [
            'education barrier', 'education learner', 'education method',
            'education readiness', 'education topic', 'teaching directed toward'
        ],
        'communication': [
            'family communication method', 'family meeting',
            'informed', 'understand & agree with plan'
        ],
        'support': [
            'support systems', 'spiritual support', 'social work consult',
            'healthcare proxy', 'spokesperson'
        ]
    }
    
    # Flatten the feature categories to get all feature names
    all_features = [feature for category in feature_categories.values() 
                   for feature in category]
    
    # Extract patient-level features
    patient_features = {}
    for patient_id, group in chartevents_df.groupby('SUBJECT_ID'):
        patient_features[patient_id] = {}
        
        # Initialize all features to 0 (absent)
        for feature in all_features:
            patient_features[patient_id][feature] = 0
            
        # Set features to 1 if present
        for _, row in group.iterrows():
            item_name = row['ITEMID']
            item_value = row['VALUE']
            
            # Match item name to features
            for feature in all_features:
                if feature.lower() in item_name.lower():
                    patient_features[patient_id][feature] = 1
                    
                    # Add specific values for certain features
                    if feature in ['riker-sas scale', 'richmond-ras scale']:
                        try:
                            # Try to extract numerical value
                            numerical_value = float(item_value)
                            # For agitation scales, higher values typically indicate agitation
                            if numerical_value > 3:  # Threshold for agitation
                                patient_features[patient_id][feature + '_agitated'] = 1
                            else:
                                patient_features[patient_id][feature + '_agitated'] = 0
                        except (ValueError, TypeError):
                            # If not numerical, check for agitation keywords
                            if 'agit' in item_value.lower():
                                patient_features[patient_id][feature + '_agitated'] = 1
                            else:
                                patient_features[patient_id][feature + '_agitated'] = 0
    
    # Convert to DataFrame
    features_df = pd.DataFrame.from_dict(patient_features, orient='index')
    
    return features_df


def extract_trust_indicators_from_notes(notes_df: pd.DataFrame) -> Dict[str, Dict[str, List[int]]]:
    """
    Extract trust-related indicators from clinical notes.
    
    Args:
        notes_df: DataFrame containing clinical notes
        
    Returns:
        Dictionary with noncompliance and autopsy indicators
    """
    # Create trust metrics object for keyword finding
    trust_metrics = TrustMetrics()
    
    # Group notes by patient
    patient_notes = {}
    for patient_id, group in notes_df.groupby('SUBJECT_ID'):
        patient_notes[patient_id] = group['TEXT'].tolist()
    
    # Initialize results
    results = {}
    
    # Process each patient's notes
    for patient_id, notes in patient_notes.items():
        noncompliance = trust_metrics.find_noncompliance_keywords(notes)
        autopsy = trust_metrics.find_autopsy_keywords(notes)
        
        results[patient_id] = {
            'noncompliance': noncompliance,
            'autopsy': autopsy
        }
    
    return results


def get_noncompliance_labels(notes_df: pd.DataFrame) -> pd.Series:
    """
    Generate noncompliance labels from clinical notes.
    
    Args:
        notes_df: DataFrame containing clinical notes
        
    Returns:
        Series with binary labels for patient noncompliance
    """
    # Create trust metrics object
    trust_metrics = TrustMetrics()
    
    # Group notes by patient
    patient_notes = {}
    for patient_id, group in notes_df.groupby('SUBJECT_ID'):
        patient_notes[patient_id] = group['TEXT'].tolist()
    
    # Generate labels
    labels = {}
    for patient_id, notes in patient_notes.items():
        noncompliance_indicators = trust_metrics.find_noncompliance_keywords(notes)
        # Label as noncompliant if any noncompliance indicators found
        labels[patient_id] = 1 if noncompliance_indicators else 0
    
    return pd.Series(labels)


def get_autopsy_labels(notes_df: pd.DataFrame) -> pd.Series:
    """
    Generate autopsy labels from clinical notes.
    
    Args:
        notes_df: DataFrame containing clinical notes
        
    Returns:
        Series with binary labels for patient autopsy
    """
    # Create trust metrics object
    trust_metrics = TrustMetrics()
    
    # Group notes by patient
    patient_notes = {}
    for patient_id, group in notes_df.groupby('SUBJECT_ID'):
        patient_notes[patient_id] = group['TEXT'].tolist()
    
    # Generate labels
    labels = {}
    for patient_id, notes in patient_notes.items():
        autopsy_indicators = trust_metrics.find_autopsy_keywords(notes)
        # Label as autopsy if any autopsy indicators found
        labels[patient_id] = 1 if autopsy_indicators else 0
    
    return pd.Series(labels)


def main():
    """
    Example usage of the trust metrics module.
    """
    # This would typically be run with actual MIMIC-III data
    # Here we just demonstrate the API with dummy data
    
    # Create dummy data
    import numpy as np
    
    # Sample features (would normally come from chartevents)
    num_patients = 100
    features = pd.DataFrame({
        'pain': np.random.randint(0, 2, num_patients),
        'agitated': np.random.randint(0, 2, num_patients),
        'restraint': np.random.randint(0, 2, num_patients),
        'education_barrier': np.random.randint(0, 2, num_patients),
        'family_meeting': np.random.randint(0, 2, num_patients)
    })
    
    # Sample labels
    noncompliance_labels = np.random.randint(0, 2, num_patients)
    autopsy_labels = np.random.randint(0, 2, num_patients)
    
    # Sample notes (simplified for demonstration)
    notes = [
        f"Patient {i} {'showing signs of noncompliance' if np.random.random() > 0.7 else 'cooperative'}"
        for i in range(num_patients)
    ]
    
    # Initialize trust metrics calculator
    trust_metrics = TrustMetrics(regularization_strength=0.1)
    
    # Train models
    trust_metrics.fit_noncompliance_metric(features, noncompliance_labels)
    trust_metrics.fit_autopsy_metric(features, autopsy_labels)
    
    # Calculate trust metrics
    noncompliance_scores = trust_metrics.calculate_noncompliance_mistrust(features)
    autopsy_scores = trust_metrics.calculate_autopsy_mistrust(features)
    sentiment_scores = trust_metrics.calculate_sentiment_mistrust(notes)
    
    # Combine metrics
    combined_scores = trust_metrics.combine_metrics(
        noncompliance_scores, autopsy_scores, sentiment_scores
    )
    
    # Get most important features
    important_features = trust_metrics.get_important_features('noncompliance')
    
    print("Noncompliance scores (first 5):", noncompliance_scores[:5])
    print("Autopsy scores (first 5):", autopsy_scores[:5])
    print("Sentiment scores (first 5):", sentiment_scores[:5])
    print("Combined scores (first 5):", combined_scores[:5])
    print("Important features for noncompliance:", important_features)


if __name__ == "__main__":
    main()
