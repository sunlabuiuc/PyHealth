"""
Synthetic EHR Data Evaluation - Example/Use Case of Pyhealth

Authors: Will Hunnius (hunnius3), Jiesen Zhang (jiesenz)
Course: CS598 DL4H - Deep Learning for Healthcare, UIUC


Description:
    Evaluates the quality of synthetic Electronic Health Record (EHR) data
    by comparing it to real data. Measures how realistic and useful the
    synthetic data is for training machine learning models.

    Based on: Lin et al. (2025) "A Case Study Exploring the Current Landscape
    of Synthetic Medical Record Generation with Commercial LLMs" - JMLR CHIL 2025
    https://arxiv.org/abs/2504.14657

Inputs:
    real_data: pandas DataFrame with real patient records
        - Must contain numerical features (age, vitals, labs, etc.)
        - Must contain a binary target column (e.g., "mortality": 0 or 1)
    
    synthetic_data: pandas DataFrame with LLM-generated patient records
        - Same columns as real_data

Outputs:
    Dictionary with two evaluation categories:
    
    fidelity: Distribution matching metrics
        - mean_kl: Average KL divergence across features (lower = better)
        - per_feature: KL divergence for each numerical column
    
    utility: Predictive performance metrics (TSTR paradigm)
        - tstr_auc: AUC when model is trained on synthetic, tested on real
        - trtr_auc: AUC when model is trained on real, tested on real (baseline)
        - utility_gap: trtr_auc - tstr_auc (smaller = synthetic data is more useful)

"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from pyhealth.datasets import MIMIC4Dataset


class SyntheticDataEvaluator:
    """Evaluates synthetic EHR data quality against real PyHealth data."""
    
    def __init__(self, target_column="mortality"):
        self.target_column = target_column

    def compute_kl_divergence(self, real, synthetic, n_bins=50):
        """Compute KL divergence between two distributions."""
        real = np.asarray(real).flatten()
        synthetic = np.asarray(synthetic).flatten()
        real = real[~np.isnan(real)]
        synthetic = synthetic[~np.isnan(synthetic)]
        
        if len(real) == 0 or len(synthetic) == 0:
            return float("inf")
        
        combined = np.concatenate([real, synthetic])
        bin_edges = np.histogram_bin_edges(combined, bins=n_bins)
        real_hist, _ = np.histogram(real, bins=bin_edges, density=True)
        synthetic_hist, _ = np.histogram(synthetic, bins=bin_edges, density=True)
        
        epsilon = 1e-10
        real_hist = (real_hist + epsilon) / (real_hist + epsilon).sum()
        synthetic_hist = (synthetic_hist + epsilon) / (synthetic_hist + epsilon).sum()
        
        return float(entropy(real_hist, synthetic_hist))

    def evaluate_fidelity(self, real_data, synthetic_data):
        """Evaluate distribution matching between real and synthetic data."""
        numerical_cols = real_data.select_dtypes(include=[np.number]).columns
        numerical_cols = [c for c in numerical_cols if c in synthetic_data.columns]
        
        kl_scores = {col: self.compute_kl_divergence(real_data[col].values, synthetic_data[col].values) 
                     for col in numerical_cols}
        
        return {"per_feature": kl_scores, "mean_kl": np.mean(list(kl_scores.values()))}

    def _prepare_features(self, df):
        """Encode categorical features and fill missing values."""
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.Categorical(df[col]).codes
        return df.fillna(df.median()).values

    def evaluate_utility(self, real_train, real_test, synthetic_train):
        """Evaluate TSTR (Train-Synthetic, Test-Real) performance."""
        feature_cols = [c for c in real_train.columns if c != self.target_column]
        
        X_real_train = self._prepare_features(real_train[feature_cols])
        X_real_test = self._prepare_features(real_test[feature_cols])
        X_syn_train = self._prepare_features(synthetic_train[feature_cols])
        
        y_real_train = real_train[self.target_column].values
        y_real_test = real_test[self.target_column].values
        y_syn_train = synthetic_train[self.target_column].values
        
        # TSTR: Train on Synthetic, Test on Real
        model_tstr = RandomForestClassifier(n_estimators=100, random_state=42)
        model_tstr.fit(X_syn_train, y_syn_train)
        tstr_auc = roc_auc_score(y_real_test, model_tstr.predict_proba(X_real_test)[:, 1])
        
        # TRTR: Train on Real, Test on Real (baseline)
        model_trtr = RandomForestClassifier(n_estimators=100, random_state=42)
        model_trtr.fit(X_real_train, y_real_train)
        trtr_auc = roc_auc_score(y_real_test, model_trtr.predict_proba(X_real_test)[:, 1])
        
        return {"tstr_auc": tstr_auc, "trtr_auc": trtr_auc, "utility_gap": trtr_auc - tstr_auc}

    def evaluate(self, real_data, synthetic_data, test_size=0.2):
        """Run full evaluation: fidelity + utility."""
        real_train, real_test = train_test_split(
            real_data, test_size=test_size, random_state=42,
            stratify=real_data[self.target_column] if self.target_column in real_data.columns else None
        )
        
        return {
            "fidelity": self.evaluate_fidelity(real_train, synthetic_data),
            "utility": self.evaluate_utility(real_train, real_test, synthetic_data),
        }


# Example: Using PyHealth to load real data and evaluate synthetic data
if __name__ == "__main__":
    
    # Load real EHR data using PyHealth's MIMIC4Dataset
    mimic4_dataset = MIMIC4Dataset(
        root="/path/to/mimic4",
        tables=["diagnoses_icd", "procedures_icd", "labevents"],
        dev=True  # Use dev=True for smaller subset during testing
    )
    
    # Convert PyHealth dataset to pandas DataFrame
    real_data = mimic4_dataset.to_dataframe()
    
    # Load synthetic data (generated by LLM like GPT-4)
    synthetic_data = pd.read_csv("synthetic_ehr_data.csv")
    
    # Evaluate synthetic data quality
    evaluator = SyntheticDataEvaluator(target_column="mortality")
    results = evaluator.evaluate(real_data, synthetic_data)
    
    print("Fidelity (KL Divergence):", results["fidelity"]["mean_kl"])
    print("Utility (TSTR AUC):", results["utility"]["tstr_auc"])
    print("Baseline (TRTR AUC):", results["utility"]["trtr_auc"])
