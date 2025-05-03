"""
Example using GPBoost for binary sleep classification with Empatica E4 data.

Implementation based on:
Wang, Z., Zeng, T., Liu, Z., & Williams, C. K. I. (2024). 
Addressing Wearable Sleep Tracking Inequity: A New Dataset and Novel Methods for a Population with Sleep Disorders. 
In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, 
PMLR 248:8716-8741. https://proceedings.mlr.press/v248/wang24a.html
Offical code repository: https://github.com/WillKeWang/DREAMT_FE
"""
import sys

from examples.synthetic_sleep_data_generator import SyntheticSleepDataGenerator
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("NumPy not installed. Install with: pip install numpy")
    sys.exit(1)
except Exception as e:
    print(f"Error importing NumPy: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError:
    print("Pandas not installed. Install with: pip install pandas")
    sys.exit(1)
except ValueError as e:
    if "numpy.dtype size changed" in str(e):
        print("\nERROR: NumPy and Pandas version incompatibility detected.")
        print("Solution options:")
        print("1. Reinstall pandas: pip install --force-reinstall pandas")
        print("2. Ensure consistent versions: pip install 'numpy==1.23.5' 'pandas==1.5.3'")
        sys.exit(1)
    else:
        print(f"Error importing Pandas: {e}")
        sys.exit(1)

import random
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

try:
    from pyhealth.datasets import SampleEHRDataset
    from pyhealth.models import GPBoostTimeSeriesModel
except ImportError as e:
    print("\nERROR: PyHealth modules not found.")
    print("\nTroubleshooting steps:")
    print("1. Confirm PyHealth is installed: pip list | grep pyhealth")
    print("2. If not installed: pip install -e .")
    sys.exit(1)
except Exception as e:
    print(f"Error importing PyHealth modules: {e}")
    sys.exit(1)

class BinaryTokenizer:
    def __init__(self):
        self.vocabulary = {"Awake": 0, "Asleep": 1}
        self.reverse_vocab = {0: "Awake", 1: "Asleep"}
    
    def encode(self, label):
        return [self.vocabulary.get(label, 0)]
    
    def decode(self, index):
        return self.reverse_vocab.get(index, "Unknown")
    
def print_sample_data_overview(data):
    print("\n=== Sample Data Overview ===")
    
    sample_patient = random.choice(data)
    patient_id = sample_patient["patient_id"]
    
    visits = sample_patient["visits"]
    total_epochs = len(visits)
    sleep_epochs = sum(1 for v in visits if v["sleep_stage"] == "Asleep")
    wake_epochs = total_epochs - sleep_epochs
    
    print(f"Patient {patient_id} - {total_epochs} epochs ({wake_epochs} awake, {sleep_epochs} asleep)")
    print(f"Random effects: Obesity = {sample_patient['obesity']}, Apnea = {sample_patient['apnea']}")
    
    print("\nFirst 5 epochs of data:")
    print("Time | Sleep Stage | Heart Rate | EDA  | Temperature | Movement")
    print("-" * 70)
    for i, visit in enumerate(visits[:5]):
        print(f"{i:4d} | {visit['sleep_stage']:11s} | {visit['heart_rate']:9.1f} | {visit['eda']:.3f} | "
                f"{visit['temperature']:10.1f} | {visit['accelerometer']:.4f}")
                
    # Show transitions: 5 epochs around a state change if possible
    print("\nSample sleep transition (if available):")
    transition_idx = None
    for i in range(1, total_epochs):
        if visits[i]['sleep_stage'] != visits[i-1]['sleep_stage']:
            transition_idx = i
            break
            
    if transition_idx and transition_idx > 2 and transition_idx < total_epochs - 2:
        print("Time | Sleep Stage | Heart Rate | EDA  | Temperature | Movement")
        print("-" * 70)
        for i in range(transition_idx - 2, transition_idx + 3):
            visit = visits[i]
            print(f"{i:4d} | {visit['sleep_stage']:11s} | {visit['heart_rate']:9.1f} | {visit['eda']:.3f} | "
                    f"{visit['temperature']:10.1f} | {visit['accelerometer']:.4f}")
    else:
        print("No clear transition found in the sample")
        
    print("\nFeature statistics across all patients:")
    all_hr = [visit['heart_rate'] for patient in data for visit in patient['visits']]
    all_eda = [visit['eda'] for patient in data for visit in patient['visits']]
    all_temp = [visit['temperature'] for patient in data for visit in patient['visits']]
    all_acc = [visit['accelerometer'] for patient in data for visit in patient['visits']]
    
    print(f"Heart Rate: min={min(all_hr):.1f}, max={max(all_hr):.1f}, mean={np.mean(all_hr):.1f}, std={np.std(all_hr):.1f}")
    print(f"EDA: min={min(all_eda):.3f}, max={max(all_eda):.3f}, mean={np.mean(all_eda):.3f}, std={np.std(all_eda):.3f}")
    print(f"Temperature: min={min(all_temp):.1f}, max={max(all_temp):.1f}, mean={np.mean(all_temp):.1f}, std={np.std(all_temp):.1f}")
    print(f"Movement: min={min(all_acc):.4f}, max={max(all_acc):.4f}, mean={np.mean(all_acc):.4f}, std={np.std(all_acc):.4f}")
    
    all_stages = [visit['sleep_stage'] for patient in data for visit in patient['visits']]
    awake_percent = 100 * all_stages.count("Awake") / len(all_stages)
    asleep_percent = 100 * all_stages.count("Asleep") / len(all_stages)
    print(f"\nSleep stages: {awake_percent:.1f}% Awake, {asleep_percent:.1f}% Asleep")

def print_model_performance_summary(model):
    """
    Print a summary of the model performance.
    
    Args:
        model: The trained GPBoost model.
    """
    print("\nModel performance summary:")
    
    re_info = model.get_random_effects_info()
    
    if 'model_params' in re_info:
        print("\nGP Model Parameters:")
        for k, v in re_info['model_params'].items():
            print(f"  {k}: {v}")

    print(f"\nModel type: GPBoost with random effects")
    print(f"Features used: {', '.join(feature_keys)}")
    print(f"Random effect features: {', '.join(random_effect_keys)}")

def print_sample_predictions(model, test_data):
    """
    Print sample predictions for the first 10 time points of the first test patient.
    
    Args:
        model: The trained GPBoost model.
        test_data: The test dataset.
    """
    df_test = model._data_to_pandas(test_data[:1])
    sample_true = df_test['label'].values[:10].astype(int)
    
    n_samples = min(10, len(sample_true))
    if len(y_prob) >= n_samples:
        sample_prob = y_prob[:n_samples, 0] if len(y_prob.shape) > 1 else y_prob[:n_samples]
        sample_pred = (sample_prob > 0.5).astype(int)
        
        print("Time | True State | Predicted State | Probability")
        print("--------------------------------------------------")
        for t in range(n_samples):
            true_state = "Asleep" if sample_true[t] == 1 else "Awake"
            pred_state = "Asleep" if sample_pred[t] == 1 else "Awake"
            print(f"{t:4d} | {true_state:10s} | {pred_state:14s} | {float(sample_prob[t]):.4f}")

def print_eval_results(y_true, y_prob):
    """
    Print evaluation results.
    
    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
    """
    y_prob_flat = y_prob.flatten() if len(y_prob.shape) > 1 else y_prob
    y_pred = (y_prob_flat > 0.5).astype(int)
        
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob_flat)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\nEvaluation Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

def create_datasets(data):
    """
    Create datasets for training and evaluation.
    
    Args:
        data: The dataset.
        feature_keys: List of feature keys.
        label_key: Label key.
        group_key: Group key.
        random_effect_keys: Random effect keys.
    
    Returns:
        train_data, val_data, test_data: The created datasets.
    """
    # Split into train, validation, and test sets
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
    print(f"Training set: {len(train_data)} patients")
    print(f"Validation set: {len(val_data)} patients")
    print(f"Test set: {len(test_data)} patients")

    return train_data, val_data, test_data

if __name__ == "__main__":
    print("\n" + "="*50)
    print("GPBoost Binary Sleep Classification Example")
    print("="*50 + "\n")
    
    print("Generating synthetic Empatica E4 sleep data...")
    generator = SyntheticSleepDataGenerator(hours=8, epoch_seconds=30, seed=123)
    data = generator.generate_data(num_patients=100)
    print(f"Generated data for {len(data)} patients")
    print(f"Average epochs per patient: {sum(len(p['visits']) for p in data)/len(data):.1f}")

    print_sample_data_overview(data)
    
    feature_keys = ["heart_rate", "eda", "temperature", "accelerometer"]
    label_key = "sleep_stage"
    group_key = "patient_id"
    random_effect_keys = ["obesity", "apnea"]
    
    print("Creating datasets...")
    train_data, val_data, test_data = create_datasets(data=data)
    
    # Hyperopt-style parameter space for optimization
    from hyperopt import hp
    
    param_space = {
        "max_depth": hp.quniform("max_depth", 3, 6, 1),
        "learning_rate": hp.uniform("learning_rate", 0.005, 0.01),
        "num_leaves": hp.quniform("num_leaves", 20, 200, 20),
        "feature_fraction": hp.uniform("feature_fraction", 0.5, 0.95),
        "lambda_l2": hp.uniform("lambda_l2", 1.0, 10.0),
        "lambda_l1": hp.quniform("lambda_l1", 10, 100, 10),
        "pos_bagging_fraction": hp.uniform("pos_bagging_fraction", 0.8, 0.95),
        "neg_bagging_fraction": hp.uniform("neg_bagging_fraction", 0.6, 0.8),
        "num_boost_round": hp.quniform("num_boost_round", 400, 1000, 100),
    }
    
    print("Training GPBoost model for binary sleep classification...")
    model = GPBoostTimeSeriesModel(
        label_tokenizer=BinaryTokenizer(),
        feature_keys=feature_keys,
        label_key=label_key,
        group_key=group_key,
        random_effect_features=random_effect_keys,
        verbose=-1
    )
    
    # Optimize hyperparameters with hyperopt
    print("Optimizing hyperparameters...")
    best_params = model.optimize_hyperparameters(
        train_data=train_data,
        val_data=val_data,
        param_space=param_space,
        n_iter=20,
        verbose=1
    )
    
    # Update model parameters with best ones
    model.kwargs.update(best_params)
    
    # Train with optimized parameters
    print("Training with optimized parameters...")
    model.train(train_data, val_data)
    print("Training complete")
    
    print("Evaluating model...")
    eval_results = model.inference(test_data)
    
    y_true = eval_results["y_true"].astype(int)
    y_prob = eval_results["y_prob"]
        
    print_eval_results(y_true, y_prob)
    print_sample_predictions(model, test_data)
    print_model_performance_summary(model)

