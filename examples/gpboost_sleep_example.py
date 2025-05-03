"""
Example using GPBoost for binary sleep classification with Empatica E4 data.
"""
import sys
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

class SyntheticSleepDataGenerator:
    """Generator for synthetic Empatica E4 sleep data with binary states."""
    
    def __init__(
        self, 
        hours: int = 8, 
        epoch_seconds: int = 30, 
        seed: int = 42
    ):
        self.hours = hours
        self.epoch_seconds = epoch_seconds
        self.seed = seed
        self.sleep_states = ["Awake", "Asleep"]
        
        self.epochs_per_hour = int(3600 / epoch_seconds)
        self.total_epochs = int(hours * self.epochs_per_hour)
        
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_data(self, num_patients: int = 100) -> list:
        """
        Generate synthetic sleep data for multiple patients.
        
        Args:
            num_patients: Number of patients to generate data for
            
        Returns:
            List of patient data dictionaries
        """
        data = []
        for i in range(num_patients):
            patient_id = f"patient_{i}"
            patient_data = self._generate_patient_data(patient_id)
            data.append(patient_data)
        
        return data
    
    def _generate_patient_data(self, patient_id: str) -> dict:
        """Generate data for a single patient."""
        # Generate random effects
        is_obese = random.choice([0, 1])
        has_apnea = random.choice([0, 1])
        
        # Generate patient-specific baseline physiological values
        baselines = self._generate_baselines(is_obese, has_apnea)
        
        # Generate state transitions and awakenings
        sleep_onset, final_wake, awakening_points, awakening_durations = self._create_sleep_structure(has_apnea)
        
        # Generate visits with physiological signals
        visits = self._generate_visits(baselines, sleep_onset, final_wake, 
                                      awakening_points, awakening_durations, has_apnea)
        
        # Create complete patient record
        patient_record = {
            "patient_id": patient_id,
            "visits": visits,
            "obesity": is_obese,
            "apnea": has_apnea
        }
        
        return patient_record
    
    def _generate_baselines(self, is_obese: int, has_apnea: int) -> dict:
        """Generate baseline physiological values with random effects."""
        # Base physiological values with variability
        baselines = {
            'hr': 60 + np.random.normal(5, 8),  # Heart rate
            'eda': 0.3 + np.random.gamma(2, 0.1),  # Skin conductance
            'temp': 36.0 + np.random.normal(0.5, 0.4),  # Temperature
            'acc': 0.01 + np.random.exponential(0.02),  # Movement
        }
        
        # Patient-specific state differentiation
        state_diffs = {
            'hr_diff': 10 + np.random.normal(0, 5),
            'eda_diff': 0.2 + np.random.normal(0, 0.1),
            'acc_diff': 0.03 + np.random.exponential(0.02),
        }
        
        # Apply random effects to base values
        if is_obese:
            baselines['hr'] += np.random.normal(3, 2)
            baselines['temp'] -= np.random.normal(0.1, 0.05)
            
        if has_apnea:
            baselines['hr'] += np.random.normal(2, 1)
            baselines['eda'] += np.random.normal(0.05, 0.03)
            
        return {**baselines, **state_diffs}
    
    def _create_sleep_structure(self, has_apnea: int) -> tuple:
        """Create sleep structure with onset, awakening, and final wake times."""
        # Sleep onset latency (time to fall asleep)
        sleep_onset = int((15 + 15 * np.random.random()) * 60 / self.epoch_seconds)
        
        # Final awakening time
        final_wake = int((10 + 20 * np.random.random()) * 60 / self.epoch_seconds)
        
        # Generate awakenings based on apnea status
        awakenings_per_hour = 0.5 + np.random.exponential(0.5)
        if has_apnea:
            awakenings_per_hour += 0.5 + np.random.exponential(1.0)
        
        # Calculate total number of awakenings
        num_awakenings = max(1, int(awakenings_per_hour * self.hours))
        sleep_period = self.total_epochs - sleep_onset - final_wake
        
        # Generate awakening timepoints and durations
        if sleep_period > num_awakenings:
            awakening_points = sorted(
                random.sample(
                    range(sleep_onset, self.total_epochs - final_wake),
                    num_awakenings
                )
            )
            
            # Duration of awakenings (30s to 3 minutes)
            awakening_durations = [
                max(1, int(np.random.exponential(3) * 60 / self.epoch_seconds))
                for _ in range(num_awakenings)
            ]
        else:
            awakening_points = []
            awakening_durations = []
            
        return sleep_onset, final_wake, awakening_points, awakening_durations
    
    def _generate_visits(self, baselines: dict, sleep_onset: int, 
                        final_wake: int, awakening_points: list, 
                        awakening_durations: list, has_apnea: int) -> list:
        """Generate visit-level time series data with physiological signals."""
        visits = []
        
        # Track signal history for realistic temporal dynamics
        signal_history = {
            'hr': baselines['hr'], 
            'eda': baselines['eda'], 
            'temp': baselines['temp'], 
            'acc': baselines['acc']
        }
        
        # Track state transitions
        last_state = 0
        transition_momentum = 0
        
        # Add momentum to signals for realistic transitions
        hr_momentum, eda_momentum = 0, 0
        temp_momentum, acc_momentum = 0, 0
        
        # Create annotation errors (5% mislabeled)
        mislabel_indices = set(random.sample(range(self.total_epochs), int(0.05 * self.total_epochs)))
        
        # Generate time-series data for each epoch
        for t in range(self.total_epochs):
            # Determine ground truth sleep state
            true_state = self._determine_sleep_state(t, sleep_onset, final_wake, 
                                                    awakening_points, awakening_durations)
            
            # Handle transitions between states
            transition_factor = self._handle_state_transition(true_state, last_state, transition_momentum)
            transition_momentum = max(0, transition_momentum - 1)
            last_state = true_state
            
            # Apply label errors
            stage = self.sleep_states[1 - true_state] if t in mislabel_indices else self.sleep_states[true_state]
            
            # Generate physiological signals for this epoch
            hr, eda, temp, acc = self._generate_signals(
                true_state, baselines, signal_history, transition_factor,
                hr_momentum, eda_momentum, temp_momentum, acc_momentum, has_apnea
            )
            
            # Update signal momentum (autocorrelation)
            hr_momentum = 0.8 * hr_momentum + 0.2 * np.random.normal(0, 2)
            eda_momentum = 0.8 * eda_momentum + 0.2 * np.random.normal(0, 0.05)
            temp_momentum = 0.95 * temp_momentum + 0.05 * np.random.normal(0, 0.1)
            acc_momentum = 0.6 * acc_momentum + 0.4 * np.random.normal(0, 0.01)
            
            # Store signal history for next epoch
            signal_history = {'hr': hr, 'eda': eda, 'temp': temp, 'acc': acc}
            
            # Create visit record
            visit_data = {
                "visit_id": f"visit_{t}",
                "heart_rate": hr,
                "eda": eda,
                "temperature": temp,
                "accelerometer": acc,
                "sleep_stage": stage
            }
            
            visits.append(visit_data)
        
        return visits
    
    def _determine_sleep_state(self, t: int, sleep_onset: int, final_wake: int, 
                             awakening_points: list, awakening_durations: list) -> int:
        """Determine sleep state (0=awake, 1=asleep) for a given time point."""
        if t < sleep_onset or t >= (self.total_epochs - final_wake):
            # Initial period or final waking: awake
            return 0
        else:
            # Middle period: mostly asleep with awakenings
            for i, awakening_time in enumerate(awakening_points):
                if 0 <= t - awakening_time < awakening_durations[i]:
                    return 0  # Awakening period
            return 1  # Default to asleep during night
    
    def _handle_state_transition(self, true_state: int, last_state: int, momentum: int) -> float:
        """Handle state transitions with realistic momentum."""
        if true_state != last_state:
            # Takes longer to fall asleep than wake up
            new_momentum = 4 if true_state == 1 else 2
            return new_momentum / 4.0
        elif momentum > 0:
            return momentum / 4.0
        else:
            return 0
    
    def _generate_signals(self, true_state: int, baselines: dict, history: dict, 
                         transition_factor: float, hr_momentum: float, eda_momentum: float, 
                         temp_momentum: float, acc_momentum: float, has_apnea: int) -> tuple:
        """Generate physiological signals for a single epoch."""
        # Calculate target values based on state
        if true_state == 0:  # Awake
            hr_target = baselines['hr'] + baselines['hr_diff'] + np.random.normal(0, 3)
            eda_target = baselines['eda'] + baselines['eda_diff'] + np.random.normal(0, 0.1)
            temp_target = baselines['temp'] + np.random.normal(0, 0.1)
            acc_target = baselines['acc'] + baselines['acc_diff'] + np.random.exponential(0.02)
        else:  # Asleep
            hr_target = baselines['hr'] - 5 + np.random.normal(0, 2)
            eda_target = baselines['eda'] - 0.1 + np.random.normal(0, 0.05)
            temp_target = baselines['temp'] - 0.2 + np.random.normal(0, 0.1)
            acc_target = baselines['acc'] + np.random.exponential(0.005)
        
        # Blend for transitions
        if transition_factor > 0:
            hr_target = hr_target * (1 - transition_factor) + history['hr'] * transition_factor
            eda_target = eda_target * (1 - transition_factor) + history['eda'] * transition_factor
            temp_target = temp_target * (1 - transition_factor) + history['temp'] * transition_factor
            acc_target = acc_target * (1 - transition_factor) + history['acc'] * transition_factor
        
        # Calculate new values with autocorrelation
        hr = 0.7 * history['hr'] + 0.3 * hr_target + hr_momentum
        eda = 0.8 * history['eda'] + 0.2 * eda_target + eda_momentum
        temp = 0.95 * history['temp'] + 0.05 * temp_target + temp_momentum
        acc = 0.6 * history['acc'] + 0.4 * acc_target + acc_momentum
        
        # Add apnea events during sleep
        if has_apnea and true_state == 1 and random.random() < 0.1:
            hr += np.random.gamma(4, 1)
            eda += np.random.gamma(3, 0.05)
            acc += np.random.exponential(0.02)
        
        # Ensure values are in reasonable ranges
        hr = max(35, min(140, hr))
        eda = max(0.05, eda)
        temp = max(34.5, min(38.5, temp))
        acc = max(0, acc)
        
        return hr, eda, temp, acc

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
            if not k.startswith('_') and v is not None:
                if isinstance(v, float):
                    val_str = f"{v:.4g}"
                elif isinstance(v, (list, tuple)) and len(v) > 6:
                    val_str = f"[{', '.join(str(x) for x in v[:3])}..., {', '.join(str(x) for x in v[-3:])}]"
                else:
                    val_str = str(v)
                print(f"  {k}: {val_str}")

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

if __name__ == "__main__":
    print("\n" + "="*50)
    print("GPBoost Binary Sleep Classification Example")
    print("="*50 + "\n")
    
    print("Generating synthetic Empatica E4 sleep data...")
    generator = SyntheticSleepDataGenerator(hours=8, epoch_seconds=30, seed=123)
    data = generator.generate_data(num_patients=100)
    print(f"Generated data for {len(data)} patients")
    print(f"Average epochs per patient: {sum(len(p['visits']) for p in data)/len(data):.1f}")
    
    feature_keys = ["heart_rate", "eda", "temperature", "accelerometer"]
    label_key = "sleep_stage"
    group_key = "patient_id"
    random_effect_keys = ["obesity", "apnea"]
    
    class BinaryTokenizer:
        def __init__(self):
            self.vocabulary = {"Awake": 0, "Asleep": 1}
            self.reverse_vocab = {0: "Awake", 1: "Asleep"}
        
        def encode(self, label):
            return [self.vocabulary.get(label, 0)]
        
        def decode(self, index):
            return self.reverse_vocab.get(index, "Unknown")
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_data)} patients, Test set: {len(test_data)} patients")
    
    try:
        input_schema = {key: "float" for key in feature_keys}
        output_schema = {label_key: ["Awake", "Asleep"]}
        
        train_dataset = SampleEHRDataset(samples=train_data, dataset_name="synth_sleep_train")
        train_dataset.label_tokenizer = BinaryTokenizer()
        train_dataset.input_schema = input_schema
        train_dataset.output_schema = output_schema
        
        test_dataset = SampleEHRDataset(samples=test_data, dataset_name="synth_sleep_test")
        test_dataset.label_tokenizer = BinaryTokenizer()
        test_dataset.input_schema = input_schema
        test_dataset.output_schema = output_schema
        
    except Exception as e:
        print(f"Error creating PyHealth datasets: {e}")
        sys.exit(1)
    
    print("Training GPBoost model for binary sleep classification...")
    try:
        model = GPBoostTimeSeriesModel(
            dataset=train_dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            group_key=group_key,
            random_effect_features=random_effect_keys,
            num_boost_round=100,
            learning_rate=0.1,
            max_depth=5,
            verbose=-1
        )
        
        model.train(train_data)
        print("Training complete")
    except Exception as e:
        print(f"Error during model training: {e}")
        sys.exit(1)
    
    print("Evaluating model...")
    eval_results = model.inference(test_data)
    
    y_true = eval_results["y_true"].astype(int)
    y_prob = eval_results["y_prob"]
        
    print_eval_results(y_true, y_prob)
    print_sample_predictions(model, test_data)
    print_model_performance_summary(model)

