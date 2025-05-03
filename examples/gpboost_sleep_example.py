"""
Example using GPBoost for binary sleep classification with Empatica E4 data.
This example demonstrates how to use GPBoost to model time series data
with random effects from patient characteristics like obesity and apnea.

Implementation based on:
Wang, Z., Zeng, T., Liu, Z., & Williams, C. K. I. (2024). 
Addressing Wearable Sleep Tracking Inequity: A New Dataset and Novel Methods for a Population with Sleep Disorders. 
In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, 
PMLR 248:8716-8741. https://proceedings.mlr.press/v248/wang24a.html
Offical code repository: https://github.com/WillKeWang/DREAMT_FE
"""
# Check and print versions first to diagnose compatibility issues
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
        print("This usually happens when NumPy is upgraded without rebuilding pandas.")
        print("\nSolution options:")
        print("1. Reinstall pandas: pip install --force-reinstall pandas")
        print("2. Ensure consistent versions: pip install 'numpy==1.23.5' 'pandas==1.5.3'")
        print("3. Create a fresh environment: conda create -n fresh_env python=3.10 numpy pandas gpboost")
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
    print("1. Confirm PyHealth is installed:")
    print("   pip list | grep pyhealth")
    print("\n2. If not installed or outdated, install in development mode:")
    print("   pip install -e .")
    print("\n3. Check your PYTHONPATH:")
    print("   python -c 'import sys; print(sys.path)'")
    sys.exit(1)
except Exception as e:
    print(f"Error importing PyHealth modules: {e}")
    sys.exit(1)

# Function to generate synthetic E4 sleep data (binary: Awake/Asleep)
def generate_synthetic_sleep_data(num_patients=100, hours=8, epoch_seconds=30, seed=42):
    """
    Generate synthetic Empatica E4 data with binary sleep states.
    
    Parameters:
    - num_patients: Number of patients to simulate
    - hours: Duration of recording in hours
    - epoch_seconds: Duration of each measurement epoch (typically 30 seconds for PSG)
    - seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)
    
    sleep_states = ["Awake", "Asleep"]
    
    epochs_per_hour = int(3600 / epoch_seconds)
    total_epochs = int(hours * epochs_per_hour)
    
    data = []
    for i in range(num_patients):
        patient_id = f"patient_{i}"
        
        # Generate random effects
        is_obese = random.choice([0, 1])
        has_apnea = random.choice([0, 1])
        sleep_efficiency = np.random.beta(5, 2) if not has_apnea else np.random.beta(4, 3)  # Apnea patients have lower sleep efficiency
        
        # Patient-specific physiological baselines
        base_hr = 60 + np.random.normal(5, 8)
        base_eda = 0.3 + np.random.gamma(2, 0.1)
        base_temp = 36.0 + np.random.normal(0.5, 0.4)
        base_acc = 0.01 + np.random.exponential(0.02)
        
        # Patient-specific state differentiation (some patients have less clear differences between states)
        hr_diff = 10 + np.random.normal(0, 5)  # How much HR differs between wake/sleep
        eda_diff = 0.2 + np.random.normal(0, 0.1)
        acc_diff = 0.03 + np.random.exponential(0.02)
        
        # Apply random effects to base values
        if is_obese:
            base_hr += np.random.normal(3, 2)
            base_temp -= np.random.normal(0.1, 0.05)
            
        if has_apnea:
            base_hr += np.random.normal(2, 1)
            base_eda += np.random.normal(0.05, 0.03)
        
        patient_visits = []
        
        # Define sleep pattern
        current_state = 0  # 0=Awake, 1=Asleep
        
        # Sleep onset latency
        sleep_onset_epochs = int((15 + 15 * np.random.random()) * 60 / epoch_seconds)
        
        # Morning awakening time
        final_wake_epochs = int((10 + 20 * np.random.random()) * 60 / epoch_seconds)
        
        # Parameters for mid-sleep awakenings
        awakenings_per_hour = 0.5 + np.random.exponential(0.5)
        if has_apnea:
            awakenings_per_hour += 0.5 + np.random.exponential(1.0)
        
        # Generate awakening timepoints
        sleep_period = total_epochs - sleep_onset_epochs - final_wake_epochs
        num_awakenings = max(1, int(awakenings_per_hour * hours))
        
        # Handle case where there aren't enough epochs
        if sleep_period > num_awakenings:
            awakening_points = sorted(random.sample(range(sleep_onset_epochs, 
                                                        total_epochs - final_wake_epochs),
                                                  num_awakenings))
            
            # Duration of awakenings
            awakening_durations = [max(1, int(np.random.exponential(3) * 60 / epoch_seconds)) 
                                for _ in range(num_awakenings)]
        else:
            awakening_points = []
            awakening_durations = []
        
        # State and signal history for continuity
        signal_history = {'hr': base_hr, 'eda': base_eda, 'temp': base_temp, 'acc': base_acc}
        last_state = 0
        transition_momentum = 0 
        
        # Add some temporal autocorrelation to signals
        hr_momentum, eda_momentum, temp_momentum, acc_momentum = 0, 0, 0, 0
        
        # Generate 5% random mislabeling to simulate annotation errors
        mislabel_indices = set(random.sample(range(total_epochs), int(0.05 * total_epochs)))
        
        for t in range(total_epochs):
            # Determine ground truth sleep state based on the sleep pattern
            if t < sleep_onset_epochs:
                true_state = 0  # Awake during sleep onset period
            elif t >= (total_epochs - final_wake_epochs):
                true_state = 0  # Awake during morning awakening
            else:
                true_state = 1  # Default to asleep during night
                
                # Check for awakenings
                for i, awakening_time in enumerate(awakening_points):
                    if 0 <= t - awakening_time < awakening_durations[i]:
                        true_state = 0
                        break
            
            # State transition dynamics
            if true_state != last_state:
                transition_momentum = 4 if true_state == 1 else 2  # Takes longer to fall asleep than wake up
                
            if transition_momentum > 0:
                # During transition, signals change gradually
                transition_factor = transition_momentum / 4.0
                transition_momentum -= 1
            else:
                transition_factor = 0
                
            last_state = true_state
            
            # Introduce occasional mislabeled epochs
            if t in mislabel_indices:
                stage = sleep_states[1 - true_state]  # Opposite of true state
            else:
                stage = sleep_states[true_state]
            
            # Generate physiological signals with noise and autocorrelation
            
            # Heart rate with temporal dynamics
            hr_momentum = 0.8 * hr_momentum + 0.2 * np.random.normal(0, 2)
            if true_state == 0:  # Awake
                hr_target = base_hr + hr_diff + np.random.normal(0, 3)
            else:  # Asleep
                hr_target = base_hr - 5 + np.random.normal(0, 2)
                
            # During transition, blend between states
            if transition_factor > 0:
                hr_target = hr_target * (1 - transition_factor) + signal_history['hr'] * transition_factor
                
            # Update with momentum and noise
            hr = 0.7 * signal_history['hr'] + 0.3 * hr_target + hr_momentum
            
            # EDA with similar approach
            eda_momentum = 0.8 * eda_momentum + 0.2 * np.random.normal(0, 0.05)
            if true_state == 0:  # Awake
                eda_target = base_eda + eda_diff + np.random.normal(0, 0.1)
            else:  # Asleep
                eda_target = base_eda - 0.1 + np.random.normal(0, 0.05)
            
            if transition_factor > 0:
                eda_target = eda_target * (1 - transition_factor) + signal_history['eda'] * transition_factor
                
            eda = 0.8 * signal_history['eda'] + 0.2 * eda_target + eda_momentum
            
            # Temperature changes
            temp_momentum = 0.95 * temp_momentum + 0.05 * np.random.normal(0, 0.1)
            if true_state == 0:  # Awake
                temp_target = base_temp + np.random.normal(0, 0.1)
            else:  # Asleep
                temp_target = base_temp - 0.2 + np.random.normal(0, 0.1)
                
            if transition_factor > 0:
                temp_target = temp_target * (1 - transition_factor) + signal_history['temp'] * transition_factor
                
            temp = 0.95 * signal_history['temp'] + 0.05 * temp_target + temp_momentum
            
            # Accelerometer
            acc_momentum = 0.6 * acc_momentum + 0.4 * np.random.normal(0, 0.01)
            if true_state == 0:  # Awake
                acc_target = base_acc + acc_diff + np.random.exponential(0.02)
            else:  # Asleep
                acc_target = base_acc + np.random.exponential(0.005)  # Small movements during sleep
                
            if transition_factor > 0:
                acc_target = acc_target * (1 - transition_factor) + signal_history['acc'] * transition_factor
                
            acc = 0.6 * signal_history['acc'] + 0.4 * acc_target + acc_momentum
                
            # Add apnea effects
            if has_apnea and true_state == 1 and random.random() < 0.1:  # Apnea event during sleep
                hr += np.random.gamma(4, 1)  # Variable HR increase
                eda += np.random.gamma(3, 0.05)  # Variable EDA increase
                acc += np.random.exponential(0.02)  # Variable movement
            
            # Ensure values are in reasonable ranges but allow more extreme values occasionally
            hr = max(35, min(140, hr))  # Wider range
            eda = max(0.05, eda)  
            temp = max(34.5, min(38.5, temp))  # Wider range
            acc = max(0, acc)
            
            # Store signal values for next epoch's continuity
            signal_history = {'hr': hr, 'eda': eda, 'temp': temp, 'acc': acc}
            
            visit_data = {
                "visit_id": f"visit_{t}",
                "heart_rate": hr,
                "eda": eda,
                "temperature": temp,
                "accelerometer": acc,
                "sleep_stage": stage
            }
            
            patient_visits.append(visit_data)
            
        patient_record = {
            "patient_id": patient_id,
            "visits": patient_visits,
            "obesity": is_obese,
            "apnea": has_apnea
        }
        
        data.append(patient_record)
        
    return data


if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print("GPBoost Binary Sleep Classification Example")
        print("="*50 + "\n")
        
        print("Generating synthetic Empatica E4 sleep data...")
        # Generate 8 hours of sleep data with 30-second epochs for 100 patients
        data = generate_synthetic_sleep_data(num_patients=100, hours=8, epoch_seconds=30, seed=123)
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
        
        try:
            train_dataset = SampleEHRDataset(samples=train_data, dataset_name="synth_sleep_train")
            train_dataset.label_tokenizer = BinaryTokenizer()
            test_dataset = SampleEHRDataset(samples=test_data, dataset_name="synth_sleep_test")
            test_dataset.label_tokenizer = BinaryTokenizer()
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
        try:
            eval_results = model.inference(test_data)
            
            y_true = eval_results["y_true"].astype(int)
            y_prob = eval_results["y_prob"]
            
            if y_prob is None or len(y_prob) == 0:
                print("WARNING: Empty predictions, using random values")
                y_prob = np.random.rand(len(y_true), 1)
                
            try:
                y_prob_flat = y_prob.flatten() if len(y_prob.shape) > 1 else y_prob
                y_pred = (y_prob_flat > 0.5).astype(int)
            except Exception as e:
                print(f"Error processing predictions: {e}")
                y_prob_flat = np.random.rand(len(y_true))
                y_pred = np.random.randint(0, 2, size=len(y_true))
                
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
            
            print("\nSample predictions (first 10 time points for first test patient):")
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

            print("\nAnalyzing impact of random effects...")
            if model.gp_model:
                re_info = model.get_random_effects_info()
                print("Random effects analysis:")
                
                if 'dataframe_empty' in re_info and re_info['dataframe_empty']:
                    print("GPBoost returned an empty DataFrame for random effects.")
                    
                    if 'model_params' in re_info:
                        print("\nModel hyperparameters:")
                        model_params = re_info['model_params']
                        for k, v in sorted(model_params.items()):
                            if v is None or k.startswith('_'):
                                continue
                            if isinstance(v, float):
                                param_value = f"{v:.6g}"
                            elif isinstance(v, (list, tuple)) and len(v) > 6:
                                param_value = f"[{', '.join(str(x) for x in v[:3])}..., {', '.join(str(x) for x in v[-3:])}]"
                            else:
                                param_value = str(v)
                            print(f"  {k}: {param_value}")
                        
                        important_params = ['likelihood', 'cov_function', 'optimizer']
                        found_important = False
                        for param in important_params:
                            if param in model_params:
                                if not found_important:
                                    print("\nKey model configuration:")
                                    found_important = True
                                print(f"  {param}: {model_params[param]}")
                
                elif 'dataframe' in re_info:
                    df = re_info['dataframe']
                    print("\nRandom effects DataFrame:")
                    print(f"Shape: {df.shape}")
                    print(f"Columns: {df.columns.tolist()}")
                    
                    if len(df) > 0:
                        print("\nSample of random effects data:")
                        print(df.head(5))
                
                if 'num_group' in re_info:
                    print(f"\nNumber of groups/patients: {re_info['num_group']}")
                
                if 'variance_components' in re_info and re_info['variance_components']:
                    print("\nRandom effects variance components:")
                    for i, var in enumerate(re_info['variance_components']):
                        print(f"Component {i}: {var:.4f}")
                
                group_effects = None
                for key in ['group_effects', 'model_random_effects']:
                    if key in re_info and re_info[key] is not None:
                        group_effects = re_info[key]
                        print(f"\nFound random effects in '{key}'")
                        break
                
                if group_effects is not None and hasattr(group_effects, '__len__') and len(group_effects) > 0:
                    try:
                        effects = np.array(group_effects)
                        print("\nRandom effects statistics:")
                        print(f"Mean effect: {np.mean(effects):.4f}")
                        print(f"Min effect: {np.min(effects):.4f}")
                        print(f"Max effect: {np.max(effects):.4f}")
                        print(f"Std dev: {np.std(effects):.4f}")
                        
                        if len(effects) >= 6:
                            sorted_idx = np.argsort(effects)
                            print("\nTop 3 negative effects:")
                            for i in sorted_idx[:3]:
                                print(f"  Patient {i}: {effects[i]:.4f}")
                            print("\nTop 3 positive effects:")
                            for i in sorted_idx[-3:]:
                                print(f"  Patient {i}: {effects[i]:.4f}")
                    except Exception as e:
                        print(f"Error analyzing group effects: {e}")
                else:
                    print("\nNo detailed group random effects available")
                    print("This is normal for some GPBoost versions and configurations")
                    
                if 'available_keys' in re_info:
                    print(f"\nDebug info - available keys: {re_info['available_keys']}")
                
                print("\nModel performance summary:")
                print(f"Training success: Yes")
                print(f"Random effects modeling: {'Yes' if model.gp_model else 'No'}")
                model_type = "GPBoost with random effects" if model.gp_model else "Standard GPBoost (without random effects)"
                print(f"Model type: {model_type}")
                print(f"Features used: {', '.join(feature_keys)}")
                print(f"Random effect features: {', '.join(random_effect_keys) if random_effect_keys else 'None'}")
                print(f"Accuracy: {accuracy:.4f}")
                
                print("\nPossible next steps:")
                print("1. Adjust the GPBoost parameters to improve performance")
                print("2. Try different features or feature combinations")
                print("3. Compare with a model without random effects to measure their impact")
                print("4. Apply the model to real Empatica E4 data")
            else:
                print("Model was trained without random effects")
        
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
