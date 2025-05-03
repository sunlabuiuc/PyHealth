import random
import numpy as np


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
