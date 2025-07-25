"""
Author: Muni Bondu
Description: This task predicts hospital length of stay (LOS) using synthetic admissions data.
This implements a regression task using admission and discharge dates.

Paper (if applicable): N/A """

import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

from pyhealth.datasets import BaseDataset
from pyhealth.tasks import BaseTask

class LOSDataset(BaseDataset):
    """Length of Stay Dataset compatible with PyHealth BaseDataset.

    Args:
        root (str): Root directory containing data files.
        dev (bool): If True, load a small sample for development.

    Attributes:
        patients (List[Dict]): List of patient data dictionaries.
    """

    def __init__(self, root: str = ".", dev: bool = False):
        tables = ["admissions"]  # Must match YAML table key
        config_path = os.path.join(root, "lengthofstay.yaml")
        
        # parent class
        super().__init__(root=root, tables=tables, config_path=config_path)
        self.dev = dev
        
        # Load and process data
        self.patients = self.load_patient_data()

    def load_patient_data(self) -> List[Dict]:
        """Load CSV and convert to internal patient data format."""
        try:
            # Get the file path from the config
            table_config = self.config.tables["admissions"]
            csv_filename = table_config.file_path
            csv_path = os.path.join(self.root, csv_filename)
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
                
            print(f"Loading data from: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            if self.dev:
                df = df.sample(n=min(5, len(df)), random_state=42)
                print(f"Development mode: Using {len(df)} samples")

            patients = []
            for idx, row in df.iterrows():
                try:
                    # Parse dates
                    admission_date = pd.to_datetime(row['admission_date']).date()
                    discharge_date = pd.to_datetime(row['discharge_date']).date()
                    
                    # Calculate length of stay
                    los = (discharge_date - admission_date).days
                    if los < 0:
                        print(f"Warning: Negative LOS for patient {row['patient_id']}, setting to 0")
                        los = 0
                    # Create patient record
                    patient_data = {
                        'patient_id': str(row['patient_id']),
                        'admission_date': admission_date,
                        'discharge_date': discharge_date,
                        'length_of_stay': los,
                        # Add features for prediction
                        'admission_day_of_year': admission_date.timetuple().tm_yday,
                        'admission_month': admission_date.month,
                        'admission_weekday': admission_date.weekday(),
                    }
                    patients.append(patient_data)
                except (ValueError, KeyError) as e:
                    print(f"Error processing row {idx} for patient {row.get('patient_id', 'unknown')}: {e}")
                    continue
            print(f"Successfully loaded {len(patients)} patient records")
            return patients
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    def get_patient_data(self) -> List[Dict]:
        """Return patient data for compatibility with PyHealth."""
        return self.patients


class LOSTask(BaseTask):
    """Length of Stay Prediction Task compatible with PyHealth BaseTask."""

    def __init__(self, dataset: LOSDataset):
        super().__init__()
        self.task_name = "length_of_stay"
        self.dataset = dataset

    def __call__(self, patient_id: str) -> Dict[str, Any]:
        """Process a single patient and return task-specific data.
        
        This method is required by PyHealth's BaseTask.
        Args:
            patient_id: The patient ID to process
            
        Returns:
            Dictionary containing patient features and target
        """
        # Find the patient in the dataset
        patient_data = None
        for p in self.dataset.patients:
            if p['patient_id'] == patient_id:
                patient_data = p
                break
        
        if patient_data is None:
            raise ValueError(f"Patient {patient_id} not found in dataset")
        
        # Return the processed data for this patient
        return {
            "patient_id": patient_data["patient_id"],
            "features": {
                "admission_day_of_year": patient_data["admission_day_of_year"],
                "admission_month": patient_data["admission_month"], 
                "admission_weekday": patient_data["admission_weekday"],
            },
            "target": patient_data["length_of_stay"],
            "metadata": {
                "admission_date": patient_data["admission_date"],
                "discharge_date": patient_data["discharge_date"],
            }
        }

    def get_targets(self) -> List[int]:
        """Get the length of stay targets for the dataset."""
        return [p['length_of_stay'] for p in self.dataset.patients]

    def get_features(self) -> List[Dict[str, Any]]:
        """Extract features to be used for prediction."""
        features = []
        for p in self.dataset.patients:
            features.append({
                "patient_id": p["patient_id"],
                "admission_day_of_year": p["admission_day_of_year"],
                "admission_month": p["admission_month"],
                "admission_weekday": p["admission_weekday"],
            })
        return features

    def get_patient_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Split patients into train/validation/test sets."""
        import random
        random.seed(42)
        
        patients = list(range(len(self.dataset.patients)))
        random.shuffle(patients)
        
        n_train = int(len(patients) * train_ratio)
        n_val = int(len(patients) * val_ratio)
        
        train_idx = patients[:n_train]
        val_idx = patients[n_train:n_train + n_val]
        test_idx = patients[n_train + n_val:]
        
        return train_idx, val_idx, test_idx


def create_sample_data():
    """Create a sample CSV file for testing."""
    csv_content = """patient_id,admission_date,discharge_date
1,2023-01-01,2023-01-05
2,2023-02-10,2023-02-15
3,2023-03-20,2023-03-22
4,2023-04-01,2023-04-10
5,2023-05-05,2023-05-07
6,2023-06-15,2023-06-18
7,2023-07-20,2023-07-25
8,2023-08-10,2023-08-12
9,2023-09-05,2023-09-08
10,2023-10-12,2023-10-20
11,2023-11-01,2023-11-03
12,2023-12-15,2023-12-18
"""
    filename = "fake_los_data.csv"
    with open(filename, "w") as f:
        f.write(csv_content)
    print(f"Created sample data file: {filename}")


def run_simple_prediction():
    """Run a simple prediction using scikit-learn."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        print("\n" + "="*50)
        print("RUNNING SIMPLE PREDICTION MODEL")
        print("="*50)
        
        # Load dataset
        dataset = LOSDataset(dev=False)
        task = LOSTask(dataset)
        
        # Get features and targets
        features = task.get_features()
        targets = task.get_targets()
        
        if not features or not targets:
            print("No data available for prediction")
            return
        
        # Convert to arrays
        X = np.array([[f['admission_day_of_year'], f['admission_month'], f['admission_weekday']] 
                     for f in features])
        y = np.array(targets)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Actual vs Predicted (first 5 test samples):")
        for i in range(min(5, len(y_test))):
            print(f"  Actual: {y_test[i]}, Predicted: {y_pred[i]:.1f}")
            
    except ImportError:
        print("Scikit-learn not available. Skipping prediction demo.")
    except Exception as e:
        print(f"Error in prediction: {e}")
def test_task_call():
    """Test the __call__ method of the task."""
    print("\n" + "="*50)
    print("TESTING TASK __call__ METHOD")
    print("="*50)
    try:
        dataset = LOSDataset(dev=True)
        task = LOSTask(dataset)
        
        if dataset.patients:
            # Test with the first patient
            first_patient_id = dataset.patients[0]['patient_id']
            result = task(first_patient_id)
            
            print(f"Patient ID: {result['patient_id']}")
            print(f"Features: {result['features']}")
            print(f"Target (LOS): {result['target']} days")
            print(f"Metadata: {result['metadata']}")
        else:
            print("No patients available for testing")      
    except Exception as e:
        print(f"Error testing task call: {e}")
def main():
    print("PyHealth Length of Stay Prediction Task")
    print("="*40)
    
    # Create sample data if it doesn't exist
    if not os.path.exists("fake_los_data.csv"):
        print("Creating sample data...")
        create_sample_data()
    
    # Check if YAML config exists
    if not os.path.exists("lengthofstay.yaml"):
        print("Error: lengthofstay.yaml not found!")
        print("Please create the YAML configuration file first.")
        print("You can use the provided lengthofstay.yaml configuration.")
        return

    try:
        print("Initializing dataset...")
        dataset = LOSDataset(dev=True)

        if not dataset.patients:
            print("No patient data loaded. Please check your data file.")
            return

        print("Creating task...")
        task = LOSTask(dataset)

        print("Getting targets...")
        targets = task.get_targets()
        print(f"Length of Stay targets: {targets}")

        print("Getting features...")
        features = task.get_features()
        print("Features:")
        for feat in features[:3]:  # Show first 3
            print(f"  {feat}")
        if len(features) > 3:
            print(f"  ... and {len(features) - 3} more")

        print(f"\nDataset Summary:")
        print(f"{'='*30}")
        print(f"Number of patients: {len(dataset.patients)}")
        if targets:
            print(f"Average LOS: {sum(targets) / len(targets):.1f} days")
            print(f"Min LOS: {min(targets)} days")
            print(f"Max LOS: {max(targets)} days")
        
        # Test the __call__ method
        test_task_call()
        
        # Run simple prediction if possible
        run_simple_prediction()
        
        print(f"\n{'='*50}")
        print("SUCCESS! Dataset and task created successfully.")
        print("You can now use this dataset with PyHealth models.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure yaml file exists and has the correct structure")
        print("2. Ensure csv file exists and is readable")
        print("3. Check that PyHealth is properly installed")
        print("4. Verify your Python environment has required dependencies")


if __name__ == "__main__":
    main()