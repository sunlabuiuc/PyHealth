import sys
import numpy as np
import pandas as pd
from pyhealth.models.base_model import BaseModel
from typing import List, Dict, Optional, Union, Any

class GPBoostTimeSeriesModel(BaseModel):
    """
    GPBoost Time Series Model for PyHealth.
    
    This model uses the gpboost library to fit a gradient boosting model
    with Gaussian Process random effects for longitudinal/time series data.
    
    Implementation based on:
    Wang, Z., Zeng, T., Liu, Z., & Williams, C. K. I. (2024). 
    Addressing Wearable Sleep Tracking Inequity: A New Dataset and Novel Methods for a Population with Sleep Disorders. 
    In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, 
    PMLR 248:8716-8741. https://proceedings.mlr.press/v248/wang24a.html
    Offical code repository: https://github.com/WillKeWang/DREAMT_FE
    
    Args:
        dataset: PyHealth dataset object
        feature_keys: List of feature keys to use
        label_key: Key for the target variable (binary: 0/1, asleep/awake)
        group_key: Key identifying the grouping variable (e.g., 'patient_id')
        random_effect_features: Features to use for random effects modeling
        **kwargs: Additional arguments passed to gpboost.train()
    """
    def __init__(
        self,
        dataset,
        feature_keys: List[str],
        label_key: str,
        group_key: str,
        random_effect_features: Optional[List[str]] = None,
        **kwargs,
    ):
        # Check if dataset has input_schema attribute, add it if missing
        if not hasattr(dataset, 'input_schema'):
            input_schema = {key: "float" for key in feature_keys}
            dataset.input_schema = input_schema
            print("Added missing input_schema to dataset")
            
        # Check if dataset has output_schema attribute, add it if missing
        if not hasattr(dataset, 'output_schema'):
            # Binary classification - specifically for sleep detection
            output_schema = {label_key: ["Awake", "Asleep"]}
            dataset.output_schema = output_schema
            print("Added missing output_schema to dataset")
        
        # Initialize BaseModel with minimal parameters
        super(GPBoostTimeSeriesModel, self).__init__(dataset=dataset)
        
        # Store the parameters specific to our model
        self.feature_keys = feature_keys
        self.label_key = label_key
        self.group_key = group_key
        self.mode = "binary"  # Fixed to binary classification
        self.random_effect_features = random_effect_features
        self.kwargs = kwargs
        self.model = None
        self.gp_model = None
        
        # Check GPBoost availability with detailed error messages
        try:
            import gpboost as gpb
            self.gpb = gpb
        except ImportError:
            print("GPBoost not installed. Install with: pip install gpboost")
            sys.exit(1)
        except Exception as e:
            print(f"Error importing GPBoost: {e}")
            
            # Handle the common macOS OpenMP dependency issue
            error_str = str(e)
            if "libomp.dylib" in error_str and "/opt/homebrew" in error_str:
                print("\n===============================================")
                print("OpenMP dependency missing for GPBoost on macOS")
                print("===============================================")
                print("This error occurs because GPBoost requires OpenMP, which isn't installed on your system.")
                print("\nTo fix this issue, install libomp via Homebrew:")
                print("  1. Install Homebrew if you don't have it: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
                print("  2. Install OpenMP: brew install libomp")
                print("  3. Try running this script again")
            else:
                print("\nUnknown error when importing GPBoost. Try reinstalling:")
                print("  pip uninstall -y gpboost")
                print("  pip install gpboost --no-cache-dir")
            
            sys.exit(1)
        
        # Set GPBoost objective to binary
        self.objective = "binary"
        self.num_classes = 1
        self.kwargs.setdefault("objective", self.objective)
        
    def _data_to_pandas(self, data: List[Dict]) -> pd.DataFrame:
        """Convert PyHealth data format to pandas DataFrame for GPBoost"""
        records = []
        for patient_data in data:
            group_id = patient_data[self.group_key]
            for i, visit in enumerate(patient_data['visits']):
                record = {'group': group_id, 'time': i}
                
                # Extract features
                for key in self.feature_keys:
                    record[key] = visit.get(key, np.nan)
                
                # Extract label - binary (0=Awake, 1=Asleep)
                label = visit.get(self.label_key)
                if label is not None:
                    if hasattr(self.dataset, "label_tokenizer"):
                        record['label'] = self.dataset.label_tokenizer.encode(label)[0]
                    else:
                        # If label is already binary (0/1)
                        record['label'] = 1 if label in ["Asleep", "1", 1, True] else 0
                else:
                    record['label'] = np.nan
                
                # Extract random effect features if specified
                if self.random_effect_features:
                    for re_key in self.random_effect_features:
                        record[re_key] = patient_data.get(re_key, np.nan)
                        
                records.append(record)
                
        df = pd.DataFrame(records)
        # Ensure group column is properly encoded for gpboost
        df['group'] = pd.factorize(df['group'])[0]
        return df
        
    def train(self, train_data: List[Dict]):
        """Train the GPBoost model"""
        df_train = self._data_to_pandas(train_data)
        y_train = df_train['label'].values
        X_train = df_train[self.feature_keys].values
        group_train = df_train['group'].values
        
        # Try to train with random effects for binary classification
        try:
            # Define GP model for random effects
            self.gp_model = self.gpb.GPModel(group_data=group_train, likelihood="binary")
            print("Using random effects model with binary likelihood")
            
            # Create dataset for training
            data_train_gpb = self.gpb.Dataset(X_train, y_train)
            
            # Train model with random effects
            self.model = self.gpb.train(
                params=self.kwargs,
                train_set=data_train_gpb,
                gp_model=self.gp_model,
                num_boost_round=self.kwargs.pop('num_boost_round') or 100,
            )
            print("Successfully trained GPBoost model with random effects")
            
        except Exception as e:
            print(f"Error training random effects model: {e}")
            print("Falling back to standard GPBoost model without random effects")
            
            # Remove potentially problematic parameters
            train_params = self.kwargs.copy()
            if 'group' in train_params:
                del train_params['group']
                
            # Create dataset without group
            data_train_gpb = self.gpb.Dataset(X_train, y_train)
            
            # Train model without random effects
            self.model = self.gpb.train(
                params=train_params,
                train_set=data_train_gpb,
                num_boost_round=self.kwargs.get('num_boost_round', 100)
            )
            self.gp_model = None
            print("Successfully trained standard binary GPBoost model")

    def inference(self, test_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Make predictions on test data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        df_test = self._data_to_pandas(test_data)
        y_true = df_test['label'].values
        X_test = df_test[self.feature_keys].values
        group_test = df_test['group'].values
        
        # Get raw predictions from model
        try:
            if self.gp_model is not None:
                print("Making predictions with random effects model")
                raw_pred = self.model.predict(
                    data=X_test,
                    group_data_pred=group_test,
                    predict_var=False
                )
            else:
                print("Making predictions with standard model")
                raw_pred = self.model.predict(data=X_test)
                
            print(f"Raw prediction type: {type(raw_pred)}")
            if isinstance(raw_pred, dict):
                print(f"Available keys: {list(raw_pred.keys())}")
        except Exception as e:
            print(f"Prediction error: {e}")
            print("Attempting basic prediction")
            try:
                raw_pred = self.model.predict(data=X_test)
            except Exception as e2:
                print(f"Basic prediction also failed: {e2}")
                # Generate fixed probabilities
                raw_pred = np.full(len(y_true), 0.5)
        
        # Initialize with default probabilities
        y_prob = np.full((len(y_true), 1), 0.5)
        
        # Process GPBoost output format
        try:
            if isinstance(raw_pred, dict):
                # Try to extract proper predictions
                pred_key = None
                # Priority order for prediction keys
                for key in ['response_mean', 'fixed_effect', 'response']:
                    if key in raw_pred and raw_pred[key] is not None:
                        pred_key = key
                        break
                
                if pred_key:
                    print(f"Using key: {pred_key}")
                    raw_values = raw_pred[pred_key]
                    
                    # Check if we got a reasonable array
                    if isinstance(raw_values, (np.ndarray, list)) and len(raw_values) == len(y_true):
                        # Good case - we got an array of correct length
                        y_prob = np.array(raw_values).reshape(-1, 1)
                        print(f"Extracted predictions with shape {y_prob.shape}")
                    else:
                        print(f"Warning: Unexpected prediction format for {pred_key}, using default values")
                else:
                    print("No usable prediction key found in dict, using default values")
                    
            elif raw_pred is not None:
                # Try to convert raw_pred to a numpy array
                try:
                    y_prob = np.array(raw_pred)
                    if len(y_prob.shape) == 1:
                        y_prob = y_prob.reshape(-1, 1)
                    print(f"Using direct prediction with shape {y_prob.shape}")
                except Exception as e:
                    print(f"Error converting prediction to array: {e}")
            
            # Validate predictions
            if y_prob.shape[0] != len(y_true):
                print(f"Warning: Prediction length mismatch: {y_prob.shape[0]} vs {len(y_true)}")
                y_prob = np.full((len(y_true), 1), 0.5)
                
            # Replace any None or NaN values
            y_prob = np.nan_to_num(y_prob, nan=0.5)
            
        except Exception as e:
            print(f"Error processing predictions: {e}")
            y_prob = np.full((len(y_true), 1), 0.5)
        
        return {
            "y_prob": y_prob,
            "y_true": y_true
        }
        
    def get_random_effects_info(self) -> Dict[str, Any]:
        """Extract and return random effects information in a consistent format"""
        if not self.gp_model:
            return {"has_random_effects": False}
        
        try:
            raw_coef = self.gp_model.get_coef()
            print(f"Random effects info: {type(raw_coef)}")
            
            result = {"has_random_effects": True}
            
            if isinstance(raw_coef, pd.DataFrame):
                if raw_coef.empty:
                    print("Warning: GPBoost returned an empty DataFrame")
                    result["dataframe_empty"] = True
                    
                    try:
                        if hasattr(self.gp_model, 'params'):
                            result['model_params'] = self.gp_model.params
                        
                        if hasattr(self.gp_model, 'num_groups'):
                            result['num_group'] = self.gp_model.num_groups
                    except Exception as e:
                        print(f"Error getting model parameters: {e}")
                else:
                    print(f"DataFrame columns: {raw_coef.columns.tolist()}")
                    result["dataframe"] = raw_coef
                    
                    if 'groups' in raw_coef.columns:
                        result['num_group'] = raw_coef['groups'].nunique()
                    else:
                        result['num_group'] = len(raw_coef)
                    
                    for col in raw_coef.columns:
                        if 'variance' in col.lower() or 'effect' in col.lower():
                            result[f'variance_{col}'] = raw_coef[col].values
            
            return result
                
        except Exception as e:
            print(f"Error getting random effects info: {e}")
            return {"has_random_effects": True, "error": str(e)}
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model:
            self.model.save_model(path)
        else:
            raise ValueError("Model has not been trained yet")
            
    def load_model(self, path: str):
        """Load a trained model"""
        self.model = self.gpb.Booster(model_file=path)
