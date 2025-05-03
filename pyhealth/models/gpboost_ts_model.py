import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Any

class GPBoostTimeSeriesModel:
    """
    GPBoost Time Series Model for PyHealth.
    
    !!! IMPORTANT NOTE !!!
    This model is NOT a PyTorch model and does NOT inherit from BaseModel.
    It will NOT work with PyHealth's standard PyTorch-based training workflows.
    
    Implementation based on:
    Sigrist, F. (2022). 
    GPBoost: Unifying Boosting and Mixed Effects Models. 
    Journal of Machine Learning Research, 23(120): 1-17.
    https://www.jmlr.org/papers/v23/20-322.html
    Official code repository: https://github.com/fabsig/GPBoost
    
    Application in time series data analysis:
    Wang, Z., Zeng, T., Liu, Z., & Williams, C. K. I. (2024). 
    Addressing Wearable Sleep Tracking Inequity: A New Dataset and Novel Methods for a Population with Sleep Disorders. 
    In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, 
    PMLR 248:8716-8741. https://proceedings.mlr.press/v248/wang24a.html
    Offical code repository: https://github.com/WillKeWang/DREAMT_FE
    
    Args:
        dataset: PyHealth dataset object with input_schema and output_schema defined
        feature_keys: List of feature keys to use
        label_key: Key for the target variable
        group_key: Key identifying the grouping variable (e.g., 'patient_id', 'subject_id')
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
        if not hasattr(dataset, 'input_schema'):
            raise ValueError("Dataset missing required 'input_schema' attribute.")
            
        if not hasattr(dataset, 'output_schema'):
            raise ValueError("Dataset missing required 'output_schema' attribute.")
        
        self.dataset = dataset
        self.feature_keys = feature_keys
        self.label_key = label_key
        self.group_key = group_key
        self.random_effect_features = random_effect_features
        self.kwargs = kwargs
        self.model = None
        self.gp_model = None
        
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
                print("\nOpenMP dependency missing for GPBoost on macOS.")
                print("Install libomp via Homebrew: brew install libomp")
            
            sys.exit(1)
        
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
                
                for key in self.feature_keys:
                    record[key] = visit.get(key, np.nan)
                
                label = visit.get(self.label_key)
                if label is not None:
                    if hasattr(self.dataset, "label_tokenizer"):
                        record['label'] = self.dataset.label_tokenizer.encode(label)[0]
                    else:
                        record['label'] = label
                else:
                    record['label'] = np.nan
                
                if self.random_effect_features:
                    for re_key in self.random_effect_features:
                        record[re_key] = patient_data.get(re_key, np.nan)
                        
                records.append(record)
                
        df = pd.DataFrame(records)
        df['group'] = pd.factorize(df['group'])[0]
        return df
        
    def train(self, train_data: List[Dict]):
        """Train the GPBoost model"""
        df_train = self._data_to_pandas(train_data)
        y_train = df_train['label'].values
        X_train = df_train[self.feature_keys].values
        group_train = df_train['group'].values
        
        self.gp_model = self.gpb.GPModel(group_data=group_train, likelihood="bernoulli_probit")
        print("Using random effects model with bernoulli_probit likelihood")
        
        data_train_gpb = self.gpb.Dataset(X_train, y_train)
        
        self.model = self.gpb.train(
            params=self.kwargs,
            train_set=data_train_gpb,
            gp_model=self.gp_model,
            num_boost_round=self.kwargs.pop('num_boost_round', 100),
        )
        print("Successfully trained GPBoost model with random effects")

    def inference(self, test_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Make predictions on test data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        df_test = self._data_to_pandas(test_data)
        y_true = df_test['label'].values
        X_test = df_test[self.feature_keys].values
        group_test = df_test['group'].values
        
        print("Making predictions with random effects model")
        raw_pred = self.model.predict(
            data=X_test,
            group_data_pred=group_test,
            predict_var=False
        )
        
        y_prob = np.full((len(y_true), 1), 0.5)
        
        pred_key = None
        for key in ['response_mean', 'fixed_effect', 'response']:
            if key in raw_pred and raw_pred[key] is not None:
                pred_key = key
                break
        
        if pred_key:
            raw_values = raw_pred[pred_key]
            
            if isinstance(raw_values, (np.ndarray, list)) and len(raw_values) == len(y_true):
                y_prob = np.array(raw_values).reshape(-1, 1)
            else:
                print(f"Warning: Unexpected prediction format")
        else:
            print("No usable prediction key found in dict")
                
            
        y_prob = np.nan_to_num(y_prob, nan=0.5)
        
        return {
            "y_prob": y_prob,
            "y_true": y_true
        }
        
    def get_random_effects_info(self) -> Dict[str, Any]:
        """
        Get basic information about the random effects component of the model.
        
        Note: Due to limitations in GPBoost with bernoulli_probit likelihood,
        detailed random effect coefficients are not directly accessible.
        """
        if not self.gp_model:
            return {"has_random_effects": False}
        
        result = {"has_random_effects": True}
        
        if hasattr(self.gp_model, 'params'):
            result['model_params'] = self.gp_model.params
            
        return result
