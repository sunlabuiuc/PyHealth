import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Any
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


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
        feature_keys: List of feature keys to use
        label_key: Key for the target variable
        group_key: Key identifying the grouping variable (e.g., 'patient_id', 'subject_id')
        random_effect_features: Features to use for random effects modeling
        **kwargs: Additional arguments passed to gpboost.train()
    """

    def __init__(
        self,
        feature_keys: List[str],
        label_key: str,
        group_key: str,
        random_effect_features: Optional[List[str]] = None,
        label_tokenizer: Optional[Any] = None,
        **kwargs,
    ):

        self.feature_keys = feature_keys
        self.label_key = label_key
        self.group_key = group_key
        self.random_effect_features = random_effect_features
        self.label_tokenizer = label_tokenizer
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
            for i, visit in enumerate(patient_data["visits"]):
                record = {"group": group_id, "time": i}

                for key in self.feature_keys:
                    record[key] = visit.get(key, np.nan)

                label = visit.get(self.label_key)
                if label is not None:
                    if self.label_tokenizer is not None:
                        record["label"] = self.label_tokenizer.encode(label)[0]
                    else:
                        record["label"] = label
                else:
                    record["label"] = np.nan

                if self.random_effect_features:
                    for re_key in self.random_effect_features:
                        record[re_key] = patient_data.get(re_key, np.nan)

                records.append(record)

        df = pd.DataFrame(records)
        df["group"] = pd.factorize(df["group"])[0]
        return df

    def optimize_hyperparameters(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        param_space: Optional[Dict] = None,
        n_iter: int = 20,
        verbose: int = 0,
    ) -> Dict:
        """
        Optimize hyperparameters using hyperopt.

        Args:
            train_data: Training data
            val_data: Validation data
            param_space: Dictionary with hyperopt-style parameter spaces
            n_iter: Number of parameter settings sampled
            verbose: Verbosity level

        Returns:
            Dictionary with best hyperparameters
        """
        if param_space is None:
            raise ValueError("missing required 'param_space' parameter.")

        df_train = self._data_to_pandas(train_data)
        X_train = df_train[self.feature_keys].values
        y_train = df_train["label"].values
        group_train = df_train["group"].values

        df_val = self._data_to_pandas(val_data)
        X_val = df_val[self.feature_keys].values
        y_val = df_val["label"].values
        group_val = df_val["group"].values

        def objective(params):
            # Create a copy of the parameters for this trial
            trial_params = params.copy()
            num_boost_round = int(trial_params.pop("num_boost_round", 100))

            # Convert int parameters from float
            for param in ["max_depth", "num_leaves"]:
                if param in trial_params:
                    trial_params[param] = int(trial_params[param])

            trial_params["objective"] = self.objective
            trial_params["metric"] = "auc"
            trial_params["verbose"] = -1

            gp_model = self.gpb.GPModel(
                group_data=group_train, likelihood="bernoulli_probit"
            )

            train_set = self.gpb.Dataset(X_train, y_train)
            val_set = self.gpb.Dataset(X_val, y_val)

            booster = self.gpb.train(
                params=trial_params,
                train_set=train_set,
                valid_sets=[val_set],
                gp_model=gp_model,
                num_boost_round=num_boost_round,
                early_stopping_rounds=20,
                verbose_eval=False,
                use_gp_model_for_validation=False,
            )

            score = booster.best_score["valid_0"]["auc"]

            # Return negative score since hyperopt minimizes
            return {"loss": -score, "status": STATUS_OK, "params": params}

        trials = Trials()
        best = fmin(
            fn=objective,
            space=param_space,
            algo=tpe.suggest,
            max_evals=n_iter,
            trials=trials,
            verbose=verbose,
        )

        if verbose > 0:
            print(f"Best parameters: {best}")
            best_trial = sorted(trials.trials, key=lambda x: x["result"]["loss"])[0]
            if "loss" in best_trial["result"]:
                print(f"Best score (AUC): {-best_trial['result']['loss']:.4f}")

        return best

    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None):
        """
        Train the GPBoost model

        Args:
            train_data: Training data
            val_data: Optional validation data
        """
        df_train = self._data_to_pandas(train_data)
        y_train = df_train["label"].values
        X_train = df_train[self.feature_keys].values
        group_train = df_train["group"].values

        eval_set = None
        eval_group = None

        if val_data is not None:
            df_val = self._data_to_pandas(val_data)
            y_val = df_val["label"].values
            X_val = df_val[self.feature_keys].values
            group_val = df_val["group"].values
            eval_set = [(X_val, y_val)]
            eval_group = [group_val]

        self.gp_model = self.gpb.GPModel(
            group_data=group_train, likelihood="bernoulli_probit"
        )
        print("Using random effects model with bernoulli_probit likelihood")

        data_train_gpb = self.gpb.Dataset(X_train, y_train)

        if eval_set is not None:
            data_val_gpb = self.gpb.Dataset(X_val, y_val)
            valid_sets = [data_val_gpb]
        else:
            valid_sets = None

        self.kwargs["metric"] = "auc"
        # Convert int parameters from float
        for param in ["max_depth", "num_leaves"]:
            if param in self.kwargs:
                self.kwargs[param] = int(self.kwargs[param])

        self.model = self.gpb.train(
            params=self.kwargs,
            train_set=data_train_gpb,
            valid_sets=valid_sets,
            gp_model=self.gp_model,
            num_boost_round=int(self.kwargs.pop("num_boost_round", 100)),
            use_gp_model_for_validation=False,
            verbose_eval=50,
        )
        print("Successfully trained GPBoost model with random effects")

    def inference(self, test_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Make predictions on test data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        df_test = self._data_to_pandas(test_data)
        y_true = df_test["label"].values
        X_test = df_test[self.feature_keys].values
        group_test = df_test["group"].values

        print("Making predictions with random effects model")
        raw_pred = self.model.predict(
            data=X_test, group_data_pred=group_test, predict_var=False
        )

        y_prob = np.full((len(y_true), 1), 0.5)

        pred_key = None
        for key in ["response_mean", "fixed_effect", "response"]:
            if key in raw_pred and raw_pred[key] is not None:
                pred_key = key
                break

        if pred_key:
            raw_values = raw_pred[pred_key]

            if isinstance(raw_values, (np.ndarray, list)) and len(raw_values) == len(
                y_true
            ):
                y_prob = np.array(raw_values).reshape(-1, 1)
            else:
                print(f"Warning: Unexpected prediction format")
        else:
            print("No usable prediction key found in dict")

        y_prob = np.nan_to_num(y_prob, nan=0.5)

        return {"y_prob": y_prob, "y_true": y_true}

    def get_random_effects_info(self) -> Dict[str, Any]:
        """
        Get basic information about the random effects component of the model.

        Note: Due to limitations in GPBoost with bernoulli_probit likelihood,
        detailed random effect coefficients are not directly accessible.
        """
        if not self.gp_model:
            return {"has_random_effects": False}

        result = {"has_random_effects": True}

        if hasattr(self.gp_model, "params"):
            result["model_params"] = self.gp_model.params

        return result
