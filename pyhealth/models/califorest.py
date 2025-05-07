"""
CaliForest: Calibrated Random Forest for Healthcare Applications

A novel implementation of calibrated random forests that balances high discrimination with accurate calibration for healthcare predictions using out-of-bag samples (no explicit calibration set needed).

Installation:
  pip install -r requirements.txt

Usage:
  python califorest_pr.py --preprocess
  python califorest_pr.py --dataset <dataset> --random_seed <seed> --n_estimators <n> --depth <d>

  Example:
    python califorest_pr.py --preprocess
    python califorest_pr.py --dataset mimic3_mort_hosp --random_seed 0 --n_estimators 300 --depth 10
    python califorest_pr.py --dataset mimic3_mort_icu --random_seed 0 --n_estimators 300 --depth 10
    python califorest_pr.py --dataset mimic3_los_3 --random_seed 0 --n_estimators 300 --depth 10
    python califorest_pr.py --dataset mimic3_los_7 --random_seed 0 --n_estimators 300 --depth 10

Data Setup:
  Place `data/all_hourly_data.h5` (with tables "patients" and "vitals_labs") in the `data/` directory.
  Run `python califorest_pr.py --preprocess` to generate `data/mimic_X.npy`, `data/mimic_Y.npy`, and `data/mimic_subjects.pkl`.

Model Variants:
  CF-Iso, CF-Logit, RC-Iso, RC-Logit, RF-NoCal

Evaluation Metrics:
  ROC AUC, Hosmer-Lemeshow test, Spiegelhalter's Z-test, Scaled Brier score, Reliability metrics

"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression as Iso
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pandas import DataFrame, IndexSlice
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_hastie_10_2, load_breast_cancer
from sklearn.metrics import roc_auc_score
import time
from scipy.stats import chi2, norm
import sklearn.metrics as skm



GAP_TIME = 6  # In hours
WINDOW_SIZE = 24  # In hours
ID_COLS = ["subject_id", "hadm_id", "icustay_id"]
TRAIN_FRAC = 0.7
TEST_FRAC = 0.3

def read_data(dataset, random_seed):
    """
    Load and preprocess the specified dataset for training and testing.

    Args:
        dataset (str): Name of the dataset to load. Options include:
            - "hastie": Hastie-10-2 synthetic dataset.
            - "breast_cancer": Breast cancer dataset.
            - "mimic3_mort_hosp": MIMIC-III dataset for hospital mortality.
            - "mimic3_mort_icu": MIMIC-III dataset for ICU mortality.
            - "mimic3_los_3": MIMIC-III dataset for length of stay (3-day threshold).
            - "mimic3_los_7": MIMIC-III dataset for length of stay (7-day threshold).
        random_seed (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) containing the training and testing data.

    Source:
        https://github.com/yubin-park/califorest/blob/master/analysis/run_chil_exp.py
    """
    X_train, X_test, y_train, y_test = None, None, None, None

    if dataset == "hastie":
        np.random.seed(random_seed)
        poly = PolynomialFeatures()
        X, y = make_hastie_10_2(n_samples=10000)
        X = poly.fit_transform(X)
        y[y < 0] = 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    elif dataset == "breast_cancer":
        np.random.seed(random_seed)
        poly = PolynomialFeatures()
        X, y = load_breast_cancer(return_X_y=True)
        X = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    elif dataset == "mimic3_mort_hosp":
        X_train, X_test, y_train, y_test = extract(random_seed, "mort_hosp")

    elif dataset == "mimic3_mort_icu":
        X_train, X_test, y_train, y_test = extract(random_seed, "mort_icu")

    elif dataset == "mimic3_los_3":
        X_train, X_test, y_train, y_test = extract(random_seed, "los_3")

    elif dataset == "mimic3_los_7":
        X_train, X_test, y_train, y_test = extract(random_seed, "los_7")

    return X_train, X_test, y_train, y_test


def init_models(n_estimators, max_depth):
    """
    Initialize a dictionary of machine learning models for comparison.

    Args:
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the trees.

    Returns:
        dict: Dictionary of initialized models with keys as model names and values as model instances.

    Source:
        https://github.com/yubin-park/califorest/blob/master/analysis/run_chil_exp.py
    """
    mss = 3  # Minimum samples required to split an internal node
    msl = 1  # Minimum samples required to be at a leaf node
    models = {
        "CF-Iso": CaliForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=mss,
            min_samples_leaf=msl,
            ctype="isotonic",
        ),
        "CF-Logit": CaliForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=mss,
            min_samples_leaf=msl,
            ctype="logistic",
        ),
        "RC-Iso": RC30(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=mss,
            min_samples_leaf=msl,
            ctype="isotonic",
        ),
        "RC-Logit": RC30(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=mss,
            min_samples_leaf=msl,
            ctype="logistic",
        ),
        "RF-NoCal": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=mss,
            min_samples_leaf=msl,
        ),
    }
    return models


def run(dataset, random_seed, n_estimators=300, depth=10):
    """
    Train and evaluate models on the specified dataset.

    Args:
        dataset (str): Name of the dataset to use.
        random_seed (int): Random seed for reproducibility.
        n_estimators (int, optional): Number of trees in the forest. Defaults to 300.
        depth (int, optional): Maximum depth of the trees. Defaults to 10.

    Returns:
        list: List of results for each model, including performance metrics.

    Source:
        https://github.com/yubin-park/califorest/blob/master/analysis/run_chil_exp.py
    """
    X_train, X_test, y_train, y_test = read_data(dataset, random_seed)

    output = []

    models = init_models(n_estimators, depth)

    for name, model in models.items():
        t_start = time.time()
        model.fit(X_train, y_train)
        t_elapsed = time.time() - t_start
        y_pred = model.predict_proba(X_test)[:, 1]

        score_auc = roc_auc_score(y_test, y_pred)
        score_hl = hosmer_lemeshow(y_test, y_pred)
        score_sh = spiegelhalter(y_test, y_pred)
        score_b, score_bs = scaled_brier_score(y_test, y_pred)
        rel_small, rel_large = reliability(y_test, y_pred)

        row = [
            dataset,
            name,
            random_seed,
            score_auc,
            score_b,
            score_bs,
            score_hl,
            score_sh,
            rel_small,
            rel_large,
        ]

        print(
            ("[info] {} {}: {:.3f} sec & BS {:.5f}").format(
                dataset, name, t_elapsed, score_b
            )
        )

        output.append(row)

    return output

def simple_imputer(df: DataFrame) -> DataFrame:
    """
    Performs simple imputation and feature engineering on time-series data.

    Specifically, it imputes missing 'mean' values using forward fill,
    group mean, and zero fill. It also creates a binary 'mask' indicating
    data presence and calculates 'time_since_measured' for each feature.

    Args:
        df (pd.DataFrame): Input DataFrame with a MultiIndex for columns.
                           Expected levels include feature names and aggregation
                           functions ('mean', 'count'). Rows should be indexed
                           by time and grouped by ID_COLS.

    Returns:
        pd.DataFrame: Processed DataFrame containing imputed 'mean' columns,
                      'mask' columns, and 'time_since_measured' columns.
                      Columns are sorted.
    """
    idx = IndexSlice
    df = df.copy()

    # Simplify column index if it has extra levels (e.g., from previous processing)
    if len(df.columns.names) > 2:
        df.columns = df.columns.droplevel(("label", "LEVEL1", "LEVEL2"))

    # Select only 'mean' and 'count' aggregations for processing
    df_out = df.loc[:, idx[:, ["mean", "count"]]]

    # --- Impute 'mean' values ---
    # Calculate the mean value for each feature within each ICU stay
    icustay_means = df_out.loc[:, idx[:, "mean"]].groupby(ID_COLS).mean()

    # Impute missing mean values in three steps:
    # 1. Forward fill within each group (carry last observation forward)
    # 2. Fill remaining NaNs with the pre-calculated group mean
    # 3. Fill any remaining NaNs (if a feature was missing for the entire group) with 0
    imputed_means = (
        df_out.loc[:, idx[:, "mean"]]
        .groupby(ID_COLS)
        .fillna(method="ffill")
        .groupby(ID_COLS)
        .fillna(icustay_means)
        .fillna(0)
    ).copy()
    df_out.loc[:, idx[:, "mean"]] = imputed_means

    # --- Create 'mask' feature ---
    # Create a binary mask: 1 if data was present (count > 0), 0 otherwise
    mask = (df.loc[:, idx[:, "count"]] > 0).astype(float).copy()
    # Replace original 'count' columns with the 'mask'
    df_out.loc[:, idx[:, "count"]] = mask
    # Rename the 'count' level in the column index to 'mask'
    df_out = df_out.rename(columns={"count": "mask"}, level="Aggregation Function")

    # --- Calculate 'time_since_measured' feature ---
    # 1 if the value was absent (masked), 0 otherwise
    is_absent = 1 - df_out.loc[:, idx[:, "mask"]].copy()
    # Cumulative sum of absence within each group gives total hours of absence so far
    hours_of_absence = is_absent.groupby(ID_COLS).cumsum()
    # Get the cumulative absence at the last point a measurement *was* present
    last_present_absence = (
        hours_of_absence[is_absent == 0].groupby(ID_COLS).fillna(method="ffill")
    )
    # Time since measured is the difference between total absence and absence at last measurement
    time_since_measured = hours_of_absence - last_present_absence.fillna(
        0
    )  # fillna(0) handles start of series
    # Rename the aggregation level for the new feature
    time_since_measured.rename(
        columns={"mask": "time_since_measured"},
        level="Aggregation Function",
        inplace=True,
    )

    # Add the 'time_since_measured' columns to the output DataFrame
    df_out = pd.concat((df_out, time_since_measured), axis=1)
    # If a value was never measured, fill 'time_since_measured' with a large value
    # (WINDOW_SIZE + 1 implies longer than the observation window)
    time_since_measured_filled = (
        df_out.loc[:, idx[:, "time_since_measured"]].fillna(WINDOW_SIZE + 1)
    ).copy()
    df_out.loc[:, idx[:, "time_since_measured"]] = time_since_measured_filled

    # Sort columns for consistent order
    df_out.sort_index(axis=1, inplace=True)
    return df_out


def extract(
    random_seed: int,
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts and preprocesses time-series and static data for model training and testing.

    This function:
        - Loads patient static and hourly time-series data from HDF5 files.
        - Filters patients with sufficient ICU stay duration.
        - Prepares target variables for prediction (mortality, length of stay, etc.).
        - Splits the data into training and testing sets by subject.
        - Normalizes the time-series features using training set statistics.
        - Applies imputation and feature engineering to handle missing values.
        - Pivots the time-series data to a flat format suitable for machine learning models.
        - Returns the processed feature arrays and target arrays for both train and test sets.

    Args:
        random_seed (int): Seed for reproducible train/test split.
        target (str): Name of the target variable to extract (e.g., 'mort_hosp', 'mort_icu', 'los_3', 'los_7').

    Returns:
        Tuple containing:
            - X_train (np.ndarray): Training set features (subjects × features × time).
            - X_test (np.ndarray): Test set features (subjects × features × time).
            - y_train (np.ndarray): Training set target values.
            - y_test (np.ndarray): Test set target values.
    """
    # Load preprocessed data
    X = np.load("data/mimic_X.npy")
    Y = np.load("data/mimic_Y.npy")
    subjects = pd.read_pickle("data/mimic_subjects.pkl")

    # Get target column index
    target_idx = {"mort_hosp": 0, "mort_icu": 1, "los_3": 2, "los_7": 3}[target]

    # Split subjects into train and test sets
    np.random.seed(random_seed)
    unique_subjects = np.array(list(set(subjects)))
    subjects = np.random.permutation(unique_subjects)
    N = len(subjects)
    N_train = int(TRAIN_FRAC * N)
    train_subj = subjects[:N_train]
    test_subj = subjects[N_train:]

    # Split data by subject
    train_mask = np.isin(subjects, train_subj)
    test_mask = np.isin(subjects, test_subj)

    return (
        X[train_mask],
        X[test_mask],
        Y[train_mask, target_idx],
        Y[test_mask, target_idx],
    )



class RC30(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators=30,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        ctype="isotonic",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ctype = ctype

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )
        if self.ctype == "logistic":
            self.calibrator = LR(C=1e20, solver="lbfgs")
        elif self.ctype == "isotonic":
            self.calibrator = Iso(y_min=0, y_max=1, out_of_bounds="clip")
        X0, X1, y0, y1 = train_test_split(X, y, test_size=0.3)
        self.model.fit(X0, y0)
        if self.ctype == "logistic":
            y_est = self.model.predict_proba(X1)[:, [1]]
            self.calibrator.fit(y_est, y1)
        elif self.ctype == "isotonic":
            y_est = self.model.predict_proba(X1)[:, 1]
            self.calibrator.fit(y_est, y1)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        if self.ctype == "logistic":
            return self.calibrator.predict_proba(self.model.predict_proba(X)[:, [1]])
        elif self.ctype == "isotonic":
            n, m = X.shape
            y = np.zeros((n, 2))
            y[:, 1] = self.calibrator.predict(self.model.predict_proba(X)[:, 1])
            y[:, 0] = 1 - y[:, 1]
            return y


class CaliForest(ClassifierMixin, BaseEstimator):
    """
    A calibrated random forest classifier that combines decision trees with post-hoc calibration.

    This classifier trains an ensemble of decision trees and applies either isotonic regression
    or logistic regression calibration to improve probability estimates.

    Parameters:
        n_estimators (int): Number of trees in the forest (default: 300).
        criterion (str): Splitting criterion ("gini" or "entropy") (default: "gini").
        max_depth (int): Maximum depth of the trees (default: 5).
        min_samples_split (int): Minimum samples required to split an internal node (default: 2).
        min_samples_leaf (int): Minimum samples required to be at a leaf node (default: 1).
        ctype (str): Calibration type ("isotonic" or "logistic") (default: "isotonic").
        alpha0 (float): Prior parameter for calibration weights (default: 100).
        beta0 (float): Prior parameter for calibration weights (default: 25).
    """

    def __init__(
        self,
        n_estimators=300,
        criterion="gini",
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        ctype="isotonic",
        alpha0=100,
        beta0=25,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ctype = ctype
        self.alpha0 = alpha0
        self.beta0 = beta0

    def fit(self, X, y):
        """
        Fit the calibrated random forest model.

        Args:
            X (array-like): Training input samples.
            y (array-like): Target values (binary).

        Returns:
            self: Returns an instance of self.
        """
        # Validate input data
        X, y = check_X_y(X, y, accept_sparse=False)

        # Initialize estimators and calibrator
        self.estimators = []
        self.calibrator = None

        # Create decision tree estimators
        for i in range(self.n_estimators):
            self.estimators.append(
                Tree(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    # Note: Using sqrt for feature subset selection
                    # max_features="auto" was deprecated at newer version
                    # for DecisionTreeClassifier
                    max_features="sqrt",  
                )
            )

        # Initialize the appropriate calibrator
        if self.ctype == "logistic":
            self.calibrator = LR(penalty=None, solver="saga", max_iter=5000)
        elif self.ctype == "isotonic":
            self.calibrator = Iso(y_min=0, y_max=1, out_of_bounds="clip")

        n, m = X.shape
        # Initialize arrays for out-of-bag predictions
        Y_oob = np.full((n, self.n_estimators), np.nan)
        n_oob = np.zeros(n)
        IB = np.zeros((n, self.n_estimators), dtype=int)
        OOB = np.full((n, self.n_estimators), True)

        # Generate bootstrap indices
        for eid in range(self.n_estimators):
            IB[:, eid] = np.random.choice(n, n)
            OOB[IB[:, eid], eid] = False

        # Train each estimator and collect out-of-bag predictions
        for eid, est in enumerate(self.estimators):
            ib_idx = IB[:, eid]  # In-bag indices
            oob_idx = OOB[:, eid]  # Out-of-bag indices
            est.fit(X[ib_idx, :], y[ib_idx])
            Y_oob[oob_idx, eid] = est.predict_proba(X[oob_idx, :])[:, 1]
            n_oob[oob_idx] += 1

        # Filter samples with sufficient out-of-bag predictions
        oob_idx = n_oob > 1
        Y_oob_ = Y_oob[oob_idx, :]
        n_oob_ = n_oob[oob_idx]
        z_hat = np.nanmean(Y_oob_, axis=1)  # Mean prediction for each sample
        z_true = y[oob_idx]  # True labels for calibration

        # Calculate calibration weights
        beta = self.beta0 + np.nanvar(Y_oob_, axis=1) * n_oob_ / 2
        alpha = self.alpha0 + n_oob_ / 2
        z_weight = alpha / beta

        # Fit the calibrator
        if self.ctype == "logistic":
            self.calibrator.fit(z_hat[:, np.newaxis], z_true, z_weight)
        elif self.ctype == "isotonic":
            self.calibrator.fit(z_hat, z_true, z_weight)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Args:
            X (array-like): Input samples.

        Returns:
            array: Array of shape (n_samples, 2) with class probabilities.
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        n, m = X.shape
        n_est = len(self.estimators)
        z = np.zeros(n)
        y_mat = np.zeros((n, 2))

        # Aggregate predictions from all estimators
        for eid, est in enumerate(self.estimators):
            z += est.predict_proba(X)[:, 1]
        z /= n_est  # Average prediction

        # Apply calibration
        if self.ctype == "logistic":
            y_mat[:, 1] = self.calibrator.predict_proba(z[:, np.newaxis])[:, 1]
        elif self.ctype == "isotonic":
            y_mat[:, 1] = self.calibrator.predict(z)

        y_mat[:, 0] = 1 - y_mat[:, 1]  # Probability of class 0
        return y_mat

    def predict(self, X):
        """
        Predict class labels for X.

        Args:
            X (array-like): Input samples.

        Returns:
            array: Predicted class labels.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)




def preprocess():
    # Load original data
    statics = pd.read_hdf("data/all_hourly_data.h5", "patients")
    data_full_lvl2 = pd.read_hdf("data/all_hourly_data.h5", "vitals_labs")

    # Filter patients
    statics = statics[statics.max_hours > WINDOW_SIZE + GAP_TIME]

    # Prepare targets
    Ys = statics.loc[:, ["mort_hosp", "mort_icu", "los_icu"]]
    Ys.loc[:, "mort_hosp"] = (Ys.loc[:, "mort_hosp"]).astype(int)
    Ys.loc[:, "mort_icu"] = (Ys.loc[:, "mort_icu"]).astype(int)
    Ys.loc[:, "los_3"] = (Ys.loc[:, "los_icu"] > 3).astype(int)
    Ys.loc[:, "los_7"] = (Ys.loc[:, "los_icu"] > 7).astype(int)
    Ys.drop(columns=["los_icu"], inplace=True)

    # Process time series data
    lvl2 = data_full_lvl2.loc[
        (
            data_full_lvl2.index.get_level_values("icustay_id").isin(
                set(Ys.index.get_level_values("icustay_id"))
            )
        )
        & (data_full_lvl2.index.get_level_values("hours_in") < WINDOW_SIZE),
        :,
    ]

    # Normalize
    idx = pd.IndexSlice
    lvl2_means = lvl2.loc[:, idx[:, "mean"]].mean(axis=0)
    lvl2_stds = lvl2.loc[:, idx[:, "mean"]].std(axis=0)
    vals_centered = lvl2.loc[:, idx[:, "mean"]] - lvl2_means
    lvl2.loc[:, idx[:, "mean"]] = vals_centered

    # Impute and pivot
    lvl2 = simple_imputer(lvl2)
    lvl2_flat = lvl2.pivot_table(
        index=["subject_id", "hadm_id", "icustay_id"], columns=["hours_in"]
    )

    # Save preprocessed data
    np.save("data/mimic_X.npy", lvl2_flat.values)
    np.save("data/mimic_Y.npy", Ys.values)
    pd.Series(Ys.index.get_level_values("subject_id")).to_pickle(
        "data/mimic_subjects.pkl"
    )


def hosmer_lemeshow(y_true, y_score):
    """
    Calculate the Hosmer Lemeshow to assess whether
    or not the observed event rates match expected
    event rates.

    Assume that there are 10 groups:
    HL = \\sum_{g=1}^G \\frac{(O_{1g} - E_{1g})^2}{N_g \\pi_g (1- \\pi_g)}
    """
    n_grp = 10  # number of groups

    # create the dataframe
    df = pd.DataFrame({"score": y_score, "target": y_true})

    # sort the values
    df = df.sort_values("score")
    # shift the score a bit
    df["score"] = np.clip(df["score"], 1e-8, 1 - 1e-8)
    df["rank"] = list(range(df.shape[0]))
    # cut them into 10 bins
    df["score_decile"] = pd.qcut(df["rank"], n_grp, duplicates="raise")
    # sum up based on each decile
    obsPos = df["target"].groupby(df.score_decile, observed=False).sum()
    obsNeg = df["target"].groupby(df.score_decile, observed=False).count() - obsPos
    exPos = df["score"].groupby(df.score_decile, observed=False).sum()
    exNeg = df["score"].groupby(df.score_decile, observed=False).count() - exPos
    hl = (((obsPos - exPos) ** 2 / exPos) + ((obsNeg - exNeg) ** 2 / exNeg)).sum()

    # https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    # Re: p-value, higher the better Goodness-of-Fit
    p_value = 1 - chi2.cdf(hl, n_grp - 2)

    return p_value


def reliability(y_true, y_score):
    """
    Calculate the reliability of the model, which measures the calibration of the model.

    The reliability is defined as:
    rel_small = \\mean((obs - exp) ** 2)
    rel_large = (\\mean(y_true) - \\mean(y_score)) ** 2
    """
    n_grp = 10
    df = pd.DataFrame({"score": y_score, "target": y_true})
    df = df.sort_values("score")
    df["rank"] = list(range(df.shape[0]))
    df["score_decile"] = pd.qcut(df["rank"], n_grp, duplicates="raise")

    obs = df["target"].groupby(df.score_decile, observed=False).mean()
    exp = df["score"].groupby(df.score_decile, observed=False).mean()

    rel_small = np.mean((obs - exp) ** 2)
    rel_large = (np.mean(y_true) - np.mean(y_score)) ** 2

    return rel_small, rel_large


def spiegelhalter(y_true, y_score):
    """
    Calculate the Spiegelhalter test statistic, which measures the calibration of the model.

    The Spiegelhalter test statistic is defined as:
    top = \\sum (y_true - y_score) * (1 - 2 * y_score)
    bot = \\sum (1 - 2 * y_score) ** 2 * y_score * (1 - y_score)
    sh = top / \\sqrt(bot)
    """
    top = np.sum((y_true - y_score) * (1 - 2 * y_score))
    bot = np.sum((1 - 2 * y_score) ** 2 * y_score * (1 - y_score))
    sh = top / np.sqrt(bot)

    # https://en.wikipedia.org/wiki/Z-test
    # Two-tailed test
    # Re: p-value, higher the better Goodness-of-Fit
    p_value = norm.sf(np.abs(sh)) * 2

    return p_value


def scaled_brier_score(y_true, y_score):
    """
    Calculate the scaled Brier score, which measures the calibration of the model.

    The scaled Brier score is defined as:
    1 - (Brier score / (mean of true labels * (1 - mean of true labels)))
    """
    brier = skm.brier_score_loss(y_true, y_score)
    # calculate the mean of the probability
    p = np.mean(y_true)
    brier_scaled = 1 - brier / (p * (1 - p))
    return brier, brier_scaled


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CaliForest processing script")
    parser.add_argument("--preprocess", action="store_true", help="Run data preprocessing to generate mimic_X.npy and mimic_Y.npy")
    parser.add_argument("--dataset", type=str, default="hastie", help="Dataset to run model on (e.g., hastie, breast_cancer, mimic3_mort_hosp)")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--n_estimators", type=int, default=300, help="Number of trees in the forest")
    parser.add_argument("--depth", type=int, default=10, help="Maximum depth of trees")
    args = parser.parse_args()

    if args.preprocess:
        print("[info] Running preprocessing...")
        preprocess()

    print(f"[info] Running training on dataset {args.dataset}")
    results = run(args.dataset, args.random_seed, args.n_estimators, args.depth)
    for row in results:
        print(row)

    # Save results to CSV
    columns = [
        "dataset",
        "model",
        "random_seed",
        "roc_auc",
        "brier_score",
        "scaled_brier_score",
        "hosmer_lemeshow_p",
        "spiegelhalter_p",
        "reliability_small",
        "reliability_large",
    ]
    df_results = pd.DataFrame(results, columns=columns)
    out_csv = f"results_{args.dataset}_seed{args.random_seed}.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"[info] Results saved to {out_csv}")
