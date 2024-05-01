import xgboost as xgb
from pyhealth.models import BaseModel

class XGBoostModel(BaseModel):
    """
    An XGBoost model integrated within the PyHealth framework, specifically designed
    for binary classification tasks in healthcare datasets. This class extends the
    BaseModel class of PyHealth, allowing it to seamlessly integrate with PyHealth's
    dataset management and training routines.

    XGBoost is a powerful, scalable machine learning algorithm that implements
    gradient boosting framework. It has proven to be highly effective in a variety
    of machine learning tasks in diverse domains. This model class uses XGBoost to
    predict binary outcomes based on a set of health-related features specified during
    the class instantiation.

    Reference: https://xgboost.readthedocs.io/en/stable/

    Args:
        dataset (SampleBaseDataset): The dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys (List[str]): List of keys in samples to use as features,
            e.g., ['urineoutput', 'lactate_min', ...].
        label_key (str): Key in samples to use as the label (e.g., 'thirtyday_expire_flag').
        mode (str): One of 'binary', 'multiclass', or 'multilabel'. This implementation specifically
            supports 'binary' for binary classification tasks.
        **xgb_params: Additional parameters to configure the XGBoost model, such as
            'objective', 'max_depth', and 'eta'.

    Example Usage:
        # Assuming 'dataset' is an instance of a class derived from SampleBaseDataset and has been properly loaded.
        xgb_model = XGBoostModel(dataset, 
                                 feature_keys=['urineoutput', 'lactate_min', 'bun_mean', 'sysbp_min', 'metastatic_cancer', 'inr_max', 'age', 'sodium_max', 'aniongap_max', 'creatinine_min', 'spo2_mean'],
                                 label_key='thirtyday_expire_flag', 
                                 mode='binary', 
                                 objective='binary:logistic', 
                                 max_depth=5, 
                                 eta=0.1)
        xgb_model.fit(dataset.dataframe, num_round=100)
        predictions = xgb_model.predict(dataset.dataframe)
        print(predictions)
    """
    def __init__(self, dataset, feature_keys, label_key, mode, **xgb_params):
        super(XGBoostModel, self).__init__(dataset, feature_keys, label_key, mode)
        self.model = None
        self.xgb_params = xgb_params

    def fit(self, train_data, num_round=100):
        """Train the XGBoost model."""
        dtrain = xgb.DMatrix(train_data[self.feature_keys], label=train_data[self.label_key])
        self.model = xgb.train(self.xgb_params, dtrain, num_round)

    def predict(self, data):
        """Make predictions with the trained XGBoost model."""
        ddata = xgb.DMatrix(data[self.feature_keys])
        return self.model.predict(ddata)

    def save_model(self, file_path):
        """Save the model to a file."""
        self.model.save_model(file_path)

    def load_model(self, file_path):
        """Load the model from a file."""
        self.model = xgb.Booster()
        self.model.load_model(file_path)