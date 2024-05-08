import pandas as pd
from pyhealth.datasets.base_ehr_dataset import BaseEHRDataset
import xgboost as xgb
from pyhealth.models import BaseModel

'''
Example: Predicting 30-Day Mortality in Sepsis Patients using PyHealth and XGBoost
This example demonstrates how to use the PyHealth framework to predict 30-day mortality among patients with sepsis-3, 
leveraging the powerful XGBoost algorithm. This use case involves loading a specific dataset, setting a prediction task, 
and training an XGBoost model.
'''

if __name__ == '__main__':
    # Step 1: load the data
    # Example of how you might use this Mimic3Sepsis3Dataset class
    dataset = Mimic3Sepsis3Dataset(root='/content/drive/My Drive/Colab Notebooks', tables=['dummy_table'])
    # Ensure this is called to load and process data
    dataset.load_data()  
    print(dataset.dataframe.head())

    # Step 2: set task
    # Example of how you might use this mortality_prediction_sepsis3_fn class
    # Simulating a patient's data
    patient_data = {
        'urineoutput': [550],
        'lactate_min': [1.8],
        'bun_mean': [15],
        'sysbp_min': [110],
        'metastatic_cancer': [1],
        'inr_max': [1.2],
        'age': [75],
        'sodium_max': [136],
        'aniongap_max': [14],
        'creatinine_min': [1.1],
        'spo2_mean': [96],
        'thirtyday_expire_flag': [1]  # Assume the patient deceased within 30 days
    }
    # Calling the function with simulated data
    sample_output = mortality_prediction_sepsis3_fn(patient_data)
    # Printing the output
    print("Sample Output:")
    for sample in sample_output:
        print(sample)
        
    # Step 3: train the model
    # Example usage of XGBoostModel class
    xgb_model = XGBoostModel(dataset, feature_keys=['urineoutput', 'lactate_min', 'bun_mean', 'sysbp_min', 'metastatic_cancer', 'inr_max', 'age', 'sodium_max', 'aniongap_max', 'creatinine_min', 'spo2_mean'], label_key='thirtyday_expire_flag', mode='binary', objective='binary:logistic', max_depth=5, eta=0.1)
    xgb_model.fit(dataset.dataframe, num_round=100)
    predictions = xgb_model.predict(dataset.dataframe)
    print(predictions)