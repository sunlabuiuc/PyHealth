from typing import Any, Dict, List
from datetime import datetime

from .base_task import BaseTask


class TransferPrediction(BaseTask):
    """Task for predicting patient transfers between care units.

    This implementation provides a baseline solution for the transfer prediction task
    discussed in the paper "Uncertainty-Aware Text-to-Program for Question Answering
    on Structured Electronic Health Records" (CHIL 2022).

    Dataset:
        The implementation uses the MIMIC-III Clinical Database (version 1.4),
        which requires credentialed access from PhysioNet:
        https://physionet.org/content/mimiciii/1.4/

    This task predicts whether a patient will be transferred to another unit
    during their hospital stay based on their current unit and admission information.
    
    The prediction is based on the TRANSFERS table, which records all unit transfers
    including:
    - Transfers between regular wards
    - Transfers to/from ICU
    - Transfers to/from emergency department
    - Transfers to/from operating room
    
    Args:
        patient: a Patient object containing admission and transfer events
    
    Returns:
        samples: a list of samples. Each sample is a dict including:
            - patient_id: patient identifier
            - admission_id: admission identifier
            - current_unit: current unit the patient is in
            - admission_type: type of admission (EMERGENCY, ELECTIVE, etc.)
            - admission_location: location of admission
            - insurance: type of insurance
            - ethnicity: patient's ethnicity
            - diagnosis: admission diagnosis
            - transfer: binary indicator of transfer (1 if transferred, 0 if not)
    """
    task_name: str = "TransferPrediction"
    input_schema: Dict[str, str] = {
        "current_unit": "sequence",
        "admission_type": "sequence",
        "admission_location": "sequence",
        "insurance": "sequence",
        "ethnicity": "sequence",
        "diagnosis": "text"
    }
    output_schema: Dict[str, str] = {"transfer": "binary"}

    def __init__(self):
        """Initialize the TransferPrediction task.
        
        Sets up the task with default parameters and configurations.
        """
        super().__init__()

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient record to extract transfer prediction samples.
        
        This method processes a patient's admission and transfer events to create
        samples for transfer prediction. Each sample represents a point in time
        where a transfer prediction can be made.
        
        Args:
            patient: A Patient object containing admission and transfer events.
            
        Returns:
            A list of samples, where each sample is a dictionary containing:
                - patient_id: patient identifier
                - admission_id: admission identifier
                - current_unit: current unit the patient is in
                - admission_type: type of admission
                - admission_location: location of admission
                - insurance: type of insurance
                - ethnicity: patient's ethnicity
                - diagnosis: admission diagnosis
                - transfer: binary indicator of transfer
        """
        # Get admission events
        admissions = patient.get_events(event_type="admissions")
        if not admissions:
            return []
        
        # Get transfer events
        transfers = patient.get_events(event_type="transfers")
        print(f"\nPatient {patient.patient_id}:")
        print(f"Number of admissions: {len(admissions)}")
        print(f"Number of transfers: {len(transfers)}")
        
        samples = []
        for admission in admissions:
            # Get admission time from timestamp
            admittime = admission.timestamp
            dischtime = datetime.strptime(
                admission.attr_dict["dischtime"], "%Y-%m-%d %H:%M:%S"
            )
            
            # Get transfers during this admission
            admission_transfers = []
            for t in transfers:
                # Skip transfers without timestamp
                if t.timestamp is None:
                    continue
                
                # Check if transfer is within admission period
                if t.timestamp >= admittime and t.timestamp <= dischtime:
                    admission_transfers.append(t)
            
            # Sort transfers by timestamp
            admission_transfers.sort(key=lambda x: x.timestamp)
            
            # Create a single sample for this admission
            if admission_transfers:
                # Use the first transfer's current unit
                current_unit = admission_transfers[0].attr_dict["curr_careunit"]
                if current_unit is None:
                    continue
                
                # Check if there was any transfer to a different unit
                has_transfer = False
                for i in range(len(admission_transfers) - 1):
                    next_unit = admission_transfers[i + 1].attr_dict["curr_careunit"]
                    if next_unit is not None and next_unit != current_unit:
                        has_transfer = True
                        break
            else:
                # If no transfers, use admission location as current unit
                current_unit = admission.attr_dict["admission_location"]
                has_transfer = False
            
            # Create sample if all required fields are present
            if all(admission.attr_dict.get(field) is not None for field in [
                "admission_type", "admission_location", "insurance", "ethnicity", "diagnosis"
            ]):
                sample = {
                    "patient_id": patient.patient_id,
                    "admission_id": admission.attr_dict["hadm_id"],
                    "current_unit": current_unit,
                    "admission_type": admission.attr_dict["admission_type"],
                    "admission_location": admission.attr_dict["admission_location"],
                    "insurance": admission.attr_dict["insurance"],
                    "ethnicity": admission.attr_dict["ethnicity"],
                    "diagnosis": admission.attr_dict["diagnosis"],
                    "transfer": 1 if has_transfer else 0
                }
                samples.append(sample)
        
        return samples


if __name__ == "__main__":
    from datetime import datetime
    from collections import defaultdict
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from pyhealth.datasets import MIMIC3Dataset

    # Load dataset
    root = "mimic-iii-clinical-database-1.4"
    dataset = MIMIC3Dataset(
        root=root,
        tables=["ADMISSIONS", "TRANSFERS"]
    )

    # Set task
    samples = dataset.set_task(TransferPrediction())
    
    # Convert samples to DataFrame
    data = []
    for sample in samples:
        data.append({
            "patient_id": sample["patient_id"],
            "admission_id": sample["admission_id"],
            "current_unit": str(sample["current_unit"]),
            "admission_type": str(sample["admission_type"]),
            "admission_location": str(sample["admission_location"]),
            "insurance": str(sample["insurance"]),
            "ethnicity": str(sample["ethnicity"]),
            "diagnosis": str(sample["diagnosis"]),
            "transfer": sample["transfer"].item()
        })
    df = pd.DataFrame(data)
    
    # Remove duplicate records
    print("\n=== Data Cleaning ===")
    print(f"Original number of records: {len(df)}")
    df = df.drop_duplicates(subset=['patient_id', 'admission_id', 'current_unit'])
    print(f"Number of records after removing duplicates: {len(df)}")
    
    # Data Analysis
    print("\n=== Data Analysis ===")
    print(f"Total samples: {len(df)}")
    print(f"Transfer samples: {df['transfer'].sum()}")
    print(f"Non-transfer samples: {len(df) - df['transfer'].sum()}")
    print(f"Transfer rate: {df['transfer'].mean()*100:.2f}%")
    
    # Check feature distributions
    print("\n=== Feature Distributions ===")
    categorical_features = ['current_unit', 'admission_type', 'admission_location', 'insurance', 'ethnicity']
    for feature in categorical_features:
        print(f"\n{feature} distribution:")
        value_counts = df[feature].value_counts()
        print(value_counts)
        print(f"Unique values: {len(value_counts)}")
    
    # Check feature-target relationships
    print("\n=== Feature-Transfer Relationships ===")
    for feature in categorical_features:
        print(f"\n{feature} transfer rates:")
        transfer_rates = df.groupby(feature)['transfer'].mean()
        print(transfer_rates.sort_values(ascending=False))
    
    # Feature Engineering
    # 1. Handle missing values
    df = df.fillna('UNKNOWN')
    
    # 2. One-hot encoding with proper feature names
    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = onehot_encoder.fit_transform(df[categorical_features])
    
    # Get feature names
    feature_names = []
    for i, feature in enumerate(categorical_features):
        categories = onehot_encoder.categories_[i]
        for category in categories:
            # Clean up category name
            if isinstance(category, str):
                clean_category = category
            else:
                # Extract numbers from tensor
                numbers = [str(x) for x in category if isinstance(x, (int, float))]
                clean_category = '_'.join(numbers) if numbers else str(category)
            feature_names.append(f"{feature}_{clean_category}")
    
    # Convert to DataFrame for better inspection
    X_df = pd.DataFrame(X_encoded.toarray(), columns=feature_names)
    
    # 3. Prepare features and labels
    X = X_df.values
    y = df['transfer']
    
    # 4. Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model Training
    # 1. Initialize model with more conservative parameters
    model = RandomForestClassifier(
        n_estimators=200,  # Increase number of trees
        max_depth=7,       # Increase depth
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42
    )
    
    # 2. Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print("\n=== Cross-validation Results ===")
    print(f"CV ROC-AUC scores: {cv_scores}")
    print(f"Mean ROC-AUC: {cv_scores.mean():.4f}")
    print(f"Std ROC-AUC: {cv_scores.std():.4f}")
    
    # 3. Train final model
    model.fit(X_train, y_train)
    
    # Model Evaluation
    # 1. Training set predictions
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    
    # 2. Test set predictions
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n=== Model Performance ===")
    print("\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    print(f"\nROC-AUC Score (Test Set): {roc_auc_score(y_test, y_test_prob):.4f}")
    
    # Feature Importance
    print("\n=== Feature Importance ===")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Overfitting Check
    print("\n=== Overfitting Check ===")
    print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
    print(f"Training ROC-AUC: {roc_auc_score(y_train, y_train_prob):.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_test_prob):.4f}")
    
    # Additional Diagnostics
    print("\n=== Additional Diagnostics ===")
    print("\nFeature correlation with target:")
    for feature in categorical_features:
        correlation = df[feature].astype('category').cat.codes.corr(df['transfer'])
        print(f"{feature}: {correlation:.4f}")
    
    print("\nSample of actual transfers:")
    print(df[df['transfer'] == 1].head())
    
    print("\nSample of non-transfers:")
    print(df[df['transfer'] == 0].head())
    
    # Check for data leakage
    print("\n=== Data Leakage Check ===")
    print("\nNumber of unique patients:", df['patient_id'].nunique())
    print("Number of unique admissions:", df['admission_id'].nunique())
    print("\nPatient-admission combinations:")
    print(df.groupby('patient_id')['admission_id'].nunique().describe()) 