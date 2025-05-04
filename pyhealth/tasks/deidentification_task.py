from .base_task import BaseTask
from typing import Dict, List, Optional
import torch
import numpy as np
import pandas as pd
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pyhealth.datasets.deidentification_dataset import DeidentificationDataset

class DeIdentificationTask(BaseTask):
    def __init__(self, dataset: DeidentificationDataset):
        self.dataset = dataset
        self.model = None  

    def pre_process_data(self):
        """
        Preprocess the dataset (clean the text and other necessary transformations).
        """
        self.dataset.preprocess_data()

    def extract_diagnosis(self):
        """Extract diagnosis from each record's text column."""
        def extract(text):
            if "Diagnosis:" in text:
                return "Diagnosis Found"
            return "Diagnosis not found"

        self.dataset.data["diagnosis_extracted"] = self.dataset.data["text"].apply(extract)

    def get_task_info(self):
        """
        Return task-related information (e.g., dataset name, task type).
        """
        return {
            'dataset_name': self.dataset.config['table_name'],
            'task_type': 'deidentification',  # Task type can be extended for different task types
        }

    def __call__(self, example):
        """
        Transform an example using the trained model.
        """
        if self.model:
            prediction = self.model.predict([example["text"]])[0]
            return {
                "input": example["text"],
                "label": prediction
            }
        else:
            return {
                "input": example["text"],
                "label": example.get("diagnosis_extracted", "Diagnosis not found")
            }

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the task's performance on a given dataset (e.g., compute accuracy, precision).
        
        Args:
            test_data (pd.DataFrame): The dataset to evaluate on.
        
        Returns:
            dict: The evaluation metrics (e.g., accuracy).
        """
        correct_predictions = 0
        total = len(test_data)
        
        # Iterate over the test data and count correct predictions
        for _, example in test_data.iterrows():
            transformed_example = self(example)
            predicted = transformed_example["label"]
            actual = example["diagnosis_extracted"]
            
            if predicted == actual:
                correct_predictions += 1

        accuracy = correct_predictions / total
        print(f"Accuracy: {accuracy}")
        return {"accuracy": accuracy, 'total': total}

    def train(self, train_data: pd.DataFrame, epochs: int = 10):
        """
        Train a logistic regression model using TF-IDF features.
        """
        print("Training with logistic regression using TF-IDF features...")

        # Prepare training data
        X_train = train_data["text"]
        y_train = train_data["diagnosis_extracted"]

        # Create a pipeline with TF-IDF vectorizer and logistic regression
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(max_iter=epochs))
        ])

        # Create a pipeline with TF-IDF vectorizer and logistic regression
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(max_iter=500, class_weight='balanced'))
        ])

        # Train the model
        self.model.fit(X_train, y_train)

        print("Training complete.")

if __name__ == "__main__":
    from pyhealth.datasets.deidentification_dataset import DeidentificationDataset

    # Configuration for the sample dataset
    dataset_config = {
        'table_name': 'discharge_summaries',
        'file_path': 'data/deid_raw/discharge/discharge_summaries.json',
        'patient_id': 'document_id',
        'timestamp': None,
        'attributes': ['document_id', 'text']
    }

    # Load dataset
    dataset = DeidentificationDataset(dataset_config)
    
    # Create DeIdentificationTask
    task = DeIdentificationTask(dataset)

    # Step 1: Preprocess the text
    print("Preprocessing data...")
    task.pre_process_data()

    # Step 2: Extract diagnosis information from the text
    print("Extracting diagnoses...")
    task.extract_diagnosis()

    # Step 3: Transform all examples in the dataset
    print("\nTransforming examples:")
    transformed_examples = []
    for _, example in dataset.data.iterrows():
        transformed = task(example)
        transformed_examples.append(transformed)

    # Step 4: Training loop (example, just printing out the epoch info)
    print("\nTraining model...")
    task.train(train_data=dataset.data, epochs=3)

    # Step 5: Evaluate the task on test data
    print("\nEvaluating model...")
    # Assuming we have test data for evaluation (you should separate train/test in practice)
    test_data = dataset.data.sample(frac=0.2, random_state=42)  # Take 20% as test data
    evaluation_results = task.evaluate(test_data)
    print("Evaluation Results:", evaluation_results)
