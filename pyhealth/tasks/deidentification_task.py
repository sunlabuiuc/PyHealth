"""
DeIdentificationTask Class
Author: Varshini R R
NetID: vrr4

This module implements the `DeIdentificationTask` class, which handles the 
training, evaluation, and preprocessing of synthetic hospital discharge summaries 
for the purpose of de-identification. The task utilizes logistic regression and 
TF-IDF features for diagnosis extraction and classification.

Usage:
    This class is used in the de-identification pipeline to load, preprocess, 
    and train a model on discharge summary data prior to further evaluation 
    or deployment.

Modules and Methods:
    - `pre_process_data`: Preprocesses the dataset by cleaning and normalizing the text.
    - `extract_diagnosis`: Extracts diagnosis mentions from the text and labels them accordingly.
    - `get_task_info`: Retrieves metadata about the task.
    - `__call__`: Inferences on a single example and provides the input and predicted label.
    - `train`: Trains a logistic regression classifier using TF-IDF features on the dataset.
    - `evaluate`: Evaluates the model using accuracy on a provided test dataset.

Dependencies:
    - pandas
    - scikit-learn
    - pyhealth

"""
from typing import Dict
import pandas as pd
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .base_task import BaseTask
from pyhealth.datasets.deidentification_dataset import DeidentificationDataset


class DeIdentificationTask(BaseTask):
    """A task class for de-identification using logistic regression and TF-IDF."""

    def __init__(self, dataset: DeidentificationDataset):
        """
        Initialize the task with a given dataset.

        Args:
            dataset (DeidentificationDataset): The dataset object to use for training and evaluation.
        """
        self.dataset = dataset
        self.model = None

    def pre_process_data(self) -> None:
        """Preprocess the dataset by cleaning and normalizing the text column."""
        self.dataset.preprocess_data()

    def extract_diagnosis(self) -> None:
        """Extract diagnosis mentions from the text and label accordingly."""
        def extract(text: str) -> str:
            return "Diagnosis Found" if "Diagnosis:" in text else "Diagnosis not found"

        self.dataset.data["diagnosis_extracted"] = self.dataset.data["text"].apply(extract)

    def get_task_info(self) -> Dict[str, str]:
        """
        Retrieve metadata about the task.

        Returns:
            Dict[str, str]: Dictionary containing dataset name and task type.
        """
        return {
            "dataset_name": self.dataset.config["table_name"],
            "task_type": "deidentification"
        }

    def __call__(self, example: Dict[str, str]) -> Dict[str, str]:
        """
        Run inference on a single example.

        Args:
            example (Dict[str, str]): A dictionary with a "text" field.

        Returns:
            Dict[str, str]: A dictionary with the input and predicted label.
        """
        if self.model:
            prediction = self.model.predict([example["text"]])[0]
            return {"input": example["text"], "label": prediction}
        return {
            "input": example["text"],
            "label": example.get("diagnosis_extracted", "Diagnosis not found")
        }

    def train(self, train_data: pd.DataFrame, epochs: int = 10) -> None:
        """
        Train a logistic regression classifier on the training data.

        Args:
            train_data (pd.DataFrame): DataFrame containing 'text' and 'diagnosis_extracted'.
            epochs (int, optional): Max iterations for training. Defaults to 10.
        """
        print("Training with logistic regression using TF-IDF features...")

        X_train = train_data["text"]
        y_train = train_data["diagnosis_extracted"]

        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])

        self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the classifier on test data using accuracy.

        Args:
            test_data (pd.DataFrame): The test dataset.

        Returns:
            Dict[str, float]: Evaluation results with accuracy and total sample count.
        """
        correct_predictions = 0
        total = len(test_data)

        for _, example in test_data.iterrows():
            transformed_example = self(example)
            if transformed_example["label"] == example["diagnosis_extracted"]:
                correct_predictions += 1

        accuracy = correct_predictions / total
        print(f"Accuracy: {accuracy:.4f}")
        return {"accuracy": accuracy, "total": total}


if __name__ == "__main__":
    dataset_config = {
        "table_name": "discharge_summaries",
        "file_path": "data/deid_raw/discharge/discharge_summaries.json",
        "patient_id": "document_id",
        "timestamp": None,
        "attributes": ["document_id", "text"]
    }

    # Load dataset
    dataset = DeidentificationDataset(dataset_config)

    # Create and run de-identification task
    task = DeIdentificationTask(dataset)

    print("Preprocessing data...")
    task.pre_process_data()

    print("Extracting diagnoses...")
    task.extract_diagnosis()

    print("Training model...")
    task.train(train_data=dataset.data, epochs=10)

    print("Evaluating model...")
    test_data = dataset.data.sample(frac=0.2, random_state=42)
    results = task.evaluate(test_data)
    print("Evaluation Results:", results)
