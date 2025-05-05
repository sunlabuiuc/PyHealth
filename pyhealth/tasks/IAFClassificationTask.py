"""
Jake Cumberland (jakepc3)
Swaroop Potdar (spotd)

For use with the IAFDBDataset object, based on the dataset object from Interpretation of Intracardiac
Electrograms Through Textual Representations: https://arxiv.org/abs/2402.01115

A task meant to prepare a sample for classification training.
"""

from base_task import BaseTask

import torch
class IAFClassificationTask(BaseTask):
    def __init__(self, dataset, task_name="iaf_classification", label_function=None):
        """
        Args:
        - dataset: IAFDBDataset object
        - task_name: string, name of the task
        - label_function: function that generates a label given a (i, j, k, 1) key and segment
        """
        self.task_name = task_name
        self.dataset = dataset

        # Default label function: classifies based on electrode index
        if label_function is None:
            def label_function(record_key, segment):
                i, j, k, _ = record_key
                return int(i % 2 == 0)
        self.label_function = label_function

        # Preprocess dataset samples into task samples
        self.samples = self.create_samples()

    def create_samples(self):
        samples = []
        for key in self.dataset.samples:
            segment = self.dataset.samples[key]
            label = self.label_function(key, segment)
            samples.append({
                "record_id": f"{key}",
                "signal": torch.tensor(segment, dtype=torch.float32),  # shape: [segment_length]
                "label": int(label)
            })
        print(f"Created {len(samples)} samples for task '{self.task_name}'")
        return samples

    def __call__(self, example):
        """
        Format each sample for model input.

        Parameters:
        - example: dict with 'signal' and 'label'

        Returns:
        - A tuple: (inputs, label)
        """
        inputs = example["signal"]  # shape [segment_length]
        label = example["label"]
        return inputs, label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]