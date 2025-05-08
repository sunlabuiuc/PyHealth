"""
Unit tests for the EHR Summarization pipeline using PyHealth.

This file validates:
1. Dataset processing: EHRSummarizationDataset correctly splits notes into sentences 
   and labels them based on future ICD codes.
2. Task interface: `ehr_summarization_task` generates samples in the expected format.
3. Model forward pass: ClinicalBertSentenceClassifier returns output of correct shape 
   when given tokenized input.

Authors: Abhitej Bokka (abhitej2), Liam Shen (liams4)
"""

from pyhealth.datasets.ehr_summarization import EHRSummarizationDataset
from pyhealth.tasks.ehr_summarization_task import ehr_summarization_task
from pyhealth.models.clinical_bert import ClinicalBertSentenceClassifier
from pyhealth.datasets import SampleDataset
from transformers import AutoTokenizer
import torch


def test_dataset_and_task():
    """
    Test the EHRSummarizationDataset and the ehr_summarization_task.

    This function:
    - Constructs a mock patient with two encounters.
    - Processes the notes into sentences and labels.
    - Runs the summarization task to generate samples.
    - Checks that each sentence has a corresponding binary label.
    """

    # Construct a minimal in-memory dataset
    dataset = EHRSummarizationDataset(root="./sample_data/")
    dataset.patients = {
        "1": {
            "patient_id": "1",
            "encounters": [
                {
                    "date": "2020-01-01",
                    "visit_id": "v1",
                    "diagnoses": ["250.00"],  # ICD-9 code for diabetes
                    "notes": [{"text": "Patient has diabetes."}]
                },
                {
                    "date": "2020-02-15",
                    "visit_id": "v2",
                    "diagnoses": ["410.90"],  # ICD-9 code for heart issues
                    "notes": [{"text": "Chest pain and shortness of breath."}]
                }
            ]
        }
    }

    # Apply spaCy-based sentence splitting and future-label assignment
    dataset.process_notes()

    # Make dataset iterable (SampleEHRDataset expects iteration for task consumption)
    import types
    dataset.__iter__ = types.MethodType(lambda self: iter(self.patients.values()), dataset)

    # Run the task to generate sentence-level samples
    samples = list(ehr_summarization_task(dataset))
    sample = samples[0]


    assert "sentences" in sample
    assert "labels" in sample
    assert len(sample["sentences"]) == len(sample["labels"])


def test_model_forward():
    """
    Test the forward pass of the ClinicalBertSentenceClassifier.

    This function:
    - Tokenizes two sentences using Bio_ClinicalBERT.
    - Passes them through the ClinicalBERT model wrapper.
    - Asserts that the output shape matches the number of input examples.
    """

    # Use Huggingface tokenizer to prepare two synthetic sentences
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    sentences = ["Patient has chest pain.", "History of diabetes."]
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

    # Construct a mock SampleDataset that matches the model input/output schema
    mock_samples = [
        {"sentences": "Patient has chest pain.", "labels": 1},
        {"sentences": "History of diabetes.", "labels": 0}
    ]

    mock_dataset = SampleDataset(
        samples=mock_samples,
        input_schema={"sentences": "text"},  # feature name and type
        output_schema={"labels": "binary"},  # label type
        dataset_name="mock",
        task_name="ehr_summarization"
    )

    # Instantiate ClinicalBERT with mock dataset and run forward pass through the model
    model = ClinicalBertSentenceClassifier(dataset=mock_dataset, feature_keys=["sentences"])
    out = model(inputs["input_ids"], inputs["attention_mask"])

    # Ensure that we get one prediction per sentence
    assert out.shape[0] == 2
