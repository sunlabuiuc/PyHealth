# pyhealth/models/clinical_t5.py
"""ClinicalT5 model for healthcare NLP tasks.
Paper: Eric Lehman et al., Do We Still Need Clinical Language Models? (2023)
Task: Supports both text generation (e.g., radiology reports) and classification (e.g., MedNLI)

Note:
    This implementation is adapted from:
    - Original paper: https://arxiv.org/abs/2302.08091
    - Clinical-T5 weights: https://physionet.org/content/clinical-t5/1.0.0/
"""
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class ClinicalT5(BaseModel):
    """ClinicalT5 model for healthcare text generation and classification tasks.
    
    This model adapts the T5 architecture for clinical NLP tasks, supporting both:
    - Sequence-to-sequence tasks (e.g., radiology report generation)
    - Classification tasks (e.g., MedNLI natural language inference)
    
    Paper: Eric Lehman et al., "Do We Still Need Clinical Language Models?" (2023)
    
    Args:
        dataset (SampleDataset): The dataset to train the model.
        model_name (str): Name of the T5 variant (e.g., "clinical-t5-large").
        mode (str): Task type, either "generation" or "classification".
        max_seq_length (int): Maximum sequence length for input text. Default is 256.
    """
    
    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str = "clinical-t5-large",
        mode: str = "generation",
        max_seq_length: int = 256,
    ):
        super(ClinicalT5, self).__init__(dataset)
        self.model_name = model_name
        self.mode = mode
        self.max_seq_length = max_seq_length
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Validate task configuration
        if self.mode == "classification":
            assert len(self.label_keys) == 1, "Classification requires single label key"
            self.label_key = self.label_keys[0]
            self.label_space = dataset.output_processors[self.label_key].get_label_space()
        
        # For generation tasks, ensure proper prompt formatting
        if self.mode == "generation":
            assert len(self.feature_keys) == 1, "Generation requires single feature key"
            self.feature_key = self.feature_keys[0]

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation for ClinicalT5.
        
        Args:
            **kwargs: Input features and labels from the dataset.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - loss: Model loss
                - y_prob: Predicted probabilities
                - y_true: Ground truth labels
        """
        if self.mode == "classification":
            return self._forward_classification(**kwargs)
        else:
            return self._forward_generation(**kwargs)

    def _forward_classification(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Handles classification tasks (e.g., MedNLI)."""
        # Prepare inputs
        inputs = [
            f"mednli premise: {premise} hypothesis: {hypothesis}"
            for premise, hypothesis in zip(
                kwargs["sentence1"], kwargs["sentence2"]
            )
        ]
        
        # Tokenize inputs and labels
        tokenized_inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)
        
        labels = kwargs[self.label_key]
        tokenized_labels = self.tokenizer(
            [self.label_space[label] for label in labels],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8,  # Short for class labels
        ).input_ids.to(self.device)
        
        # Model forward pass
        outputs = self.model(
            input_ids=tokenized_inputs.input_ids,
            attention_mask=tokenized_inputs.attention_mask,
            labels=tokenized_labels,
        )
        
        return {
            "loss": outputs.loss,
            "y_prob": self._prepare_classification_probs(outputs.logits),
            "y_true": labels,
        }

    def _forward_generation(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Handles generation tasks (e.g., radiology report generation)."""
        # Prepare inputs with task prompt
        inputs = [
            f"generate radiology report: {findings}"
            for findings in kwargs[self.feature_key]
        ]
        
        # Tokenize inputs and labels
        tokenized_inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)
        
        tokenized_labels = self.tokenizer(
            kwargs[self.label_key],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).input_ids.to(self.device)
        
        # Model forward pass
        outputs = self.model(
            input_ids=tokenized_inputs.input_ids,
            attention_mask=tokenized_inputs.attention_mask,
            labels=tokenized_labels,
        )
        
        return {
            "loss": outputs.loss,
            "y_prob": outputs.logits,
            "y_true": tokenized_labels,
        }

    def _prepare_classification_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Converts logits to probabilities for classification tasks."""
        # Get probabilities by taking softmax over label tokens
        probs = torch.softmax(logits[:, 0, :], dim=-1)  # Use first token position
        return probs

    def predict(self, input_text: str, **kwargs) -> str:
        """Generates predictions for inference.
        
        Args:
            input_text (str): Input text to generate prediction for.
            **kwargs: Additional generation parameters.
            
        Returns:
            str: Generated output text.
        """
        input_text = f"generate radiology report: {input_text}" if self.mode == "generation" else input_text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
        ).to(self.device)
        
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=self.max_seq_length,
            **kwargs,
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)