"""
Model: ClinicalBERT for Sentence-Level ICD Relevance Classification
Authors: Abhitej Bokka (abhitej2), Liam Shen (liams4)
Description: Uses Bio_ClinicalBERT to predict whether each sentence is relevant
based on future ICD codes. Supports CLS or average pooling.
"""

from pyhealth.models import BaseModel
from transformers import AutoModel
import torch.nn as nn
import torch


class ClinicalBertSentenceClassifier(BaseModel):
    """
    Sentence-level classifier using ClinicalBERT.
    Outputs a probability score for each sentence being ICD-relevant.

    Args:
        dataset: PyHealth-compatible dataset (or None during testing)
        feature_keys: List of input feature keys (e.g., ["sentences"])
        pooling_type: 'cls' (default) or 'avg' for sentence embedding aggregation
    """

    def __init__(self, dataset, feature_keys, pooling_type="cls"):
        super().__init__(dataset)
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.pooling_type = pooling_type
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for sentence classification.

        Args:
            input_ids: Tensor of token IDs [B, T]
            attention_mask: Binary mask tensor [B, T]

        Returns:
            Tensor of predicted probabilities [B]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling_type == "cls":
            pooled = outputs.last_hidden_state[:, 0]  # CLS token
        else:
            pooled = outputs.last_hidden_state.mean(dim=1)  # Avg pooling
        return torch.sigmoid(self.fc(pooled)).squeeze(-1)

