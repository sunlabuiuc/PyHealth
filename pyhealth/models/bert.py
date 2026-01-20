"""BERT and BioBERT models for text classification in healthcare.

This module provides BERT-based models specifically designed for healthcare NLP tasks.
It supports the following pre-trained models:
- bert-base-uncased, bert-base-cased
- dmis-lab/biobert-v1.1 (BioBERT)
- Any other HuggingFace BERT-compatible model
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


# Supported BERT models with shorthand aliases
BIOMEDICAL_MODELS = {
    "bert-base-uncased": "bert-base-uncased",
    "bert-base-cased": "bert-base-cased",
    "biobert": "dmis-lab/biobert-v1.1",
}


class BERTLayer(nn.Module):
    """BERT layer for encoding text into fixed-size representations.
    
    This layer wraps HuggingFace's BERT models and can be used standalone or
    as part of a larger model. It supports different pooling strategies and
    can optionally freeze the encoder for feature extraction.
    
    Args:
        model_name: Name of the pre-trained BERT model. Can be a HuggingFace
            model name or a shorthand from BIOMEDICAL_MODELS.
        pooling: Pooling strategy for obtaining sentence embeddings.
            - "cls": Use the [CLS] token embedding (default)
            - "mean": Mean of all token embeddings (excluding padding)
            - "max": Max pooling of all token embeddings (excluding padding)
        max_length: Maximum sequence length for tokenization. Default is 512.
        dropout: Dropout probability applied after the encoder. Default is 0.1.
        freeze_encoder: Whether to freeze the BERT encoder weights. Default is False.
        freeze_layers: Number of encoder layers to freeze from the bottom. 
            Only used if freeze_encoder is False. Default is 0.

    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pooling: Literal["cls", "mean", "max"] = "cls",
        max_length: int = 512,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        freeze_layers: int = 0,
    ):
        super(BERTLayer, self).__init__()
        
        # Resolve model name if using shorthand
        self.model_name = BIOMEDICAL_MODELS.get(model_name.lower(), model_name)
        self.pooling = pooling
        self.max_length = max_length
        self.freeze_encoder = freeze_encoder
        self.freeze_layers = freeze_layers
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        
        # Get hidden size from config
        self.hidden_size = self.config.hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Apply freezing if requested
        if freeze_encoder:
            self._freeze_all_layers()
        elif freeze_layers > 0:
            self._freeze_bottom_layers(freeze_layers)
    
    def _freeze_all_layers(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def _freeze_bottom_layers(self, num_layers: int):
        """Freeze the embeddings and bottom N encoder layers.
        
        Args:
            num_layers: Number of layers to freeze from the bottom.
        """
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze specified number of encoder layers
        if hasattr(self.encoder, 'encoder'):
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def get_output_size(self) -> int:
        """Returns the output embedding dimension."""
        return self.hidden_size
    
    def forward(
        self,
        texts: Union[str, List[str]],
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode texts into fixed-size embeddings.
        
        Args:
            texts: A single text string or list of text strings.
            return_attention: Whether to return attention weights. Default is False.
        
        Returns:
            embeddings: Tensor of shape [batch_size, hidden_size].
            attention_weights: (Optional) Tensor of attention weights if
                return_attention is True.
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Move to same device as model
        encoding = {k: v.to(self.encoder.device) for k, v in encoding.items()}
        
        # Forward pass through encoder
        outputs = self.encoder(
            **encoding,
            output_attentions=return_attention,
        )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        attention_mask = encoding["attention_mask"]  # [batch, seq_len]
        
        # Apply pooling strategy
        if self.pooling == "cls":
            embeddings = hidden_states[:, 0, :]  # [batch, hidden]
        elif self.pooling == "mean":
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask  # [batch, hidden]
        elif self.pooling == "max":
            # Max pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states[~mask_expanded.bool()] = float('-inf')
            embeddings = torch.max(hidden_states, dim=1)[0]  # [batch, hidden]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        if return_attention:
            return embeddings, outputs.attentions
        return embeddings


class BERT(BaseModel):
    """BERT model for text classification in healthcare applications.
    
    This model uses pre-trained BERT models (including BioBERT) for text 
    classification tasks. It supports fine-tuning the entire model or using 
    it as a feature extractor with frozen weights.
    
    The model expects text input and produces classification outputs for
    binary, multiclass, or multilabel tasks.
    
    Note:
        This model is designed for single text input features. For multiple
        text features, each feature should have its own text field in the
        dataset schema.
    
    Args:
        dataset: SampleDataset containing the training data. Used to infer
            vocabulary size, label space, and feature keys.
        model_name: Name of the pre-trained BERT model. Can be a HuggingFace
            model identifier or a shorthand name (e.g., "biobert").
            See BIOMEDICAL_MODELS for available shorthands. Default is "bert-base-uncased".
        pooling: Pooling strategy for sentence embeddings. Options:
            - "cls": Use [CLS] token embedding (default, recommended for classification)
            - "mean": Mean of all token embeddings
            - "max": Max pooling of all token embeddings
        max_length: Maximum sequence length for tokenization. Default is 512.
            Longer sequences will be truncated.
        dropout: Dropout probability applied to encoder output. Default is 0.1.
        freeze_encoder: If True, freeze all BERT encoder weights and use it
            as a fixed feature extractor. Default is False.
        freeze_layers: Number of encoder layers to freeze from the bottom.
            Only effective if freeze_encoder is False. Useful for gradual
            unfreezing during fine-tuning. Default is 0.
        classifier_hidden_dim: Hidden dimension of the classification head.
            If None, a single linear layer is used. Default is None.
    
    Attributes:
        model_name: Resolved HuggingFace model name.
        feature_key: Key for the text feature in input data.
        label_key: Key for the label in input data.
        encoder: The BERTLayer encoder.
        classifier: Classification head.
    """
    
    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str = "bert-base-uncased",
        pooling: Literal["cls", "mean", "max"] = "cls",
        max_length: int = 512,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        freeze_layers: int = 0,
        classifier_hidden_dim: Optional[int] = None,
    ):
        super(BERT, self).__init__(dataset=dataset)
        
        # Validate feature keys - expect single text feature
        assert len(self.feature_keys) == 1, (
            f"BERT model expects a single text feature, got {len(self.feature_keys)} features: "
            f"{self.feature_keys}. For multiple text features, consider using separate "
            "BERT encoders or concatenating texts."
        )
        self.feature_key = self.feature_keys[0]
        
        # Validate label keys
        assert len(self.label_keys) == 1, (
            f"BERT model expects a single label key, got {len(self.label_keys)}: {self.label_keys}"
        )
        self.label_key = self.label_keys[0]
        
        # Store configuration
        self.model_name = BIOMEDICAL_MODELS.get(model_name.lower(), model_name)
        self.pooling = pooling
        self.max_length = max_length
        self.freeze_encoder = freeze_encoder
        self.freeze_layers = freeze_layers
        self.classifier_hidden_dim = classifier_hidden_dim
        
        # Initialize encoder
        self.encoder = BERTLayer(
            model_name=model_name,
            pooling=pooling,
            max_length=max_length,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
            freeze_layers=freeze_layers,
        )
        
        # Get output size from dataset
        output_size = self.get_output_size()
        hidden_size = self.encoder.hidden_size
        
        # Build classifier head
        if classifier_hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, output_size),
            )
        else:
            self.classifier = nn.Linear(hidden_size, output_size)
    
    def get_encoder_parameters(self):
        """Returns parameters of the BERT encoder.
        
        Useful for applying different learning rates to encoder and classifier.
        
        Returns:
            Iterator of encoder parameters.
        """
        return self.encoder.parameters()
    
    def get_classifier_parameters(self):
        """Returns parameters of the classification head.
        
        Useful for applying different learning rates to encoder and classifier.
        
        Returns:
            Iterator of classifier parameters.
        """
        return self.classifier.parameters()
    
    def forward(
        self,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.
        
        Args:
            **kwargs: Keyword arguments containing feature and label keys.
                Must include the text feature key and label key defined in
                the dataset schema.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - loss: Scalar loss tensor
                - y_prob: Predicted probabilities [batch_size, num_classes]
                - y_true: Ground truth labels [batch_size, num_classes]
                - logit: Raw logits before activation [batch_size, num_classes]
                - embed (optional): Embeddings if embed=True in kwargs
        """
        # Get text input
        texts = kwargs[self.feature_key]
        
        # Handle different input types
        if isinstance(texts, torch.Tensor):
            # Convert tensor to list of strings if needed
            texts = [str(t) for t in texts.tolist()]
        elif not isinstance(texts, list):
            texts = [texts]
        
        # Encode texts
        embeddings = self.encoder(texts)  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(embeddings)  # [batch_size, output_size]
        
        # Get labels and compute loss
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        
        # Compute probabilities
        y_prob = self.prepare_y_prob(logits)
        
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        
        # Optionally return embeddings
        if kwargs.get("embed", False):
            results["embed"] = embeddings
        
        return results
    
    def encode(
        self,
        texts: Union[str, List[str]],
    ) -> torch.Tensor:
        """Encode texts into embeddings without classification.
        
        This method is useful for getting text representations for downstream
        tasks or analysis.
        
        Args:
            texts: Single text string or list of text strings.
        
        Returns:
            torch.Tensor: Embeddings of shape [batch_size, hidden_size].
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.encoder(texts)
        return embeddings
    
    def __repr__(self) -> str:
        return (
            f"BERT(\n"
            f"  model_name={self.model_name},\n"
            f"  pooling={self.pooling},\n"
            f"  max_length={self.max_length},\n"
            f"  freeze_encoder={self.freeze_encoder},\n"
            f"  freeze_layers={self.freeze_layers},\n"
            f"  hidden_size={self.encoder.hidden_size},\n"
            f"  output_size={self.get_output_size()},\n"
            f")"
        )

