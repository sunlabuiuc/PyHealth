"""
DescEmbBERT: BERT-based text encoder for medical descriptions

Contributor:
Valentin Burkin
vburkin2 (UIUC student)

Paper: Unifying Heterogeneous Electronic Health Records Systems via Text-Based Code Embedding
Authors: Kyunghoon Hur, Jiyoung Lee, Jungwoo Oh, Wesley Price, Young-Hak Kim, Edward Choi
Link: https://arxiv.org/abs/2108.03625

This implementation is a reproduction of the DescEmbBERT model from the paper.
The model uses a pre-trained BERT model to encode medical text descriptions,
supporting various BERT model variants and tasks including masked language modeling (MLM) and classification.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from ..datasets import SampleDataset
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class IdentityLayer(nn.Module):
    """A simple identity layer that returns the input unchanged."""
    
    def forward(self, source):
        return source


class SubwordInputLayer(nn.Module):
    """Embedding layer for subword tokens."""
    
    def __init__(self, enc_embed_dim: int):
        """
        Initialize the subword input layer.
        
        Args:
            enc_embed_dim (int): The embedding dimension for the tokens.
        """
        super().__init__()
        # Fixed vocabulary size for BERT
        index_size = 28996
        self.embedding = nn.Embedding(index_size, enc_embed_dim, padding_idx=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token indices to embeddings.
        
        Args:
            x (torch.Tensor): Input token indices of shape (batch_size, seq_length).
            
        Returns:
            torch.Tensor: Token embeddings of shape (batch_size, seq_length, enc_embed_dim).
        """
        return self.embedding(x)


class DescEmbBERT(BaseModel):
    """
    BERT-based text encoder for embedding medical descriptions.
    
    This model uses a pre-trained BERT model to encode medical text descriptions.
    It supports various BERT model variants and can be used for different tasks
    including masked language modeling (MLM) and classification.
    
    Args:
        dataset (SampleDataset): The dataset to train the model.
        bert_model (str): The name of the BERT model to use.
        pred_embed_dim (int): The dimension of the prediction embeddings.
        init_bert_params (bool): Whether to initialize BERT parameters.
        init_bert_params_with_freeze (bool): Whether to initialize BERT parameters and freeze them.
        task (str): The task to perform ('mlm' or 'classification').
        value_mode (str): The value mode to use ('DSVA_DPE' or 'standard').
        load_pretrained_weights (bool): Whether to load pre-trained weights.
        model_path (str): The path to the pre-trained model weights.
    """
    
    # Dictionary mapping model names to their HuggingFace paths and embedding dimensions
    BERT_MODEL_CONFIGS = {
        'bert': ["bert-base-uncased", 768],
        'bio_clinical_bert': ["emilyalsentzer/Bio_ClinicalBERT", 768],
        'pubmed_bert': ["microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 768],
        'blue_bert': ["bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12", 768],
        'bio_bert': ["dmis-lab/biobert-v1.1", 768],
        'bert_tiny': ["google/bert_uncased_L-2_H-128_A-2", 128],
        'bert_mini': ["google/bert_uncased_L-4_H-256_A-4", 256],
        'bert_small': ["google/bert_uncased_L-4_H-512_A-8", 512]
    }
    
    def __init__(
        self,
        dataset: SampleDataset,
        bert_model: str = 'bio_clinical_bert',
        pred_embed_dim: int = 768,
        init_bert_params: bool = True,
        init_bert_params_with_freeze: bool = False,
        task: str = 'classification',
        value_mode: str = 'standard',
        load_pretrained_weights: bool = False,
        model_path: Optional[str] = None
    ):
        """
        Initialize the DescEmbBERT model.
        
        Args:
            dataset (SampleDataset): The dataset to train the model.
            bert_model (str): The name of the BERT model to use.
            pred_embed_dim (int): The dimension of the prediction embeddings.
            init_bert_params (bool): Whether to initialize BERT parameters.
            init_bert_params_with_freeze (bool): Whether to initialize BERT parameters and freeze them.
            task (str): The task to perform ('mlm' or 'classification').
            value_mode (str): The value mode to use ('DSVA_DPE' or 'standard').
            load_pretrained_weights (bool): Whether to load pre-trained weights.
            model_path (str): The path to the pre-trained model weights.
        """
        super(DescEmbBERT, self).__init__(dataset)
        self.pred_embed_dim = pred_embed_dim
        self.task = task
        self.value_mode = value_mode
        
        # Get model configuration based on the specified BERT model
        if bert_model not in self.BERT_MODEL_CONFIGS:
            raise ValueError(f"Unsupported BERT model: {bert_model}. Supported models: {list(self.BERT_MODEL_CONFIGS.keys())}")
        
        model_name, hidden_size = self.BERT_MODEL_CONFIGS[bert_model]
        
        # Initialize BERT model based on configuration
        if not init_bert_params:
            # Random initialization
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(config)
        elif init_bert_params_with_freeze:
            # Load pre-trained with frozen parameters
            with torch.no_grad():
                self.model = AutoModel.from_pretrained(model_name)
                # Freeze all parameters
                for param in self.model.parameters():
                    param.requires_grad = False
        else:
            # Load pre-trained with trainable parameters
            self.model = AutoModel.from_pretrained(model_name)
            self.model = nn.ModuleList([self.model, IdentityLayer()])
        
        # Initialize MLM projection if needed
        self.mlm_proj = None
        if task == "mlm":
            self.mlm_proj = nn.Linear(hidden_size, 28996)  # BERT vocabulary size
        
        # Projection layer for encoding
        self.post_encode_proj = nn.Linear(hidden_size, self.pred_embed_dim)
        
        # Resize token type embeddings for DSVA_DPE mode
        if value_mode == 'DSVA_DPE':
            old_token_type_embeddings = self.model.embeddings.token_type_embeddings
            new_token_type_embeddings = self.model._get_resized_embeddings(old_token_type_embeddings, 28)
            self.model.embeddings.token_type_embeddings = new_token_type_embeddings
        
        # Load pre-trained weights if specified
        if load_pretrained_weights and model_path:
            self._load_pretrained_weights(model_path)
    
    def _load_pretrained_weights(self, model_path: str) -> None:
        """
        Load pre-trained weights from a checkpoint.
        
        Args:
            model_path (str): The path to the pre-trained model weights.
        """
        assert model_path, "Model path must be provided"
        logger.info(f"Preparing to load pre-trained checkpoint {model_path}")
        
        state_dict = torch.load(model_path)['model_state_dict']
        # Filter out MLM-related weights
        state_dict = {k: v for k, v in state_dict.items() if 'mlm' not in k}
        
        self.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded checkpoint {model_path}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the BERT encoder.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_length).
            token_type_ids (torch.Tensor): Token type IDs of shape (batch_size, seq_length).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_length).
            **kwargs: Additional keyword arguments.
            
        Returns:
            torch.Tensor: The model output.
                - For MLM task: Shape (batch_size, seq_length, vocab_size)
                - For classification task: Shape (batch_size, pred_embed_dim)
        """
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
        
        bsz, _, word_max_len = input_ids.shape
        
        # Prepare inputs for BERT
        bert_inputs = {
            "input_ids": input_ids.view(-1, word_max_len),
            "token_type_ids": token_type_ids.view(-1, word_max_len),
            "attention_mask": attention_mask.view(-1, word_max_len)
        }
        
        # Get BERT outputs
        if isinstance(self.model, nn.ModuleList):
            bert_outputs = self.model[0](**bert_inputs)
        else:
            bert_outputs = self.model(**bert_inputs)
        
        # For MLM task, return MLM projections
        if self.mlm_proj:
            mlm_output = self.mlm_proj(bert_outputs[0])  # (B x S, W, H) -> (B x S, W, Bert-vocab)
            return mlm_output
        
        # For other tasks, return CLS token embedding with projection
        net_output = self.post_encode_proj(
            bert_outputs[0][:, 0, :]  # Use CLS token
        ).view(bsz, -1, self.pred_embed_dim)
        
        return net_output


# Example test case
if __name__ == "__main__":
    import unittest
    from ..datasets import SampleDataset
    
    class DummyDataset(SampleDataset):
        def __init__(self):
            self.input_schema = {
                "input_ids": "token",
                "token_type_ids": "token",
                "attention_mask": "token"
            }
            self.output_schema = {
                "label": "binary"
            }
            self.output_processors = {
                "label": type("DummyProcessor", (), {"size": lambda: 1})()
            }
    
    class TestDescEmbBERT(unittest.TestCase):
        def test_forward_pass(self):
            # Create a dummy dataset
            dataset = DummyDataset()
            
            # Create the model
            model = DescEmbBERT(
                dataset=dataset,
                bert_model="bert_tiny",  # Use a small model for testing
                pred_embed_dim=128,
                init_bert_params=True,
                task="classification"
            )
            
            # Create dummy inputs
            batch_size = 2
            seq_length = 10
            input_ids = torch.randint(0, 1000, (batch_size, seq_length))
            token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
            
            # Forward pass
            output = model(input_ids, token_type_ids, attention_mask)
            
            # Check output shape
            self.assertEqual(output.shape, (batch_size, 1, 128))
            
            # Test MLM task
            model_mlm = DescEmbBERT(
                dataset=dataset,
                bert_model="bert_tiny",
                pred_embed_dim=128,
                init_bert_params=True,
                task="mlm"
            )
            
            output_mlm = model_mlm(input_ids, token_type_ids, attention_mask)
            
            # Check MLM output shape
            self.assertEqual(output_mlm.shape, (batch_size, seq_length, 28996))
    
    unittest.main()
