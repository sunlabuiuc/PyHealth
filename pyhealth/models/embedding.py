from typing import Dict

import torch
import torch.nn as nn

from ..datasets import SampleDataset
from ..processors import (
    MultiHotProcessor,
    SequenceProcessor,
    TensorProcessor,
    TimeseriesProcessor,
)
from .base_model import BaseModel


class EmbeddingModel(BaseModel):
    """
    EmbeddingModel is responsible for creating embedding layers for different types of input data.

    This model automatically creates appropriate embedding transformations based on the processor type:
    
    - SequenceProcessor: Creates nn.Embedding for categorical sequences (e.g., diagnosis codes)
      Input: (batch, seq_len) with integer indices
      Output: (batch, seq_len, embedding_dim)
    
    - TimeseriesProcessor: Creates nn.Linear for time series features
      Input: (batch, seq_len, num_features)
      Output: (batch, seq_len, embedding_dim)
    
    - TensorProcessor: Creates nn.Linear for fixed-size numerical features
      Input: (batch, feature_size)
      Output: (batch, embedding_dim)
    
    - MultiHotProcessor: Creates nn.Linear for multi-hot encoded categorical features
      Input: (batch, num_categories) binary tensor
      Output: (batch, embedding_dim)
      Note: Converts sparse categorical representations to dense embeddings
    
    - Other processors with size(): Creates nn.Linear if processor reports a positive size
      Input: (batch, size)
      Output: (batch, embedding_dim)

    Attributes:
        dataset (SampleDataset): The dataset containing input processors.
        embedding_layers (nn.ModuleDict): A dictionary of embedding layers for each input field.
        embedding_dim (int): The target embedding dimension for all features.
    """

    def __init__(self, dataset: SampleDataset, embedding_dim: int = 128):
        """
        Initializes the EmbeddingModel with the given dataset and embedding dimension.

        Args:
            dataset (SampleDataset): The dataset containing input processors.
            embedding_dim (int): The dimension of the embedding space. Default is 128.
        """
        super().__init__(dataset)
        self.embedding_dim = embedding_dim
        self.embedding_layers = nn.ModuleDict()
        for field_name, processor in self.dataset.input_processors.items():
            if isinstance(processor, SequenceProcessor):
                vocab_size = len(processor.code_vocab)
                self.embedding_layers[field_name] = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_dim,
                    padding_idx=0,
                )
            elif isinstance(processor, TimeseriesProcessor):
                self.embedding_layers[field_name] = nn.Linear(
                    in_features=processor.size, out_features=embedding_dim
                )
            elif isinstance(processor, TensorProcessor):
                # For tensor processor, we need to determine the input size
                # from the first sample in the dataset
                sample_tensor = None
                for sample in dataset.samples:
                    if field_name in sample:
                        sample_tensor = processor.process(sample[field_name])
                        break
                if sample_tensor is not None:
                    input_size = (
                        sample_tensor.shape[-1] if sample_tensor.dim() > 0 else 1
                    )
                    self.embedding_layers[field_name] = nn.Linear(
                        in_features=input_size, out_features=embedding_dim
                    )
            elif isinstance(processor, MultiHotProcessor):
                # MultiHotProcessor produces fixed-size binary vectors
                # Use processor.size() to get the vocabulary size (num_categories)
                num_categories = processor.size()
                self.embedding_layers[field_name] = nn.Linear(
                    in_features=num_categories, out_features=embedding_dim
                )
            else:
                # Handle other processors with a size() method
                size_attr = getattr(processor, "size", None)
                if callable(size_attr):
                    size_value = size_attr()
                else:
                    size_value = size_attr
                
                if isinstance(size_value, int) and size_value > 0:
                    self.embedding_layers[field_name] = nn.Linear(
                        in_features=size_value, out_features=embedding_dim
                    )
                else:
                    # No valid size() method found - raise an error
                    raise ValueError(
                        f"Processor for field '{field_name}' (type: {type(processor).__name__}) "
                        f"does not have a valid size() method or it returned an invalid value. "
                        f"To use this processor with EmbeddingModel, it must either:\n"
                        f"  1. Be a recognized processor type (SequenceProcessor, TimeseriesProcessor, "
                        f"TensorProcessor, MultiHotProcessor), or\n"
                        f"  2. Implement a size() method that returns a positive integer representing "
                        f"the feature dimension."
                    )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute embeddings for the input data.

        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary of input tensors.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of embedded tensors.
        """
        embedded = {}
        for field_name, tensor in inputs.items():
            tensor = tensor.to(self.device)
            if field_name in self.embedding_layers:
                embedded[field_name] = self.embedding_layers[field_name](tensor)
            else:
                embedded[field_name] = tensor  # passthrough for continuous features
        return embedded

    def __repr__(self) -> str:
        """
        Returns a string representation of the EmbeddingModel.

        Returns:
            str: A string representation of the model.
        """
        return f"EmbeddingModel(embedding_layers={self.embedding_layers})"
