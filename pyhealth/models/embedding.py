from typing import Dict

import torch
import torch.nn as nn

from ..datasets import SampleDataset
from ..processors import SequenceProcessor, TimeseriesProcessor, TensorProcessor
from .base_model import BaseModel


class EmbeddingModel(BaseModel):
    """
    EmbeddingModel is responsible for creating embedding layers for different types of input data.

    Attributes:
        dataset (SampleDataset): The dataset containing input processors.
        embedding_layers (nn.ModuleDict): A dictionary of embedding layers for each input field.
    """

    def __init__(self, dataset: SampleDataset, embedding_dim: int = 128):
        """
        Initializes the EmbeddingModel with the given dataset and embedding dimension.

        Args:
            dataset (SampleDataset): The dataset containing input processors.
            embedding_dim (int): The dimension of the embedding space. Default is 128.
        """
        super().__init__(dataset)
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
            if field_name in self.embedding_layers:
                tensor = tensor.to(self.device)
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
