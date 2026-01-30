from __future__ import annotations

from typing import Dict, Any, Optional, Union
import os

import torch
import torch.nn as nn

from ..datasets import SampleDataset
from ..processors import (
    MultiHotProcessor,
    NestedFloatsProcessor,
    NestedSequenceProcessor,
    SequenceProcessor,
    StageNetProcessor,
    StageNetTensorProcessor,
    TensorProcessor,
    TimeseriesProcessor,
    DeepNestedSequenceProcessor,
    DeepNestedFloatsProcessor,
)
from .base_model import BaseModel


def _iter_text_vectors(
    path: str,
    embedding_dim: int,
    wanted_tokens: set[str],
    encoding: str = "utf-8",
) -> Dict[str, torch.Tensor]:
    """Loads word vectors from a text file (e.g., GloVe) for a subset of tokens.

    Expected format: one token per line followed by embedding_dim floats.

    This function reads the file line-by-line and only retains vectors for
    tokens present in `wanted_tokens`.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"pretrained embedding file not found: {path}")

    vectors: Dict[str, torch.Tensor] = {}
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # token + embedding_dim values
            if len(parts) < embedding_dim + 1:
                continue
            token = parts[0]
            if token not in wanted_tokens:
                continue
            try:
                vec = torch.tensor(
                    [float(x) for x in parts[1 : embedding_dim + 1]],
                    dtype=torch.float,
                )
            except ValueError:
                continue
            vectors[token] = vec
    return vectors


def init_embedding_with_pretrained(
    embedding: nn.Embedding,
    code_vocab: Dict[Any, int],
    pretrained_path: str,
    embedding_dim: int,
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
    normalize: bool = False,
    freeze: bool = False,
) -> int:
    """Initializes an nn.Embedding from a pretrained text-vector file.

    Tokens not found in the pretrained file are left as the module's existing
    random initialization.

    Returns:
        int: number of tokens successfully initialized from the file.
    """

    # Build wanted token set (stringified)
    vocab_tokens = {str(t) for t in code_vocab.keys()}
    vectors = _iter_text_vectors(pretrained_path, embedding_dim, vocab_tokens)

    loaded = 0
    with torch.no_grad():
        for tok, idx in code_vocab.items():
            tok_s = str(tok)
            if tok_s in vectors:
                vec = vectors[tok_s]
                if normalize:
                    vec = vec / (vec.norm(p=2) + 1e-12)
                embedding.weight[idx].copy_(vec)
                loaded += 1

        # Ensure pad row is zero
        if pad_token in code_vocab:
            embedding.weight[code_vocab[pad_token]].zero_()
        # If embedding has a padding_idx, keep it consistent
        if embedding.padding_idx is not None:
            embedding.weight[embedding.padding_idx].zero_()

    if freeze:
        embedding.weight.requires_grad_(False)

    return loaded

class EmbeddingModel(BaseModel):
    """
    EmbeddingModel is responsible for creating embedding layers for different types of input data.

    This model automatically creates appropriate embedding transformations based on the processor type:

    - SequenceProcessor: nn.Embedding
        Input: (batch, seq_len)
        Output: (batch, seq_len, embedding_dim)

    - NestedSequenceProcessor: nn.Embedding
        Input: (batch, num_visits, max_codes_per_visit)
        Output: (batch, num_visits, max_codes_per_visit, embedding_dim)

    - DeepNestedSequenceProcessor: nn.Embedding
        Input: (batch, num_groups, num_visits, max_codes_per_visit)
        Output: (batch, num_groups, num_visits, max_codes_per_visit, embedding_dim)

    - TimeseriesProcessor / NestedFloatsProcessor / DeepNestedFloatsProcessor / StageNetTensorProcessor:
        nn.Linear over the last dimension
        Input: (..., size)
        Output: (..., embedding_dim)

    - TensorProcessor: nn.Linear (size inferred from first sample)

    - MultiHotProcessor: nn.Linear over multi-hot vector
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        pretrained_emb_path: Optional[Union[str, Dict[str, str]]] = None,
        freeze_pretrained: bool = False,
        normalize_pretrained: bool = False,
    ):
        super().__init__(dataset)
        self.embedding_dim = embedding_dim
        self.embedding_layers = nn.ModuleDict()

        for field_name, processor in self.dataset.input_processors.items():
            # Deep categorical: use special module that collapses last dim to embedding_dim

            # Regular categorical sequences -> nn.Embedding (adds embedding dim)
            if isinstance(
                processor,
                (
                    SequenceProcessor,
                    StageNetProcessor,
                    NestedSequenceProcessor,
                    DeepNestedSequenceProcessor
                ),
            ):
                vocab_size = len(processor.code_vocab)

                # For NestedSequenceProcessor and DeepNestedSequenceProcessor, don't use padding_idx
                # because empty visits/groups need non-zero embeddings.
                if isinstance(processor, (NestedSequenceProcessor, DeepNestedSequenceProcessor)):
                    self.embedding_layers[field_name] = nn.Embedding(
                        num_embeddings=vocab_size,
                        embedding_dim=embedding_dim,
                        padding_idx=None,
                    )
                else:
                    self.embedding_layers[field_name] = nn.Embedding(
                        num_embeddings=vocab_size,
                        embedding_dim=embedding_dim,
                        padding_idx=0,
                    )

                # Optional pretrained initialization (e.g., GloVe).
                if pretrained_emb_path is not None:
                    if isinstance(pretrained_emb_path, str):
                        path = pretrained_emb_path
                    else:
                        path = pretrained_emb_path.get(field_name)
                    if path:
                        init_embedding_with_pretrained(
                            self.embedding_layers[field_name],
                            processor.code_vocab,
                            path,
                            embedding_dim=embedding_dim,
                            normalize=normalize_pretrained,
                            freeze=freeze_pretrained,
                        )

            # Numeric features (including deep nested floats) -> nn.Linear over last dim
            elif isinstance(
                processor,
                (
                    TimeseriesProcessor,
                    StageNetTensorProcessor,
                    NestedFloatsProcessor,
                    DeepNestedFloatsProcessor,
                ),
            ):
                # Assuming processor.size() returns the last-dim size
                in_features = processor.size()
                self.embedding_layers[field_name] = nn.Linear(
                    in_features=in_features, out_features=embedding_dim
                )

            elif isinstance(processor, TensorProcessor):
                # Infer size from first sample
                sample_tensor = None
                for sample in dataset:
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
                num_categories = processor.size()
                self.embedding_layers[field_name] = nn.Linear(
                    in_features=num_categories, out_features=embedding_dim
                )

            else:
                print(
                    "Warning: No embedding created for field due to lack of compatible processor:",
                    field_name,
                )

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                output_mask: bool = False
                ) -> Dict[str, torch.Tensor] | tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        
        embedded: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {} if output_mask else None
        
        for field_name, tensor in inputs.items():
            processor = self.dataset.input_processors.get(field_name, None)
            
            if field_name not in self.embedding_layers:
                # No embedding layer -> passthrough
                embedded[field_name] = tensor
                continue
            
            tensor = tensor.to(self.device)            
            embedded[field_name] = self.embedding_layers[field_name](tensor)
            
            if output_mask:
                # Generate a mask for this field
                if hasattr(processor, "code_vocab"):
                    pad_idx = processor.code_vocab.get("<pad>", 0)
                else:
                    pad_idx = 0
                    
                masks[field_name] = (tensor != pad_idx)
        
        if output_mask:
            return embedded, masks
        else:
            return embedded

    def __repr__(self) -> str:
        return f"EmbeddingModel(embedding_layers={self.embedding_layers})"
