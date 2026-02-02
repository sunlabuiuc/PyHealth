"""Text embedding module for multimodal PyHealth pipelines.

This module provides a Transformer-based text encoder for clinical/medical text.
It is designed to integrate with PyHealth's multimodal fusion architecture.

Key Features:
    - Uses pretrained medical language models (default: Bio_ClinicalBERT)
    - Splits long texts into 128-token chunks for consistent encoding
    - Projects embeddings to a shared dimension for multimodal concatenation
    - Returns attention masks compatible with PyHealth's TransformerLayer

Dependencies:
    - transformers >= 4.20.0 (pinned as ~=4.53.2 in pyproject.toml)
    - torch

Example:
    >>> from pyhealth.models.text_embedding import TextEmbedding
    >>> encoder = TextEmbedding(embedding_dim=256)
    >>> embeddings, mask = encoder(["Patient has fever.", "Follow-up."])
    >>> embeddings.shape  # [2, T, 256]
"""

from typing import List, Optional, Tuple, Union
import logging
import warnings

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


logger = logging.getLogger(__name__)


class TextEmbedding(nn.Module):
    """Encodes clinical text into embeddings for multimodal fusion.

    This module wraps a pretrained Hugging Face transformer (default:
    Bio_ClinicalBERT) and handles the complexities of encoding long clinical
    notes that exceed typical transformer context windows.

    Input/Output:
        Input:  List[str] or str - raw text strings
        Output: (embeddings, mask) tuple or just embeddings tensor

        embeddings: torch.Tensor of shape [batch, seq_len, embedding_dim]
        mask: torch.BoolTensor of shape [batch, seq_len] where True=valid

    Chunking Behavior:
        Long texts are split into non-overlapping chunks. Each chunk:
        1. Contains at most (chunk_size - 2) content tokens
        2. Gets [CLS] prepended and [SEP] appended
        3. Is encoded independently by the transformer
        4. Embeddings are concatenated along the sequence dimension

        Example: A 300-token note with chunk_size=128 becomes 3 chunks:
            Chunk 1: [CLS] + tokens[0:126] + [SEP]   = 128 tokens
            Chunk 2: [CLS] + tokens[126:252] + [SEP] = 128 tokens  
            Chunk 3: [CLS] + tokens[252:300] + [SEP] = 50 tokens

    Pooling Modes:
        - "none" (default): All token embeddings → [B, T, E'] where T = total tokens
        - "cls": One [CLS] embedding per chunk → [B, C, E'] where C = num chunks
        - "mean": Mean-pooled embedding per chunk → [B, C, E']

    Design Decisions:

        return_mask parameter (backward compatibility):
            Earlier versions returned only embeddings. Adding mask return as
            default would break existing callers. The `return_mask=True` default
            provides the mask for new code, while `return_mask=False` preserves
            old behavior.

        max_chunks parameter (performance guardrail):
            Clinical notes can be extremely long (10,000+ tokens). Without a
            limit, this causes:
            - Memory explosion (O(chunks * chunk_size * hidden_dim))
            - Silent OOMs in production
            - Unexpectedly slow inference

            Default max_chunks=64 caps output at ~8K tokens. A UserWarning
            alerts when truncation occurs, allowing users to:
            - Increase max_chunks if memory permits
            - Pre-summarize long notes
            - Use chunk-level pooling instead

        freeze parameter:
            Medical transformers are expensive to fine-tune. For multimodal
            fusion where the text encoder is just one component, freezing
            prevents catastrophic forgetting and reduces GPU memory by ~50%.

    Mask Convention:
        Returns torch.bool tensor matching PyHealth's TransformerLayer:
        - True (or 1) = valid token position
        - False (or 0) = padding position

        TransformerLayer uses: scores.masked_fill(mask == 0, -1e9)
        So True positions are attended, False positions are masked out.

    Args:
        model_name: Hugging Face model identifier.
            Default: "emilyalsentzer/Bio_ClinicalBERT" (clinical domain)
            Alternatives: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        embedding_dim: Output embedding dimension (E'). Default: 128.
            Should match other modality encoders for concatenation.
        chunk_size: Tokens per chunk including [CLS]/[SEP]. Default: 128.
            Matches the "128 token text bits" from the multimodal study.
        max_chunks: Maximum chunks to keep. Default: 64.
            Set to None for unlimited (use with caution on long texts).
        pooling: How to aggregate token embeddings. Default: "none".
            "none" = all tokens, "cls" = [CLS] per chunk, "mean" = mean per chunk.
        freeze: If True, freeze transformer weights. Default: False.
            Recommended for multimodal fusion to prevent overfitting.
        return_mask: If True, return (embeddings, mask) tuple. Default: True.
            Set to False for backward compatibility with single-tensor callers.

    Example:
        Basic usage with default parameters:

        >>> encoder = TextEmbedding(embedding_dim=256)
        >>> texts = ["Patient presents with chest pain.", "Routine checkup."]
        >>> embeddings, mask = encoder(texts)
        >>> embeddings.shape
        torch.Size([2, 12, 256])  # [batch=2, tokens, dim=256]
        >>> mask.shape
        torch.Size([2, 12])  # [batch=2, tokens]

        Using chunk-level pooling for efficiency:

        >>> encoder = TextEmbedding(pooling="cls", embedding_dim=128)
        >>> long_note = "..." * 1000  # Very long clinical note
        >>> emb, mask = encoder([long_note])
        >>> emb.shape  # [1, num_chunks, 128] instead of [1, thousands, 128]

        Backward-compatible single tensor return:

        >>> encoder = TextEmbedding(return_mask=False)
        >>> embeddings = encoder(["Test"])  # Just tensor, no tuple
    """

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embedding_dim: int = 128,
        chunk_size: int = 128,
        max_chunks: Optional[int] = 64,
        pooling: str = "none",
        freeze: bool = False,
        return_mask: bool = True,
    ):
        """Initialize the text embedding module.

        Loads the pretrained tokenizer and transformer model from Hugging Face.
        Creates a projection layer to map transformer hidden size to embedding_dim.

        Raises:
            ValueError: If pooling is not one of "none", "cls", "mean".
            ValueError: If chunk_size < 4 (need room for [CLS], [SEP], content).
        """
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.pooling = pooling
        self.return_mask = return_mask

        if pooling not in ("none", "cls", "mean"):
            raise ValueError(f"pooling must be 'none', 'cls', or 'mean', got {pooling}")

        if chunk_size < 4:
            raise ValueError(f"chunk_size must be >= 4, got {chunk_size}")

        # Load tokenizer and model from Hugging Face
        # First use downloads ~420MB to HF_HOME or ~/.cache/huggingface/
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Projection: transformer hidden_size (e.g., 768) → embedding_dim (e.g., 128)
        # This aligns text embeddings with other modalities in a shared E' space
        self.fc = nn.Linear(self.transformer.config.hidden_size, embedding_dim)

    def _chunk_and_encode(
        self, text: str, device: torch.device
    ) -> torch.Tensor:
        """Tokenize, chunk, and encode a single text string.

        This is the core encoding logic. For a single text:
        1. Tokenize without truncation to get all tokens
        2. Split into chunks of (chunk_size - 2) tokens each
        3. Add [CLS] and [SEP] to each chunk
        4. Batch-encode all chunks through the transformer
        5. Apply the projection layer
        6. Return embeddings based on pooling mode

        Args:
            text: A single text string to encode. Can be empty.
            device: torch.device to place output tensors on.

        Returns:
            torch.Tensor: Encoded embeddings.
                - pooling="none": shape [total_tokens, embedding_dim]
                - pooling="cls": shape [num_chunks, embedding_dim]
                - pooling="mean": shape [num_chunks, embedding_dim]

        Side Effects:
            Emits UserWarning if text exceeds max_chunks and gets truncated.
        """
        # Step 1: Tokenize without truncation to get ALL tokens
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,  # We add [CLS]/[SEP] manually per chunk
            return_tensors=None,       # Return Python list, not tensor
        )
        input_ids = tokens["input_ids"]

        # Step 2: Split into chunks, reserving 2 tokens for [CLS] and [SEP]
        effective_chunk_size = self.chunk_size - 2
        chunks = []
        for i in range(0, len(input_ids), effective_chunk_size):
            chunk_ids = input_ids[i : i + effective_chunk_size]
            # Add special tokens: [CLS] content... [SEP]
            chunk_ids = [self.tokenizer.cls_token_id] + chunk_ids + [self.tokenizer.sep_token_id]
            chunks.append(chunk_ids)

        # Handle empty text edge case
        if not chunks:
            chunks = [[self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]]

        # Step 3: Apply max_chunks limit (performance guardrail)
        # Rationale: Clinical notes can be 10K+ tokens. Without a cap:
        # - Memory usage explodes (each chunk needs transformer forward pass)
        # - Silent OOMs in production environments
        # - Inference time becomes unpredictable
        # We warn rather than silently truncate so users can adjust.
        if self.max_chunks is not None and len(chunks) > self.max_chunks:
            original_chunks = len(chunks)
            chunks = chunks[: self.max_chunks]
            warnings.warn(
                f"Text produced {original_chunks} chunks, truncated to {self.max_chunks}. "
                f"Consider increasing max_chunks or summarizing input.",
                UserWarning,
            )

        # Step 4: Pad chunks to uniform length for batched encoding
        max_len = max(len(c) for c in chunks)
        padded = []
        attention_masks = []
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0  # Fallback for tokenizers without explicit pad token

        for c in chunks:
            pad_len = max_len - len(c)
            attention_masks.append([1] * len(c) + [0] * pad_len)
            padded.append(c + [pad_token_id] * pad_len)

        input_ids_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long, device=device)

        # Step 5: Encode all chunks through transformer in one forward pass
        outputs = self.transformer(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
        )
        hidden_states = outputs.last_hidden_state  # [num_chunks, max_len, hidden_size]

        # Step 6: Project from hidden_size to embedding_dim
        projected = self.fc(hidden_states)  # [num_chunks, max_len, embedding_dim]

        # Step 7: Apply pooling strategy
        if self.pooling == "cls":
            # Return [CLS] token (position 0) from each chunk
            return projected[:, 0, :]  # [num_chunks, embedding_dim]

        elif self.pooling == "mean":
            # Mean pool over non-padding positions
            mask = attention_mask_tensor.unsqueeze(-1).float()  # [num_chunks, max_len, 1]
            summed = (projected * mask).sum(dim=1)              # [num_chunks, embedding_dim]
            lengths = mask.sum(dim=1).clamp(min=1)              # [num_chunks, 1]
            return summed / lengths                              # [num_chunks, embedding_dim]

        else:  # pooling == "none"
            # Concatenate all non-padding tokens from all chunks
            all_embeddings = []
            for i, mask in enumerate(attention_masks):
                valid_len = sum(mask)
                all_embeddings.append(projected[i, :valid_len, :])
            return torch.cat(all_embeddings, dim=0)  # [total_tokens, embedding_dim]

    def forward(
        self, text: Union[List[str], str]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode a batch of texts into embeddings.

        Main entry point for encoding text. Handles batching, padding across
        samples, and optionally returning attention masks.

        Args:
            text: Single string or list of strings to encode.
                Each string can be any length; chunking handles overflow.

        Returns:
            Depends on self.return_mask setting:

            If return_mask=True (default):
                Tuple[torch.Tensor, torch.Tensor]:
                    embeddings: Shape [B, T, E'] where:
                        B = batch size (number of input strings)
                        T = max sequence length across batch (tokens or chunks)
                        E' = embedding_dim
                    mask: Shape [B, T], dtype=torch.bool
                        True at positions with valid embeddings
                        False at padding positions

            If return_mask=False (backward compatibility):
                torch.Tensor: Just the embeddings tensor [B, T, E']

        Note:
            The return_mask parameter exists for backward compatibility.
            New code should use the default return_mask=True to get masks
            needed for downstream attention layers.

        Example:
            >>> encoder = TextEmbedding(embedding_dim=128)
            >>> emb, mask = encoder(["Hello world", "A longer text here"])
            >>> emb.shape   # [2, T, 128] where T is max tokens
            >>> mask.shape  # [2, T]
            >>> mask[0].sum()  # Number of valid tokens in first sample
        """
        # Normalize single string to list
        if isinstance(text, str):
            text = [text]

        # Get device from transformer parameters
        device = next(self.transformer.parameters()).device

        # Encode each text independently (chunking happens inside)
        batch_embeddings = []
        for t in text:
            emb = self._chunk_and_encode(t, device)
            batch_embeddings.append(emb)

        # Find max sequence length for padding across batch
        max_seq = max(e.shape[0] for e in batch_embeddings)

        # Pad all samples to max_seq and build masks
        padded = []
        masks = []
        for e in batch_embeddings:
            seq_len = e.shape[0]
            pad_len = max_seq - seq_len

            # Pad embedding tensor with zeros
            if pad_len > 0:
                padding = torch.zeros(pad_len, self.embedding_dim, device=device)
                e = torch.cat([e, padding], dim=0)
            padded.append(e)

            # Build boolean mask: True for valid positions, False for padding
            mask = torch.cat([
                torch.ones(seq_len, dtype=torch.bool, device=device),
                torch.zeros(pad_len, dtype=torch.bool, device=device),
            ])
            masks.append(mask)

        # Stack into batch tensors
        embeddings = torch.stack(padded, dim=0)  # [B, max_seq, embedding_dim]
        mask = torch.stack(masks, dim=0)         # [B, max_seq]

        # Return based on backward compat setting
        if self.return_mask:
            return embeddings, mask
        else:
            return embeddings
