"""
Transformer-based models for synthetic EHR generation.

This module implements autoregressive generative models for creating synthetic
Electronic Health Records. The models learn to generate realistic patient visit
sequences by training on real EHR data.
"""

from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.processors import NestedSequenceProcessor


class TransformerEHRGenerator(BaseModel):
    """Transformer-based autoregressive model for synthetic EHR generation.

    This model uses a decoder-only transformer architecture (similar to GPT) to learn
    patient visit sequence patterns. It can generate synthetic patient histories by
    sampling from the learned distribution.

    The model processes nested sequences of medical codes (visits containing diagnosis
    codes) and learns to predict future codes autoregressively.

    Architecture:
        - Token embedding layer for medical codes
        - Positional encoding for sequential modeling
        - Multi-layer transformer decoder
        - Output projection to vocabulary

    Args:
        dataset: SampleDataset containing training data
        embedding_dim: Dimension of code embeddings. Default is 256.
        num_heads: Number of attention heads. Default is 8.
        num_layers: Number of transformer layers. Default is 6.
        dim_feedforward: Hidden dimension of feedforward network. Default is 1024.
        dropout: Dropout probability. Default is 0.1.
        max_seq_length: Maximum sequence length for positional encoding. Default is 512.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import SyntheticEHRGenerationMIMIC3
        >>> from pyhealth.datasets import get_dataloader
        >>>
        >>> # Load dataset and apply task
        >>> base_dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic3",
        ...     tables=["DIAGNOSES_ICD"],
        ... )
        >>> task = SyntheticEHRGenerationMIMIC3(min_visits=2)
        >>> sample_dataset = base_dataset.set_task(task)
        >>>
        >>> # Create model
        >>> model = TransformerEHRGenerator(
        ...     dataset=sample_dataset,
        ...     embedding_dim=256,
        ...     num_heads=8,
        ...     num_layers=6,
        ... )
        >>>
        >>> # Training
        >>> train_loader = get_dataloader(sample_dataset, batch_size=32)
        >>> for batch in train_loader:
        ...     output = model(**batch)
        ...     loss = output["loss"]
        ...     loss.backward()
        >>>
        >>> # Generation
        >>> synthetic_codes = model.generate(num_samples=100, max_visits=10)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__(dataset)

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_length = max_seq_length

        # Get vocabulary size from the processor
        input_processor = dataset.input_processors["visit_codes"]
        assert isinstance(
            input_processor, NestedSequenceProcessor
        ), "Expected NestedSequenceProcessor for visit_codes"

        self.vocab_size = input_processor.vocab_size()
        self.pad_idx = input_processor.code_vocab.get("<pad>", 0)

        # Special tokens
        self.bos_token = input_processor.code_vocab.get("<bos>", self.vocab_size)
        self.eos_token = input_processor.code_vocab.get("<eos>", self.vocab_size + 1)
        self.visit_delim_token = input_processor.code_vocab.get(
            "VISIT_DELIM", self.vocab_size + 2
        )

        # Adjust vocab size to include special tokens if needed
        extended_vocab_size = max(
            self.vocab_size, self.bos_token + 1, self.eos_token + 1, self.visit_delim_token + 1
        )

        # Token embedding
        self.token_embedding = nn.Embedding(
            extended_vocab_size, embedding_dim, padding_idx=self.pad_idx
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length, embedding_dim)
        )
        nn.init.normal_(self.pos_encoding, std=0.02)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Linear(embedding_dim, extended_vocab_size)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.pad_idx is not None:
            self.token_embedding.weight.data[self.pad_idx].zero_()
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)

    def flatten_nested_sequence(
        self, nested_seq: torch.Tensor, visit_delim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flatten nested visit sequences into 1D sequences with visit delimiters.

        Args:
            nested_seq: Tensor of shape (batch, num_visits, codes_per_visit)
            visit_delim: Token ID for visit delimiter

        Returns:
            Tuple of:
                - Flattened sequence (batch, seq_len)
                - Attention mask (batch, seq_len)
        """
        batch_size, num_visits, codes_per_visit = nested_seq.shape
        device = nested_seq.device

        # Initialize output sequence
        max_seq_len = num_visits * (codes_per_visit + 1)  # +1 for delimiter
        flat_seq = torch.full(
            (batch_size, max_seq_len),
            self.pad_idx,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)

        for b in range(batch_size):
            pos = 0
            for v in range(num_visits):
                # Add codes from this visit
                visit_codes = nested_seq[b, v]
                valid_codes = visit_codes[visit_codes != self.pad_idx]

                if len(valid_codes) > 0:
                    flat_seq[b, pos : pos + len(valid_codes)] = valid_codes
                    mask[b, pos : pos + len(valid_codes)] = True
                    pos += len(valid_codes)

                # Add visit delimiter (except after last visit)
                if v < num_visits - 1 and len(valid_codes) > 0:
                    flat_seq[b, pos] = visit_delim
                    mask[b, pos] = True
                    pos += 1

        return flat_seq, mask

    def unflatten_to_nested_sequence(
        self, flat_seq: torch.Tensor, visit_delim: int, max_codes_per_visit: int
    ) -> List[List[List[int]]]:
        """Convert flattened sequences back to nested visit structure.

        Args:
            flat_seq: Flattened sequence (batch, seq_len)
            visit_delim: Token ID for visit delimiter
            max_codes_per_visit: Maximum codes per visit

        Returns:
            List of patient histories, each containing visits, each containing codes
        """
        batch_size = flat_seq.shape[0]
        nested_sequences = []

        for b in range(batch_size):
            seq = flat_seq[b].cpu().tolist()
            patient_visits = []
            current_visit = []

            for token in seq:
                if token == self.pad_idx or token == self.eos_token:
                    # End of sequence
                    if current_visit:
                        patient_visits.append(current_visit)
                    break
                elif token == visit_delim:
                    # End of visit
                    if current_visit:
                        patient_visits.append(current_visit)
                        current_visit = []
                elif token != self.bos_token:
                    # Regular code
                    current_visit.append(token)

            # Add last visit if exists
            if current_visit:
                patient_visits.append(current_visit)

            nested_sequences.append(patient_visits)

        return nested_sequences

    def forward(self, visit_codes: torch.Tensor, future_codes: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for training or generation.

        Args:
            visit_codes: Input nested sequences (batch, num_visits, codes_per_visit)
            future_codes: Target nested sequences for teacher forcing (batch, num_visits, codes_per_visit)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary containing:
                - logit: Raw predictions (batch, seq_len, vocab_size)
                - loss: Cross-entropy loss (scalar) if future_codes provided
                - y_true: True next tokens if future_codes provided
                - y_prob: Predicted probabilities (batch, seq_len, vocab_size)
        """
        # Flatten nested sequences
        flat_input, input_mask = self.flatten_nested_sequence(
            visit_codes, self.visit_delim_token
        )

        # Get sequence length
        seq_len = flat_input.size(1)
        if seq_len > self.max_seq_length:
            flat_input = flat_input[:, : self.max_seq_length]
            input_mask = input_mask[:, : self.max_seq_length]
            seq_len = self.max_seq_length

        # Embed tokens
        embeddings = self.token_embedding(flat_input)  # (batch, seq_len, embed_dim)

        # Add positional encoding
        embeddings = embeddings + self.pos_encoding[:, :seq_len, :]
        embeddings = self.dropout_layer(embeddings)

        # Create causal mask for autoregressive generation
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=embeddings.device
        )

        # Create padding mask
        padding_mask = ~input_mask  # Invert: True = padding

        # Pass through transformer decoder
        # For decoder-only, memory is None, so it uses self-attention
        transformer_out = self.transformer_decoder(
            tgt=embeddings,
            memory=embeddings,  # Use same sequence as memory for self-attention
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        # Project to vocabulary
        logits = self.output_projection(transformer_out)  # (batch, seq_len, vocab_size)

        # Prepare output dictionary
        output = {
            "logit": logits,
            "y_prob": F.softmax(logits, dim=-1),
        }

        # Calculate loss if targets provided
        if future_codes is not None:
            flat_target, target_mask = self.flatten_nested_sequence(
                future_codes, self.visit_delim_token
            )

            if flat_target.size(1) > self.max_seq_length:
                flat_target = flat_target[:, : self.max_seq_length]
                target_mask = target_mask[:, : self.max_seq_length]

            # Shift target by 1 for next-token prediction
            target_shifted = flat_target[:, 1:]  # Remove first token
            logits_shifted = logits[:, :-1, :]  # Remove last prediction
            mask_shifted = target_mask[:, 1:]

            # Flatten for loss calculation
            logits_flat = logits_shifted.reshape(-1, logits_shifted.size(-1))
            target_flat = target_shifted.reshape(-1)
            mask_flat = mask_shifted.reshape(-1)

            # Calculate loss only on non-padded tokens
            loss = F.cross_entropy(
                logits_flat[mask_flat], target_flat[mask_flat], ignore_index=self.pad_idx
            )

            output["loss"] = loss
            output["y_true"] = flat_target

        return output

    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        max_visits: int = 10,
        max_codes_per_visit: int = 20,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> List[List[List[int]]]:
        """Generate synthetic patient histories.

        Args:
            num_samples: Number of synthetic patients to generate
            max_visits: Maximum number of visits per patient
            max_codes_per_visit: Maximum codes per visit
            max_length: Maximum total sequence length
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)

        Returns:
            List of synthetic patient histories, each containing visits with diagnosis codes
        """
        self.eval()
        device = self.device

        generated_sequences = []

        for _ in range(num_samples):
            # Start with BOS token
            current_seq = torch.tensor([[self.bos_token]], dtype=torch.long, device=device)

            for step in range(max_length - 1):
                # Get sequence length
                seq_len = current_seq.size(1)

                # Embed and add positional encoding
                embeddings = self.token_embedding(current_seq)
                embeddings = embeddings + self.pos_encoding[:, :seq_len, :]

                # Create causal mask
                causal_mask = nn.Transformer.generate_square_subsequent_mask(
                    seq_len, device=device
                )

                # Pass through transformer
                transformer_out = self.transformer_decoder(
                    tgt=embeddings,
                    memory=embeddings,
                    tgt_mask=causal_mask,
                )

                # Get logits for next token
                logits = self.output_projection(transformer_out[:, -1, :])  # (1, vocab_size)

                # Apply temperature
                logits = logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                current_seq = torch.cat([current_seq, next_token], dim=1)

                # Stop if EOS token generated
                if next_token.item() == self.eos_token:
                    break

            generated_sequences.append(current_seq)

        # Convert to nested structure
        generated_sequences = torch.cat(generated_sequences, dim=0)
        nested_output = self.unflatten_to_nested_sequence(
            generated_sequences, self.visit_delim_token, max_codes_per_visit
        )

        return nested_output
