"""Data collator for PromptEHR with corruption strategies.

This module provides batching and data augmentation through corruption.
"""

import torch
import numpy as np
from typing import List, Dict
import logging
from pyhealth.tokenizer import Tokenizer
from .promptehr_dataset import CorruptionFunctions


class EHRDataCollator:
    """Data collator for batching EHR patient data with corruptions.

    Generates training samples using corruption strategies to improve robustness:
    - Mask infilling: Replace code spans with <mask> token
    - Token deletion: Randomly delete codes
    - Token replacement: Replace codes with random alternatives

    Args:
        tokenizer: PyHealth Tokenizer configured for PromptEHR
        max_seq_length: Maximum sequence length for padding/truncation
        logger: Logger instance
        corruption_prob: Probability of applying corruption (default: 0.5)
        lambda_poisson: Poisson lambda for span masking (default: 3.0)
        del_probability: Token deletion probability (default: 0.15)
        rep_probability: Token replacement probability (default: 0.15)
        use_mask_infilling: Enable mask infilling (default: True)
        use_token_deletion: Enable token deletion (default: True)
        use_token_replacement: Enable token replacement (default: True)
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_seq_length: int,
        logger: logging.Logger,
        corruption_prob: float = 0.5,
        lambda_poisson: float = 3.0,
        del_probability: float = 0.15,
        rep_probability: float = 0.15,
        use_mask_infilling: bool = True,
        use_token_deletion: bool = True,
        use_token_replacement: bool = True
    ):
        """Initialize collator."""
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.vocabulary("<pad>")
        self.logger = logger
        self.corruption_prob = corruption_prob

        # Corruption flags
        self.use_mask_infilling = use_mask_infilling
        self.use_token_deletion = use_token_deletion
        self.use_token_replacement = use_token_replacement

        # Initialize corruption functions
        self.corruption_funcs = CorruptionFunctions(
            tokenizer=tokenizer,
            lambda_poisson=lambda_poisson,
            del_probability=del_probability,
            rep_probability=rep_probability
        )

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with optional corruption.

        Args:
            batch: List of dictionaries from PromptEHRDataset.__getitem__

        Returns:
            Dictionary with batched tensors:
                - x_num: [batch, 1] age values
                - x_cat: [batch, 1] gender IDs
                - input_ids: [batch, max_seq_len] padded token sequences
                - attention_mask: [batch, max_seq_len] attention masks
                - labels: [batch, max_seq_len] labels with -100 for padding
        """
        processed_samples = []

        for item in batch:
            # Apply corruption with probability
            if np.random.rand() < self.corruption_prob:
                # Randomly select one corruption type
                available_corruptions = []
                if self.use_mask_infilling:
                    available_corruptions.append('mask_infilling')
                if self.use_token_deletion:
                    available_corruptions.append('token_deletion')
                if self.use_token_replacement:
                    available_corruptions.append('token_replacement')

                if len(available_corruptions) > 0:
                    corruption_type = np.random.choice(available_corruptions)

                    if corruption_type == 'mask_infilling':
                        corrupted_visits, _ = self.corruption_funcs.mask_infill(item['visit_codes'])
                    elif corruption_type == 'token_deletion':
                        corrupted_visits = self.corruption_funcs.del_token(item['visit_codes'])
                    else:  # token_replacement
                        corrupted_visits = self.corruption_funcs.rep_token(item['visit_codes'])
                else:
                    corrupted_visits = item['visit_codes']
            else:
                # No corruption - teacher forcing
                corrupted_visits = item['visit_codes']

            # Shuffle code order within each visit (treats codes as unordered sets)
            shuffled_visits = []
            for visit in corrupted_visits:
                if len(visit) > 0:
                    shuffled_visit = list(np.random.choice(visit, len(visit), replace=False))
                else:
                    shuffled_visit = []
                shuffled_visits.append(shuffled_visit)

            # Encode visits to token IDs
            # Note: Do NOT prepend <s> here - shift_tokens_right will add it during training
            # Labels should be the target sequence without BOS
            token_sequence = []
            for visit in shuffled_visits:
                token_sequence.append("<v>")
                token_sequence.extend(visit)
                token_sequence.append("</v>")
            token_sequence.append("</s>")

            token_ids = self.tokenizer.convert_tokens_to_indices(token_sequence)

            processed_samples.append({
                'x_num': item['x_num'],
                'x_cat': item['x_cat'],
                'token_ids': np.array(token_ids, dtype=np.int64)
            })

        # Collate all samples
        return self._collate_samples(processed_samples)

    def _collate_samples(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Batch multiple samples with padding.

        Args:
            samples: List of sample dictionaries

        Returns:
            Batched tensors
        """
        # Stack demographic features
        x_num = torch.stack([torch.from_numpy(s['x_num']) for s in samples])
        x_cat = torch.stack([torch.from_numpy(s['x_cat']) for s in samples])

        # Pad token sequences
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for sample in samples:
            token_ids = sample['token_ids']
            seq_len = len(token_ids)

            # Truncate if too long, ensuring </s> token is preserved
            if seq_len > self.max_seq_length:
                end_token_id = self.tokenizer.vocabulary("</s>")
                token_ids = np.concatenate([
                    token_ids[:self.max_seq_length - 1],
                    np.array([end_token_id], dtype=np.int64)
                ])
                seq_len = self.max_seq_length

            # Create attention mask
            attention_mask = np.ones(seq_len, dtype=np.int64)

            # Pad to max_seq_length
            num_padding = self.max_seq_length - seq_len
            if num_padding > 0:
                token_ids = np.concatenate([
                    token_ids,
                    np.full(num_padding, self.pad_token_id, dtype=np.int64)
                ])
                attention_mask = np.concatenate([
                    attention_mask,
                    np.zeros(num_padding, dtype=np.int64)
                ])

            # Create labels (mask padding with -100)
            labels = token_ids.copy()
            labels[labels == self.pad_token_id] = -100

            input_ids_list.append(torch.from_numpy(token_ids))
            attention_mask_list.append(torch.from_numpy(attention_mask))
            labels_list.append(torch.from_numpy(labels))

        return {
            'x_num': x_num,
            'x_cat': x_cat,
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'labels': torch.stack(labels_list)
        }
