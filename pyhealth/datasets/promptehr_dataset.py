"""PromptEHR dataset for synthetic EHR generation.

This module provides the dataset class for training and generating with PromptEHR.
"""

from typing import Optional, List, Dict
import torch
from torch.utils.data import Dataset
from pyhealth.tokenizer import Tokenizer


def create_promptehr_tokenizer(diagnosis_codes: List[str]) -> Tokenizer:
    """Create a tokenizer for PromptEHR with special generation tokens.

    This function creates a PyHealth Tokenizer configured for PromptEHR,
    with 7 special tokens that are compatible with the pehr_scratch implementation.

    Special tokens (IDs 0-6):
        - <pad> (0): Padding token
        - <s> (1): Start of sequence (BOS)
        - </s> (2): End of sequence (EOS)
        - <unk> (3): Unknown token
        - <v> (4): Visit start marker
        - </v> (5): Visit end marker
        - <mask> (6): Masking token for corruption

    Medical diagnosis codes will start at ID 7 (code_offset=7).

    Args:
        diagnosis_codes: List of unique diagnosis code strings (e.g., ["401.9", "427.31", ...])

    Returns:
        Configured PyHealth Tokenizer with 1:1 code-to-token mapping.

    Example:
        >>> codes = ["401.9", "427.31", "250.00"]
        >>> tokenizer = create_promptehr_tokenizer(codes)
        >>> tokenizer.get_vocabulary_size()
        10  # 7 special tokens + 3 diagnosis codes
        >>> tokenizer.convert_tokens_to_indices(["<v>", "401.9", "</v>"])
        [4, 7, 5]  # <v>=4, first code=7, </v>=5

    Note:
        This maintains compatibility with pehr_scratch checkpoint token IDs.
        The order of special tokens MUST NOT be changed.
    """
    # Define special tokens in exact order (IDs will be 0-6)
    # CRITICAL: Order must match pehr_scratch for checkpoint compatibility
    special_tokens = [
        "<pad>",   # ID 0 - padding
        "<s>",     # ID 1 - start of sequence (BART BOS)
        "</s>",    # ID 2 - end of sequence (BART EOS)
        "<unk>",   # ID 3 - unknown token
        "<v>",     # ID 4 - visit start marker
        "</v>",    # ID 5 - visit end marker
        "<mask>",  # ID 6 - masking token for corruption
    ]

    # Create tokenizer with special tokens first, then diagnosis codes
    # PyHealth's Vocabulary adds special_tokens first, preserving order
    # This automatically creates code_offset=7 (len(special_tokens))
    tokenizer = Tokenizer(
        tokens=diagnosis_codes,
        special_tokens=special_tokens
    )

    return tokenizer


class PromptEHRDataset(Dataset):
    """Dataset for PromptEHR training and generation.

    This dataset handles MIMIC-III data with proper patient record structure
    and demographic information for conditional generation.

    Args:
        TODO: Add arguments after porting from pehr_scratch

    Examples:
        TODO: Add usage examples
    """

    def __init__(self, **kwargs):
        super(PromptEHRDataset, self).__init__()
        # TODO: Port from ~/pehr_scratch/data_loader.py and dataset.py
        raise NotImplementedError("PromptEHRDataset porting in progress")

    def __len__(self):
        raise NotImplementedError("PromptEHRDataset porting in progress")

    def __getitem__(self, idx):
        raise NotImplementedError("PromptEHRDataset porting in progress")
