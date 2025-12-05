"""
LabTOP: Lab Test Outcome Prediction using GPT-2 with Digit-Wise Tokenization

Paper: Im et al. "LabTOP: A Unified Model for Lab Test Outcome Prediction 
       on Electronic Health Records" CHIL 2025 (Best Paper Award)
       https://arxiv.org/abs/2502.14259

This implementation follows the PyHealth BaseModel structure.
"""

from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from pyhealth.models import BaseModel
    from pyhealth.datasets import SampleDataset
except ImportError:
    # Fallback for standalone use
    class BaseModel(nn.Module):
        def __init__(self, dataset=None, feature_keys=None, label_key=None, mode="regression"):
            super().__init__()
            self.dataset = dataset
            self.feature_keys = feature_keys
            self.label_key = label_key
            self.mode = mode


class DigitWiseTokenizer:
    """
    Tokenizer that converts numerical values to digit sequences.
    
    This is LabTOP's key innovation for preserving exact numerical precision
    while maintaining a compact vocabulary.
    
    Example:
        >>> tokenizer = DigitWiseTokenizer(precision=2)
        >>> tokens = tokenizer.number_to_tokens(123.45)
        >>> # Returns: ['1', '2', '3', '.', '4', '5']
    
    Args:
        precision: Number of decimal places to keep (default: 2)
    """
    
    def __init__(self, precision: int = 2):
        self.precision = precision
        
        # Special tokens
        self.special_tokens = {
            'PAD': '<|pad|>',
            'EOS': '<|endoftext|>',
            'SEP': '|endofevent|',
            'LAB': '<|lab|>',
            'AGE': '<|age|>',
            'GENDER_M': '<|gender_m|>',
            'GENDER_F': '<|gender_f|>',
        }
        
        # Digit tokens (0-9, '.', '-')
        self.digit_tokens = [str(i) for i in range(10)] + ['.', '-']
        
        # Build vocabulary
        self.vocab = {}
        self.id_to_token = {}
        
        # Add special tokens
        idx = 0
        for token in self.special_tokens.values():
            self.vocab[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        # Add digit tokens
        for token in self.digit_tokens:
            self.vocab[token] = idx
            self.id_to_token[idx] = token
            idx += 1
    
    def number_to_tokens(self, number: float) -> List[str]:
        """Convert a number to list of digit tokens."""
        number = round(float(number), self.precision)
        num_str = f"{number:.{self.precision}f}"
        return list(num_str)
    
    def tokens_to_number(self, tokens: List[str]) -> Optional[float]:
        """Convert list of digit tokens back to number."""
        num_str = ''.join(tokens)
        try:
            return float(num_str)
        except ValueError:
            return None
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs."""
        return [self.vocab.get(token, self.vocab[self.special_tokens['PAD']]) 
                for token in tokens]
    
    def decode(self, ids: List[int]) -> List[str]:
        """Convert token IDs back to tokens."""
        return [self.id_to_token.get(id, self.special_tokens['PAD']) 
                for id in ids]
    
    def __len__(self) -> int:
        return len(self.vocab)


class LabTOPVocabulary:
    """
    Complete vocabulary for LabTOP including special tokens, digit tokens,
    and lab item codes.
    
    Args:
        lab_items: List of unique lab item IDs
        digit_tokenizer: DigitWiseTokenizer instance
    """
    
    def __init__(self, lab_items: List[int], digit_tokenizer: DigitWiseTokenizer):
        self.digit_tokenizer = digit_tokenizer
        
        # Start with digit tokenizer vocab
        self.vocab = dict(digit_tokenizer.vocab)
        self.id_to_token = dict(digit_tokenizer.id_to_token)
        
        # Add lab item codes
        idx = len(self.vocab)
        for lab_id in sorted(lab_items):
            token = f"<|lab_{lab_id}|>"
            self.vocab[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        # Store special token IDs
        self.pad_token_id = self.vocab[digit_tokenizer.special_tokens['PAD']]
        self.eos_token_id = self.vocab[digit_tokenizer.special_tokens['EOS']]
        self.sep_token_id = self.vocab[digit_tokenizer.special_tokens['SEP']]
        self.lab_token_id = self.vocab[digit_tokenizer.special_tokens['LAB']]
    
    def encode_event(self, event: Dict) -> List[int]:
        """
        Encode a lab event into token IDs.
        
        Event format: <|lab|> <|lab_50912|> 1 . 2 3 |endofevent|
        
        Args:
            event: Dict with 'code' (itemid) and 'value' (number)
        
        Returns:
            List of token IDs
        """
        tokens = []
        
        # Lab type marker
        tokens.append(self.digit_tokenizer.special_tokens['LAB'])
        
        # Lab item code
        lab_token = f"<|lab_{event['code']}|>"
        tokens.append(lab_token)
        
        # Lab value (digit-wise)
        value_tokens = self.digit_tokenizer.number_to_tokens(event['value'])
        tokens.extend(value_tokens)
        
        # Separator
        tokens.append(self.digit_tokenizer.special_tokens['SEP'])
        
        # Convert to IDs
        return [self.vocab[t] for t in tokens]
    
    def encode_demographics(self, age: Optional[int], gender: Optional[str]) -> List[int]:
        """
        Encode patient demographics.
        
        Format: <|age|> 6 5 <|gender_m|>
        """
        tokens = []
        
        # Age
        if age is not None:
            tokens.append(self.digit_tokenizer.special_tokens['AGE'])
            age_tokens = self.digit_tokenizer.number_to_tokens(int(age))
            tokens.extend(age_tokens)
        
        # Gender
        if gender == 'M':
            tokens.append(self.digit_tokenizer.special_tokens['GENDER_M'])
        elif gender == 'F':
            tokens.append(self.digit_tokenizer.special_tokens['GENDER_F'])
        
        # Convert to IDs
        return [self.vocab[t] for t in tokens]
    
    def __len__(self) -> int:
        return len(self.vocab)


class LabTOP(BaseModel):
    """
    LabTOP: Lab Test Outcome Prediction Model
    
    A GPT-2 based transformer that predicts lab test outcomes using
    digit-wise tokenization for continuous numerical predictions.
    
    Paper: Im et al. "LabTOP: A Unified Model for Lab Test Outcome Prediction 
           on Electronic Health Records" CHIL 2025 (Best Paper Award)
           https://arxiv.org/abs/2502.14259
    
    Key Innovation:
        - Digit-wise tokenization: Represents numerical values as sequences
          of individual digits (e.g., 123.45 → ['1','2','3','.','4','5'])
        - Preserves exact numerical precision
        - Small vocabulary (only ~20-50 tokens total)
        - Unified model for all lab test types
    
    Args:
        dataset: PyHealth dataset object
        feature_keys: List of input feature names 
        label_key: Target lab test name
        mode: Prediction mode (default: "regression")
        n_layers: Number of transformer layers (default: 12)
        n_heads: Number of attention heads (default: 12)
        embedding_dim: Embedding dimension (default: 768)
        max_seq_length: Maximum sequence length (default: 1024)
        digit_precision: Decimal precision for values (default: 2)
        dropout: Dropout rate (default: 0.1)
        **kwargs: Additional arguments
    
    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.models import LabTOP
        >>> 
        >>> # Load dataset
        >>> dataset = MIMIC4Dataset(root="/data/mimic4")
        >>> 
        >>> # Initialize model
        >>> model = LabTOP(
        ...     dataset=dataset,
        ...     feature_keys=["demographics", "lab_history"],
        ...     label_key="lab_value",
        ...     mode="regression",
        ...     embedding_dim=768,
        ...     n_layers=12
        ... )
        >>> 
        >>> # Forward pass
        >>> outputs = model(**batch)
        >>> loss = outputs["loss"]
    
    References:
        - Paper: https://arxiv.org/abs/2502.14259
        - Code: https://github.com/sujeongim/LabTOP
    """
    
    def __init__(
        self,
        dataset,
        feature_keys: List[str],
        label_key: str,
        mode: str = "regression",
        n_layers: int = 12,
        n_heads: int = 12,
        embedding_dim: int = 768,
        max_seq_length: int = 1024,
        digit_precision: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        super(LabTOP, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.digit_precision = digit_precision
        
        # Initialize digit-wise tokenizer
        self.digit_tokenizer = DigitWiseTokenizer(precision=digit_precision)
        
        # Build vocabulary (will be updated with lab items from dataset)
        # For now, use a placeholder - should be set with build_vocabulary()
        self.vocabulary = None
        self.vocab_size = len(self.digit_tokenizer)  # Base vocab size
        
        # GPT-2 configuration
        self.gpt2_config = GPT2Config(
            vocab_size=self.vocab_size,  # Will be updated after vocab built
            n_positions=max_seq_length,
            n_embd=embedding_dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_inner=embedding_dim * 4,
            activation_function='gelu_new',
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
        )
        
        # Initialize GPT-2 model (will be rebuilt after vocabulary is set)
        self.model = None
    
    def build_vocabulary(self, lab_items: List[int]) -> None:
        """
        Build complete vocabulary including lab item codes.
        
        This should be called after determining unique lab items from dataset.
        
        Args:
            lab_items: List of unique lab item IDs
        """
        self.vocabulary = LabTOPVocabulary(lab_items, self.digit_tokenizer)
        self.vocab_size = len(self.vocabulary)
        
        # Update GPT-2 config with actual vocab size
        self.gpt2_config.vocab_size = self.vocab_size
        self.gpt2_config.bos_token_id = self.vocabulary.eos_token_id
        self.gpt2_config.eos_token_id = self.vocabulary.eos_token_id
        self.gpt2_config.pad_token_id = self.vocabulary.pad_token_id
        
        # Build GPT-2 model
        self.model = GPT2LMHeadModel(self.gpt2_config)
    
    def prepare_input(
        self, 
        demographics: Dict,
        lab_history: List[Dict],
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare input sequence from patient data.
        
        Args:
            demographics: Dict with 'age' and 'gender'
            lab_history: List of events with 'code', 'value', 'timestamp'
            max_length: Maximum sequence length (uses self.max_seq_length if None)
        
        Returns:
            Dict with 'input_ids', 'attention_mask'
        """
        if self.vocabulary is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        max_length = max_length or self.max_seq_length
        
        # Start with demographics
        token_ids = self.vocabulary.encode_demographics(
            demographics.get('age'),
            demographics.get('gender')
        )
        
        # Add lab events (sorted by timestamp)
        for event in lab_history:
            event_ids = self.vocabulary.encode_event(event)
            token_ids.extend(event_ids)
            
            if len(token_ids) >= max_length - 1:
                break
        
        # Add EOS token
        token_ids.append(self.vocabulary.eos_token_id)
        
        # Truncate if needed
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(self.vocabulary.pad_token_id)
            attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor([token_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LabTOP model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target token IDs for training [batch_size, seq_len]
        
        Returns:
            Dict with 'logits', 'loss' (if labels provided), 'y_prob', 'y_true'
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_vocabulary() first.")
        
        # Forward through GPT-2
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        result = {
            'logits': outputs.logits,  # [batch_size, seq_len, vocab_size]
        }
        
        if labels is not None:
            result['loss'] = outputs.loss
            # For PyHealth compatibility
            result['y_true'] = labels
            result['y_prob'] = F.softmax(outputs.logits, dim=-1)
        
        return result
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate lab value prediction autoregressively.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated token IDs
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_vocabulary() first.")
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=self.vocabulary.pad_token_id,
            eos_token_id=self.vocabulary.eos_token_id,
            **kwargs
        )
    
    def decode_prediction(self, token_ids: List[int]) -> Optional[float]:
        """
        Decode generated token IDs back to numerical value.
        
        Args:
            token_ids: List of generated token IDs
        
        Returns:
            Predicted numerical value or None if invalid
        """
        if self.vocabulary is None:
            raise ValueError("Vocabulary not built.")
        
        # Find digit tokens between lab marker and separator
        digit_tokens = []
        in_value = False
        
        for token_id in token_ids:
            token = self.vocabulary.id_to_token.get(token_id, '')
            
            if token == self.digit_tokenizer.special_tokens['LAB']:
                in_value = True
                continue
            
            if token == self.digit_tokenizer.special_tokens['SEP']:
                break
            
            if in_value and token in self.digit_tokenizer.digit_tokens:
                digit_tokens.append(token)
        
        # Convert to number
        return self.digit_tokenizer.tokens_to_number(digit_tokens)


# For backward compatibility and standalone testing
if __name__ == "__main__":
    print("LabTOP Model")
    print("=" * 70)
    print("A GPT-2 based model for lab test outcome prediction")
    print("with digit-wise tokenization.")
    print()
    print("Paper: Im et al. CHIL 2025 (Best Paper Award)")
    print("https://arxiv.org/abs/2502.14259")
    print("=" * 70)
    
    # Example usage
    print("\nExample: Building vocabulary and model")
    
    # Mock dataset
    class MockDataset:
        pass
    
    # Initialize model
    model = LabTOP(
        dataset=MockDataset(),
        feature_keys=["demographics", "lab_history"],
        label_key="lab_value",
        mode="regression",
        n_layers=12,
        embedding_dim=768
    )
    
    # Build vocabulary with example lab items
    lab_items = [50912, 50931, 50971]  # Creatinine, Glucose, Potassium
    model.build_vocabulary(lab_items)
    
    print(f"✅ Model built with {len(model.vocabulary)} vocab tokens")
    print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # Test tokenization
    print("\nExample: Digit-wise tokenization")
    tokenizer = DigitWiseTokenizer(precision=2)
    value = 123.45
    tokens = tokenizer.number_to_tokens(value)
    print(f"   Value: {value}")
    print(f"   Tokens: {tokens}")
    print(f"   Reconstructed: {tokenizer.tokens_to_number(tokens)}")