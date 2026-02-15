"""Processor for tuple time-based text data with temporal information.

This processor handles clinical notes or text entries paired with temporal
information (time differences), preparing them for multimodal fusion where
different modality types need to be distinguished automatically.

Input/Output:
    Input:  Tuple[List[str], List[float]]
            - List[str]: Clinical text entries (e.g., discharge notes, progress notes)
            - List[float]: Time differences between entries (in any time unit)

    Output: Tuple[dict, torch.Tensor, str]
            - dict: HuggingFace tokenizer output (input_ids, attention_mask, etc.)
            - torch.Tensor: 1D float tensor of time differences [shape: (N,)]
            - str: Type tag for automatic modality routing (default: "note")

Use Case:
    This processor enables automatic modality bucketing in multimodal pipelines.
    The type_tag allows downstream models to automatically route different feature
    types to appropriate encoders without hardcoding feature names:

    - type_tag="note" routes to text encoder
    - type_tag="image" routes to vision encoder
    - type_tag="ehr" routes to EHR encoder

    This design eliminates the need to manually map task schema feature_keys to
    specific model components.

Example:
    >>> from pyhealth.processors import TupleTimeTextProcessor
    >>> processor = TupleTimeTextProcessor(type_tag="note", tokenizer_name="bert-base-uncased")
    >>>
    >>> # Clinical notes with time differences
    >>> texts = [
    ...     "Patient admitted with chest pain.",
    ...     "Follow-up: symptoms improved.",
    ...     "Discharge: stable condition."
    ... ]
    >>> time_diffs = [0.0, 2.5, 5.0]  # hours since admission
    >>>
    >>> result = processor.process((texts, time_diffs))
    >>> token_ids, time_tensor, tag = result
    >>> print(f"Token IDs shape: {token_ids['input_ids'].shape}")
    >>> print(f"Time tensor: {time_tensor}")
    >>> print(f"Type tag: {tag}")

Args:
    type_tag (str): Modality identifier for automatic routing in multimodal
        models. Common values: "note", "image", "ehr", "signal".
        Default: "note"
    tokenizer_name (str): HuggingFace model name for the tokenizer.
        Default: "bert-base-uncased"
"""

from typing import Any, Dict, List, Tuple
import torch
from transformers import AutoTokenizer
from .base_processor import FeatureProcessor
from . import register_processor


@register_processor("tuple_time_text")
class TupleTimeTextProcessor(FeatureProcessor):
    """Processes (text, time_diff) tuples for multimodal temporal fusion.

    Tokenizes text entries using a HuggingFace tokenizer and converts
    temporal data into tensors for downstream model consumption.
    """

    def __init__(self, type_tag: str = "note", tokenizer_name: str = "bert-base-uncased"):
        """Initialize the processor.

        Args:
            type_tag: Modality identifier for automatic routing. Default: "note"
            tokenizer_name: HuggingFace model name for the tokenizer.
                Default: "bert-base-uncased"
        """
        super().__init__()
        self.type_tag = type_tag
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def process(self, value: Tuple[List[str], List[float]]) -> Tuple[Dict[str, Any], Any, str]:
        """Process a tuple of texts and time differences.

        Tokenizes the text entries using the HuggingFace tokenizer and
        converts time differences to a float tensor.

        Args:
            value: Tuple containing:
                - List[str]: Text entries (clinical notes, observations, etc.)
                - List[float]: Time differences corresponding to each text entry

        Returns:
            Tuple containing:
                - dict: Tokenizer output with keys like 'input_ids',
                    'attention_mask', etc. (padded and truncated)
                - torch.Tensor: 1D float tensor of time differences [shape: (N,)]
                - str: Type tag for modality routing
        """
        texts, time_diffs = value
        token_ids = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        time_tensor = torch.tensor(time_diffs, dtype=torch.float32)
        return token_ids, time_tensor, self.type_tag

    def size(self):
        """Return the vocabulary size of the tokenizer."""
        return self.tokenizer.vocab_size

    def __repr__(self):
        return f"TupleTimeTextProcessor(type_tag='{self.type_tag}', tokenizer='{self.tokenizer.name_or_path}')"
