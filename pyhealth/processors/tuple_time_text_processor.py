"""Processor for tuple time-based text data with temporal information.

This processor handles clinical notes or text entries paired with temporal
information (time differences), preparing them for multimodal fusion where
different modality types need to be distinguished automatically.

Input/Output:
    Input:  Tuple[List[str], List[float]]
            - List[str]: Clinical text entries (e.g., discharge notes, progress notes)
            - List[float]: Time differences between entries (in any time unit)
    
    Output: Tuple[List[str], torch.Tensor, str]
            - List[str]: Same text entries (unmodified)
            - torch.Tensor: 1D float tensor of time differences
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
    >>> processor = TupleTimeTextProcessor(type_tag="note")
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
    >>> texts_out, time_tensor, tag = result
    >>> print(f"Texts: {texts_out}")
    >>> print(f"Time tensor: {time_tensor}")
    >>> print(f"Type tag: {tag}")
    
Args:
    type_tag (str): Modality identifier for automatic routing in multimodal
        models. Common values: "note", "image", "ehr", "signal".
        Default: "note"
"""

from typing import Any, List, Tuple
import torch
from .base_processor import FeatureProcessor
from . import register_processor


@register_processor("tuple_time_text")
class TupleTimeTextProcessor(FeatureProcessor):
    """Processes (text, time_diff) tuples for multimodal temporal fusion.
    
    Converts paired text and temporal data into a format suitable for models
    that need to distinguish between different modality types automatically.
    """
    
    def __init__(self, type_tag: str = "note"):
        """Initialize the processor.
        
        Args:
            type_tag: Modality identifier for automatic routing. Default: "note"
        """
        super().__init__()
        self.type_tag = type_tag

    def process(self, value: Tuple[List[str], List[float]]) -> Tuple[List[str], torch.Tensor, str]:
        """Process a tuple of texts and time differences.
        
        Args:
            value: Tuple containing:
                - List[str]: Text entries (clinical notes, observations, etc.)
                - List[float]: Time differences corresponding to each text entry
        
        Returns:
            Tuple containing:
                - List[str]: Original text entries (unmodified)
                - torch.Tensor: 1D float tensor of time differences [shape: (N,)]
                - str: Type tag for modality routing
                
        Example:
            >>> processor = TupleTimeTextProcessor(type_tag="clinical_note")
            >>> texts = ["Note 1", "Note 2"]
            >>> times = [0.0, 24.0]  # hours
            >>> result = processor.process((texts, times))
            >>> print(result[1])  # tensor([0., 24.])
        """
        texts, time_diffs = value
        time_tensor = torch.tensor(time_diffs, dtype=torch.float32)
        return texts, time_tensor, self.type_tag
    
    def size(self):
        """Return the size of the processor vocabulary (not applicable for this processor)."""
        return None
    
    def __repr__(self):
        return f"TupleTimeTextProcessor(type_tag='{self.type_tag}')"
