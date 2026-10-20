from typing import Any

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("raw")
class RawProcessor(FeatureProcessor):
    """
    Processor that returns values unchanged (pass-through).

    Use it for fields a model or task reads as Python objects rather than
    tensors: strings, numbers, lists whose length varies between samples (notes
    per admission, measurement times, codes per visit), nested lists, tuples,
    dicts, ``None``, bytes or NumPy arrays.

    In the disk cache (``set_task``, ``create_sample_dataset(in_memory=False)``)
    raw values are stored as one pickled item per sample, so they may vary in
    length, shape and type between samples, and ``dataset[i][field]`` returns
    exactly the object that went in (lists stay lists, tuples stay tuples).
    Pickled values are as trusted as the rest of the cache directory, which
    already holds pickled processors in ``schema.pkl``: only load caches you
    created or trust.

    In PyHealth 2.0.2 and earlier, raw values whose length or shape varied
    between samples failed to write to the disk cache, or were misread.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p1", "notes": ["admit", "discharge"], "label": 0},
        ...     {"patient_id": "p2", "notes": ["admit"], "label": 1},
        ... ]
        >>> dataset = create_sample_dataset(  # doctest: +SKIP
        ...     samples, {"notes": "raw"}, {"label": "binary"}, in_memory=False
        ... )
        >>> dataset[0]["notes"]  # doctest: +SKIP
        ['admit', 'discharge']
    """

    def process(self, value: Any) -> Any:
        return value

    def size(self):
        return None

    def __repr__(self):
        return "RawProcessor()"
