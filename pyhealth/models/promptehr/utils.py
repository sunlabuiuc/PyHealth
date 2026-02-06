"""Utility functions and classes for PromptEHR.

This module contains:
    - VisitStructureSampler: Samples realistic visit structures for generation
    - Data collation functions
    - Helper utilities
"""

import torch
import torch.nn as nn


class VisitStructureSampler:
    """Samples realistic visit structures from training data.

    This is a critical component added Nov 21, 2025 that solves the
    over-generation problem. Reduces codes/patient from 18.1 → 11.97 (34%).

    Args:
        TODO: Add arguments after porting from pehr_scratch
    """

    def __init__(self, **kwargs):
        # TODO: Port from ~/pehr_scratch/visit_structure_sampler.py
        raise NotImplementedError("VisitStructureSampler porting in progress")

    def sample(self, **kwargs):
        """Sample a visit structure."""
        raise NotImplementedError("VisitStructureSampler porting in progress")
