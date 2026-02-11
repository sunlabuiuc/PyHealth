"""Conditional prompt encoder for demographic conditioning.

This module provides demographic conditioning through prompt-based learning
with reparameterization to prevent overfitting.

Ported from pehr_scratch/conditional_prompt.py (lines 1-219).
"""

import torch
import torch.nn as nn
from typing import Optional


class NumericalConditionalPrompt(nn.Module):
    """Embeds continuous numerical features (e.g., age) with reparameterization.

    Uses intermediate d_hidden=128 dimension for better gradient flow and
    regularization, following PromptEHR's architecture.
    """

    def __init__(
        self,
        n_num_features: int,
        hidden_dim: int,
        d_hidden: int = 128,
        prompt_length: int = 1
    ):
        """Initialize numerical prompt encoder with reparameterization.

        Args:
            n_num_features: Number of continuous features (1 for age only)
            hidden_dim: Output dimension size (768 for BART-base)
            d_hidden: Intermediate reparameterization dimension (default: 128)
            prompt_length: Number of prompt vectors per feature (default: 1)
        """
        super().__init__()
        self.n_num_features = n_num_features
        self.hidden_dim = hidden_dim
        self.d_hidden = d_hidden
        self.prompt_length = prompt_length

        # Reparameterization: learned weight and bias in d_hidden space
        self.weight = nn.Parameter(torch.Tensor(n_num_features, d_hidden))
        self.bias = nn.Parameter(torch.Tensor(n_num_features, d_hidden))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.bias)

        # Project from d_hidden to output dimension
        self.proj = nn.Linear(d_hidden, hidden_dim, bias=False)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """Embed numerical features with reparameterization.

        Args:
            x_num: [batch, n_num_features] continuous values

        Returns:
            [batch, prompt_length * n_num_features, hidden_dim] embeddings
        """
        # Reparameterization: weight * value + bias
        # x_num: [batch, n_num_features]
        # weight: [n_num_features, d_hidden]
        # Result: [batch, n_num_features, d_hidden]
        x = self.weight[None] * x_num[..., None]
        x = x + self.bias[None]

        # Project to output dimension
        # x: [batch, n_num_features, d_hidden] → [batch, n_num_features, hidden_dim]
        x = self.proj(x)

        # Output: [batch, n_num_features * prompt_length, hidden_dim]
        return x


class CategoricalConditionalPrompt(nn.Module):
    """Embeds categorical features with offset-based indexing and reparameterization.

    Uses single embedding table with offset-based indexing to prevent category
    collision, following PromptEHR's architecture.
    """

    def __init__(
        self,
        cat_cardinalities: list,
        hidden_dim: int,
        d_hidden: int = 128,
        prompt_length: int = 1
    ):
        """Initialize categorical prompt encoder with reparameterization.

        Args:
            cat_cardinalities: List of category counts for each feature
                              [2] for gender (M/F) - ethnicity removed
            hidden_dim: Output dimension size (768 for BART-base)
            d_hidden: Intermediate reparameterization dimension (default: 128)
            prompt_length: Number of prompt vectors per feature (default: 1)
        """
        super().__init__()
        assert cat_cardinalities, 'cat_cardinalities must be non-empty'
        self.cat_cardinalities = cat_cardinalities
        self.hidden_dim = hidden_dim
        self.d_hidden = d_hidden
        self.prompt_length = prompt_length

        # Compute offset indices to prevent category collision
        # Example: [2] → offsets = [0]
        # Gender 0 (M) → index 0, Gender 1 (F) → index 1
        category_offsets = torch.tensor([0] + cat_cardinalities[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets, persistent=False)

        # Single embedding table for all categories
        total_categories = sum(cat_cardinalities)
        self.embeddings = nn.Embedding(total_categories, d_hidden)

        # Learned bias per feature (not per category)
        self.bias = nn.Parameter(torch.Tensor(len(cat_cardinalities), d_hidden))
        nn.init.xavier_uniform_(self.bias)

        # Project from d_hidden to output dimension
        self.proj = nn.Linear(d_hidden, hidden_dim, bias=False)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Embed categorical features with offset-based indexing.

        Args:
            x_cat: [batch, n_cat_features] categorical IDs

        Returns:
            [batch, n_cat_features * prompt_length, hidden_dim] embeddings
        """
        # Add offsets to prevent category collision
        # x_cat: [batch, n_cat_features]
        # category_offsets: [n_cat_features]
        x = self.embeddings(x_cat + self.category_offsets[None])

        # Add learned bias per feature
        # x: [batch, n_cat_features, d_hidden]
        # bias: [n_cat_features, d_hidden]
        x = x + self.bias[None]

        # Project to output dimension
        # x: [batch, n_cat_features, d_hidden] → [batch, n_cat_features, hidden_dim]
        x = self.proj(x)

        # Output: [batch, n_cat_features * prompt_length, hidden_dim]
        return x


class ConditionalPromptEncoder(nn.Module):
    """Combined prompt encoder for both numerical and categorical features.

    Encodes patient demographics (age + gender) into prompt vectors that
    condition the BART encoder and decoder.

    Example:
        >>> # For PromptEHR: age (continuous) + gender (categorical)
        >>> encoder = ConditionalPromptEncoder(
        ...     n_num_features=1,           # age
        ...     cat_cardinalities=[2],      # gender (M/F)
        ...     hidden_dim=768,             # BART dimension
        ...     d_hidden=128                # reparameterization
        ... )
        >>> # Batch of 16 patients
        >>> age = torch.randn(16, 1)        # Normalized ages
        >>> gender = torch.randint(0, 2, (16, 1))  # 0=M, 1=F
        >>> prompts = encoder(x_num=age, x_cat=gender)
        >>> prompts.shape  # [16, 2, 768] - 2 prompts (age + gender)
    """

    def __init__(
        self,
        n_num_features: Optional[int] = None,
        cat_cardinalities: Optional[list] = None,
        hidden_dim: int = 768,
        d_hidden: int = 128,
        prompt_length: int = 1
    ):
        """Initialize combined prompt encoder.

        Args:
            n_num_features: Number of continuous features (None to disable)
            cat_cardinalities: Category counts for each categorical feature (None to disable)
            hidden_dim: Hidden dimension size (768 for BART-base)
            d_hidden: Intermediate reparameterization dimension (default: 128)
            prompt_length: Number of prompt vectors per feature (default: 1)
        """
        super().__init__()
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        self.hidden_dim = hidden_dim
        self.d_hidden = d_hidden
        self.prompt_length = prompt_length

        # Initialize numerical prompt encoder (age)
        if n_num_features is not None and n_num_features > 0:
            self.num_prompt = NumericalConditionalPrompt(
                n_num_features, hidden_dim, d_hidden, prompt_length
            )
        else:
            self.num_prompt = None

        # Initialize categorical prompt encoder (gender)
        if cat_cardinalities is not None and len(cat_cardinalities) > 0:
            self.cat_prompt = CategoricalConditionalPrompt(
                cat_cardinalities, hidden_dim, d_hidden, prompt_length
            )
        else:
            self.cat_prompt = None

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode demographics to prompt embeddings.

        Args:
            x_num: [batch, n_num_features] continuous values (optional)
            x_cat: [batch, n_cat_features] categorical IDs (optional)

        Returns:
            [batch, total_prompts, hidden_dim] combined prompt embeddings
        """
        prompts = []

        if x_num is not None and self.num_prompt is not None:
            num_embeds = self.num_prompt(x_num)
            prompts.append(num_embeds)

        if x_cat is not None and self.cat_prompt is not None:
            cat_embeds = self.cat_prompt(x_cat)
            prompts.append(cat_embeds)

        if len(prompts) == 0:
            raise ValueError("No prompt embeddings generated. Provide x_num or x_cat.")

        # Concatenate along prompt dimension
        combined_prompts = torch.cat(prompts, dim=1)
        return combined_prompts

    def get_num_prompts(self) -> int:
        """Calculate total number of prompt tokens."""
        num_prompts = 0

        if self.num_prompt is not None:
            num_prompts += self.n_num_features * self.prompt_length

        if self.cat_prompt is not None:
            num_prompts += len(self.cat_cardinalities) * self.prompt_length

        return num_prompts
