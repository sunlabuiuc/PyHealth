"""Label-free prompts and frozen text encoding for CLIPStress (Yang et al., 2025).

Contributor: Siqi Zhu (UIUC NetID: siqizhu4).
Paper: Contrastive Pretraining for Stress Detection with Multimodal Wearable
Sensor Data and Surveys (CHIL 2025).

Paper: https://proceedings.mlr.press/v287/yang25a.html
"""

import logging
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import torch


def build_stress_prompt(
    signal: np.ndarray,
    channel_names: Sequence[str],
    survey: Mapping[str, str | float | int],
    target_keys: Sequence[str] = (),
) -> str:
    """Describe a wearable window using eleven statistics and survey predictors.

    Args:
        signal: Finite array shaped (time, channels), before padding. Generate
            descriptions in original units before fitting train-only scalers.
        channel_names: One descriptive name per sensor channel.
        survey: Explicitly selected predictor fields, never a complete sample.
        target_keys: Additional target/target-derived names to reject. Common
            stress label names are always rejected in both channels and surveys.

    Returns:
        A newline-separated description. Statistics use population moments;
        constant channels have zero skewness and excess kurtosis by convention.

    Raises:
        ValueError: For invalid windows, names, nonfinite values, or label fields.

    Examples:
        >>> prompt = build_stress_prompt(
        ...     np.array([[1.], [3.]]), ["steps"], {"age": 30})
        >>> "mean is 2" in prompt and "age: 30" in prompt
        True
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError("signal must be a finite, nonempty (time, channels) array.")
    if len(channel_names) != x.shape[1] or len(set(channel_names)) != x.shape[1]:
        raise ValueError("Provide one unique channel name per signal column.")
    forbidden = {
        "label",
        "labels",
        "target",
        "stress",
        "stress_score",
        "stress_level",
        "stai_stress",
        "stai_stress_category",
        *[k.strip().casefold() for k in target_keys],
    }
    for key in [*channel_names, *survey]:
        if not isinstance(key, str) or not key.strip() or "\n" in key:
            raise ValueError("Feature names must be nonempty single-line strings.")
        if key.strip().casefold() in forbidden:
            raise ValueError(f"Target field {key!r} must not enter a prompt.")
    lines = []
    for index, name in enumerate(channel_names):
        values = x[:, index]
        mean = values.mean()
        std = values.std()
        centered = (values - mean) / std if std > 0 else np.zeros_like(values)
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        stats = {
            "mean": mean,
            "std": std,
            "min": values.min(),
            "max": values.max(),
            "25th percentile": q25,
            "median": median,
            "75th percentile": q75,
            "root mean square": np.sqrt(np.mean(values**2)),
            "kurtosis": np.mean(centered**4) - 3 if std > 0 else 0,
            "skewness": np.mean(centered**3),
            "IQR": q75 - q25,
        }
        lines.append(
            name
            + ": "
            + ", ".join(f"{stat} is {value:.6g}" for stat, value in stats.items())
        )
    for key, value in survey.items():
        if isinstance(value, (int, float, np.number)) and not np.isfinite(value):
            raise ValueError(f"Survey value {key!r} must be finite.")
        if not isinstance(value, (str, int, float, np.number)) or "\n" in str(value):
            raise ValueError("Survey values must be single-line strings or numbers.")
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def encode_stress_prompts(
    prompts: Sequence[str],
    model_name: str = "bert-base-uncased",
    batch_size: int = 32,
    device: str = "cpu",
    max_length: int = 512,
    local_files_only: bool = False,
    truncation: str = "warn",
) -> torch.Tensor:
    """Cache mean-pooled embeddings from a frozen pretrained text model.

    Args:
        prompts: Nonempty sequence of feature-only descriptions.
        model_name: Hugging Face model identifier or local model directory.
        batch_size: Number of descriptions encoded at once.
        device: Torch device for the text encoder.
        max_length: Token truncation limit, including special tokens.
        local_files_only: Require already downloaded model/tokenizer assets.
        truncation: ``warn`` reports descriptions exceeding max_length;
            ``error`` rejects them before loading encoder weights; ``chunk``
            encodes nonoverlapping windows and pools all their valid tokens.
            Chunking preserves feature coverage but changes the text context
            relative to encoding a single truncated sequence.

    Returns:
        CPU tensor shaped (number of prompts, language model hidden width).
        Padding is excluded from pooling; model dropout and gradients are off.

    Raises:
        ValueError: If prompts or batching parameters are invalid.

    Examples:
        >>> # Explicitly downloads model weights if they are not cached.
        >>> vectors = encode_stress_prompts(["steps: mean is 12"])  # doctest: +SKIP
        >>> vectors.shape  # doctest: +SKIP
        torch.Size([1, 768])
    """
    if (
        isinstance(prompts, str)
        or not prompts
        or any(not isinstance(p, str) or not p.strip() for p in prompts)
    ):
        raise ValueError("prompts must be a nonempty sequence of nonempty strings.")
    if batch_size <= 0 or max_length < 2:
        raise ValueError("batch_size must be positive and max_length at least 2.")
    if truncation not in ("warn", "error", "chunk"):
        raise ValueError("truncation must be warn, error, or chunk.")
    # Import and download only when this explicit preprocessing helper is called.
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    truncated = 0
    longest = 0
    for start in range(0, len(prompts), batch_size):
        lengths = tokenizer(
            list(prompts[start : start + batch_size]),
            truncation=False,
            padding=False,
            return_length=True,
            verbose=False,
        )["length"]
        truncated += sum(length > max_length for length in lengths)
        longest = max(longest, max(lengths))
    if truncated and truncation != "chunk":
        message = (
            f"{truncated} of {len(prompts)} prompts exceed max_length={max_length} "
            f"(longest: {longest} tokens). Truncation can omit survey fields; "
            "shorten descriptions or review feature coverage before training."
        )
        if truncation == "error":
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    encoder = AutoModel.from_pretrained(
        model_name, local_files_only=local_files_only
    ).to(device)
    encoder.eval().requires_grad_(False)
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            descriptions = list(prompts[start : start + batch_size])
            extra = {"return_overflowing_tokens": True} if truncation == "chunk" else {}
            tokens = tokenizer(
                descriptions,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                **extra,
            )
            if truncation == "chunk":
                mapping = tokens.pop("overflow_to_sample_mapping").to(device)
                order = tokens["attention_mask"].sum(1).argsort(descending=True)
                tokens = {key: value[order] for key, value in tokens.items()}
                mapping = mapping[order.to(device)]
                sums = None
                counts = torch.zeros(len(descriptions), 1, device=device)
                for offset in range(0, len(mapping), batch_size):
                    chunk = {
                        key: value[offset : offset + batch_size].to(device)
                        for key, value in tokens.items()
                    }
                    # Short overflow windows need not carry 512 padded positions.
                    columns = chunk["attention_mask"].bool().any(0).nonzero().flatten()
                    first, last = int(columns[0]), int(columns[-1]) + 1
                    chunk = {key: value[:, first:last] for key, value in chunk.items()}
                    hidden = encoder(**chunk).last_hidden_state
                    mask = chunk["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    if sums is None:
                        sums = hidden.new_zeros(len(descriptions), hidden.shape[-1])
                    indices = mapping[offset : offset + batch_size]
                    sums.index_add_(0, indices, (hidden * mask).sum(1))
                    counts.index_add_(0, indices, mask.sum(1).to(counts.dtype))
                embeddings.append((sums / counts.clamp_min(1)).cpu())
                logging.getLogger(__name__).debug(
                    "Encoded %d/%d prompts",
                    min(start + batch_size, len(prompts)),
                    len(prompts),
                )
                continue
            tokens = {key: value.to(device) for key, value in tokens.items()}
            hidden = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            embeddings.append(pooled.cpu())
    return torch.cat(embeddings)
