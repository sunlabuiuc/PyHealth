"""Text prompt construction and encoding for MedTsLLM.

Mirrors the prompt structure from the original paper implementation
(Chan et al., MLHC 2024, https://github.com/flixpar/med-ts-llm):

    [BOS] [dataset desc] [clip desc] [input stats] [task desc] [Time series:]

The four content segments (dataset, clip, stats, task) are toggled
independently by the caller.
"""

from typing import Any

import torch
from torch import Tensor


def compute_lags(x: Tensor, n_lags: int = 5) -> Tensor:
    """Compute top-N autocorrelation lags via FFT.

    Matches the original paper's ``calcute_lags``. Power spectral
    density is computed as ``rfft(x) * conj(rfft(x))``, inverse-
    transformed back to autocorrelation, averaged across features,
    and the top-N lag indices are selected.

    Args:
        x: (batch, seq_len) or (batch, seq_len, n_features).
        n_lags: Number of top lags to return.

    Returns:
        (batch, n_lags) long tensor of lag indices.
    """
    if x.ndim == 3:
        x = x.permute(0, 2, 1).contiguous()
    else:
        x = x.unsqueeze(1)
    q_fft = torch.fft.rfft(x, dim=-1)
    corr = torch.fft.irfft(q_fft * torch.conj(q_fft), dim=-1)
    mean_corr = corr.mean(dim=1)
    _, lags = torch.topk(mean_corr, n_lags, dim=-1)
    return lags


def build_prompt(
    inputs: dict[str, Any],
    *,
    dataset_description: str = "",
    task_description: str = "",
    include_dataset: bool = True,
    include_task: bool = True,
    include_clip: bool = False,
    include_stats: bool = False,
    n_lags: int = 5,
    bos_token: str = "",
) -> list[list[str]]:
    """Build text prompts for a batch of inputs.

    Args:
        inputs: Dict with ``"x_enc"`` (batch, seq_len, n_features). May
            include ``"descriptions"`` (per-sample strings) when
            ``include_clip`` is True.
        dataset_description: Dataset-level description string.
        task_description: Task-level description string.
        include_dataset: Whether to include dataset description.
        include_task: Whether to include task description.
        include_clip: Whether to include per-sample descriptions.
        include_stats: Whether to include per-sample input statistics
            (min/max/median/trend/top-N autocorr lags).
        n_lags: Number of autocorrelation lags for the stats prompt.
        bos_token: Beginning-of-sequence token string.

    Returns:
        List of string lists, one per batch element.
    """
    x_enc = inputs["x_enc"]
    bs = x_enc.shape[0]

    dataset_prompt = (
        f"Dataset: {dataset_description}" if include_dataset else ""
    )
    task_prompt = f"Task: {task_description}" if include_task else ""

    clip_prompts = (
        inputs.get("descriptions", [""] * bs) if include_clip else [""] * bs
    )

    if include_stats:
        stats_prompts = _build_stats_prompts(x_enc, n_lags)
    else:
        stats_prompts = [""] * bs

    prompts = []
    for b in range(bs):
        parts = [
            bos_token,
            dataset_prompt,
            clip_prompts[b],
            stats_prompts[b],
            task_prompt,
            "Time series:",
        ]
        parts = [p for p in parts if p != ""]
        parts = [
            (p + " " if isinstance(p, str) and i > 0 else p)
            for i, p in enumerate(parts)
        ]
        prompts.append(parts)

    return prompts


def encode_prompts(
    prompts: list[list[str]],
    tokenizer,
    embedding_layer,
    device: torch.device,
) -> Tensor:
    """Encode text prompts into embedding tensors.

    Args:
        prompts: List of string lists from ``build_prompt``.
        tokenizer: HuggingFace tokenizer.
        embedding_layer: LLM's input embedding layer.
        device: Device for tensors.

    Returns:
        Padded prompt embeddings: (batch, max_tokens, d_llm).
    """
    batch_embeddings = []

    for parts in prompts:
        part_embeddings = []
        for part in parts:
            ids = tokenizer(
                part,
                return_tensors="pt",
                padding=False,
                truncation=False,
            ).input_ids.to(device)
            emb = embedding_layer(ids)
            part_embeddings.append(emb)

        combined = torch.cat(part_embeddings, dim=1)
        batch_embeddings.append(combined)

    max_len = max(e.shape[1] for e in batch_embeddings)
    d_llm = batch_embeddings[0].shape[2]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    pad_emb = embedding_layer(torch.tensor([pad_id], device=device))

    padded = []
    for emb in batch_embeddings:
        if emb.shape[1] < max_len:
            pad_len = max_len - emb.shape[1]
            pad = pad_emb.expand(1, pad_len, d_llm)
            emb = torch.cat([pad, emb], dim=1)
        padded.append(emb)

    return torch.cat(padded, dim=0)


def _build_stats_prompts(x_enc: Tensor, n_lags: int) -> list[str]:
    """Build per-sample input statistics prompt strings.

    Reports per-feature min, max, median, trend direction, and the
    top-N autocorrelation lags. Matches the format in the original
    paper implementation.
    """
    xs = x_enc.detach()
    if xs.ndim == 2:
        xs = xs.unsqueeze(-1)

    with torch.no_grad():
        mins = torch.min(xs, dim=1).values.tolist()
        maxs = torch.max(xs, dim=1).values.tolist()
        medians = torch.median(xs.float(), dim=1).values.tolist()
        trends = (xs.diff(dim=1).sum(dim=1) > 0).tolist()
        lags = compute_lags(xs.float(), n_lags).tolist()

    def fmt(v: Any) -> str:
        if isinstance(v, list):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        if isinstance(v, bool):
            return "upward" if v else "downward"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    prompts = []
    for b in range(xs.shape[0]):
        prompt = (
            f"Input statistics (per feature): "
            f"min value = {fmt(mins[b])}, "
            f"max value = {fmt(maxs[b])}, "
            f"median value = {fmt(medians[b])}, "
            f"the trend of input is {fmt(trends[b])}, "
            f"the top {n_lags} lags are {lags[b]}."
        )
        prompts.append(prompt)

    return prompts
