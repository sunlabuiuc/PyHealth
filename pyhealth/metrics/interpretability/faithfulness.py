"""Comprehensiveness and sufficiency as computed by Edin et al. (EMNLP 2024).

These follow the authors' evaluation code (``faithfulness_metrics.py`` and
``PermuteDataset`` in github.com/JoakimEdin/explainable-medical-coding), not
the generic :class:`ComprehensivenessMetric` / :class:`SufficiencyMetric`,
which use percentage bins, zero ablation and no normalisation.

For one (note, code) pair with predicted probability ``p = sigmoid(logit)``:

* Features are words (sub-word tokens summed per word) or tokens, ranked by
  attribution. Removing a feature replaces its tokens with the baseline token
  (the tokenizer's mask token); special tokens and padding stay unchanged.
* Comprehensiveness, step ``i``: remove the top ``i * step_size`` features.
* Sufficiency, step ``i``: keep only the top ``max_features - i * step_size``
  features (remove the rest).
* With ``max_features=100`` and ``step_size=2`` there are 50 steps of each.
* ``comprehensiveness = mean_i(p - p_i) / p``;
  ``sufficiency = mean_i(max(0, p - p_i)) / p``.

The score of a method is the mean over all evaluated pairs. The authors keep
only pairs whose code is predicted (``p`` above the decision boundary).
"""

import math
import re
from typing import Optional, Sequence

import numpy as np
import torch


def roberta_word_map(tokens: Sequence[str]) -> torch.Tensor:
    """Maps byte-level BPE tokens to word indices (authors' word map).

    Port of ``get_word_map_roberta`` (include_space=True): a token starting
    with ``Ġ`` opens a new word, whitespace/punctuation-only tokens join the
    preceding word and ``<...>`` tokens are their own word.
    """
    word_map = []
    idx = 0
    space_between_words = False
    for token in tokens:
        if re.match(r"^<.*>$", token):
            if idx in word_map:
                idx += 1
            word_map.append(idx)
            idx += 1
            space_between_words = False
            continue
        if re.match(r"^[ĊĠ:.,]+$", token):
            space_between_words = True
            word_map.append(idx)
            continue
        if token.startswith("Ġ"):
            space_between_words = True
        if space_between_words:
            idx += 1
            space_between_words = False
        word_map.append(idx)
    return torch.tensor(word_map)


def word_feature_map(tokens: Sequence[str]) -> torch.Tensor:
    """Word index per non-special token, as the authors compute it.

    ``tokens`` are the note's tokens without ``<s>``, ``</s>`` and padding.
    The map is built with ``<s>``/``</s>`` around them and then stripped and
    shifted by one, exactly like ``PermuteDataset``.
    """
    return roberta_word_map(["<s>", *tokens, "</s>"])[1:-1] - 1


@torch.no_grad()
def _target_probs(model, input_ids, attention_mask, class_idx, batch_size):
    device = next(model.parameters()).device
    probs = []
    for start in range(0, len(input_ids), batch_size):
        ids = input_ids[start : start + batch_size].to(device)
        mask = attention_mask.expand(len(ids), -1).to(device)
        out = model.forward_from_embedding(model.get_input_embeddings()(ids), mask)
        probs.append(torch.sigmoid(out["logit"][:, class_idx].float()).cpu())
    return torch.cat(probs)


def perturbation_curve(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    attributions,
    class_idx: int,
    baseline_token_id: int,
    positions: Optional[torch.Tensor] = None,
    feature_map: Optional[torch.Tensor] = None,
    descending: bool = True,
    step_size: int = 2,
    max_features: Optional[int] = 100,
    batch_size: int = 16,
) -> tuple[float, np.ndarray]:
    """Probability drop for one class at each removal step.

    Args:
        model: Exposes ``get_input_embeddings()`` and
            ``forward_from_embedding(inputs_embeds, attention_mask)`` returning
            a dict with ``logit`` ``[B, J]`` (e.g. PLMCA, CAML). Put it in
            eval mode first.
        input_ids: ``[N]`` token ids of one note (padding allowed).
        attention_mask: ``[N]`` padding mask.
        attributions: ``[len(positions)]`` scores of the explained tokens.
        class_idx: Output column of the explained code.
        baseline_token_id: Id that replaces removed tokens (mask token).
        positions: Indices in ``input_ids`` that ``attributions`` refer to and
            that may be removed. Defaults to the non-padding positions
            without the first and last (``<s>``/``</s>``).
        feature_map: ``[len(positions)]`` feature (word) index per token;
            ``None`` treats every token as a feature.
        descending: True for comprehensiveness, False for sufficiency.
        step_size: Features removed per step (authors: 2).
        max_features: Number of features considered, K (authors: 100).
        batch_size: Perturbed copies per forward pass.

    Returns:
        ``(p, deltas)``: the unperturbed probability and ``p - p_i`` for each
        of the ``ceil(max_features / step_size)`` steps.
    """
    input_ids = input_ids.reshape(-1).cpu()
    attention_mask = attention_mask.reshape(1, -1).cpu()
    if positions is None:
        positions = attention_mask[0].nonzero().flatten()[1:-1]
    positions = torch.as_tensor(positions).cpu()
    attributions = torch.as_tensor(np.asarray(attributions), dtype=torch.float)
    attributions = attributions.flatten().cpu()
    if len(attributions) != len(positions):
        raise ValueError("attributions and positions must have the same length.")
    if feature_map is None:
        feature_map = torch.arange(len(positions))
    feature_map = torch.as_tensor(feature_map).cpu()
    num_features = int(feature_map.max()) + 1
    feature_scores = torch.zeros(num_features).index_add_(0, feature_map, attributions)
    order = torch.argsort(feature_scores, descending=descending, stable=True)

    num_steps = math.ceil((max_features or num_features) / step_size)
    # Sufficiency starts by removing everything outside the top max_features.
    offset = max(num_features - max_features, 0) if not descending and max_features else 0
    perturbed = input_ids.repeat(num_steps, 1)
    for step in range(num_steps):
        removed = order[: min(offset + step * step_size, num_features)]
        perturbed[step, positions[torch.isin(feature_map, removed)]] = baseline_token_id

    p = _target_probs(model, input_ids[None], attention_mask, class_idx, 1)[0]
    p_steps = _target_probs(model, perturbed, attention_mask, class_idx, batch_size)
    return p.item(), (p - p_steps).double().numpy()


def faithfulness(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    attributions,
    class_idx: int,
    baseline_token_id: int,
    positions: Optional[torch.Tensor] = None,
    feature_map: Optional[torch.Tensor] = None,
    step_size: int = 2,
    max_features: Optional[int] = 100,
    batch_size: int = 16,
) -> dict[str, float]:
    """Comprehensiveness and sufficiency of one (note, code) explanation.

    Arguments are those of :func:`perturbation_curve`. Costs
    ``2 * ceil(max_features / step_size) + 2`` forward passes (102 with the
    authors' settings).

    Returns:
        ``{"y_prob", "comprehensiveness", "sufficiency"}``; comprehensiveness
        is higher-is-better, sufficiency lower-is-better.

    Example:
        >>> ids, mask = model.tokenize(text)
        >>> faithfulness(model, ids[0], mask[0], scores, class_idx,
        ...              model.tokenizer.mask_token_id)
    """
    args = (model, input_ids, attention_mask, attributions, class_idx, baseline_token_id)
    kwargs = dict(
        positions=positions,
        feature_map=feature_map,
        step_size=step_size,
        max_features=max_features,
        batch_size=batch_size,
    )
    p, comp = perturbation_curve(*args, descending=True, **kwargs)
    _, suff = perturbation_curve(*args, descending=False, **kwargs)
    return {
        "y_prob": p,
        "comprehensiveness": float(comp.mean() / p),
        "sufficiency": float(np.clip(suff, 0, 100).mean() / p),
    }
