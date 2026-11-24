"""Token masking (TM): adversarial-robustness training for explainability.

Edin et al., "An Unsupervised Approach to Achieve Supervised-Level
Explainability in Healthcare Records", EMNLP 2024, Section 2.2 and
Appendix D (https://arxiv.org/abs/2406.08958).

For each training batch an inner optimiser learns the smallest soft mask
over the input tokens that keeps the model's predictions (the masked tokens
are mixed with a baseline token's embedding). The rounded mask is then used
to drop the unimportant tokens, and the model is trained to predict the same
on the masked note as on the full note, while staying close to a frozen
teacher (the model TM fine-tuning started from). The result is a model that
relies on fewer tokens, which makes its explanations sparser and more
plausible.

The model needs ``tokenizer``, ``get_input_embeddings()`` and
``forward_from_embedding(inputs_embeds, attention_mask)`` returning a dict
with ``logit``; :class:`~pyhealth.models.PLMCA` and
:class:`~pyhealth.models.CAML` have all three.
"""

import torch
from torch import nn


def _forward(model: nn.Module, embeds: torch.Tensor, attention_mask: torch.Tensor):
    return model.forward_from_embedding(embeds, attention_mask)["logit"]


@torch.enable_grad()
def optimize_token_mask(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    baseline_embeds: torch.Tensor,
    diet_constant: float = 0.01,
    lr: float = 0.1,
    max_iter: int = 20,
    min_iter: int = 0,
    init: float = 2.0,
) -> tuple[torch.Tensor, int]:
    """Learns a soft token mask that keeps the predictions with few tokens.

    Minimises ``mean(sigmoid(m)) + diet_constant * |p(x_m) - p(x)|_1 / B``
    over ``m`` with AdamW, where ``x_m = x * sigmoid(m) + b * (1 - sigmoid(m))``
    mixes the token embeddings ``x`` with the baseline embeddings ``b``. It
    stops after ``max_iter`` steps, or earlier once the loss is within 3% of
    its value two steps before. The model runs in eval mode and its
    parameters receive no gradients.

    Args:
        model: Model with ``get_input_embeddings`` and
            ``forward_from_embedding``.
        input_ids: ``[batch, N]`` token ids.
        attention_mask: ``[batch, N]`` padding mask.
        baseline_embeds: ``[batch, N, E]`` embeddings of the baseline input.
        diet_constant: Weight of the prediction-change term.
        lr: AdamW learning rate for the mask.
        max_iter: Maximum number of mask updates.
        min_iter: Updates done before the convergence check may stop the
            loop. The authors' code uses 0; with their settings the check
            fires at the third update, while every mask value is still
            about 0.85, so rounding keeps every token. A larger value (or a
            larger ``lr``) lets the mask actually move.
        init: Starting value of the mask logits (the authors use 2.0, i.e.
            ``sigmoid = 0.88``).

    Returns:
        ``(mask, iterations)``: the soft mask ``sigmoid(m)`` of shape
        ``[batch, N]`` (detached) and the number of updates done.

    Example:
        >>> ids, mask = model.tokenize(["Sepsis with fever."])
        >>> embed = model.get_input_embeddings()
        >>> baseline = embed(torch.full_like(ids, model.tokenizer.mask_token_id))
        >>> soft_mask, steps = optimize_token_mask(model, ids, mask, baseline)
        >>> soft_mask.shape == ids.shape
        True
    """
    was_training = model.training
    model.eval()
    embeds = model.get_input_embeddings()(input_ids).detach()
    baseline_embeds = baseline_embeds.detach()
    with torch.no_grad():
        original = torch.sigmoid(_forward(model, embeds, attention_mask))

    mask = torch.full((*input_ids.shape, 1), float(init), device=embeds.device)
    mask.requires_grad_(True)
    optimizer = torch.optim.AdamW([mask], lr=lr)
    prev_loss = prev_prev_loss = float("inf")
    iterations = 0
    for _ in range(max_iter):
        soft = torch.sigmoid(mask)
        mixed = embeds * soft + baseline_embeds * (1 - soft)
        probs = torch.sigmoid(_forward(model, mixed, attention_mask))
        loss = soft.sum() / soft[..., 0].numel() + diet_constant * (
            probs - original
        ).abs().sum() / probs.size(0)
        value = loss.item()
        # Only the mask gets a gradient, so gradients the outer training
        # loop is accumulating on the model are left untouched.
        (mask.grad,) = torch.autograd.grad(loss, mask)
        optimizer.step()
        optimizer.zero_grad()
        iterations += 1
        converged = 0.97 * prev_prev_loss < value < 1.03 * prev_prev_loss
        if iterations >= min_iter and converged:
            break
        prev_prev_loss, prev_loss = prev_loss, value
    model.train(was_training)
    return torch.sigmoid(mask.detach())[..., 0], iterations


def token_masking_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    teacher_logits: torch.Tensor,
    lambda_1: float = 1.0,
    diet_constant: float = 0.01,
    lr: float = 0.1,
    max_iter: int = 20,
    baseline_token_id: int | None = None,
    min_iter: int = 0,
    init: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Token-masking training loss with distillation from a frozen teacher.

    ``loss = |p(x) - p_teacher(x)|_1 / B + lambda_1 * |p(x) - p(x_masked)|_1 / B``
    where ``x_masked`` replaces the tokens dropped by the rounded mask from
    :func:`optimize_token_mask` with ``baseline_token_id``. ``p(x)`` is
    detached in the second term, so it only pulls the masked prediction
    towards the full one. Special tokens and padding are never masked.

    Args:
        model: Model being trained (``tokenizer``, ``get_input_embeddings``,
            ``forward_from_embedding``).
        input_ids: ``[batch, N]`` token ids from ``model.tokenize``.
        attention_mask: ``[batch, N]`` padding mask.
        teacher_logits: ``[batch, J]`` logits of the frozen starting model
            on the same notes.
        lambda_1: Weight of the masked term.
        diet_constant: Weight of the prediction-change term in the mask
            optimisation.
        lr: Mask learning rate.
        max_iter: Maximum mask updates per batch.
        baseline_token_id: Token that replaces masked tokens; defaults to
            the tokenizer's mask token.
        min_iter: See :func:`optimize_token_mask` (0 = authors' behaviour).
        init: See :func:`optimize_token_mask` (2.0 = authors' behaviour).

    Returns:
        A dict with ``loss`` (scalar), ``logit`` ``[batch, J]`` (unmasked
        student), ``mask`` ``[batch, N]`` (1 = token kept) and
        ``iterations`` (mask updates used).

    Example:
        >>> ids, mask = model.tokenize(["Sepsis with fever."])
        >>> with torch.no_grad():
        ...     embeds = frozen_model.get_input_embeddings()(ids)
        ...     teacher = frozen_model.forward_from_embedding(embeds, mask)["logit"]
        >>> out = token_masking_loss(model, ids, mask, teacher)
        >>> out["loss"].backward()
    """
    tokenizer = model.tokenizer
    if baseline_token_id is None:
        baseline_token_id = tokenizer.mask_token_id
    embed = model.get_input_embeddings()
    baseline_ids = torch.full_like(input_ids, baseline_token_id)
    soft, iterations = optimize_token_mask(
        model,
        input_ids,
        attention_mask,
        embed(baseline_ids),
        diet_constant=diet_constant,
        lr=lr,
        max_iter=max_iter,
        min_iter=min_iter,
        init=init,
    )
    keep = soft.round().bool() | (attention_mask == 0)
    keep |= torch.isin(
        input_ids, torch.tensor(tokenizer.all_special_ids, device=input_ids.device)
    )
    masked_ids = torch.where(keep, input_ids, baseline_ids)

    batch_size = input_ids.size(0)
    logits = _forward(model, embed(input_ids), attention_mask)
    probs = torch.sigmoid(logits)
    masked_probs = torch.sigmoid(_forward(model, embed(masked_ids), attention_mask))
    teacher_probs = torch.sigmoid(teacher_logits.detach())
    distill = (probs - teacher_probs).abs().sum() / batch_size
    masked = (probs.detach() - masked_probs).abs().sum() / batch_size
    return {
        "loss": distill + lambda_1 * masked,
        "logit": logits,
        "mask": keep.to(soft.dtype),
        "iterations": iterations,
    }
