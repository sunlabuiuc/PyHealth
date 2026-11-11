"""AttInGrad: attention multiplied by input x gradient.

Contributors: Juri Moon (jmoon326)
Paper: Edin et al., "An Unsupervised Approach to Achieve Supervised-Level
    Explainability in Healthcare Records", EMNLP 2024.
Paper link: https://arxiv.org/abs/2406.08958

Description:
    Token-level explanations for medical coding models that expose a
    class-wise attention matrix and can run from token embeddings, such as
    :class:`~pyhealth.models.PLMCA`. Three methods from the paper share one
    forward and backward pass:

    * ``"attention"``: the decoder attention ``A[j, n]`` of code ``j``.
    * ``"inputxgrad"``: ``||X_n * dlogit_j / dX_n||_2``, the L2 norm over the
      embedding dimension of the input embeddings times their gradient.
    * ``"attingrad"``: the product of the two. The paper argues attention
      alone is misaligned with the input words because every encoder
      position mixes the whole chunk, while InputXGrad is measured at the
      input, so their product keeps tokens that both point to.

    Following the authors' code, gradients are taken with respect to the
    logit (not the probability), and each explanation is L1-normalized so
    the scores of one note sum to 1.
"""

from typing import Dict, Optional

import torch

from pyhealth.interpret.methods.base_interpreter import BaseInterpreter

METHODS = ("attention", "inputxgrad", "attingrad")


class AttInGrad(BaseInterpreter):
    """Attention x InputXGrad explanations for class-wise attention models.

    Args:
        model: A trained model with ``tokenize(texts)``,
            ``get_input_embeddings()`` and
            ``forward_from_embedding(inputs_embeds, attention_mask)``
            returning ``logit`` ``[batch, J]`` and ``attention``
            ``[batch, J, N]``, for example :class:`~pyhealth.models.PLMCA`.
        method: One of ``"attention"``, ``"inputxgrad"`` or ``"attingrad"``.

    Raises:
        ValueError: If ``method`` is unknown or the model lacks the methods
            listed above.

    Example:
        >>> interpreter = AttInGrad(model)  # model is a trained PLMCA
        >>> batch = next(iter(test_loader))
        >>> scores = interpreter.attribute(**batch, target_class_idx=3)
        >>> scores["text"].shape  # [batch, N], one score per token
        >>> input_ids, _ = model.tokenize(batch["text"])  # tokens they score
    """

    def __init__(self, model, method: str = "attingrad") -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
        for name in ("tokenize", "get_input_embeddings", "forward_from_embedding"):
            if not callable(getattr(model, name, None)):
                raise ValueError(
                    f"AttInGrad needs model.{name}(); {type(model).__name__} "
                    "does not have it."
                )
        super().__init__(model)
        self.method = method

    def attribute(
        self, target_class_idx: Optional[int] = None, **data
    ) -> Dict[str, torch.Tensor]:
        """Scores every token of each note for one code.

        Args:
            target_class_idx: Index of the code to explain. If ``None``, the
                code with the highest logit is explained for each note.
            **data: A dataloader batch containing the model's text feature.

        Returns:
            ``{feature_key: scores}`` where ``scores`` has shape
            ``[batch, N]`` and matches the tokens of
            ``model.tokenize(texts)``. Padding gets 0 and each row sums to 1.
        """
        feature_key = self.model.feature_keys[0]
        input_ids, attention_mask = self.model.tokenize(data[feature_key])
        embeddings = self.model.get_input_embeddings()(input_ids).detach()
        embeddings.requires_grad_(True)

        with torch.enable_grad():
            out = self.model.forward_from_embedding(embeddings, attention_mask)
            logits = out["logit"]
            targets = self._resolve_target_indices(logits, target_class_idx)
            rows = torch.arange(logits.shape[0], device=logits.device)
            # Notes are independent in eval mode, so the gradient of the sum
            # gives each note the gradient of its own target logit.
            (grad,) = torch.autograd.grad(logits[rows, targets].sum(), embeddings)

        input_x_grad = (embeddings.detach() * grad).norm(p=2, dim=-1)
        attention = out["attention"].detach()[rows, targets]
        if self.method == "attention":
            scores = attention
        elif self.method == "inputxgrad":
            scores = input_x_grad
        else:
            scores = attention * input_x_grad

        scores = scores * attention_mask
        total = scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return {feature_key: scores / total}
