"""AttInGrad: attention multiplied by input x gradient.

Contributors: Juri Moon (jmoon326)
Paper: Edin et al., "An Unsupervised Approach to Achieve Supervised-Level
    Explainability in Healthcare Records", EMNLP 2024.
Paper link: https://arxiv.org/abs/2406.08958

Description:
    Token-level explanations for medical coding models that expose a
    class-wise attention matrix and can run from token embeddings, such as
    :class:`~pyhealth.models.PLMCA`:

    * ``"attention"``: the decoder attention ``A[j, n]`` of code ``j``.
    * ``"inputxgrad"``: ``||X_n * dlogit_j / dX_n||_2``, the L2 norm over the
      embedding dimension of the input embeddings times their gradient.
    * ``"attingrad"``: attention times InputXGrad. The paper argues attention
      alone is misaligned with the input words because every encoder
      position mixes the whole chunk, while InputXGrad is measured at the
      input, so their product keeps tokens that both point to.
    * ``"integrated_gradients"``: Integrated Gradients on the embeddings,
      from a baseline where every real token is replaced by the mask token
      (special tokens kept), reduced per token with the L2 norm.
    * ``"att_ig"``: attention times Integrated Gradients. Not in the paper;
      used to ablate whether AttInGrad's gain needs InputXGrad specifically.
    * ``"deeplift"``: DeepLIFT on the embeddings from the same mask-token
      baseline, reduced per token with the L2 norm. It reuses the Rescale
      hooks of :class:`~pyhealth.interpret.methods.DeepLift`: ``nn.ReLU``,
      ``nn.Sigmoid`` and ``nn.Tanh`` modules get the multiplier
      ``delta_out / delta_in``; every other operation (functional
      ``torch.tanh``, softmax, GELU, LayerNorm) uses the linear rule, i.e.
      its gradient at the input. This is also what captum's ``DeepLift``,
      used by the authors, does with operations it does not know.
    * ``"att_deeplift"``: attention times DeepLIFT, for the same ablation.

    Following the authors' code, gradients are taken with respect to the
    logit (not the probability), and each explanation is L1-normalized so
    the scores of one note sum to 1.
"""


import torch

from pyhealth.interpret.methods.base_interpreter import BaseInterpreter
# Intentional reuse of DeepLift's private hook context: att_deeplift then uses
# the same Rescale rules and activation swapping as pyhealth's DeepLift instead
# of a second copy that could drift from it.
from pyhealth.interpret.methods.deeplift import _DeepLiftHookContext

METHODS = (
    "attention",
    "inputxgrad",
    "attingrad",
    "integrated_gradients",
    "att_ig",
    "deeplift",
    "att_deeplift",
)


class AttInGrad(BaseInterpreter):
    """Attention x input-attribution explanations for class-wise attention models.

    Args:
        model: A trained model with ``tokenize(texts)``,
            ``get_input_embeddings()``, a Hugging Face ``tokenizer`` and
            ``forward_from_embedding(inputs_embeds, attention_mask)``
            returning ``logit`` ``[batch, J]`` and ``attention``
            ``[batch, J, N]``, for example :class:`~pyhealth.models.PLMCA`.
        method: One of ``"attention"``, ``"inputxgrad"``, ``"attingrad"``,
            ``"integrated_gradients"``, ``"att_ig"``, ``"deeplift"`` or
            ``"att_deeplift"``.
        steps: Riemann steps for Integrated Gradients.

    Raises:
        ValueError: If ``method`` is unknown or ``steps`` is less than 1.
        TypeError: If the model lacks ``tokenize``,
            ``get_input_embeddings`` or ``forward_from_embedding``.

    Example:
        >>> interpreter = AttInGrad(model)  # model is a trained PLMCA
        >>> batch = next(iter(test_loader))
        >>> scores = interpreter.attribute(**batch, target_class_idx=3)
        >>> scores["text"].shape  # [batch, N], one score per token
        >>> input_ids, _ = model.tokenize(batch["text"])  # tokens they score
    """

    def __init__(self, model, method: str = "attingrad", steps: int = 20) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
        for name in ("tokenize", "get_input_embeddings", "forward_from_embedding"):
            if not callable(getattr(model, name, None)):
                raise TypeError(
                    f"AttInGrad needs model.{name}(); {type(model).__name__} "
                    "does not have it."
                )
        if steps < 1:
            raise ValueError("steps must be at least 1.")
        super().__init__(model)
        self.method = method
        self.steps = steps

    def attribute(
        self, target_class_idx: int | None = None, **data
    ) -> dict[str, torch.Tensor]:
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
        input_ids, mask = self.model.tokenize(data[feature_key])
        embed = self.model.get_input_embeddings()
        embeddings = embed(input_ids).detach()

        with torch.enable_grad():
            out, grad = self._logit_gradient(embeddings, mask, target_class_idx)
        targets = out["targets"]
        rows = torch.arange(len(targets), device=targets.device)
        attention = out["attention"].detach()[rows, targets]

        if self.method in ("integrated_gradients", "att_ig"):
            baseline = embed(self._baseline_ids(input_ids, mask)).detach()
            with torch.enable_grad():
                raw = self.integrated_gradients(embeddings, baseline, mask, targets)
            inputs = raw.norm(p=2, dim=-1)
        elif self.method in ("deeplift", "att_deeplift"):
            baseline = embed(self._baseline_ids(input_ids, mask)).detach()
            with torch.enable_grad():
                raw = self.deeplift(embeddings, baseline, mask, targets)
            inputs = raw.norm(p=2, dim=-1)
        else:
            inputs = (embeddings * grad).norm(p=2, dim=-1)

        if self.method == "attention":
            scores = attention
        elif self.method in ("inputxgrad", "integrated_gradients", "deeplift"):
            scores = inputs
        else:
            scores = attention * inputs

        scores = scores * mask
        total = scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return {feature_key: scores / total}

    def integrated_gradients(
        self,
        embeddings: torch.Tensor,
        baseline: torch.Tensor,
        mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Integrated Gradients per embedding dimension, ``[batch, N, E]``.

        Uses the midpoint Riemann sum, so the attributions summed over all
        tokens and dimensions approach ``logit(x) - logit(baseline)``.
        """
        total = torch.zeros_like(embeddings)
        for step in range(self.steps):
            alpha = (step + 0.5) / self.steps
            point = baseline + alpha * (embeddings - baseline)
            _, grad = self._logit_gradient(point, mask, targets)
            total += grad
        return (embeddings - baseline) * total / self.steps

    def deeplift(
        self,
        embeddings: torch.Tensor,
        baseline: torch.Tensor,
        mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """DeepLIFT (Rescale rule) per embedding dimension, ``[batch, N, E]``.

        A reference pass on ``baseline`` records the activations, then the
        backward pass on ``embeddings`` swaps each hooked activation's
        derivative for its secant multiplier. For a model whose only
        nonlinearities are hooked modules, the attributions summed over all
        tokens and dimensions equal ``logit(x) - logit(baseline)``. With no
        hooked modules this is ``(x - baseline) * dlogit/dx`` at ``x``.
        """
        with _DeepLiftHookContext(self.model) as hooks:
            hooks.start_baseline()
            with torch.no_grad():
                self.model.forward_from_embedding(baseline, mask)
            hooks.start_actual()
            _, grad = self._logit_gradient(embeddings, mask, targets)
        return (embeddings - baseline) * grad

    def _logit_gradient(self, embeddings, mask, targets):
        """Gradient of each note's target logit w.r.t. its embeddings."""
        embeddings = embeddings.detach().requires_grad_(True)
        out = self.model.forward_from_embedding(embeddings, mask)
        logits = out["logit"]
        if targets is None or isinstance(targets, int):
            targets = self._resolve_target_indices(logits, targets)
        rows = torch.arange(logits.shape[0], device=logits.device)
        # Notes are independent in eval mode, so the gradient of the sum
        # gives each note the gradient of its own target logit.
        (grad,) = torch.autograd.grad(logits[rows, targets].sum(), embeddings)
        out["targets"] = targets
        return out, grad

    def _baseline_ids(self, input_ids: torch.Tensor, mask: torch.Tensor):
        """Input ids with every real, non-special token set to the mask token."""
        tokenizer = self.model.tokenizer
        fill = tokenizer.mask_token_id
        if fill is None:
            fill = tokenizer.pad_token_id or 0
        special = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in tokenizer.all_special_ids:
            special |= input_ids == token_id
        replace = mask.bool() & ~special
        return torch.where(replace, torch.full_like(input_ids, fill), input_ids)
