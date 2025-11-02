import torch
from typing import Dict, Optional, Tuple

from pyhealth.models import BaseModel


class DeepLift:
    """DeepLIFT-style attribution for PyHealth models.

    This implementation provides difference-from-baseline attributions for
    models with discrete inputs (via embeddings) or continuous inputs. It
    mirrors the Integrated Gradients interface and focuses on StageNet and
    similar models that expose ``forward_from_embedding``.
    """

    def __init__(self, model: BaseModel, use_embeddings: bool = True):
        self.model = model
        self.model.eval()
        self.use_embeddings = use_embeddings

        if use_embeddings:
            assert hasattr(
                model, "forward_from_embedding"
            ), f"Model {type(model).__name__} must implement forward_from_embedding()"

    def attribute(
        self,
        baseline: Optional[Dict[str, torch.Tensor]] = None,
        target_class_idx: Optional[int] = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute DeepLIFT attributions for a single batch."""
        device = next(self.model.parameters()).device

        feature_inputs: Dict[str, torch.Tensor] = {}
        time_info: Dict[str, torch.Tensor] = {}
        label_data: Dict[str, torch.Tensor] = {}

        for key in self.model.feature_keys:
            if key not in data:
                continue

            value = data[key]
            if isinstance(value, tuple):
                time_tensor, feature_tensor = value
                time_info[key] = time_tensor.to(device)
                value = feature_tensor

            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            feature_inputs[key] = value.to(device)

        for key in self.model.label_keys:
            if key in data:
                label_val = data[key]
                if not isinstance(label_val, torch.Tensor):
                    label_val = torch.tensor(label_val)
                label_data[key] = label_val.to(device)

        if self.use_embeddings:
            return self._deeplift_embeddings(
                feature_inputs,
                baseline=baseline,
                target_class_idx=target_class_idx,
                time_info=time_info,
                label_data=label_data,
            )

        return self._deeplift_continuous(
            feature_inputs,
            baseline=baseline,
            target_class_idx=target_class_idx,
            time_info=time_info,
            label_data=label_data,
        )

    # --------------------------------------------------------------------- #
    # Embedding-based DeepLIFT (for discrete features)
    # --------------------------------------------------------------------- #

    def _deeplift_embeddings(
        self,
        inputs: Dict[str, torch.Tensor],
        baseline: Optional[Dict[str, torch.Tensor]],
        target_class_idx: Optional[int],
        time_info: Dict[str, torch.Tensor],
        label_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        input_embs, baseline_embs, input_shapes = self._prepare_embeddings_and_baselines(
            inputs, baseline
        )

        # Prepare delta variables with gradients
        delta_embeddings: Dict[str, torch.Tensor] = {}
        current_embeddings: Dict[str, torch.Tensor] = {}
        for key in input_embs:
            delta = (input_embs[key] - baseline_embs[key]).detach()
            delta.requires_grad_(True)
            delta_embeddings[key] = delta
            current_embeddings[key] = baseline_embs[key].detach() + delta

        forward_kwargs = {**label_data} if label_data else {}
        current_output = self.model.forward_from_embedding(
            feature_embeddings=current_embeddings,
            time_info=time_info,
            **forward_kwargs,
        )
        logits = current_output["logit"]
        target_idx = self._determine_target_index(logits, target_class_idx)
        target_logit = self._gather_target_logit(logits, target_idx)

        with torch.no_grad():
            baseline_output = self.model.forward_from_embedding(
                feature_embeddings=baseline_embs,
                time_info=time_info,
                **forward_kwargs,
            )
            baseline_logit = self._gather_target_logit(
                baseline_output["logit"], target_idx
            )

        self.model.zero_grad()
        target_logit.sum().backward()

        emb_contribs = {}
        for key, delta in delta_embeddings.items():
            grad = delta.grad
            if grad is None:
                emb_contribs[key] = torch.zeros_like(delta)
            else:
                emb_contribs[key] = grad.detach() * delta.detach()

        emb_contribs = self._enforce_completeness(
            emb_contribs, target_logit, baseline_logit
        )
        return self._map_embeddings_to_inputs(emb_contribs, input_shapes)

    def _prepare_embeddings_and_baselines(
        self,
        inputs: Dict[str, torch.Tensor],
        baseline: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, tuple]]:
        input_embeddings: Dict[str, torch.Tensor] = {}
        baseline_embeddings: Dict[str, torch.Tensor] = {}
        input_shapes: Dict[str, tuple] = {}

        for key, value in inputs.items():
            input_shapes[key] = value.shape
            embedded = self.model.embedding_model({key: value})[key]
            input_embeddings[key] = embedded

            if baseline is None:
                baseline_embeddings[key] = torch.zeros_like(embedded)
            else:
                if key not in baseline:
                    raise ValueError(f"Baseline missing key '{key}'")
                baseline_embeddings[key] = baseline[key].to(embedded.device)

        return input_embeddings, baseline_embeddings, input_shapes

    # --------------------------------------------------------------------- #
    # Continuous DeepLIFT fallback
    # --------------------------------------------------------------------- #

    def _deeplift_continuous(
        self,
        inputs: Dict[str, torch.Tensor],
        baseline: Optional[Dict[str, torch.Tensor]],
        target_class_idx: Optional[int],
        time_info: Dict[str, torch.Tensor],
        label_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device

        if baseline is None:
            baseline = {key: torch.zeros_like(val) for key, val in inputs.items()}

        delta_inputs: Dict[str, torch.Tensor] = {}
        current_inputs: Dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            delta = (value - baseline[key]).detach()
            delta.requires_grad_(True)
            delta_inputs[key] = delta

            if key in time_info:
                current_inputs[key] = (
                    time_info[key],
                    baseline[key].detach() + delta,
                )
            else:
                current_inputs[key] = baseline[key].detach() + delta

        model_inputs = {**current_inputs, **label_data}
        current_output = self.model(**model_inputs)
        logits = current_output["logit"]
        target_idx = self._determine_target_index(logits, target_class_idx)
        target_logit = self._gather_target_logit(logits, target_idx)

        with torch.no_grad():
            baseline_inputs = {
                key: (time_info[key], baseline[key])
                if key in time_info
                else baseline[key]
                for key in inputs
            }
            baseline_output = self.model(**{**baseline_inputs, **label_data})
            baseline_logit = self._gather_target_logit(
                baseline_output["logit"], target_idx
            )

        self.model.zero_grad()
        target_logit.sum().backward()

        contribs = {}
        for key, delta in delta_inputs.items():
            grad = delta.grad
            if grad is None:
                contribs[key] = torch.zeros_like(delta)
            else:
                contribs[key] = grad.detach() * delta.detach()

        contribs = self._enforce_completeness(contribs, target_logit, baseline_logit)

        # Ensure outputs match original input structures
        mapped = {}
        for key, contrib in contribs.items():
            if key in time_info:
                mapped[key] = contrib
            else:
                mapped[key] = contrib
            mapped[key] = mapped[key].to(device)
        return mapped

    # --------------------------------------------------------------------- #
    # Utility helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _determine_target_index(
        logits: torch.Tensor, target_class_idx: Optional[int]
    ) -> torch.Tensor:
        if target_class_idx is None:
            if logits.dim() >= 2 and logits.size(-1) > 1:
                target_idx = torch.argmax(logits, dim=-1)
            else:
                target_idx = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        else:
            if isinstance(target_class_idx, int):
                target_idx = torch.full(
                    (logits.size(0),),
                    target_class_idx,
                    dtype=torch.long,
                    device=logits.device,
                )
            elif isinstance(target_class_idx, torch.Tensor):
                target_idx = target_class_idx.to(logits.device).long()
                if target_idx.dim() == 0:
                    target_idx = target_idx.expand(logits.size(0))
            else:
                raise ValueError("target_class_idx must be int or Tensor")
        return target_idx

    @staticmethod
    def _gather_target_logit(
        logits: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        if logits.dim() == 2 and logits.size(-1) > 1:
            return logits.gather(1, target_idx.unsqueeze(-1)).squeeze(-1)
        return logits.squeeze(-1)

    @staticmethod
    def _enforce_completeness(
        contributions: Dict[str, torch.Tensor],
        target_logit: torch.Tensor,
        baseline_logit: torch.Tensor,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        delta_output = (target_logit - baseline_logit).detach()

        total = None
        for contrib in contributions.values():
            flat_sum = contrib.view(contrib.size(0), -1).sum(dim=1)
            total = flat_sum if total is None else total + flat_sum

        scale = torch.ones_like(delta_output)
        if total is not None:
            denom = total.abs()
            mask = denom > eps
            scale[mask] = delta_output[mask] / total[mask]

        for key, contrib in contributions.items():
            reshape_dims = [contrib.size(0)] + [1] * (contrib.dim() - 1)
            contributions[key] = contrib * scale.view(*reshape_dims)

        return contributions

    @staticmethod
    def _map_embeddings_to_inputs(
        emb_contribs: Dict[str, torch.Tensor],
        input_shapes: Dict[str, tuple],
    ) -> Dict[str, torch.Tensor]:
        mapped = {}
        for key, contrib in emb_contribs.items():
            if contrib.dim() == 4:
                token_attr = contrib.sum(dim=-1)
            elif contrib.dim() == 3:
                token_attr = contrib.sum(dim=-1)
            elif contrib.dim() == 2:
                token_attr = contrib.sum(dim=-1)
            else:
                token_attr = contrib

            orig_shape = input_shapes[key]
            if token_attr.shape != orig_shape:
                reshaped = token_attr
                while len(reshaped.shape) < len(orig_shape):
                    reshaped = reshaped.unsqueeze(-1)
                reshaped = reshaped.expand(orig_shape)
                token_attr = reshaped

            mapped[key] = token_attr.detach()
        return mapped
