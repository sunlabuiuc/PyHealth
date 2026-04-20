# Paper: Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization
# Paper link: https://arxiv.org/abs/1610.02391
# Description: Grad-CAM attribution method for CNN-based medical image classification in PyHealth.

"""Grad-CAM for CNN-based image classification models."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_interpreter import BaseInterpreter


class GradCAM(BaseInterpreter):
    """Compute Grad-CAM heatmaps for CNN-based image classifiers.

    Grad-CAM generates a class-conditional localization map by combining
    gradients with the activations from a target convolutional layer.
    This implementation is designed for PyHealth image workflows and
    returns CAMs using the same feature-keyed dict convention as other
    interpretability methods.

    Args:
        model: Trained model to interpret.
        target_layer: Target convolutional layer as an ``nn.Module`` or a
            dotted string path (for example ``"model.layer4.1.conv2"``).
        input_key: Batch key containing the image tensor. Default is ``"image"``.

    Examples:
        >>> gradcam = GradCAM(model, target_layer=model.model.layer4[-1].conv2)
        >>> batch = next(iter(test_loader))
        >>> cams = gradcam.attribute(**batch)
        >>> cams["image"].shape
        torch.Size([1, 224, 224])
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str | nn.Module,
        input_key: str = "image",
    ) -> None:
        super().__init__(model)
        self.input_key = input_key
        self.target_layer = self._resolve_target_layer(target_layer)
        self.last_target_class: Optional[torch.Tensor] = None

    def attribute(
        self,
        class_index: Optional[int | torch.Tensor] = None,
        normalize: bool = True,
        upsample: bool = True,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute Grad-CAM heatmaps for a batch of images.

        Args:
            class_index: Target class index to explain. If ``None``, uses the
                model's predicted class. For binary score tensors with a single
                output channel, the only valid explicit value is ``0``.
            normalize: If ``True``, normalize each heatmap to ``[0, 1]``.
            upsample: If ``True``, resize CAMs to the input image size.
            **data: Batched model inputs, including the image tensor under
                ``input_key`` and any required labels/metadata.

        Returns:
            Dict[str, torch.Tensor]: Dictionary keyed by ``input_key`` with
            CAM tensors of shape ``[B, H, W]``.

        Raises:
            KeyError: If ``input_key`` is missing from ``data``.
            ValueError: If the target layer cannot be resolved, does not
                produce a 4D activation map, or the model output lacks both
                ``logit`` and ``y_prob``.
        """
        if self.input_key not in data:
            raise KeyError(
                f"Expected input key '{self.input_key}' in the attribution batch."
            )
        image_tensor = data[self.input_key]
        if not torch.is_tensor(image_tensor):
            raise ValueError(
                f"Grad-CAM requires '{self.input_key}' to be a batched image tensor."
            )
        if image_tensor.dim() != 4:
            raise ValueError("Grad-CAM requires image tensors with shape [B, C, H, W].")

        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def forward_hook(_, __, output):
            if not torch.is_tensor(output):
                raise ValueError("Grad-CAM target layer must output a tensor.")
            if output.dim() != 4:
                raise ValueError(
                    "Grad-CAM requires a 4D convolutional activation map from "
                    "the target layer."
                )
            if not output.requires_grad:
                raise RuntimeError(
                    "Grad-CAM requires gradients. Do not call attribute() inside "
                    "torch.no_grad()."
                )
            activations["value"] = output
            output.register_hook(lambda grad: gradients.__setitem__("value", grad))

        hook = self.target_layer.register_forward_hook(forward_hook)
        self.model.zero_grad()
        try:
            score_tensor = self._forward_score_tensor(data)
            target_indices, target_scores = self._select_target_scores(
                score_tensor=score_tensor,
                class_index=class_index,
            )

            self.model.zero_grad()
            target_scores.sum().backward()

            if "value" not in activations or "value" not in gradients:
                raise RuntimeError(
                    "Grad-CAM hooks did not capture activations and gradients."
                )

            cams = self._compute_cam(
                activations=activations["value"],
                gradients=gradients["value"],
            )

            if upsample:
                cams = F.interpolate(
                    cams.unsqueeze(1),
                    size=image_tensor.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            if normalize:
                cams = self._normalize_cam(cams)

            # Keep target indices accessible for debugging and examples.
            self.last_target_class = target_indices.detach().cpu()
            return {self.input_key: cams}
        finally:
            hook.remove()
            self.model.zero_grad()

    def _resolve_target_layer(self, target_layer: str | nn.Module) -> nn.Module:
        if isinstance(target_layer, nn.Module):
            if not any(module is target_layer for module in self.model.modules()):
                raise ValueError(
                    "Grad-CAM target_layer must be a submodule of the model."
                )
            return target_layer
        if not isinstance(target_layer, str) or not target_layer:
            raise ValueError("target_layer must be a non-empty string or nn.Module.")

        current = self.model
        for part in target_layer.split("."):
            if part.isdigit():
                try:
                    current = current[int(part)]
                except Exception as exc:
                    raise ValueError(
                        f"Could not resolve target layer index '{part}' in "
                        f"path '{target_layer}'."
                    ) from exc
            else:
                if not hasattr(current, part):
                    raise ValueError(
                        f"Could not resolve target layer path '{target_layer}'. "
                        f"Missing attribute '{part}'."
                    )
                current = getattr(current, part)

        if not isinstance(current, nn.Module):
            raise ValueError(
                f"Resolved target layer '{target_layer}' is not an nn.Module."
            )
        return current

    def _forward_score_tensor(self, data: dict) -> torch.Tensor:
        score_tensor = self._forward_torchvision_logits(data)
        if score_tensor is not None:
            return score_tensor
        outputs = self.model(**data)
        return self._resolve_score_tensor(outputs)

    def _forward_torchvision_logits(self, data: dict) -> Optional[torch.Tensor]:
        try:
            from pyhealth.models.torchvision_model import TorchvisionModel
        except Exception:
            return None

        if not isinstance(self.model, TorchvisionModel):
            return None

        image_tensor = data[self.input_key].to(self.model.device)
        if image_tensor.shape[1] == 1:
            image_tensor = image_tensor.repeat((1, 3, 1, 1))
        return self.model.model(image_tensor)

    @staticmethod
    def _resolve_score_tensor(outputs: dict) -> torch.Tensor:
        if not isinstance(outputs, dict):
            raise ValueError(
                "Grad-CAM expects model outputs to be a dict containing "
                "'logit' or 'y_prob'."
            )
        if "logit" in outputs:
            score_tensor = outputs["logit"]
        elif "y_prob" in outputs:
            score_tensor = outputs["y_prob"]
        else:
            raise ValueError(
                "Grad-CAM requires model outputs to contain 'logit' or 'y_prob'."
            )
        if not torch.is_tensor(score_tensor):
            raise ValueError(
                "Grad-CAM requires 'logit' or 'y_prob' to be a torch.Tensor."
            )
        if score_tensor.dim() not in (1, 2):
            raise ValueError(
                "Grad-CAM requires classification scores shaped [B] or [B, C]."
            )
        return score_tensor

    @staticmethod
    def _select_target_scores(
        score_tensor: torch.Tensor,
        class_index: Optional[int | torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if score_tensor.dim() == 1:
            score_tensor = score_tensor.unsqueeze(-1)

        batch_size = score_tensor.shape[0]
        device = score_tensor.device

        if score_tensor.shape[-1] == 1:
            if class_index is not None:
                if isinstance(class_index, int):
                    if class_index != 0:
                        raise ValueError(
                            "Single-output Grad-CAM only supports class_index=0."
                        )
                else:
                    class_index = class_index.to(device)
                    if torch.any(class_index != 0):
                        raise ValueError(
                            "Single-output Grad-CAM only supports class_index=0."
                        )
            target_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            target_scores = score_tensor.reshape(batch_size)
            return target_indices, target_scores

        num_classes = score_tensor.shape[-1]
        if class_index is None:
            target_indices = torch.argmax(score_tensor, dim=-1)
        elif isinstance(class_index, int):
            if class_index < 0 or class_index >= num_classes:
                raise ValueError(
                    f"class_index must be in [0, {num_classes - 1}] for this model."
                )
            target_indices = torch.full(
                (batch_size,),
                class_index,
                dtype=torch.long,
                device=device,
            )
        else:
            target_indices = class_index.to(device).long()
            if target_indices.dim() == 0:
                target_indices = torch.full(
                    (batch_size,),
                    int(target_indices.item()),
                    dtype=torch.long,
                    device=device,
                )
            elif target_indices.dim() != 1:
                raise ValueError("Tensor class_index must be a scalar or a 1D tensor.")
            elif target_indices.shape[0] != batch_size:
                raise ValueError(
                    "Tensor class_index must have one target per batch element."
                )
            if torch.any((target_indices < 0) | (target_indices >= num_classes)):
                raise ValueError(
                    f"class_index values must be in [0, {num_classes - 1}]."
                )

        target_scores = score_tensor.gather(1, target_indices.unsqueeze(1)).squeeze(1)
        return target_indices, target_scores

    @staticmethod
    def _compute_cam(
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cams = torch.relu((weights * activations).sum(dim=1))
        return cams

    @staticmethod
    def _normalize_cam(cams: torch.Tensor) -> torch.Tensor:
        flat = cams.flatten(start_dim=1)
        min_vals = flat.min(dim=1).values.view(-1, 1, 1)
        max_vals = flat.max(dim=1).values.view(-1, 1, 1)
        denom = (max_vals - min_vals).clamp_min(1e-8)
        return (cams - min_vals) / denom
