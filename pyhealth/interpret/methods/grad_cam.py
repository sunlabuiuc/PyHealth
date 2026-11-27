"""Grad-CAM for convolutional model interpretability.

This module implements Gradient-weighted Class Activation Mapping (Grad-CAM),
which explains a prediction of a convolutional network by weighting the
feature maps of a convolutional layer with the gradient of the target class
score. It works with PyHealth's convolutional models for signals, images and
sequences, such as :class:`~pyhealth.models.CNN`,
:class:`~pyhealth.models.SparcNet` and
:class:`~pyhealth.models.TorchvisionModel` (ResNet and DenseNet backbones).

Paper:
    Selvaraju, Ramprasaath R., Michael Cogswell, Abhishek Das, Ramakrishna
    Vedantam, Devi Parikh, and Dhruv Batra.
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
    Localization."
    Proceedings of the IEEE International Conference on Computer Vision
    (ICCV), 2017.
    https://arxiv.org/abs/1610.02391

Reference implementation: https://github.com/jacobgil/pytorch-grad-cam
"""

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.models.base_model import BaseModel

from .base_interpreter import BaseInterpreter

_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
_INTERPOLATION_MODES = {1: "linear", 2: "bilinear", 3: "trilinear"}


class GradCAM(BaseInterpreter):
    """Grad-CAM attributions for convolutional PyHealth models.

    Implements Grad-CAM from Selvaraju et al., "Grad-CAM: Visual Explanations
    from Deep Networks via Gradient-based Localization" (ICCV 2017),
    https://arxiv.org/abs/1610.02391.

    For each explained feature, Grad-CAM takes the activations ``A`` of a
    convolutional target layer, shaped ``[batch, channels, *spatial]``, and
    the gradient of the target class logit with respect to them. Each channel
    is weighted by its gradient averaged over the spatial positions, the
    weighted channels are summed, and a ReLU keeps the positions that raise
    the class score:

    ``cam = ReLU(sum_k mean_spatial(dy/dA_k) * A_k)``

    The map is then resized to the input's spatial size and repeated over the
    input's other axes, so each attribution has the same shape as its input.

    Target layers are chosen as follows when ``target_layers`` is ``None``:

    - :class:`~pyhealth.models.CNN`: the last convolutional block of each
      feature (``model.cnn[feature_key].cnn[-1]``), so every feature is
      explained.
    - Any other model with a single feature key: its last ``Conv1d``,
      ``Conv2d`` or ``Conv3d`` module. For torchvision backbones, the last
      stage is usually a better choice than the last convolution inside it:
      pass ``"model.layer4"`` for a ResNet (as in the reference
      implementation) or ``"model.features"`` for a DenseNet, whose last
      convolution only outputs the final dense layer's new channels.

    Models without convolutions, and :class:`~pyhealth.models.TorchvisionModel`
    with a ViT or Swin backbone, are rejected; use
    :class:`~pyhealth.interpret.methods.CheferRelevance` or
    :class:`~pyhealth.interpret.methods.AttentionRollout` for those.

    How the map lines up with the input:

    - For :class:`~pyhealth.models.CNN` sequence-style features, the
      convolution runs along the input's sequence axis (axis 1), so a
      ``[batch, seq_len]`` code sequence gets one score per code and a
      ``[batch, time, features]`` tensor gets one score per time step,
      repeated over the features. A feature the CNN sees as a single
      position, such as a multi-hot vector, gets a uniform score.
    - Otherwise inputs are taken to be channels-first, ``[batch, channels,
      *spatial]`` (signals ``[batch, channels, time]``, images ``[batch,
      channels, height, width]``): the map is interpolated to the trailing
      spatial axes and repeated over the channels. An input whose rank
      doesn't match the map's (for example a spectrogram model such as
      :class:`~pyhealth.models.ContraWR` fed a raw signal) raises
      ``ValueError``; pass ``upsample=False`` to get the raw maps.

    Deep layers can have a much coarser resolution than the input (for
    example 7x7 for a ResNet on a 224x224 image, or a few time steps for
    SparcNet), and the resized map is correspondingly coarse. Pick an earlier
    layer for finer maps.

    Args:
        model (BaseModel): A trained PyHealth model with a convolutional
            layer to explain.
        target_layers: The layer to explain, either as a module or as its
            dotted name in ``model.named_modules()`` (e.g. ``"model.layer4"``)
            for a model with one feature key, or as a dict mapping feature
            keys to modules or names. Only the features in the dict are
            explained. If the layer runs more than once in a forward pass,
            its last call is used. Defaults to ``None``, which chooses the
            layers as described above.
        relu (bool): Apply the ReLU of the paper, keeping only positive
            evidence for the target class. Defaults to ``True``.
        upsample (bool): Resize and expand each map to its input's shape.
            If ``False``, the raw maps are returned with shape ``[batch,
            *spatial]`` of the target layer, which also works for models
            whose layer doesn't line up with the input, such as a
            spectrogram model fed a raw signal. Defaults to ``True``.

    Example:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import SparcNet
        >>> from pyhealth.interpret.methods import GradCAM
        >>>
        >>> samples = [
        ...     {
        ...         "patient_id": f"p{i}",
        ...         "visit_id": "v0",
        ...         "signal": np.random.randn(2, 256).astype(np.float32),
        ...         "label": i % 3,
        ...     }
        ...     for i in range(3)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": "multiclass"},
        ...     dataset_name="signals",
        ... )
        >>> model = SparcNet(dataset=dataset)
        >>> # ... train the model ...
        >>>
        >>> grad_cam = GradCAM(model)  # last Conv1d of SparcNet
        >>> batch = next(iter(get_dataloader(dataset, batch_size=3)))
        >>> attributions = grad_cam.attribute(**batch)
        >>> attributions["signal"].shape  # same as the input signal
        torch.Size([3, 2, 256])
        >>>
        >>> # explain class 2 instead of the predicted class
        >>> class_2 = grad_cam.attribute(target_class_idx=2, **batch)
    """

    def __init__(
        self,
        model: BaseModel,
        target_layers: nn.Module | str | dict[str, nn.Module | str] | None = None,
        relu: bool = True,
        upsample: bool = True,
    ):
        super().__init__(model)
        self.relu = relu
        self.upsample = upsample
        self.target_layers = self._resolve_target_layers(target_layers)

    def attribute(
        self,
        target_class_idx: int | None = None,
        **data,
    ) -> dict[str, torch.Tensor]:
        """Compute Grad-CAM attributions for a batch.

        Runs one forward pass with hooks on the target layers and takes the
        gradient of the target class logit with respect to their activations.
        Gradients are computed with :func:`torch.autograd.grad`, so the
        model's parameter ``.grad`` fields are left untouched, and the call
        works inside ``torch.no_grad()`` too.

        Args:
            target_class_idx: Class to explain. For multiclass and multilabel
                models, defaults to the predicted class of each sample; for
                binary and regression models (one logit) it is ignored.
            **data: Batch input passed directly to the model.

        Returns:
            Dict[str, torch.Tensor]: Non-negative attributions (with
                ``relu=True``) keyed by the explained feature keys, each with
                the shape of its input value, or ``[batch, *spatial]`` of the
                target layer when ``upsample=False``.
        """
        activations: dict[str, torch.Tensor] = {}
        handles = [
            layer.register_forward_hook(self._save_activation(key, activations))
            for key, layer in self.target_layers.items()
        ]
        try:
            with torch.enable_grad():
                logits = self.model(**data)["logit"]
                missing = [k for k in self.target_layers if k not in activations]
                if missing:
                    raise RuntimeError(
                        "GradCAM's target layer did not run in the forward pass "
                        f"for feature(s): {', '.join(missing)}."
                    )
                target = self._resolve_target_indices(logits, target_class_idx)
                score = logits.gather(1, target.unsqueeze(1)).sum()
                keys = list(activations)
                grads = torch.autograd.grad(
                    score,
                    [activations[k] for k in keys],
                    allow_unused=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        cams = {}
        for key, grad in zip(keys, grads):
            act = activations[key].detach()
            if grad is None:
                # the layer doesn't affect the logit: no evidence anywhere
                grad = torch.zeros_like(act)
            spatial_dims = tuple(range(2, act.dim()))
            weights = grad.mean(dim=spatial_dims, keepdim=True)
            cam = (weights * act).sum(dim=1)
            cams[key] = F.relu(cam) if self.relu else cam

        if not self.upsample:
            return cams
        return self._map_to_input_shapes(cams, data)

    def _resolve_target_layers(self, spec) -> dict[str, nn.Module]:
        """Turn the ``target_layers`` argument into ``{feature_key: module}``."""
        feature_keys = list(self.model.feature_keys)

        if spec is None:
            if self._is_feature_cnn():
                return {k: self.model.cnn[k].cnn[-1] for k in feature_keys}
            model_name = str(getattr(self.model, "model_name", ""))
            if model_name.startswith(("vit", "swin")):
                raise TypeError(
                    f"GradCAM can't explain a '{model_name}' backbone, which has "
                    "no convolutional feature maps. Use CheferRelevance or "
                    "AttentionRollout for attention models."
                )
            if len(feature_keys) != 1:
                raise ValueError(
                    "GradCAM can only choose a target layer by itself for a "
                    "model with one feature key or for CNN; pass target_layers "
                    f"as a dict for the features {feature_keys}."
                )
            convs = [m for m in self.model.modules() if isinstance(m, _CONV_TYPES)]
            if not convs:
                raise TypeError(
                    f"GradCAM requires a convolutional layer, but "
                    f"{type(self.model).__name__} has none."
                )
            return {feature_keys[0]: convs[-1]}

        if not isinstance(spec, dict):
            if len(feature_keys) != 1:
                raise ValueError(
                    "target_layers must be a dict of feature keys to layers for "
                    f"a model with several features: {feature_keys}."
                )
            spec = {feature_keys[0]: spec}

        unknown = [k for k in spec if k not in feature_keys]
        if unknown:
            raise ValueError(
                f"Unknown feature key(s) in target_layers: {', '.join(unknown)}. "
                f"The model's feature keys are {feature_keys}."
            )
        return {key: self._get_layer(layer) for key, layer in spec.items()}

    def _get_layer(self, layer: nn.Module | str) -> nn.Module:
        """Look up a target layer given as a module or a dotted name."""
        if isinstance(layer, str):
            try:
                return self.model.get_submodule(layer)
            except AttributeError as e:
                raise ValueError(f"The model has no submodule named '{layer}'.") from e
        if not isinstance(layer, nn.Module):
            raise TypeError(
                "Target layers must be nn.Module instances or their names, got "
                f"{type(layer).__name__}."
            )
        if not any(m is layer for m in self.model.modules()):
            raise ValueError("The target layer is not a submodule of the model.")
        return layer

    def _is_feature_cnn(self) -> bool:
        """Whether the model is laid out like CNN: one CNNLayer per feature."""
        cnn = getattr(self.model, "cnn", None)
        conv_dims = getattr(self.model, "feature_conv_dims", None)
        return (
            isinstance(cnn, nn.ModuleDict)
            and isinstance(conv_dims, dict)
            and all(
                k in cnn
                and isinstance(getattr(cnn[k], "cnn", None), nn.ModuleList)
                and len(cnn[k].cnn) > 0
                for k in self.model.feature_keys
            )
        )

    @staticmethod
    def _save_activation(key: str, store: dict[str, torch.Tensor]):
        """Forward hook that keeps a layer's output for the gradient step."""

        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"GradCAM's target layer for '{key}' must return a tensor, "
                    f"got {type(output).__name__}."
                )
            if output.dim() < 3:
                raise ValueError(
                    f"GradCAM's target layer for '{key}' must output "
                    "[batch, channels, *spatial], got shape "
                    f"{tuple(output.shape)}."
                )
            if not output.requires_grad:
                # frozen layers before the target: still differentiate from here
                output = output.detach().requires_grad_()
            store[key] = output
            # the model continues on a copy, so in-place ops after the layer
            # (e.g. DenseNet's ReLU) can't change the stored activation
            return output.clone()

        return hook

    def _map_to_input_shapes(
        self,
        cams: dict[str, torch.Tensor],
        data: dict,
    ) -> dict[str, torch.Tensor]:
        """Resize each map to its input and repeat it over the other axes.

        Args:
            cams: Per-feature maps of shape ``[batch, *spatial]``.
            data: Original ``**data`` kwargs from the dataloader batch.

        Returns:
            Maps with the shape of each feature's input value tensor.
        """
        result: dict[str, torch.Tensor] = {}
        for key, cam in cams.items():
            value = self._input_value(key, data.get(key))
            if value is None:
                raise ValueError(
                    f"Can't find the input tensor of '{key}' to line its "
                    "Grad-CAM map up with; pass upsample=False to get the raw "
                    "maps."
                )

            spatial = cam.dim() - 1
            if self._is_feature_cnn() and self.model.feature_conv_dims[key] == 1:
                axes = [1]
            elif value.dim() == spatial + 2:
                axes = list(range(2, value.dim()))
            else:
                raise ValueError(
                    f"Can't line up the {spatial}-D Grad-CAM map of '{key}' with "
                    f"its input of shape {tuple(value.shape)}; pass "
                    "upsample=False to get the raw maps."
                )

            size = [value.shape[a] for a in axes]
            if list(cam.shape[1:]) != size:
                cam = F.interpolate(
                    cam.unsqueeze(1).float(),
                    size=size,
                    mode=_INTERPOLATION_MODES[spatial],
                    align_corners=False,
                ).squeeze(1)

            shape = [value.shape[0]] + [1] * (value.dim() - 1)
            for axis in axes:
                shape[axis] = value.shape[axis]
            result[key] = cam.reshape(shape).expand(value.shape)
        return result

    def _input_value(self, key: str, feature) -> torch.Tensor | None:
        """The value tensor of a feature as given to the model."""
        if feature is None or isinstance(feature, torch.Tensor):
            return feature
        if isinstance(feature, dict):
            value = feature.get("value")
            return value if isinstance(value, torch.Tensor) else None
        processor = self.model.dataset.input_processors.get(key)
        try:
            schema = processor.schema()
        except (AttributeError, NotImplementedError):
            return None
        if "value" in schema and isinstance(feature, (tuple, list)):
            value = feature[schema.index("value")]
            return value if isinstance(value, torch.Tensor) else None
        return None
