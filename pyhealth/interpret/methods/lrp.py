"""Layer-wise Relevance Propagation (LRP) for PyHealth models.

Implements epsilon-rule and alphabeta-rule LRP for computing feature
attributions. Supports embedding-based models (discrete medical codes)
and CNN-based models (images).

References:
    Binder et al., "Layer-wise Relevance Propagation for Neural Networks
    with Local Renormalization Layers", arXiv:1604.00825, 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal

from pyhealth.models import BaseModel
from pyhealth.interpret.methods.base_interpreter import BaseInterpreter
from pyhealth.interpret.methods.lrp_base import stabilize_denominator


class LayerwiseRelevancePropagation(BaseInterpreter):
    """LRP attribution method for PyHealth models.

    Decomposes a model's prediction into per-feature relevance scores
    via backward propagation. Satisfies the conservation property:
    sum of relevances ~ f(x).

    Args:
        model: Trained PyHealth model to interpret.
        rule: Propagation rule ('epsilon' or 'alphabeta').
        epsilon: Stabilizer for epsilon-rule (default: 0.01).
        alpha: Positive weight for alphabeta-rule (default: 1.0).
        beta: Negative weight for alphabeta-rule (default: 0.0).
        use_embeddings: If True, propagate from embedding layer.

    Examples:
        >>> lrp = LayerwiseRelevancePropagation(model, rule="epsilon")
        >>> attributions = lrp.attribute(**test_batch)
        >>> for key, rel in attributions.items():
        ...     print(f"{key}: sum={rel.sum().item():.4f}")
    """

    def __init__(
        self,
        model: BaseModel,
        rule: Literal["epsilon", "alphabeta"] = "epsilon",
        epsilon: float = 0.01,
        alpha: float = 1.0,
        beta: float = 0.0,
        use_embeddings: bool = True,
    ):
        super().__init__(model)
        self.rule = rule
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.use_embeddings = use_embeddings

        self.hooks = []
        self.activations = {}
        self._residual_blocks = {}  # block_name -> {input, has_downsample}

        if use_embeddings:
            assert hasattr(model, "forward_from_embedding"), (
                f"Model {type(model).__name__} must implement "
                "forward_from_embedding() for embedding-level LRP. "
                "Set use_embeddings=False for continuous features only."
            )

    def attribute(
        self,
        target_class_idx: Optional[int] = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute LRP attributions for input features.

        Args:
            target_class_idx: Target class for attribution (None = predicted).
            **data: Input data from a dataloader batch.

        Returns:
            Dict mapping each feature key to its relevance tensor.
        """
        feature_keys = getattr(self.model, "feature_keys", list(data.keys()))
        device = next(self.model.parameters()).device
        has_processors = hasattr(self.model, "dataset") and hasattr(
            getattr(self.model, "dataset", None), "input_processors"
        )

        # Keep original input tuples (matching processor schema) and extract values
        inputs, values, label_data = {}, {}, {}
        for key in feature_keys:
            if key not in data:
                continue
            v = data[key]
            if isinstance(v, torch.Tensor):
                v = (v,)
            inputs[key] = tuple(
                t.to(device) if isinstance(t, torch.Tensor) else t for t in v
            )
            if has_processors and key in self.model.dataset.input_processors:
                schema = self.model.dataset.input_processors[key].schema()
                values[key] = inputs[key][schema.index("value")]
            else:
                # Fallback for models without processor schema
                values[key] = inputs[key][0]

        for key in getattr(self.model, "label_keys", []):
            if key in data:
                val = data[key]
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val)
                label_data[key] = val.to(device)

        if self.use_embeddings:
            return self._compute_from_embeddings(
                inputs, values, target_class_idx, label_data
            )
        return self._compute_from_inputs(inputs, values, target_class_idx, label_data)

    def visualize(self, plt, image, relevance, title=None, method="overlay", **kwargs):
        """Visualize LRP relevance maps using SaliencyVisualizer."""
        from pyhealth.interpret.methods.saliency_visualization import visualize_attribution

        if title is None:
            title = f"LRP Attribution ({self.rule}-rule)"
        visualize_attribution(plt, image, relevance, title=title, method=method, **kwargs)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_from_embeddings(self, inputs, values, target_class_idx, label_data):
        """LRP starting from embedding layer (for discrete inputs)."""
        input_embeddings, input_shapes = {}, {}

        for key in values:
            input_shapes[key] = values[key].shape
            embedded = self.model.embedding_model({key: values[key]})
            x = embedded[key]
            if x.dim() == 4:
                x = x.sum(dim=2)
            input_embeddings[key] = x

        # Reconstruct input tuples with embedded values spliced in
        forward_kwargs = {}
        has_processors = hasattr(self.model, "dataset") and hasattr(
            getattr(self.model, "dataset", None), "input_processors"
        )
        for key in inputs:
            if has_processors and key in self.model.dataset.input_processors:
                schema = self.model.dataset.input_processors[key].schema()
                val_idx = schema.index("value")
                original = inputs[key]
                forward_kwargs[key] = (
                    *original[:val_idx],
                    input_embeddings[key],
                    *original[val_idx + 1:],
                )
            else:
                # Fallback: pass bare embedding tensor
                forward_kwargs[key] = input_embeddings[key]
        if label_data:
            forward_kwargs.update(label_data)

        self._register_hooks()
        try:
            with torch.no_grad():
                output = self.model.forward_from_embedding(
                    **forward_kwargs,
                )
            logits = output["logit"]
            output_relevance = self._init_output_relevance(logits, target_class_idx)

            relevance_at_emb = self._propagate_relevance_backward(
                output_relevance, input_embeddings
            )

            result = {}
            for key in input_embeddings:
                rel = relevance_at_emb.get(key)
                if rel is None:
                    continue
                # Sum over embedding dim to get per-token relevance
                if rel.dim() >= 2:
                    result[key] = rel.sum(dim=-1)
                else:
                    result[key] = rel
                # Expand to match original input shape if needed
                orig = input_shapes[key]
                if result[key].shape != orig and len(orig) == 3 and result[key].dim() == 2:
                    result[key] = result[key].unsqueeze(-1).expand(orig)
        finally:
            self._remove_hooks()
        return result

    def _compute_from_inputs(self, inputs, values, target_class_idx, label_data):
        """LRP starting from continuous inputs (e.g., images)."""
        self.model.eval()
        self._register_hooks()
        try:
            forward_kwargs = {**values}
            if label_data:
                forward_kwargs.update(label_data)
            with torch.no_grad():
                output = self.model(**forward_kwargs)

            logits = output.get("logit", output.get("y_prob", output.get("y_pred")))
            output_relevance = self._init_output_relevance(logits, target_class_idx)

            result = self._propagate_relevance_backward(output_relevance, values)
            if not isinstance(result, dict):
                result = {list(values.keys())[0]: result}
        finally:
            self._remove_hooks()
        return result

    def _init_output_relevance(self, logits, target_class_idx):
        """Initialize relevance at the output layer."""
        if target_class_idx is None:
            target_class_idx = torch.argmax(logits, dim=-1)
        elif not isinstance(target_class_idx, torch.Tensor):
            target_class_idx = torch.tensor(target_class_idx, device=logits.device)

        if logits.dim() == 2 and logits.size(-1) > 1:
            batch_size = logits.size(0)
            output_relevance = torch.zeros_like(logits)
            output_relevance[range(batch_size), target_class_idx] = logits[
                range(batch_size), target_class_idx
            ]
            return output_relevance
        return logits

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        self._residual_blocks = {}

        def save_activation(name):
            def hook(module, input, output):
                in_t = input[0] if isinstance(input, tuple) else input
                out_t = output[0] if isinstance(output, tuple) else output
                self.activations[name] = {
                    "input": in_t,
                    "output": out_t,
                    "module": module,
                }
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.LSTM, nn.GRU,
                                   nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d,
                                   nn.AdaptiveAvgPool2d, nn.BatchNorm2d)):
                handle = module.register_forward_hook(save_activation(name))
                self.hooks.append(handle)

        # Hook residual blocks (torchvision ResNet BasicBlock / Bottleneck) so we can
        # correctly split relevance at the skip-connection addition during backward pass.
        try:
            from torchvision.models.resnet import BasicBlock, Bottleneck

            def save_residual_block(block_name):
                def hook(module, input, output):
                    x = input[0] if isinstance(input, tuple) else input
                    self._residual_blocks[block_name] = {
                        "input": x,
                        "has_downsample": module.downsample is not None,
                    }
                return hook

            for name, module in self.model.named_modules():
                if isinstance(module, (BasicBlock, Bottleneck)):
                    handle = module.register_forward_hook(save_residual_block(name))
                    self.hooks.append(handle)
        except (ImportError, AttributeError):
            pass

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self._residual_blocks = {}

    # ------------------------------------------------------------------
    # Backward propagation
    # ------------------------------------------------------------------

    def _get_block_for_layer(self, layer_name: str) -> Optional[str]:
        """Return the residual block prefix if this layer belongs to a tracked block."""
        for bname in self._residual_blocks:
            if layer_name.startswith(bname + "."):
                return bname
        return None

    def _propagate_through_residual_block(
        self,
        block_name: str,
        block_layer_names_rev: list,
        r_in: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate relevance through a residual block with correct LRP skip-connection split.

        At each residual addition ``out = main_path + identity`` the epsilon rule requires
        splitting the incoming relevance proportionally:
            R_main     = (main_path / stabilise(main_path + identity)) * R_in
            R_identity = (identity  / stabilise(main_path + identity)) * R_in

        R_main is then propagated back through the block's convolutional layers while
        R_identity bypasses them (or is propagated through the downsample layers when
        the block includes a projection shortcut).
        """
        has_downsample = self._residual_blocks[block_name]["has_downsample"]
        block_input = self._residual_blocks[block_name]["input"]

        # Partition layers into main path vs. downsample path
        main_layers = [l for l in block_layer_names_rev if ".downsample." not in l]
        ds_layers = [l for l in block_layer_names_rev if ".downsample." in l]

        # main_out = output of the last main-path BN (first encountered in reversed order,
        # e.g. bn2 for BasicBlock, bn3 for Bottleneck)
        main_out = self.activations[main_layers[0]]["output"] if main_layers else None

        # identity = output of the final downsample BN (projection shortcut) OR raw input
        if has_downsample and ds_layers:
            identity = self.activations[ds_layers[0]]["output"]
        else:
            identity = block_input

        # LRP-epsilon split at the residual addition
        if (main_out is not None
                and identity is not None
                and main_out.shape == identity.shape
                and r_in.shape == main_out.shape):
            total = main_out + identity
            sign = total.sign()
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            denom = total + self.epsilon * sign
            r_for_main = (main_out / denom) * r_in
            r_for_identity = (identity / denom) * r_in
        else:
            r_for_main = r_in
            r_for_identity = None

        # Propagate r_for_main backward through the main-path layers
        r_at_input = r_for_main
        for ln in main_layers:
            ai = self.activations[ln]
            if r_at_input.shape != ai["output"].shape:
                r_at_input = self._match_shapes(r_at_input, ai["output"].shape)
            r_at_input = self._propagate_through_layer(ai["module"], ai, r_at_input)

        # Propagate r_for_identity through downsample layers (projection shortcut)
        if has_downsample and ds_layers and r_for_identity is not None:
            r_ds = r_for_identity
            for ln in ds_layers:
                ai = self.activations[ln]
                if r_ds.shape != ai["output"].shape:
                    r_ds = self._match_shapes(r_ds, ai["output"].shape)
                r_ds = self._propagate_through_layer(ai["module"], ai, r_ds)
            if r_ds.shape == r_at_input.shape:
                r_at_input = r_at_input + r_ds
        elif not has_downsample and r_for_identity is not None:
            # No projection: identity goes directly to block input
            if r_for_identity.shape == r_at_input.shape:
                r_at_input = r_at_input + r_for_identity

        return r_at_input

    def _propagate_relevance_backward(self, output_relevance, input_embeddings):
        """Propagate relevance from output back to input embeddings.

        Iterates layers in reverse-forward order.  When a layer belongs to a
        tracked residual block (BasicBlock / Bottleneck), all sub-layers of that
        block are handled by ``_propagate_through_residual_block`` which applies
        a proper LRP-epsilon split at the skip-connection addition.
        """
        current_relevance = output_relevance
        layer_names = list(reversed(list(self.activations.keys())))

        feature_relevances = {}
        concat_detected = False
        blocks_handled: set = set()

        idx = 0
        while idx < len(layer_names):
            layer_name = layer_names[idx]

            # ---- Residual block handling ----
            block_name = self._get_block_for_layer(layer_name)
            if block_name is not None and block_name not in blocks_handled:
                # Collect all consecutive layers that belong to this block
                block_layers = []
                j = idx
                while j < len(layer_names) and self._get_block_for_layer(layer_names[j]) == block_name:
                    block_layers.append(layer_names[j])
                    j += 1

                current_relevance = self._propagate_through_residual_block(
                    block_name, block_layers, current_relevance
                )
                blocks_handled.add(block_name)
                idx = j
                continue

            # ---- Regular layer handling ----
            activation_info = self.activations[layer_name]
            module = activation_info["module"]
            output_tensor = activation_info["output"]

            # Detect concatenation point (PyHealth MLP multi-feature pattern)
            if (not concat_detected and isinstance(module, nn.Linear)
                    and hasattr(self.model, "feature_keys")
                    and len(self.model.feature_keys) > 1
                    and idx + 1 < len(layer_names)):
                next_name = layer_names[idx + 1]
                if "mlp." in next_name and any(
                    f in next_name for f in self.model.feature_keys
                ):
                    concat_detected = True

            if current_relevance.shape != output_tensor.shape:
                current_relevance = self._match_shapes(
                    current_relevance, output_tensor.shape
                )

            current_relevance = self._propagate_through_layer(
                module, activation_info, current_relevance
            )

            if concat_detected and current_relevance.dim() == 2:
                n_features = len(self.model.feature_keys)
                dim = current_relevance.size(1) // n_features
                for i, fk in enumerate(self.model.feature_keys):
                    feature_relevances[fk] = current_relevance[
                        :, i * dim : (i + 1) * dim
                    ]
                # Process remaining feature-specific layers
                for fk in self.model.feature_keys:
                    cur = feature_relevances[fk]
                    for ln in layer_names[idx + 1:]:
                        if fk not in ln:
                            continue
                        ai = self.activations[ln]
                        m = ai["module"]
                        if cur.shape != ai["output"].shape:
                            cur = self._match_shapes(cur, ai["output"].shape)
                        cur = self._propagate_through_layer(m, ai, cur)
                    feature_relevances[fk] = cur
                return self._split_relevance_to_features(
                    feature_relevances, input_embeddings
                )

            idx += 1

        return self._split_relevance_to_features(
            current_relevance, input_embeddings
        )

    def _propagate_through_layer(self, module, activation_info, relevance):
        """Route relevance propagation to the correct layer handler."""
        if isinstance(module, nn.Linear):
            return self._lrp_linear(module, activation_info, relevance)
        elif isinstance(module, nn.Conv2d):
            return self._lrp_conv2d(module, activation_info, relevance)
        elif isinstance(module, nn.ReLU):
            return relevance
        elif isinstance(module, nn.MaxPool2d):
            return self._lrp_maxpool2d(module, activation_info, relevance)
        elif isinstance(module, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            return self._lrp_avgpool2d(module, activation_info, relevance)
        elif isinstance(module, nn.BatchNorm2d):
            gamma = module.weight.view(1, -1, 1, 1) if module.weight is not None else 1.0
            return relevance * gamma
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            return self._lrp_rnn(module, activation_info, relevance)
        return relevance

    # ------------------------------------------------------------------
    # Layer-specific LRP rules
    # ------------------------------------------------------------------

    def _get_input(self, activation_info):
        x = activation_info["input"]
        if isinstance(x, tuple):
            x = x[0]
        return x

    def _lrp_linear(self, module, activation_info, relevance_output):
        if self.rule == "epsilon":
            return self._lrp_linear_epsilon(module, activation_info, relevance_output)
        return self._lrp_linear_alphabeta(module, activation_info, relevance_output)

    def _pad_to_match(self, x, expected_size):
        """Pad or truncate x along dim=1 to match expected_size."""
        if x.size(1) == expected_size:
            return x, x.size(1)
        orig = x.size(1)
        if x.size(1) < expected_size:
            pad = torch.zeros(
                x.size(0), expected_size - x.size(1),
                device=x.device, dtype=x.dtype,
            )
            return torch.cat([x, pad], dim=1), orig
        return x[:, :expected_size], orig

    def _match_relevance_dim(self, relevance, target_size):
        """Pad or truncate relevance along dim=1."""
        if relevance.size(1) == target_size:
            return relevance
        if relevance.size(1) < target_size:
            pad = torch.zeros(
                relevance.size(0), target_size - relevance.size(1),
                device=relevance.device, dtype=relevance.dtype,
            )
            return torch.cat([relevance, pad], dim=1)
        return relevance[:, :target_size]

    def _lrp_linear_epsilon(self, module, activation_info, relevance_output):
        x = self._get_input(activation_info)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x_padded, orig_size = self._pad_to_match(x, module.weight.size(1))
        z = F.linear(x_padded, module.weight, module.bias)
        relevance_output = self._match_relevance_dim(relevance_output, z.size(1))

        z = stabilize_denominator(z, self.epsilon, rule="epsilon")
        c = torch.einsum("bo,oi->bi", relevance_output / z, module.weight)
        result = x_padded * c

        if result.size(1) != orig_size:
            result = result[:, :orig_size]
        return result

    def _lrp_linear_alphabeta(self, module, activation_info, relevance_output):
        x = self._get_input(activation_info)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x_padded, orig_size = self._pad_to_match(x, module.weight.size(1))
        W_pos, W_neg = torch.clamp(module.weight, min=0), torch.clamp(module.weight, max=0)
        b_pos = torch.clamp(module.bias, min=0) if module.bias is not None else None
        b_neg = torch.clamp(module.bias, max=0) if module.bias is not None else None

        z_pos = F.linear(x_padded, W_pos, b_pos) + 1e-9
        z_neg = F.linear(x_padded, W_neg, b_neg) - 1e-9
        relevance_output = self._match_relevance_dim(relevance_output, z_pos.size(1))

        c_pos = torch.einsum("bo,oi->bi", relevance_output / z_pos, W_pos)
        c_neg = torch.einsum("bo,oi->bi", relevance_output / z_neg, W_neg)
        result = x_padded * (self.alpha * c_pos - self.beta * c_neg)

        if result.size(1) != orig_size:
            result = result[:, :orig_size]
        return result

    def _lrp_conv2d(self, module, activation_info, relevance_output):
        if self.rule == "epsilon":
            return self._lrp_conv2d_epsilon(module, activation_info, relevance_output)
        return self._lrp_conv2d_alphabeta(module, activation_info, relevance_output)

    @staticmethod
    def _conv_output_padding(module, z_shape, x_shape):
        pads = []
        for i in range(2):
            s = module.stride[i] if isinstance(module.stride, tuple) else module.stride
            p = module.padding[i] if isinstance(module.padding, tuple) else module.padding
            k = module.weight.shape[2 + i]
            expected = (z_shape[2 + i] - 1) * s - 2 * p + k
            pads.append(max(0, x_shape[2 + i] - expected))
        return tuple(pads)

    @staticmethod
    def _crop_spatial(tensor, target_shape):
        if tensor.shape[2:] == target_shape[2:]:
            return tensor
        if tensor.shape[2] > target_shape[2] or tensor.shape[3] > target_shape[3]:
            return tensor[:, :, : target_shape[2], : target_shape[3]]
        pad_h = target_shape[2] - tensor.shape[2]
        pad_w = target_shape[3] - tensor.shape[3]
        return F.pad(tensor, (0, pad_w, 0, pad_h))

    def _lrp_conv2d_epsilon(self, module, activation_info, relevance_output):
        x = self._get_input(activation_info)
        conv_kw = dict(
            stride=module.stride, padding=module.padding,
            dilation=module.dilation, groups=module.groups,
        )
        z = F.conv2d(x, module.weight, module.bias, **conv_kw)
        z = stabilize_denominator(z, self.epsilon, rule="epsilon")
        s = relevance_output / z

        out_pad = self._conv_output_padding(module, z.shape, x.shape)
        c = F.conv_transpose2d(s, module.weight, stride=module.stride,
                               padding=module.padding, output_padding=out_pad,
                               dilation=module.dilation, groups=module.groups)
        return x * self._crop_spatial(c, x.shape)

    def _lrp_conv2d_alphabeta(self, module, activation_info, relevance_output):
        x = self._get_input(activation_info)
        conv_kw = dict(
            stride=module.stride, padding=module.padding,
            dilation=module.dilation, groups=module.groups,
        )
        W_pos, W_neg = torch.clamp(module.weight, min=0), torch.clamp(module.weight, max=0)
        b_pos = torch.clamp(module.bias, min=0) if module.bias is not None else None
        b_neg = torch.clamp(module.bias, max=0) if module.bias is not None else None

        z_pos = F.conv2d(x, W_pos, b_pos, **conv_kw)
        z_neg = F.conv2d(x, W_neg, b_neg, **conv_kw)
        z_sum = z_pos + z_neg
        sign = z_sum.sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        z_total = z_sum + self.epsilon * sign

        s = relevance_output / z_total
        out_pad = self._conv_output_padding(module, z_pos.shape, x.shape)
        trans_kw = dict(stride=module.stride, padding=module.padding,
                        output_padding=out_pad, dilation=module.dilation,
                        groups=module.groups)

        c_pos = F.conv_transpose2d(s, W_pos, **trans_kw)
        c_neg = F.conv_transpose2d(s, W_neg, **trans_kw)
        c_pos = self._crop_spatial(c_pos, x.shape)
        c_neg = self._crop_spatial(c_neg, x.shape)
        return x * (self.alpha * c_pos + self.beta * c_neg)

    def _lrp_maxpool2d(self, module, activation_info, relevance_output):
        x = self._get_input(activation_info)
        _, indices = F.max_pool2d(
            x, kernel_size=module.kernel_size, stride=module.stride,
            padding=module.padding, dilation=module.dilation,
            return_indices=True,
        )
        return F.max_unpool2d(
            relevance_output, indices, kernel_size=module.kernel_size,
            stride=module.stride, padding=module.padding, output_size=x.size(),
        )

    def _lrp_avgpool2d(self, module, activation_info, relevance_output):
        x = self._get_input(activation_info)
        if isinstance(module, nn.AdaptiveAvgPool2d):
            return F.interpolate(
                relevance_output, size=x.shape[2:], mode="bilinear",
                align_corners=False,
            )
        channels = relevance_output.size(1)
        kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
        padding = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
        weight = torch.ones(channels, 1, *kernel_size, device=x.device) / (kernel_size[0] * kernel_size[1])
        result = F.conv_transpose2d(relevance_output, weight, stride=stride,
                                     padding=padding, groups=channels)
        return self._crop_spatial(result, x.shape)

    def _lrp_rnn(self, module, activation_info, relevance_output):
        x = self._get_input(activation_info)
        if x.dim() == 3:
            return relevance_output.unsqueeze(1).expand(x.shape[0], x.shape[1], -1)
        return relevance_output

    # ------------------------------------------------------------------
    # Shape utilities
    # ------------------------------------------------------------------

    def _match_shapes(self, relevance, target_shape):
        if relevance.shape == target_shape:
            return relevance
        batch_size = relevance.shape[0]

        # 2D -> 4D
        if relevance.dim() == 2 and len(target_shape) == 4:
            total = target_shape[1] * target_shape[2] * target_shape[3]
            if relevance.shape[1] == total:
                return relevance.view(batch_size, *target_shape[1:])
            return (relevance.sum(dim=1, keepdim=True) / total).view(
                batch_size, 1, 1, 1
            ).expand(batch_size, *target_shape[1:])

        # 4D -> 4D
        if relevance.dim() == 4 and len(target_shape) == 4:
            if relevance.shape[1] != target_shape[1]:
                relevance = relevance.mean(dim=1, keepdim=True).expand(
                    -1, target_shape[1], -1, -1
                )
            if relevance.shape[2:] != target_shape[2:]:
                relevance = F.interpolate(
                    relevance, size=target_shape[2:], mode="bilinear",
                    align_corners=False,
                )
            return relevance

        # 3D -> 4D
        if relevance.dim() == 3 and len(target_shape) == 4:
            relevance = relevance.unsqueeze(1).expand(
                -1, target_shape[1], -1, -1
            )
            if relevance.shape[2:] != target_shape[2:]:
                relevance = F.interpolate(
                    relevance, size=target_shape[2:], mode="bilinear",
                    align_corners=False,
                )
            return relevance

        return relevance

    def _split_relevance_to_features(self, relevance, input_embeddings):
        """Split combined relevance back to individual features."""
        result = {}

        # Already split per feature
        if isinstance(relevance, dict):
            for key, rel in relevance.items():
                if key not in input_embeddings:
                    continue
                emb = input_embeddings[key]
                if emb.dim() == 3 and rel.dim() == 2:
                    rel = rel.unsqueeze(1).expand_as(emb)
                result[key] = rel
            return result

        # Compute per-feature sizes (after pooling)
        feature_sizes = {}
        for key, emb in input_embeddings.items():
            if emb.dim() == 3:
                feature_sizes[key] = emb.size(2)
            elif emb.dim() == 2:
                feature_sizes[key] = emb.size(1)
            else:
                feature_sizes[key] = emb.numel() // emb.size(0)

        total = sum(feature_sizes.values())
        if relevance.dim() != 2 or relevance.size(1) != total:
            # Fallback: distribute equally
            for key in input_embeddings:
                result[key] = relevance / len(input_embeddings)
            return result

        idx = 0
        for key in self.model.feature_keys:
            if key not in input_embeddings:
                continue
            size = feature_sizes[key]
            chunk = relevance[:, idx : idx + size]
            emb = input_embeddings[key]
            if emb.dim() == 3:
                chunk = chunk.unsqueeze(1).expand_as(emb)
            result[key] = chunk
            idx += size

        return result