from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit

from .embedding import EmbeddingModel


class StageNetLayer(nn.Module):
    """StageNet layer.

    Paper: Stagenet: Stage-aware neural networks for health risk prediction. WWW 2020.

    This layer is used in the StageNet model. But it can also be used as a
    standalone layer.

    Args:
        input_dim: dynamic feature size.
        chunk_size: the chunk size for the StageNet layer. Default is 128.
        levels: the number of levels for the StageNet layer. levels * chunk_size = hidden_dim in the RNN. Smaller chunk size and more levels can capture more detailed patient status variations. Default is 3.
        conv_size: the size of the convolutional kernel. Default is 10.
        dropconnect: the dropout rate for the dropconnect. Default is 0.3.
        dropout: the dropout rate for the dropout. Default is 0.3.
        dropres: the dropout rate for the residual connection. Default is 0.3.

    Examples:
        >>> from pyhealth.models import StageNetLayer
        >>> input = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = StageNetLayer(64)
        >>> c, _, _ = layer(input)
        >>> c.shape
        torch.Size([3, 384])
    """

    def __init__(
        self,
        input_dim: int,
        chunk_size: int = 128,
        conv_size: int = 10,
        levels: int = 3,
        dropconnect: int = 0.3,
        dropout: int = 0.3,
        dropres: int = 0.3,
    ):
        super(StageNetLayer, self).__init__()

        self.dropout = dropout
        self.dropconnect = dropconnect
        self.dropres = dropres
        self.input_dim = input_dim
        self.hidden_dim = chunk_size * levels
        self.conv_dim = self.hidden_dim
        self.conv_size = conv_size
        # self.output_dim = output_dim
        self.levels = levels
        self.chunk_size = chunk_size

        self.kernel = nn.Linear(
            int(input_dim + 1), int(self.hidden_dim * 4 + levels * 2)
        )
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.zeros_(self.kernel.bias)
        self.recurrent_kernel = nn.Linear(
            int(self.hidden_dim + 1), int(self.hidden_dim * 4 + levels * 2)
        )
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.recurrent_kernel.bias)

        self.nn_scale = nn.Linear(int(self.hidden_dim), int(self.hidden_dim // 6))
        self.nn_rescale = nn.Linear(int(self.hidden_dim // 6), int(self.hidden_dim))
        self.nn_conv = nn.Conv1d(
            int(self.hidden_dim), int(self.conv_dim), int(conv_size), 1
        )
        # self.nn_output = nn.Linear(int(self.conv_dim), int(output_dim))

        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=dropout)
            self.nn_dropres = nn.Dropout(p=dropres)

        # Hooks for interpretability (e.g., DeepLIFT) default to None
        self._activation_hooks = None

    def set_activation_hooks(self, hooks) -> None:
        """Registers activation hooks for interpretability methods.

        Args:
            hooks: Object exposing ``apply(name, tensor, **kwargs)``. When
                provided, activation functions inside the layer will be
                routed through ``hooks`` instead of raw torch.ops. Passing
                ``None`` disables the hooks.
        """

        self._activation_hooks = hooks

    def _apply_activation(self, name: str, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._activation_hooks is not None and hasattr(self._activation_hooks, "apply"):
            return self._activation_hooks.apply(name, tensor, **kwargs)
        fn = getattr(torch, name)
        return fn(tensor, **kwargs)

    def cumax(self, x, mode="l2r"):
        if mode == "l2r":
            x = self._apply_activation("softmax", x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return x
        elif mode == "r2l":
            x = torch.flip(x, [-1])
            x = self._apply_activation("softmax", x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return torch.flip(x, [-1])
        else:
            return x

    def step(self, inputs, c_last, h_last, interval, device):
        x_in = inputs.to(device=device)

        # Integrate inter-visit time intervals
        interval = interval.unsqueeze(-1).to(device=device)
        x_out1 = self.kernel(torch.cat((x_in, interval), dim=-1)).to(device)
        x_out2 = self.recurrent_kernel(
            torch.cat((h_last.to(device=device), interval), dim=-1)
        )

        if self.dropconnect:
            x_out1 = self.nn_dropconnect(x_out1)
            x_out2 = self.nn_dropconnect_r(x_out2)
        x_out = x_out1 + x_out2
        f_master_gate = self.cumax(x_out[:, : self.levels], "l2r")
        f_master_gate = f_master_gate.unsqueeze(2).to(device=device)
        i_master_gate = self.cumax(x_out[:, self.levels : self.levels * 2], "r2l")
        i_master_gate = i_master_gate.unsqueeze(2)
        x_out = x_out[:, self.levels * 2 :]
        x_out = x_out.reshape(-1, self.levels * 4, self.chunk_size)
        f_gate = self._apply_activation("sigmoid", x_out[:, : self.levels]).to(
            device=device
        )
        i_gate = self._apply_activation(
            "sigmoid", x_out[:, self.levels : self.levels * 2]
        ).to(device=device)
        o_gate = self._apply_activation(
            "sigmoid", x_out[:, self.levels * 2 : self.levels * 3]
        )
        c_in = self._apply_activation("tanh", x_out[:, self.levels * 3 :]).to(
            device=device
        )
        c_last = c_last.reshape(-1, self.levels, self.chunk_size).to(device=device)
        overlap = (f_master_gate * i_master_gate).to(device=device)
        c_out = (
            overlap * (f_gate * c_last + i_gate * c_in)
            + (f_master_gate - overlap) * c_last
            + (i_master_gate - overlap) * c_in
        )
        h_out = o_gate * self._apply_activation("tanh", c_out)
        c_out = c_out.reshape(-1, self.hidden_dim)
        h_out = h_out.reshape(-1, self.hidden_dim)
        out = torch.cat([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1)
        return out, c_out, h_out

    def forward(
        self,
        x: torch.tensor,
        time: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input_dim].
            static: a tensor of shape [batch size, static_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            last_output: a tensor of shape [batch size, chunk_size*levels] representing the
                patient embedding.
            outputs: a tensor of shape [batch size, sequence len, chunk_size*levels] representing the patient at each time step.
        """
        # rnn will only apply dropout between layers
        batch_size, time_step, feature_dim = x.size()
        device = x.device
        if time == None:
            time = torch.ones(batch_size, time_step)
        time = time.reshape(batch_size, time_step)
        c_out = torch.zeros(batch_size, self.hidden_dim)
        h_out = torch.zeros(batch_size, self.hidden_dim)

        tmp_h = (
            torch.zeros_like(h_out, dtype=torch.float32)
            .view(-1)
            .repeat(self.conv_size)
            .view(self.conv_size, batch_size, self.hidden_dim)
        )
        tmp_dis = torch.zeros((self.conv_size, batch_size))
        h = []
        origin_h = []
        distance = []
        for t in range(time_step):
            out, c_out, h_out = self.step(x[:, t, :], c_out, h_out, time[:, t], device)
            cur_distance = 1 - torch.mean(
                out[..., self.hidden_dim : self.hidden_dim + self.levels], -1
            )
            origin_h.append(out[..., : self.hidden_dim])
            tmp_h = torch.cat(
                (
                    tmp_h[1:].to(device=device),
                    out[..., : self.hidden_dim].unsqueeze(0).to(device=device),
                ),
                0,
            )
            tmp_dis = torch.cat(
                (
                    tmp_dis[1:].to(device=device),
                    cur_distance.unsqueeze(0).to(device=device),
                ),
                0,
            )
            distance.append(cur_distance)

            # Re-weighted convolution operation
            local_dis = tmp_dis.permute(1, 0)
            local_dis = torch.cumsum(local_dis, dim=1)
            local_dis = self._apply_activation("softmax", local_dis, dim=1)
            local_h = tmp_h.permute(1, 2, 0)
            local_h = local_h * local_dis.unsqueeze(1)

            # Re-calibrate Progression patterns
            local_theme = torch.mean(local_h, dim=-1)
            local_theme = self.nn_scale(local_theme).to(device)
            local_theme = self._apply_activation("relu", local_theme)
            local_theme = self.nn_rescale(local_theme).to(device)
            local_theme = self._apply_activation("sigmoid", local_theme)

            local_h = self.nn_conv(local_h).squeeze(-1)
            local_h = local_theme * local_h
            h.append(local_h)

        origin_h = torch.stack(origin_h).permute(1, 0, 2)
        rnn_outputs = torch.stack(h).permute(1, 0, 2)
        if self.dropres > 0.0:
            origin_h = self.nn_dropres(origin_h)
        rnn_outputs = rnn_outputs + origin_h
        rnn_outputs = rnn_outputs.contiguous().view(-1, rnn_outputs.size(-1))
        if self.dropout > 0.0:
            rnn_outputs = self.nn_dropout(rnn_outputs)

        output = rnn_outputs.contiguous().view(batch_size, time_step, self.hidden_dim)
        last_output = get_last_visit(output, mask)

        return last_output, output, torch.stack(distance)


class StageNet(BaseModel):
    """StageNet model.

    Paper: Junyi Gao et al. Stagenet: Stage-aware neural networks for health
    risk prediction. WWW 2020.

    This model uses the StageNetProcessor which expects inputs in the format:
        {"value": [...], "time": [...]}

    The processor handles various input types:
        - Code sequences (with/without time intervals)
        - Nested code sequences (with/without time intervals)
        - Numeric feature vectors (with/without time intervals)

    Time intervals are optional and represent inter-event delays. If not
    provided, all events are treated as having uniform time intervals.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        chunk_size: the chunk size for the StageNet layer. Default is 128.
        levels: the number of levels for the StageNet layer.
            levels * chunk_size = hidden_dim in the RNN. Smaller chunk_size
            and more levels can capture more detailed patient status
            variations. Default is 3.
        **kwargs: other parameters for the StageNet layer.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "codes": {
        ...             "value": ["505800458", "50580045810", "50580045811"],
        ...             "time": [0.0, 2.0, 1.3],
        ...         },
        ...         "procedures": {
        ...             "value": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],
        ...             "time": [0.0, 1.5],
        ...         },
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-1",
        ...         "codes": {
        ...             "value": ["55154191800", "551541928", "55154192800"],
        ...             "time": [0.0, 2.0, 1.3],
        ...         },
        ...         "procedures": {
        ...             "value": [["A04A", "B035", "C129"]],
        ...             "time": [0.0],
        ...         },
        ...         "label": 0,
        ...     },
        ... ]
        >>>
        >>> # dataset
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "codes": "stagenet",
        ...         "procedures": "stagenet",
        ...     },
        ...     output_schema={"label": "binary"},
        ...     dataset_name="test"
        ... )
        >>>
        >>> # data loader
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>>
        >>> # model
        >>> model = StageNet(dataset=dataset)
        >>>
        >>> # data batch
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> # try the model
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(...),
            'y_prob': tensor(...),
            'y_true': tensor(...),
            'logit': tensor(...)
        }
        >>>

    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        chunk_size: int = 128,
        levels: int = 3,
        **kwargs,
    ):
        super(StageNet, self).__init__(
            dataset=dataset,
        )
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.levels = levels

        # validate kwargs for StageNet layer
        if "input_dim" in kwargs:
            raise ValueError("input_dim is determined by embedding_dim")

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        # Use EmbeddingModel for unified embedding handling
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Create StageNet layers for each feature
        self.stagenet = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.stagenet[feature_key] = StageNetLayer(
                input_dim=embedding_dim,
                chunk_size=self.chunk_size,
                levels=self.levels,
                **kwargs,
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(
            len(self.feature_keys) * self.chunk_size * self.levels, output_size
        )

        self._deeplift_hooks = None

    # ------------------------------------------------------------------
    # Interpretability support (e.g., DeepLIFT)
    # ------------------------------------------------------------------
    def set_deeplift_hooks(self, hooks) -> None:
        """Attach activation hooks for interpretability algorithms.

        Args:
            hooks: Object exposing ``apply(name, tensor, **kwargs)`` which
                will be invoked for activation calls within StageNet layers.
        """

        self._deeplift_hooks = hooks
        for layer in self.stagenet.values():
            if hasattr(layer, "set_activation_hooks"):
                layer.set_activation_hooks(hooks)

    def clear_deeplift_hooks(self) -> None:
        """Remove previously registered interpretability hooks."""

        self._deeplift_hooks = None
        for layer in self.stagenet.values():
            if hasattr(layer, "set_activation_hooks"):
                layer.set_activation_hooks(None)

    def forward_from_embedding(
        self,
        feature_embeddings: Dict[str, torch.Tensor],
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass starting from feature embeddings.

        This method bypasses the embedding layers but still performs
        temporal processing through StageNet layers. This is useful for
        interpretability methods like Integrated Gradients that need to
        interpolate in embedding space.

        Args:
            feature_embeddings: Dictionary mapping feature keys to their
                embedded representations. Each tensor should have shape
                [batch_size, seq_len, embedding_dim].
            time_info: Optional dictionary mapping feature keys to their
                time information tensors of shape [batch_size, seq_len].
                If None, uniform time intervals are assumed.
            **kwargs: Additional keyword arguments, must include the label
                key for loss computation.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the final loss.
                y_prob: a tensor of predicted probabilities.
                y_true: a tensor representing the true labels.
                logit: the raw logits before activation.
                embed: (if embed=True in kwargs) the patient embedding.
        """
        patient_emb = []
        distance = []

        for feature_key in self.feature_keys:
            # Get embedded feature
            x = feature_embeddings[feature_key].to(self.device)
            # x: [batch, seq_len, embedding_dim] or 4D nested

            # Handle nested sequences (4D) by pooling over inner dim
            # This matches forward() processing for consistency
            if x.dim() == 4:  # [batch, seq_len, inner_len, embedding_dim]
                # Sum pool over inner dimension
                x = x.sum(dim=2)  # [batch, seq_len, embedding_dim]

            # Get time information if available
            time = None
            if time_info is not None and feature_key in time_info:
                if time_info[feature_key] is not None:
                    time = time_info[feature_key].to(self.device)
                    # Ensure time is 2D [batch, seq_len]
                    if time.dim() == 1:
                        time = time.unsqueeze(0)

            # Create mask from embedded values
            mask = (x.sum(dim=-1) != 0).int()  # [batch, seq_len]

            # Pass through StageNet layer with embedded features
            last_output, _, cur_dis = self.stagenet[feature_key](
                x, time=time, mask=mask
            )

            patient_emb.append(last_output)
            distance.append(cur_dis)

        # Concatenate all feature embeddings
        patient_emb = torch.cat(patient_emb, dim=1)

        # Register hook if needed for gradient tracking
        if patient_emb.requires_grad:
            patient_emb.register_hook(lambda grad: grad)

        # Pass through final classification layer
        logits = self.fc(patient_emb)

        # Obtain y_true, loss, y_prob
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)

        y_prob = self.prepare_y_prob(logits)
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }

        # Optionally return embeddings
        if kwargs.get("embed", False):
            results["embed"] = patient_emb

        return results

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each
        patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key. Feature keys should
                contain tuples of (time, values) from temporal processors.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the final loss.
                distance: list of tensors of stage variation.
                y_prob: a tensor of predicted probabilities.
                y_true: a tensor representing the true labels.
        """
        patient_emb = []
        distance = []

        for feature_key in self.feature_keys:
            # Extract (time, values) tuple
            feature = kwargs[feature_key]

            # Get value and time tensors from tuple
            if isinstance(feature, tuple) and len(feature) == 2:
                time, x = feature  # Unpack (time, values)
                # x: [batch, seq_len] or [batch, seq_len, dim]
                # time: [batch, seq_len] or None

                # Warn if time information is missing
                if time is None:
                    import warnings

                    warnings.warn(
                        f"Feature '{feature_key}' does not have time "
                        f"intervals. StageNet's temporal modeling "
                        f"capabilities will be limited. Consider using "
                        f"StageNet format with time intervals for "
                        f"better performance.",
                        UserWarning,
                    )
            else:
                # Fallback for backward compatibility
                import warnings

                warnings.warn(
                    f"Feature '{feature_key}' is not a temporal tuple. "
                    f"Using fallback mode without time intervals. "
                    f"The model may not learn temporal patterns properly. "
                    f"Please use 'stagenet' or 'stagenet_tensor' "
                    f"processors in your input schema.",
                    UserWarning,
                )
                x = feature
                time = None

            # Embed the values using EmbeddingModel
            # Need to pass as dict for EmbeddingModel
            embedded = self.embedding_model({feature_key: x})
            x = embedded[feature_key]  # [batch, seq_len, embedding_dim]
            # Handle nested sequences (2D codes -> need pooling on inner dim)
            if x.dim() == 4:  # [batch, seq_len, inner_len, embedding_dim]
                # Sum pool over inner dimension
                x = x.sum(dim=2)  # [batch, seq_len, embedding_dim]

            # Create mask from embedded values
            mask = (x.sum(dim=-1) != 0).int()  # [batch, seq_len]

            # Move time to correct device if present
            if time is not None:
                time = time.to(self.device)
                # Ensure time is 2D [batch, seq_len]
                if time.dim() == 1:
                    time = time.unsqueeze(0)

            # Pass through StageNet layer
            last_output, _, cur_dis = self.stagenet[feature_key](
                x, time=time, mask=mask
            )

            patient_emb.append(last_output)

            distance.append(cur_dis)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)

        # obtain y_true, loss, y_prob
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)

        y_prob = self.prepare_y_prob(logits)
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


if __name__ == "__main__":
    from pyhealth.datasets import create_sample_dataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "codes": (
                [0.0, 2.0, 1.3],
                ["505800458", "50580045810", "50580045811"],
            ),
            "procedures": (
                [0.0, 1.5],
                [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],
            ),
            "label": 1,
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            "codes": (
                [0.0, 2.0, 1.3, 1.0, 2.0],
                [
                    "55154191800",
                    "551541928",
                    "55154192800",
                    "705182798",
                    "70518279800",
                ],
            ),
            "procedures": (
                [0.0],
                [["A04A", "B035", "C129"]],
            ),
            "label": 0,
        },
    ]

    # dataset
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={
            "codes": "stagenet",
            "procedures": "stagenet",
        },
        output_schema={"label": "binary"},
        dataset_name="test",
    )

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = StageNet(dataset=dataset)

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
