"""GRU-D model for multivariate time series with missing values.

This module implements GRU-D (Gated Recurrent Unit with Decay), a recurrent
neural network designed specifically for irregularly sampled clinical time
series. Unlike standard RNNs, GRU-D explicitly models two representations of
informative missingness: the observed mask and the time since last measurement.

References:
    Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018).
    Recurrent neural networks for multivariate time series with missing
    values. Scientific Reports, 8(1), 6085.
    https://doi.org/10.1038/s41598-018-24271-9

    Nestor, B., McDermott, M. B. A., Boag, W., Berner, G., Naumann, T.,
    Hughes, M. C., Goldenberg, A., & Ghassemi, M. (2019). Feature
    robustness in non-stationary health records: Caveats to deployable
    model performance in common clinical machine learning tasks.
    arXiv:1908.00690. https://arxiv.org/abs/1908.00690
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class FilterLinear(nn.Module):
    """A linear layer with a binary filter mask applied to its weights.

    Implements the input decay weight matrix in GRU-D (Wgamma_x) as a
    diagonal structure via element-wise multiplication with a fixed binary
    filter. This ensures each feature's decay rate is learned independently,
    preventing cross-feature interference in the decay computation.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        filter_square_matrix: Binary tensor of shape
            ``(out_features, in_features)`` where each element is 0 or 1.
            Typically an identity matrix for diagonal decay.
        bias: If ``True``, adds a learnable bias to the layer.
            Default is ``True``.

    Attributes:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        filter_square_matrix: Non-learnable binary filter mask.
        weight: Learnable weight matrix of shape
            ``(out_features, in_features)``.
        bias: Optional learnable bias of shape ``(out_features,)``.

    Example:
        >>> import torch
        >>> filt = torch.eye(5)
        >>> layer = FilterLinear(5, 5, filt)
        >>> x = torch.randn(3, 5)
        >>> out = layer(x)
        >>> out.shape
        torch.Size([3, 5])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        filter_square_matrix: torch.Tensor,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("filter_square_matrix", filter_square_matrix)
        self.weight = Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialises weights and bias with uniform distribution.

        Uses the standard deviation ``1 / sqrt(in_features)`` following
        the default PyTorch Linear layer initialisation scheme.
        """
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the filtered linear transformation.

        Multiplies the weight matrix element-wise with the binary filter
        before computing the linear transformation, effectively zeroing out
        off-diagonal weights.

        Args:
            x: Input tensor of shape ``(batch_size, in_features)``.

        Returns:
            Output tensor of shape ``(batch_size, out_features)``.
        """
        return F.linear(
            x, self.filter_square_matrix * self.weight, self.bias
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None})"
        )


class GRUDLayer(nn.Module):
    """Core GRU-D recurrent layer with temporal decay for missing data.

    GRU-D extends the standard GRU with two decay mechanisms that model
    informative missingness in irregularly sampled clinical time series:

    - **Input decay** (``gamma_x``): Decays the last observed value of
      each feature towards the global training mean as the time since
      last observation increases. Implemented via ``FilterLinear`` to
      ensure per-feature independent decay rates.
    - **Hidden decay** (``gamma_h``): Decays the hidden state towards
      zero to reduce the influence of stale patient state information.

    At each timestep the effective input is computed as:

    .. math::

        \\tilde{x}_t = m_t \\odot x_t
            + (1 - m_t) \\odot (\\gamma_x \\odot x_{t-1}^{\\prime}
            + (1 - \\gamma_x) \\odot \\bar{x})

    where :math:`m_t` is the observed mask, :math:`x_{t-1}^{\\prime}`
    is the forward-filled last observation, and :math:`\\bar{x}` is the
    global training mean.

    **Input convention:** Both ``x`` and ``x_last_obsv`` are expected to
    contain forward-filled values — i.e. simple imputation has already been
    applied upstream by the processing pipeline. The ``mask`` tensor
    distinguishes positions where a value was directly observed (``mask=1``)
    from positions where the value was imputed by forward-filling
    (``mask=0``). This matches the interleaved format produced by
    :meth:`GRUD.prepare_input` and consumed by :meth:`GRUD.forward`.

    Args:
        input_size: Number of input features (clinical variables) per
            timestep.
        hidden_size: Dimensionality of the GRU hidden state.
        x_mean: Global mean tensor of shape ``(1, 1, input_size)``
            computed from the training set by averaging over both samples
            and timesteps. Used as the long-term imputation target for
            input decay, matching scalar x-bar in Che et al. (2018) Eq. 3.
        use_input_decay: If ``True``, applies learned input decay.
            If ``False``, uses simple forward filling — ablation mode
            that removes the input decay contribution. Default is
            ``True``.
        use_hidden_decay: If ``True``, applies learned hidden state
            decay proportional to elapsed time. If ``False``, the
            hidden state is passed unchanged — ablation mode that
            removes the hidden decay contribution. Default is ``True``.

    Attributes:
        input_size: Number of input features per timestep.
        hidden_size: Dimensionality of the hidden state.
        use_input_decay: Whether input decay is active.
        use_hidden_decay: Whether hidden state decay is active.
        update_gate: Linear layer for the GRU update gate.
        reset_gate: Linear layer for the GRU reset gate.
        new_gate: Linear layer for the GRU candidate hidden state.
        gamma_x: ``FilterLinear`` layer for per-feature input decay.
        gamma_h: Linear layer for hidden state decay.

    Example:
        >>> layer = GRUDLayer(
        ...     input_size=10,
        ...     hidden_size=32,
        ...     x_mean=torch.zeros(1, 1, 10),
        ... )
        >>> x      = torch.randn(4, 24, 10)
        >>> x_last = torch.randn(4, 24, 10)
        >>> mask   = torch.randint(0, 2, (4, 24, 10)).float()
        >>> delta  = torch.rand(4, 24, 10) * 5
        >>> out    = layer(x, x_last, mask, delta)
        >>> out.shape
        torch.Size([4, 32])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        x_mean: torch.Tensor,
        use_input_decay: bool = True,
        use_hidden_decay: bool = True,
    ) -> None:
        super().__init__()
        self.input_size       = input_size
        self.hidden_size      = hidden_size
        self.use_input_decay  = use_input_decay
        self.use_hidden_decay = use_hidden_decay

        # Registered as buffers so they move with .to(device) automatically
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("zeros_input", torch.zeros(input_size))
        self.register_buffer("zeros_hidden", torch.zeros(hidden_size))

        # GRU gate layers — input is [x_tilde, h, mask] concatenated
        gate_input_size  = input_size + hidden_size + input_size
        self.update_gate = nn.Linear(gate_input_size, hidden_size)
        self.reset_gate  = nn.Linear(gate_input_size, hidden_size)
        self.new_gate    = nn.Linear(gate_input_size, hidden_size)

        # Decay rate layers (Eq. 3-4 in Che et al. 2018)
        self.gamma_x = FilterLinear(
            input_size, input_size, torch.eye(input_size)
        )
        self.gamma_h = nn.Linear(input_size, hidden_size)

    def _decay_step(
        self,
        x: torch.Tensor,
        x_last_obsv: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Performs one GRU-D recurrent step with temporal decay.

        Implements Equations 3-8 from Che et al. (2018):
            1. Compute input decay gamma_x from elapsed time delta.
            2. Impute missing values using decayed forward fill.
            3. Compute hidden decay gamma_h and apply to hidden state.
            4. Run standard GRU gates with mask concatenated as input.

        Args:
            x: Forward-filled measurement values at the current timestep,
                shape ``(batch_size, input_size)``. Observed positions
                contain the true measurement; imputed positions contain
                the forward-filled value from the last observation.
            x_last_obsv: Forward-filled measurements, same shape and
                semantics as ``x``. Both ``x`` and ``x_last_obsv`` carry
                forward-filled values; ``mask`` distinguishes which
                positions were directly observed versus imputed.
            mask: Binary observation mask at current timestep, shape
                ``(batch_size, input_size)``. 1 = directly observed,
                0 = imputed by forward-fill upstream.
            delta: Hours since last observation per feature, shape
                ``(batch_size, input_size)``.
            h: Previous hidden state, shape
                ``(batch_size, hidden_size)``.

        Returns:
            Updated hidden state of shape ``(batch_size, hidden_size)``.
        """
        x_mean_t = self.x_mean.squeeze(0).squeeze(0)  # (input_size,)

        # Input decay: gamma_x in (0, 1] — approaches 1 when freshly
        # observed, approaches 0 as time since last observation grows
        if self.use_input_decay:
            gamma_x = torch.exp(
                -torch.max(self.zeros_input, self.gamma_x(delta))
            )
            x_tilde = (
                mask * x
                + (1 - mask)
                * (gamma_x * x_last_obsv + (1 - gamma_x) * x_mean_t)
            )
        else:
            # Ablation: simple forward fill, no temporal decay
            x_tilde = mask * x + (1 - mask) * x_last_obsv

        # Hidden decay: gamma_h in (0, 1] — reduces hidden state
        # influence proportional to elapsed time since last observation
        if self.use_hidden_decay:
            gamma_h = torch.exp(
                -torch.max(self.zeros_hidden, self.gamma_h(delta))
            )
            h = gamma_h * h
        # else: h unchanged — no hidden state decay (ablation mode)

        # Standard GRU gates with mask as additional input (Eq. 5-8)
        combined   = torch.cat([x_tilde, h, mask], dim=1)
        z          = torch.sigmoid(self.update_gate(combined))
        r          = torch.sigmoid(self.reset_gate(combined))
        combined_r = torch.cat([x_tilde, r * h, mask], dim=1)
        h_tilde    = torch.tanh(self.new_gate(combined_r))
        h          = (1 - z) * h + z * h_tilde
        return h

    def forward(
        self,
        x: torch.Tensor,
        x_last_obsv: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Processes the full time series through GRU-D.

        Iterates over each timestep, calling ``_decay_step`` to apply
        temporal decay and update the hidden state. The final hidden
        state captures the patient's clinical trajectory.

        Args:
            x: Raw measurements, shape
                ``(batch_size, seq_len, input_size)``.
            x_last_obsv: Forward-filled measurements, shape
                ``(batch_size, seq_len, input_size)``.
            mask: Binary observation mask, shape
                ``(batch_size, seq_len, input_size)``.
            delta: Time since last observation in hours, shape
                ``(batch_size, seq_len, input_size)``.

        Returns:
            Final hidden state of shape ``(batch_size, hidden_size)``,
            representing the patient's summarised clinical state.
        """
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(x.size(1)):
            h = self._decay_step(
                x[:, t, :],
                x_last_obsv[:, t, :],
                mask[:, t, :],
                delta[:, t, :],
                h,
            )
        return h


class GRUD(BaseModel):
    """GRU-D model for multivariate time series with missing values.

    GRU-D (Gated Recurrent Unit with Decay) is a recurrent model designed
    for clinical EHR time series where data is irregularly sampled and
    frequently missing. It extends the standard GRU with two learned
    temporal decay mechanisms that explicitly model informative missingness:

    - **Input decay** decays the last observed value towards the global
      training mean as time since last measurement increases, capturing
      the clinical intuition that old measurements become less reliable.
    - **Hidden state decay** reduces the influence of the previous hidden
      state proportional to elapsed time, modelling increasing uncertainty
      about patient state when no measurements are available.

    The model satisfies PyHealth's ``BaseModel`` interface and infers
    feature keys and label keys automatically from ``dataset.input_schema``
    and ``dataset.output_schema`` respectively.

    Input features must use the interleaved channel format produced by the
    simple imputer pipeline:
    ``[mask_0, mean_0, time_since_0, mask_1, mean_1, time_since_1, ...]``

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset` whose
            ``input_schema`` maps feature keys to ``"tensor"`` and
            ``output_schema`` maps the label key to the task type
            (e.g. ``"binary"``). Each input tensor must have shape
            ``(seq_len, n_vars * 3)`` with interleaved channels.

            .. warning::

                If ``x_mean`` is not supplied, it is computed from
                ``dataset`` automatically. To avoid data leakage, always
                pass a dataset that contains **only training samples** —
                i.e. call ``GRUD(dataset=train_dataset)`` rather than
                ``GRUD(dataset=full_dataset)``. Computing ``x_mean`` over
                validation or test data leaks target-distribution
                information into the imputation fallback used at every
                timestep.

        hidden_size: Dimensionality of the GRU-D hidden state.
            Default is ``64``.
        dropout: Dropout probability applied to the concatenated hidden
            state before the output classifier. Default is ``0.5``.
        use_input_decay: If ``True`` (default), applies the learned input
            decay mechanism (gamma_x). Set to ``False`` to ablate input
            decay, reducing GRU-D to a GRU with forward-fill imputation.
        use_hidden_decay: If ``True`` (default), applies the learned
            hidden state decay mechanism (gamma_h). Set to ``False`` to
            ablate hidden decay.
        x_mean: Optional pre-computed global mean tensor of shape
            ``(1, 1, n_vars)``. When provided, this value is used
            directly as the imputation fallback in the input decay
            mechanism and ``_compute_x_mean`` is not called. Use this
            to supply a mean computed from a training-only split when
            ``dataset`` contains the full dataset, or to share the same
            mean across multiple folds in cross-validation:

            .. code-block:: python

                train_mean = GRUD(dataset=train_dataset)._compute_x_mean(
                    train_dataset, "time_series"
                )
                model = GRUD(dataset=full_dataset, x_mean=train_mean)

            Default is ``None`` (computed from ``dataset``).

    Attributes:
        hidden_size: Dimensionality of the GRU-D hidden state.
        dropout_rate: Dropout probability before the classifier.
        use_input_decay: Whether input decay is active.
        use_hidden_decay: Whether hidden state decay is active.
        input_size: Number of clinical variables inferred from the data.
        grud_layers: ``ModuleDict`` of ``GRUDLayer`` per feature key.
        dropout: Dropout module applied before classification.
        batch_norm: Batch normalisation on the concatenated embeddings.
        fc: Output linear classifier.

    Raises:
        ValueError: If the number of feature channels is not divisible
            by 3, indicating the expected interleaved format is violated.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> import numpy as np
        >>> def make_ts(n_vars=2, seq_len=3):
        ...     x = np.zeros((seq_len, n_vars * 3), dtype=np.float32)
        ...     x[:, 0::3] = 1.0
        ...     x[:, 1::3] = np.random.randn(seq_len, n_vars)
        ...     return x.tolist()
        >>> samples = [
        ...     {"patient_id": f"p{i}", "visit_id": f"v{i}",
        ...      "time_series": make_ts(), "label": i % 2}
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"time_series": "tensor"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="demo",
        ... )
        >>> model = GRUD(dataset=dataset, hidden_size=32, dropout=0.3)
        >>> from pyhealth.datasets import get_dataloader
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        >>> batch = next(iter(loader))
        >>> output = model(**batch)
        >>> print(list(output.keys()))
        ['loss', 'y_prob', 'y_true', 'logit']

    References:
        Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018).
        Recurrent neural networks for multivariate time series with missing
        values. Scientific Reports, 8(1), 6085.
        https://doi.org/10.1038/s41598-018-24271-9

        Nestor, B., McDermott, M. B. A., Boag, W., Berner, G., Naumann,
        T., Hughes, M. C., Goldenberg, A., & Ghassemi, M. (2019). Feature
        robustness in non-stationary health records: Caveats to deployable
        model performance in common clinical machine learning tasks.
        arXiv:1908.00690. https://arxiv.org/abs/1908.00690
    """

    def __init__(
        self,
        dataset: SampleDataset,
        hidden_size: int = 64,
        dropout: float = 0.5,
        use_input_decay: bool = True,
        use_hidden_decay: bool = True,
        x_mean: torch.Tensor = None,
    ) -> None:
        super().__init__(dataset=dataset)
        self.hidden_size      = hidden_size
        self.dropout_rate     = dropout
        self.use_input_decay  = use_input_decay
        self.use_hidden_decay = use_hidden_decay

        # feature_keys and label_keys are populated by BaseModel.__init__
        # from dataset.input_schema and dataset.output_schema respectively
        first_key = self.feature_keys[0]

        # Infer input_size from the first sample's feature tensor shape.
        # Channels are interleaved as (mask, mean, time_since) per variable
        # so total_channels must be divisible by 3.
        first_sample   = next(iter(dataset))
        sample_feature = torch.as_tensor(
            first_sample[first_key], dtype=torch.float32
        )
        total_channels = sample_feature.shape[-1]
        if total_channels % 3 != 0:
            raise ValueError(
                f"Feature '{first_key}' has {total_channels} channels, "
                "which is not divisible by 3. Expected interleaved "
                "(mask, mean, time_since_measured) channels."
            )
        self.input_size = total_channels // 3

        # Use caller-supplied x_mean if provided, otherwise compute from
        # dataset. The caller is responsible for ensuring x_mean is derived
        # from training data only to avoid leaking val/test statistics into
        # the imputation fallback used at every timestep.
        if x_mean is None:
            x_mean = self._compute_x_mean(dataset, first_key)

        # One GRUDLayer per feature key, sharing the same hyperparameters
        self.grud_layers = nn.ModuleDict(
            {
                key: GRUDLayer(
                    input_size=self.input_size,
                    hidden_size=hidden_size,
                    x_mean=x_mean,
                    use_input_decay=use_input_decay,
                    use_hidden_decay=use_hidden_decay,
                )
                for key in self.feature_keys
            }
        )

        self.dropout    = nn.Dropout(p=dropout)
        self.batch_norm = nn.BatchNorm1d(
            hidden_size * len(self.feature_keys)
        )
        self.fc = nn.Linear(
            hidden_size * len(self.feature_keys),
            self.get_output_size(),
        )

    def _compute_x_mean(
        self,
        dataset: SampleDataset,
        feature_key: str,
    ) -> torch.Tensor:
        """Computes per-feature global mean from the dataset.

        Extracts the mean channel (index 1 of every interleaved triplet)
        from all samples and averages over both samples and timesteps.
        This tensor is registered as a non-learnable buffer in
        ``GRUDLayer`` and used as the imputation fallback in input decay.

        Args:
            dataset: Dataset from which to compute the global mean.
                Should be the training dataset only to prevent leakage.
            feature_key: Key identifying the time series feature in
                each sample dictionary.

        Returns:
            Mean tensor of shape ``(1, 1, input_size)`` — a single
            per-feature global mean averaged over both samples and
            timesteps, matching the scalar x-bar in Che et al. (2018).
            Shape broadcasts over ``(batch_size, seq_len, input_size)``
            in ``GRUDLayer._decay_step``.
        """
        all_means = []
        for sample in dataset:
            feature = torch.as_tensor(
                sample[feature_key], dtype=torch.float32
            )
            # Mean channel is at every 3rd index starting from 1
            mean_channels = feature[:, 1::3]  # (seq_len, input_size)
            all_means.append(mean_channels)
        stacked = torch.stack(all_means, dim=0)       # (n, seq_len, input_size)
        return stacked.mean(dim=(0, 1), keepdim=True) # (1, 1, input_size)

    @staticmethod
    def _split_channels(
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splits interleaved (mask, mean, time_since) channels.

        The simple imputer pipeline stores three channels per variable
        in interleaved order:
        ``[mask_0, mean_0, tsm_0, mask_1, mean_1, tsm_1, ...]``

        This method extracts the three channel types back into separate
        tensors for use in ``GRUDLayer``.

        Args:
            x: Input tensor of shape
                ``(batch_size, seq_len, n_vars * 3)``.

        Returns:
            Tuple of three tensors each of shape
            ``(batch_size, seq_len, n_vars)``:

            - ``mask``: Binary observation indicator (1 = observed,
              0 = missing).
            - ``mean``: Forward-filled measurement values.
            - ``delta``: Hours since last observation per feature.
        """
        mask  = x[:, :, 0::3]
        mean  = x[:, :, 1::3]
        delta = x[:, :, 2::3]
        return mask, mean, delta

    @staticmethod
    def prepare_input(
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Converts separate channel tensors into GRU-D interleaved format.

        PyHealth time-series data is typically stored as separate tensors
        for values, missingness mask, and time since last observation.
        This helper interleaves them into the format expected by GRU-D:
        ``[mask_0, mean_0, time_since_0, mask_1, mean_1, time_since_1, ...]``

        This is the exact inverse of :meth:`_split_channels` — the two
        methods form a round-trip:

        .. code-block:: python

            x = GRUD.prepare_input(values, mask, delta)
            mask2, values2, delta2 = GRUD._split_channels(x)
            assert torch.equal(mask2,   mask)
            assert torch.equal(values2, values)
            assert torch.equal(delta2,  delta)

        Args:
            values: Forward-filled measurement values, shape
                ``(batch_size, seq_len, n_vars)`` or
                ``(seq_len, n_vars)`` for a single sample.
            mask: Binary observation indicator (1 = observed, 0 = missing),
                same shape as ``values``.
            delta: Hours since last observation per feature,
                same shape as ``values``.

        Returns:
            Interleaved tensor of shape ``(..., seq_len, n_vars * 3)``
            suitable for passing directly to :meth:`forward`.

        Example:
            >>> import torch
            >>> values = torch.randn(4, 24, 10)   # (batch, seq, vars)
            >>> mask   = torch.randint(0, 2, (4, 24, 10)).float()
            >>> delta  = torch.rand(4, 24, 10) * 5
            >>> x = GRUD.prepare_input(values, mask, delta)
            >>> x.shape
            torch.Size([4, 24, 30])
            >>> # Verify channel order: mask=0, mean=1, delta=2
            >>> assert torch.equal(x[..., 0::3], mask)
            >>> assert torch.equal(x[..., 1::3], values)
            >>> assert torch.equal(x[..., 2::3], delta)
        """
        n_vars = values.shape[-1]
        out    = torch.zeros(
            values.shape[:-1] + (n_vars * 3,),
            dtype=values.dtype,
            device=values.device,
        )
        out[..., 0::3] = mask
        out[..., 1::3] = values
        out[..., 2::3] = delta
        return out

    def forward(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs the GRU-D forward pass and computes loss and predictions.

        Satisfies the PyHealth ``BaseModel`` abstract ``forward`` method.
        For each feature key, extracts the three interleaved channel types,
        runs the corresponding ``GRUDLayer``, and concatenates the resulting
        patient embeddings before the output classifier.

        The batch dictionary is produced by PyHealth's ``get_dataloader``
        and contains one tensor per feature key plus the label key.

        Args:
            **kwargs: Batch dictionary where each feature key maps to a
                tensor of shape ``(batch_size, seq_len, n_vars * 3)``
                with interleaved ``(mask, mean, time_since_measured)``
                channels, and the label key maps to ground-truth labels.

        Returns:
            Dictionary with the following entries:

            - ``"loss"`` (:class:`torch.Tensor`): Scalar training loss
              computed by ``BaseModel.get_loss_function()``.
            - ``"y_prob"`` (:class:`torch.Tensor`): Predicted
              probabilities from ``BaseModel.prepare_y_prob()``.
            - ``"y_true"`` (:class:`torch.Tensor`): Ground-truth labels
              passed through from the input batch.
            - ``"logit"`` (:class:`torch.Tensor`): Raw pre-activation
              logits from the output linear layer.
        """
        patient_emb_list: List[torch.Tensor] = []

        for key in self.feature_keys:
            x = kwargs[key]
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(self.device).float()

            mask, mean_vals, delta = self._split_channels(x)
            # x_last_obsv is the forward-filled mean (already imputed)
            x_last_obsv = mean_vals.clone()

            h = self.grud_layers[key](
                x=mean_vals,
                x_last_obsv=x_last_obsv,
                mask=mask,
                delta=delta,
            )
            patient_emb_list.append(h)

        # Concatenate per-feature embeddings and apply regularisation
        patient_emb = torch.cat(patient_emb_list, dim=1)
        patient_emb = self.dropout(self.batch_norm(patient_emb))
        logits      = self.fc(patient_emb)

        label_key = self.label_keys[0]
        y_true    = kwargs[label_key]
        loss      = self.get_loss_function()(logits, y_true)
        y_prob    = self.prepare_y_prob(logits)

        return {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}