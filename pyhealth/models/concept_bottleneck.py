"""Concept bottleneck models for vasopressor onset prediction.

Port of Wu et al. (2022), "Learning Optimal Summaries of Clinical
Time-series with Concept Bottleneck Models" (MLHC 2022), original code at
https://github.com/dtak/optimal-summaries-public (``vasopressor/``).

Model: ``y_hat = f(g(x, h(x)))`` — :class:`SummaryEncoder` computes
differentiable interpretable summaries ``h`` of each variable, a linear
layer ``g`` maps summaries (+ raw last-hour values) to ``k`` concepts,
and a logistic head ``f`` predicts vasopressor onset.

The encoder, the concept bottleneck model, the Johnson et al. (2021)
logistic-regression baseline, the LSTM baseline and the greedy-selection
helpers are ports of ``vasopressor/models.py`` in the original
repository. Paper: https://proceedings.mlr.press/v182/wu22a.html

Deliberate differences from the original, each documented on the class
it concerns: a single output logit instead of two (PyHealth's binary
label convention), and ``batch_first=True`` in the LSTM baseline (the
original feeds batch-first tensors to a time-first LSTM).
"""

import itertools
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

# The nine summary groups that use a learned soft time window. The
# original code calls this ``num_cutoff_times`` and hardcodes 9 in every
# training script; it must equal the number of windowed groups.
WINDOWED_GROUPS: tuple[str, ...] = (
    "mean",
    "var",
    "ever_measured",
    "mean_indicators",
    "var_indicators",
    "switches",
    "slope",
    "hours_above_threshold",
    "hours_below_threshold",
)

# Full encoding order: raw last-hour row first, then the twelve summary
# groups, matching ``add_all_parsers`` + ``encode_patient_batch`` in the
# original ``models.py``.
ENCODING_GROUPS: tuple[str, ...] = (
    "raw_last_hour",
    "mean",
    "var",
    "ever_measured",
    "mean_indicators",
    "var_indicators",
    "switches",
    "slope",
    "slope_stderr",
    "first_time_measured",
    "last_time_measured",
    "hours_above_threshold",
    "hours_below_threshold",
)


class _SummaryLayout:
    """Index layout of the summary encoding (port of ``WeightsParser``).

    The windowed groups share one flat cutoff-weight vector of length
    ``len(WINDOWED_GROUPS) * changing_dim``; the full encoding has length
    ``input_dim + 12 * changing_dim``.

    Args:
        changing_dim: number of time-varying variables (28 in the paper).
        static_dim: number of static variables (10 in the original code;
            the paper counts 8, leaving out the two weight columns).

    Examples:
        >>> layout = _SummaryLayout(changing_dim=2, static_dim=1)
        >>> layout.encoding_dim
        29
        >>> layout.window_slice("mean")
        (0, 2)
    """

    def __init__(self, changing_dim: int, static_dim: int) -> None:
        self.changing_dim = changing_dim
        self.static_dim = static_dim
        self.input_dim = 2 * changing_dim + static_dim
        self.num_window_weights = len(WINDOWED_GROUPS) * changing_dim
        self.encoding_dim = self.input_dim + 12 * changing_dim
        self._window_slices = {}
        start = 0
        for group in WINDOWED_GROUPS:
            self._window_slices[group] = (start, start + changing_dim)
            start += changing_dim

    def window_slice(self, group: str) -> tuple[int, int]:
        """Returns the ``(start, end)`` slice of ``group`` in the flat
        cutoff-weight vector.

        Args:
            group: one of :data:`WINDOWED_GROUPS`.

        Returns:
            Tuple of ``(start, end)`` indices into the weight vector.
        """
        return self._window_slices[group]


def _init_cutoffs(kind: str, size: int) -> torch.Tensor:
    """Initial cutoff-time values (port of ``param_initializations.py``).

    Args:
        kind: ``"zero"``, ``"twelve"``, ``"twentyfour"`` or ``"random"``.
        size: number of cutoff parameters (``9 * changing_dim``).

    Returns:
        Tensor of shape ``(1, size)`` with the initial cutoff times.

    Examples:
        >>> _init_cutoffs("twelve", 3)
        tensor([[12., 12., 12.]])
    """
    if kind == "zero":
        return torch.zeros(1, size)
    if kind == "twelve":
        return torch.full((1, size), 12.0)
    if kind == "twentyfour":
        return torch.full((1, size), 24.0)
    if kind == "random":
        return torch.randint(0, 24, (1, size)).float()
    raise ValueError(f"Unknown cutoff init: {kind}")


class SummaryEncoder(nn.Module):
    """Differentiable interpretable summaries ``h`` (Wu et al. 2022).

    For each time-varying variable, computes window-weighted summaries
    over the first ``time_len`` hours with a learned soft window
    ``w = sigmoid((t - cutoff) / tau)`` per (variable, summary group),
    plus learned lower/upper thresholds for the time-above/below
    summaries. The output concatenates the raw last-hour row
    (values + indicators + statics) with the twelve summary groups.

    Args:
        changing_dim: number of time-varying variables. Default is 28.
        static_dim: number of static variables. Default is 10.
        num_cutoff_times: number of learned cutoff times per variable.
            Must equal 9 (the number of windowed summary groups).
            Default is 9.
        time_len: number of hourly steps per patient. Default is 6.
        cutoff_times_temperature: temperature ``tau`` of the soft
            window. Default is 1.0.
        thresholds_temperature: temperature of the threshold summaries.
            Default is 0.1.
        ever_measured_temperature: temperature of the ever-measured
            summary. Default is 0.1.
        switch_temperature: temperature of the switch summary (kept
            for API parity; the original does not use it in the
            computation). Default is 0.1.
        init_cutoffs: ``"zero"``, ``"twelve"``, ``"twentyfour"`` or
            ``"random"``. Default is ``"zero"``.
        init_thresholds: ``"rand"`` (lower ~ U(-1, 0), upper ~ U(0, 1))
            or ``"zeros"``. Default is ``"rand"``.
        differentiate_cutoffs: whether the cutoff times are learned.
            Default is True.

    Examples:
        >>> import torch
        >>> encoder = SummaryEncoder(changing_dim=2, static_dim=1)
        >>> values = torch.zeros(2, 6, 2)
        >>> indicators = torch.ones(2, 6, 2)
        >>> statics = torch.zeros(2, 1)
        >>> encoder(values, indicators, statics).shape
        torch.Size([2, 29])
    """

    def __init__(
        self,
        changing_dim: int = 28,
        static_dim: int = 10,
        num_cutoff_times: int = 9,
        time_len: int = 6,
        cutoff_times_temperature: float = 1.0,
        thresholds_temperature: float = 0.1,
        ever_measured_temperature: float = 0.1,
        switch_temperature: float = 0.1,
        init_cutoffs: str = "zero",
        init_thresholds: str = "rand",
        differentiate_cutoffs: bool = True,
    ) -> None:
        super().__init__()
        if num_cutoff_times != len(WINDOWED_GROUPS):
            raise ValueError(
                "num_cutoff_times must equal the number of windowed "
                f"summary groups ({len(WINDOWED_GROUPS)})"
            )
        self.changing_dim = changing_dim
        self.static_dim = static_dim
        self.num_cutoff_times = num_cutoff_times
        self.time_len = time_len
        self.layout = _SummaryLayout(changing_dim, static_dim)
        self.cutoff_times_temperature = cutoff_times_temperature
        self.thresholds_temperature = thresholds_temperature
        self.ever_measured_temperature = ever_measured_temperature
        self.switch_temperature = switch_temperature

        num_weights = changing_dim * num_cutoff_times
        if differentiate_cutoffs:
            self.cutoff_times = nn.Parameter(
                _init_cutoffs(init_cutoffs, num_weights)
            )
        else:
            # Fixed cutoffs of -12 include every timestep, as in the
            # original.
            self.register_buffer(
                "cutoff_times", torch.full((1, num_weights), -12.0)
            )
        # Hour index per (time, variable), repeated per cutoff group.
        times = torch.tensor(
            [[t for _ in range(changing_dim)] for t in range(time_len)],
            dtype=torch.float32,
        )
        self.register_buffer(
            "times", times.repeat(1, num_cutoff_times)
        )
        if init_thresholds == "rand":
            lower = torch.rand(changing_dim) - 1.0
            upper = torch.rand(changing_dim)
        elif init_thresholds == "zeros":
            lower = torch.zeros(changing_dim)
            upper = torch.zeros(changing_dim)
        else:
            raise ValueError(f"Unknown threshold init: {init_thresholds}")
        self.lower_thresholds = nn.Parameter(lower)
        self.upper_thresholds = nn.Parameter(upper)

    @property
    def encoding_dim(self) -> int:
        """Dimensionality of the encoder output (402 with defaults)."""
        return self.layout.encoding_dim

    def _window_weights(self) -> torch.Tensor:
        """Soft window weights of shape ``(1, time_len, 9 * C)``.

        Examples:
            >>> encoder = SummaryEncoder(changing_dim=2, static_dim=1)
            >>> encoder._window_weights().shape
            torch.Size([1, 6, 18])
        """
        temperatures = torch.full(
            (1, self.layout.num_window_weights),
            self.cutoff_times_temperature,
            device=self.cutoff_times.device,
        )
        weights = torch.sigmoid(
            (self.times - self.cutoff_times) / temperatures
        )
        return weights.reshape(1, self.time_len, -1)

    def forward(
        self,
        values: torch.Tensor,
        indicators: torch.Tensor,
        statics: torch.Tensor,
        epsilon_denom: float = 0.01,
    ) -> torch.Tensor:
        """Computes the summary encoding for a batch of patients.

        Args:
            values: standardized changing variables, shape
                ``(batch, time_len, changing_dim)``.
            indicators: measurement indicators (0/1), same shape.
            statics: static variables, shape ``(batch, static_dim)``.
            epsilon_denom: stabilizer added to denominators.
                Default is 0.01.

        Returns:
            Encoding of shape ``(batch, encoding_dim)`` in the order of
            :data:`ENCODING_GROUPS`.
        """
        eps = epsilon_denom
        weight_vector = self._window_weights()
        layout = self.layout

        def _slice(group: str) -> torch.Tensor:
            start, end = layout.window_slice(group)
            return weight_vector[:, :, start:end]

        # Weighted mean of each variable.
        mean_w = _slice("mean")
        mean_feats = torch.sum(
            mean_w * (values * indicators), dim=1
        ) / (torch.sum(mean_w, dim=1) + eps)

        # Weighted variance of each variable.
        var_w = _slice("var")
        x_mean = torch.mean(values * indicators, dim=1, keepdim=True)
        weighted_var = torch.sum(
            indicators * var_w * (values - x_mean) ** 2, dim=1
        )
        var_norm = torch.sum(indicators * var_w, dim=1) ** 2 / (
            torch.sum(indicators * var_w, dim=1) ** 2
            + torch.sum(indicators * var_w**2, dim=1)
            + eps
        )
        var_feats = weighted_var / (var_norm + eps)

        # Soft indicator for ever being measured.
        ever_w = _slice("ever_measured")
        ever_feats = torch.sigmoid(
            torch.sum(ever_w * indicators, dim=1)
            / (
                self.ever_measured_temperature
                * torch.sum(ever_w, dim=1)
                + eps
            )
        ) - 0.5

        # Weighted mean / variance of the indicator sequence.
        mean_ind_w = _slice("mean_indicators")
        mean_ind_feats = torch.sum(
            mean_ind_w * indicators, dim=1
        ) / (torch.sum(mean_ind_w, dim=1) + eps)
        var_ind_w = _slice("var_indicators")
        ind_mean = torch.mean(indicators, dim=1, keepdim=True)
        weighted_var_ind = torch.sum(
            var_ind_w * (indicators - ind_mean) ** 2, dim=1
        )
        var_ind_norm = torch.sum(var_ind_w, dim=1) ** 2 / (
            torch.sum(var_ind_w, dim=1) ** 2
            + torch.sum(var_ind_w**2, dim=1)
            + eps
        )
        var_ind_feats = weighted_var_ind / (var_ind_norm + eps)

        # The original calls this "switches" of the indicators, but its
        # code uses |x[t+1] - x[t]| on the *values*. Kept as the code
        # has it.
        switches_w = _slice("switches")[:, :-1, :]
        switch_feats = torch.sum(
            switches_w * torch.abs(values[:, 1:, :] - values[:, :-1, :]),
            dim=1,
        ) / (torch.sum(switches_w, dim=1) + eps)

        # Weighted linear-regression slope (and its standard error).
        slope_w = _slice("slope")
        linreg_y = values * indicators
        linreg_x = (
            torch.arange(
                self.time_len, device=values.device, dtype=torch.float32
            )
            .unsqueeze(1)
            .expand(self.time_len, self.changing_dim)
            .repeat(values.shape[0], 1, 1)
            * indicators
        )
        weighted_x = torch.unsqueeze(
            torch.sum(slope_w * linreg_x, dim=1)
            / (torch.sum(slope_w, dim=1) + eps),
            1,
        )
        weighted_y = torch.unsqueeze(
            torch.sum(slope_w * linreg_y, dim=1)
            / (torch.sum(slope_w, dim=1) + eps),
            1,
        )
        slope_num = torch.sum(
            slope_w * (linreg_x - weighted_x) * (linreg_y - weighted_y),
            dim=1,
        )
        slope_den = torch.sum(
            slope_w * (linreg_x - weighted_x) ** 2, dim=1
        )
        slope_feats = slope_num / (slope_den + eps)
        slope_stderr_feats = torch.where(
            slope_den > 0, 1.0 / (slope_den + eps), slope_den
        )

        # First / last hour each variable was measured.
        mask_max_values, mask_max_indices = torch.max(indicators, dim=1)
        first_indices = mask_max_indices.clone()
        first_indices[mask_max_values == 0] = -1
        first_feats = first_indices.float() / float(self.time_len)
        flipped = torch.flip(indicators, [1])
        last_values, last_indices = torch.max(flipped, dim=1)
        last_indices = last_indices.clone()
        last_indices[last_values == 0] = self.time_len
        last_feats = (
            float(self.time_len) - last_indices.float()
        ) / float(self.time_len)

        # Time above / below the learned thresholds.
        above_w = _slice("hours_above_threshold")
        below_w = _slice("hours_below_threshold")
        upper = torch.sigmoid(
            (values - self.upper_thresholds) / self.thresholds_temperature
        )
        lower = torch.sigmoid(
            (self.lower_thresholds - values) / self.thresholds_temperature
        )
        above_feats = torch.sum(
            indicators * above_w * upper, dim=1
        ) / (torch.sum(indicators * above_w, dim=1) + eps)
        below_feats = torch.sum(
            indicators * below_w * lower, dim=1
        ) / (torch.sum(indicators * below_w, dim=1) + eps)

        # Raw last-hour row: values + indicators + statics.
        last_hour = torch.cat(
            (values[:, -1, :], indicators[:, -1, :], statics), dim=1
        )
        return torch.cat(
            (
                last_hour.float(),
                mean_feats.float(),
                var_feats.float(),
                ever_feats.float(),
                mean_ind_feats.float(),
                var_ind_feats.float(),
                switch_feats.float(),
                slope_feats.float(),
                slope_stderr_feats.float(),
                first_feats.float(),
                last_feats.float(),
                above_feats.float(),
                below_feats.float(),
            ),
            dim=1,
        )


class ConceptBottleneckModel(BaseModel):
    """Concept bottleneck model ``y_hat = f(g(x, h(x)))`` (Wu et al. 2022).

    A linear layer maps the :class:`SummaryEncoder` output to ``k``
    concepts, a sigmoid turns them into concept scores, and a logistic
    head predicts vasopressor onset. Loss is class-weighted BCE plus
    ``l1_lambda`` L1 on the concept weights plus ``cos_sim_lambda``
    times the summed absolute cosine similarity between concept weight
    vectors (skipped when ``k == 1``).

    Adaptation vs the original: the original uses 2-class logits with
    one-hot targets and ``p_weight = [1/(1-p), 1/p]``; PyHealth's binary
    label processor and ``Trainer`` metrics expect a single logit, so
    this port uses one logit and scalar ``pos_weight = (1 - p) / p``.

    Args:
        dataset: the dataset to train the model (used for the input
            schema and label processor).
        num_concepts: number of concepts ``k``. Default is 4.
        l1_lambda: L1 coefficient on the bottleneck weights.
            Default is 0.0.
        cos_sim_lambda: cosine-similarity coefficient between concept
            weight vectors. Default is 0.0.
        pos_weight: scalar positive-class weight for the BCE loss
            (``(1 - p) / p`` with ``p`` the training prevalence).
            Default is None (unweighted).
        zero_weight: if True, adds an extra all-zero concept (k + 1
            total, last concept frozen at zero), as in the original.
            Default is False.
        **kwargs: passed to :class:`SummaryEncoder`.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.datasets import get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "values": [[0.0, 0.0]] * 6,
        ...         "indicators": [[1.0, 1.0]] * 6,
        ...         "statics": [0.0],
        ...         "vasopressor_onset": 0,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "values": [[1.0, 0.0]] * 6,
        ...         "indicators": [[1.0, 1.0]] * 6,
        ...         "statics": [1.0],
        ...         "vasopressor_onset": 1,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "values": "tensor",
        ...         "indicators": "tensor",
        ...         "statics": "tensor",
        ...     },
        ...     output_schema={"vasopressor_onset": "binary"},
        ...     dataset_name="test",
        ... )
        >>> model = ConceptBottleneckModel(
        ...     dataset=dataset, changing_dim=2, static_dim=1
        ... )
        >>> loader = get_dataloader(dataset, batch_size=2)
        >>> ret = model(**next(iter(loader)))
        >>> sorted(ret.keys())
        ['concepts', 'logit', 'loss', 'summaries', 'y_prob', 'y_true']
        >>> ret["logit"].shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        num_concepts: int = 4,
        l1_lambda: float = 0.0,
        cos_sim_lambda: float = 0.0,
        pos_weight: float | None = None,
        zero_weight: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(dataset)
        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        self.num_concepts = num_concepts + (1 if zero_weight else 0)
        self.zero_weight = zero_weight
        self.l1_lambda = l1_lambda
        self.cos_sim_lambda = cos_sim_lambda
        self.pos_weight = pos_weight
        self.encoder = SummaryEncoder(**kwargs)
        self.bottleneck = nn.Linear(self.encoder.encoding_dim, self.num_concepts)
        self.head = nn.Linear(self.num_concepts, 1)
        if zero_weight:
            with torch.no_grad():
                self.bottleneck.weight[self.num_concepts - 1].fill_(0.0)
                self.bottleneck.bias[self.num_concepts - 1].fill_(0.0)
                self.head.weight[:, self.num_concepts - 1].fill_(0.0)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass; returns ``logit``, ``y_prob`` and, when the
        label is present, ``loss`` and ``y_true``. Also returns the
        concept scores (``concepts``) and the summary encoding
        (``summaries``).

        Args:
            **kwargs: keyword arguments with all feature keys and,
                optionally, the label key.

        Returns:
            Dict with ``logit``, ``y_prob``, ``concepts``,
            ``summaries`` and, when the label is given, ``loss`` and
            ``y_true``.
        """
        values = kwargs["values"].to(self.device)
        indicators = kwargs["indicators"].to(self.device)
        statics = kwargs["statics"].to(self.device)
        summaries = self.encoder(values, indicators, statics)
        concepts = torch.sigmoid(self.bottleneck(summaries))
        logits = self.head(concepts)
        y_prob = torch.sigmoid(logits)
        results: dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": y_prob,
            "concepts": concepts,
            "summaries": summaries,
        }
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device).float()
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(1)
            pos_weight = (
                torch.tensor(
                    self.pos_weight, device=self.device
                )
                if self.pos_weight is not None
                else None
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, y_true, pos_weight=pos_weight
            )
            loss = loss + self.l1_lambda * torch.norm(
                self.bottleneck.weight, 1
            )
            if self.num_concepts != 1 and self.cos_sim_lambda:
                cos_sim = torch.tensor(0.0, device=self.device)
                for i, j in itertools.combinations(
                    range(self.num_concepts), 2
                ):
                    cos_sim = cos_sim + torch.abs(
                        F.cosine_similarity(
                            self.bottleneck.weight[i],
                            self.bottleneck.weight[j],
                            dim=0,
                        )
                    )
                loss = loss + self.cos_sim_lambda * cos_sim
            results["loss"] = loss
            results["y_true"] = y_true
        return results


class LogisticRegressionWithSummaries(BaseModel):
    """Johnson et al. (2021) baseline: logistic regression on summaries.

    Same :class:`SummaryEncoder` as the bottleneck model, with a single
    linear head. Loss is class-weighted BCE plus the horseshoe penalty
    of ``custom_bce_horseshoe`` (original ``custom_losses.py``) on the
    head weights — with a single logit, the penalty applies to the head
    weight vector itself, the analogue of the original
    ``weight[1] - weight[0]`` difference.

    Args:
        dataset: the dataset to train the model.
        alpha: horseshoe coefficient. Default is 1e-4.
        tau: horseshoe constant. Default is 10.0.
        pos_weight: scalar positive-class weight for the BCE loss.
            Default is None (unweighted).
        **kwargs: passed to :class:`SummaryEncoder`.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.datasets import get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "values": [[0.0, 0.0]] * 6,
        ...         "indicators": [[1.0, 1.0]] * 6,
        ...         "statics": [0.0],
        ...         "vasopressor_onset": 0,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "values": [[1.0, 0.0]] * 6,
        ...         "indicators": [[1.0, 1.0]] * 6,
        ...         "statics": [1.0],
        ...         "vasopressor_onset": 1,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "values": "tensor",
        ...         "indicators": "tensor",
        ...         "statics": "tensor",
        ...     },
        ...     output_schema={"vasopressor_onset": "binary"},
        ...     dataset_name="test",
        ... )
        >>> model = LogisticRegressionWithSummaries(
        ...     dataset=dataset, changing_dim=2, static_dim=1
        ... )
        >>> loader = get_dataloader(dataset, batch_size=2)
        >>> ret = model(**next(iter(loader)))
        >>> ret["logit"].shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        alpha: float = 1e-4,
        tau: float = 10.0,
        pos_weight: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(dataset)
        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        self.alpha = alpha
        self.tau = tau
        self.pos_weight = pos_weight
        self.encoder = SummaryEncoder(**kwargs)
        self.head = nn.Linear(self.encoder.encoding_dim, 1)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass; returns ``logit``, ``y_prob`` and, when the
        label is present, ``loss`` and ``y_true``.

        Args:
            **kwargs: keyword arguments with all feature keys and,
                optionally, the label key.

        Returns:
            Dict with ``logit``, ``y_prob`` and, when the label is
            given, ``loss`` and ``y_true``.
        """
        values = kwargs["values"].to(self.device)
        indicators = kwargs["indicators"].to(self.device)
        statics = kwargs["statics"].to(self.device)
        summaries = self.encoder(values, indicators, statics)
        logits = self.head(summaries)
        y_prob = torch.sigmoid(logits)
        results: dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": y_prob,
        }
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device).float()
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(1)
            pos_weight = (
                torch.tensor(self.pos_weight, device=self.device)
                if self.pos_weight is not None
                else None
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, y_true, pos_weight=pos_weight
            )
            # Horseshoe penalty on the head weights.
            eps = 0.001
            weights = self.head.weight.squeeze(0)
            frac = 1.0 / (weights**2 + eps)
            term = 1.0 + 2.0 * self.tau**2 * frac
            loss = loss + self.alpha * (
                -torch.sum(torch.log(torch.log(term)))
            )
            results["loss"] = loss
            results["y_true"] = y_true
        return results


class LSTMNextStateBaseline(BaseModel):
    """LSTM baseline (Harutyunyan et al. 2019 style, original
    ``LSTM_NextState`` / ``LSTM_Baseline`` in ``models.py``).

    An LSTM reads the changing variables of the first five hours and
    predicts the next state at each step; a second head predicts the
    outcome from the last hidden state concatenated with the raw
    last-hour row. Loss is the compound loss of ``custom_losses.py``:
    masked next-state MSE + ``beta`` * BCE + ``alpha`` * L2 over all
    parameters.

    Adaptation vs the original: the outcome head uses a single logit
    (see :class:`ConceptBottleneckModel`). The original also constructs
    its ``nn.LSTM`` without ``batch_first=True`` while feeding it
    ``(batch, time, features)`` tensors — the time and batch axes are
    then silently swapped inside the LSTM. This port uses
    ``batch_first=True`` by default; that is a deliberate fix, and the
    LSTM baseline's numbers should be compared against the original
    with that difference in mind. ``batch_first=False`` reproduces the
    original's behaviour, for comparison.

    Args:
        dataset: the dataset to train the model.
        lstm_hidden_dim: LSTM hidden size. Default is 32.
        hidden_next_state_dim: hidden size of the next-state head.
            Default is 256.
        hidden_outcome_dim: hidden size of the outcome head.
            Default is 32.
        alpha: L2 coefficient in the compound loss. Default is 1e-4.
        beta: BCE coefficient in the compound loss. Default is 10.0.
        pos_weight: scalar positive-class weight for the BCE term.
            Default is None (unweighted).
        batch_first: whether the LSTM reads its input as
            ``(batch, time, features)``. Default is True; False
            reproduces the original's axis swap.
        changing_dim: number of time-varying variables. Default is 28.
        static_dim: number of static variables. Default is 10.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.datasets import get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "values": [[0.0, 0.0]] * 6,
        ...         "indicators": [[1.0, 1.0]] * 6,
        ...         "statics": [0.0],
        ...         "vasopressor_onset": 0,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "values": [[1.0, 0.0]] * 6,
        ...         "indicators": [[1.0, 1.0]] * 6,
        ...         "statics": [1.0],
        ...         "vasopressor_onset": 1,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "values": "tensor",
        ...         "indicators": "tensor",
        ...         "statics": "tensor",
        ...     },
        ...     output_schema={"vasopressor_onset": "binary"},
        ...     dataset_name="test",
        ... )
        >>> model = LSTMNextStateBaseline(
        ...     dataset=dataset, changing_dim=2, static_dim=1
        ... )
        >>> loader = get_dataloader(dataset, batch_size=2)
        >>> ret = model(**next(iter(loader)))
        >>> ret["logit"].shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        lstm_hidden_dim: int = 32,
        hidden_next_state_dim: int = 256,
        hidden_outcome_dim: int = 32,
        alpha: float = 1e-4,
        beta: float = 10.0,
        pos_weight: float | None = None,
        batch_first: bool = True,
        changing_dim: int = 28,
        static_dim: int = 10,
    ) -> None:
        super().__init__(dataset)
        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight
        self.batch_first = batch_first
        self.changing_dim = changing_dim
        self.input_dim = 2 * changing_dim + static_dim
        self.lstm = nn.LSTM(
            changing_dim, lstm_hidden_dim, batch_first=batch_first
        )
        self.next_state_1 = nn.Linear(lstm_hidden_dim, hidden_next_state_dim)
        self.next_state_2 = nn.Linear(hidden_next_state_dim, changing_dim)
        self.outcome_1 = nn.Linear(
            lstm_hidden_dim + self.input_dim, hidden_outcome_dim
        )
        self.outcome_2 = nn.Linear(hidden_outcome_dim, 1)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass; returns ``logit``, ``y_prob``, the next-state
        predictions (``next_state``) and, when the label is present,
        ``loss`` and ``y_true``.

        Args:
            **kwargs: keyword arguments with all feature keys and,
                optionally, the label key.

        Returns:
            Dict with ``logit``, ``y_prob``, ``next_state`` and, when
            the label is given, ``loss`` and ``y_true``.
        """
        values = kwargs["values"].to(self.device)
        indicators = kwargs["indicators"].to(self.device)
        statics = kwargs["statics"].to(self.device)
        # LSTM over all hours but the last; predict the next hour.
        lstm_out, _ = self.lstm(values[:, :-1, :])
        next_state = self.next_state_2(
            torch.relu(self.next_state_1(lstm_out))
        )
        last_hour = torch.cat(
            (values[:, -1, :], indicators[:, -1, :], statics), dim=1
        )
        outcome_cov = torch.cat((lstm_out[:, -1, :], last_hour), dim=1)
        logits = self.outcome_2(torch.relu(self.outcome_1(outcome_cov)))
        y_prob = torch.sigmoid(logits)
        results: dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": y_prob,
            "next_state": next_state,
        }
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device).float()
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(1)
            # Masked next-state MSE (mask = indicators from hour 1 on).
            target = values[:, 1:, :]
            mask = indicators[:, 1:, :] > 0
            mse = ((target - next_state) ** 2).masked_select(mask).mean()
            pos_weight = (
                torch.tensor(self.pos_weight, device=self.device)
                if self.pos_weight is not None
                else None
            )
            bce = F.binary_cross_entropy_with_logits(
                logits, y_true, pos_weight=pos_weight
            )
            l2 = torch.tensor(0.0, device=self.device)
            for param in self.parameters():
                l2 = l2 + torch.norm(param)
            results["loss"] = mse + self.beta * bce + self.alpha * l2
            results["y_true"] = y_true
        return results


def top_features_by_weight_mass(
    bottleneck_weight: torch.Tensor, mass: float = 0.90
) -> list[list[int]]:
    """Per-concept candidate features covering ``mass`` of weight mass.

    For each concept (row of the bottleneck weight matrix), sorts
    features by descending absolute weight and keeps the smallest
    prefix whose cumulative mass reaches ``mass`` of the row total.
    The original code reads these candidate lists from a CSV produced
    by ``model-verification.ipynb``; this computes them from the
    trained weights instead.

    Args:
        bottleneck_weight: bottleneck weight matrix of shape
            ``(num_concepts, encoding_dim)``.
        mass: fraction of absolute weight mass to cover.
            Default is 0.90.

    Returns:
        List (one per concept) of feature indices, in descending
        absolute-weight order.

    Examples:
        >>> import torch
        >>> w = torch.tensor([[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
        >>> top_features_by_weight_mass(w, mass=0.90)
        [[0, 1], [1, 2]]
    """
    candidates: list[list[int]] = []
    weights = bottleneck_weight.detach().abs()
    for row in weights:
        total = row.sum()
        order = torch.argsort(row, descending=True)
        cumulative = torch.cumsum(row[order], dim=0)
        keep = (cumulative < mass * total).sum().item() + 1
        candidates.append(order[:keep].tolist())
    return candidates


def greedy_select_features(
    candidates: list[list[int]],
    score_fn: Callable[[torch.Tensor], float],
    budget: int,
    encoding_dim: int | None = None,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Greedy (feature, concept) selection by AUROC (original
    ``gpu_greedy_top_concepts.py``).

    Starts from an all-zero bottleneck mask and, for ``budget`` rounds,
    tries every not-yet-selected candidate, keeps the one whose
    ``score_fn(mask)`` (AUROC with that mask applied) is highest, and
    fixes it in the mask.

    Args:
        candidates: per-concept candidate feature indices, as returned
            by :func:`top_features_by_weight_mass`.
        score_fn: callable taking a boolean mask of shape
            ``(num_concepts, encoding_dim)`` and returning the AUROC
            with that mask applied to the bottleneck weights.
        budget: number of (concept, feature) pairs to select
            (``10 * num_concepts`` in the original).
        encoding_dim: width of the mask. Default is None, the largest
            candidate index + 1; pass the encoder's ``encoding_dim`` to
            get masks shaped like the bottleneck weights.

    Returns:
        Tuple of the ordered selections ``[(concept, feature), ...]``
        and the AUROC after each selection.

    Examples:
        >>> import torch
        >>> def fake_score(mask):
        ...     return float(mask.sum()) / 10.0
        >>> selections, aucs = greedy_select_features(
        ...     [[0, 1], [2]], fake_score, budget=2
        ... )
        >>> selections
        [(0, 0), (0, 1)]
        >>> aucs
        [0.1, 0.2]
    """
    num_concepts = len(candidates)
    if encoding_dim is None:
        encoding_dim = max(max(c) for c in candidates if c) + 1
    mask = torch.zeros(num_concepts, encoding_dim, dtype=torch.bool)
    selections: list[tuple[int, int]] = []
    aucs: list[float] = []
    for _ in range(budget):
        best_auc = -1.0
        best_pair: tuple[int, int] | None = None
        for concept, inds in enumerate(candidates):
            for ind in inds:
                if mask[concept, ind]:
                    continue
                mask[concept, ind] = True
                auc = score_fn(mask)
                if auc > best_auc:
                    best_auc = auc
                    best_pair = (concept, ind)
                mask[concept, ind] = False
        if best_pair is None:
            break
        concept, ind = best_pair
        mask[concept, ind] = True
        selections.append(best_pair)
        aucs.append(best_auc)
    return selections, aucs


def apply_topk_mask(
    model: ConceptBottleneckModel,
    selections: list[tuple[int, int]],
) -> torch.Tensor:
    """Masks a trained bottleneck to the greedy selections (in place).

    Zeroes every bottleneck weight except the selected
    (concept, feature) entries and registers a gradient hook so
    fine-tuning keeps them at zero (the original re-zeroes the
    gradients each step in ``fit``).

    Args:
        model: trained :class:`ConceptBottleneckModel`.
        selections: ordered ``(concept, feature)`` pairs from
            :func:`greedy_select_features`.

    Returns:
        The boolean mask of shape ``(num_concepts, encoding_dim)``.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "values": [[0.0, 0.0]] * 6,
        ...         "indicators": [[1.0, 1.0]] * 6,
        ...         "statics": [0.0],
        ...         "vasopressor_onset": 0,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "values": [[1.0, 0.0]] * 6,
        ...         "indicators": [[1.0, 1.0]] * 6,
        ...         "statics": [1.0],
        ...         "vasopressor_onset": 1,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "values": "tensor",
        ...         "indicators": "tensor",
        ...         "statics": "tensor",
        ...     },
        ...     output_schema={"vasopressor_onset": "binary"},
        ...     dataset_name="test",
        ... )
        >>> model = ConceptBottleneckModel(
        ...     dataset=dataset, changing_dim=2, static_dim=1
        ... )
        >>> mask = apply_topk_mask(model, [(0, 0), (1, 2)])
        >>> mask.shape
        torch.Size([4, 29])
        >>> bool(model.bottleneck.weight[0, 1] == 0)
        True
    """
    num_concepts = model.num_concepts
    encoding_dim = model.encoder.encoding_dim
    mask = torch.zeros(num_concepts, encoding_dim, dtype=torch.bool)
    for concept, ind in selections:
        mask[concept, ind] = True
    with torch.no_grad():
        model.bottleneck.weight.data = torch.where(
            mask.to(model.bottleneck.weight.device),
            model.bottleneck.weight.data,
            torch.tensor(0.0, device=model.bottleneck.weight.device),
        )
    mask_device = mask.to(model.bottleneck.weight.device)

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        return grad * mask_device

    model.bottleneck.weight.register_hook(_hook)
    return mask
