"""MotionAge: wear-aware recurrent mortality model for wearable activity.

Contributors: Inoh Pak (inohpak2), Yejin Kim (yejink6)

Description: Model contribution for the MotionAge reproduction (CS 598 DLH).
A GRU reads epoch-level activity with hour/day embeddings, pools hidden states
over worn epochs only, and optionally fuses static covariates (late fusion)
to predict five-year mortality.

Paper: Song, Y., Yin, J., Hu, Y., and Gu, T. MotionAge: A Deep Learning
Framework for Biological Age Prediction from Wearable Activity. Proceedings of
the 11th Machine Learning for Healthcare Conference (PMLR 340), 2026.

Reference implementation (MIT license):
https://github.com/jackie-jiaqi-yin/MotionAge
"""

import logging
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

logger = logging.getLogger(__name__)

PoolingMode = Literal["masked", "mean", "last"]
_POOLING_MODES = ("masked", "mean", "last")
_NUM_HOURS = 24
_NUM_DAYS = 7


def _build_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, dropout: float
) -> nn.Module:
    """Builds a linear layer, or a two-layer MLP when ``hidden_dim > 0``.

    Args:
        input_dim: Input feature size.
        hidden_dim: Hidden layer size. ``0`` or less gives a single linear layer.
        output_dim: Output feature size.
        dropout: Dropout applied after the hidden activation.

    Returns:
        The MLP module.
    """
    if hidden_dim <= 0:
        return nn.Linear(input_dim, output_dim)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class MotionAgeLayer(nn.Module):
    """Wear-aware recurrent encoder for epoch-level activity sequences.

    Each epoch is represented by its activity intensity concatenated with
    learned hour-of-day and day-of-week embeddings. A GRU (or LSTM) runs over
    the sequence, and its hidden states are pooled into one vector.

    With ``pooling="masked"`` only epochs whose wear mask is 1 contribute to
    the pooled vector, so device non-wear is not averaged in as if it were
    inactivity. ``"mean"`` averages over every epoch and ``"last"`` takes the
    final hidden state; both ignore the mask and exist for ablations.

    Args:
        hidden_dim: Hidden size of the recurrent network. Default is 64.
        num_layers: Number of stacked recurrent layers. Default is 2.
        dropout: Dropout between recurrent layers (only used when
            ``num_layers > 1``). Default is 0.117.
        hour_emb_dim: Size of the hour-of-day embedding. Default is 8.
        day_emb_dim: Size of the day-of-week embedding. Default is 4.
        rnn_type: ``"gru"`` or ``"lstm"``. Default is ``"gru"``.
        pooling: ``"masked"``, ``"mean"`` or ``"last"``. Default is
            ``"masked"``.

    Examples:
        >>> layer = MotionAgeLayer(hidden_dim=16, num_layers=1)
        >>> intensity = torch.rand(2, 12)
        >>> hour = torch.randint(0, 24, (2, 12))
        >>> day = torch.randint(0, 7, (2, 12))
        >>> mask = torch.ones(2, 12)
        >>> layer(intensity, hour, day, mask).shape
        torch.Size([2, 16])
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.117,
        hour_emb_dim: int = 8,
        day_emb_dim: int = 4,
        rnn_type: Literal["gru", "lstm"] = "gru",
        pooling: PoolingMode = "masked",
    ) -> None:
        """Builds the embeddings and the recurrent network.

        The arguments are described in the class docstring.

        Raises:
            ValueError: If a size is not positive, or ``rnn_type`` or
                ``pooling`` is not supported.
        """
        super().__init__()
        for name, value in {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "hour_emb_dim": hour_emb_dim,
            "day_emb_dim": day_emb_dim,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if pooling not in _POOLING_MODES:
            raise ValueError(
                f"pooling must be one of {_POOLING_MODES}, got '{pooling}'."
            )
        rnn_classes = {"gru": nn.GRU, "lstm": nn.LSTM}
        if rnn_type not in rnn_classes:
            raise ValueError(f"rnn_type must be 'gru' or 'lstm', got '{rnn_type}'.")

        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.hour_embedding = nn.Embedding(_NUM_HOURS, hour_emb_dim)
        self.day_embedding = nn.Embedding(_NUM_DAYS, day_emb_dim)
        self.rnn = rnn_classes[rnn_type](
            input_size=1 + hour_emb_dim + day_emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        intensity: torch.Tensor,
        hour: torch.Tensor,
        day: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encodes a batch of activity sequences.

        Args:
            intensity: Float tensor of shape ``[batch, seq_len]``.
            hour: Hour-of-day indices in ``[0, 23]``, shape ``[batch, seq_len]``.
            day: Day-of-week indices in ``[0, 6]``, shape ``[batch, seq_len]``.
            mask: Optional wear mask of shape ``[batch, seq_len]`` with 1 for
                valid (worn) epochs. ``None`` treats every epoch as valid.

        Returns:
            Pooled hidden states of shape ``[batch, hidden_dim]``.

        Raises:
            ValueError: If the input shapes differ or indices are out of range.
        """
        if intensity.dim() != 2:
            raise ValueError(
                "intensity must have shape [batch, seq_len], "
                f"got {tuple(intensity.shape)}."
            )
        if mask is None:
            mask = torch.ones_like(intensity)
        for name, tensor in {"hour": hour, "day": day, "mask": mask}.items():
            if tensor.shape != intensity.shape:
                raise ValueError(
                    f"{name} must have shape {tuple(intensity.shape)}, "
                    f"got {tuple(tensor.shape)}."
                )
        hour = hour.long()
        day = day.long()
        if hour.numel() and (hour.min() < 0 or hour.max() >= _NUM_HOURS):
            raise ValueError("hour indices must be in [0, 23].")
        if day.numel() and (day.min() < 0 or day.max() >= _NUM_DAYS):
            raise ValueError("day indices must be in [0, 6].")

        x = torch.cat(
            [
                intensity.unsqueeze(-1).float(),
                self.hour_embedding(hour),
                self.day_embedding(day),
            ],
            dim=-1,
        )
        h_seq, _ = self.rnn(x)

        if self.pooling == "last":
            return h_seq[:, -1, :]
        if self.pooling == "mean":
            return h_seq.mean(dim=1)
        weights = mask.unsqueeze(-1).to(h_seq.dtype)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (h_seq * weights).sum(dim=1) / denom


class MotionAge(BaseModel):
    """MotionAge wear-aware mortality model (Song et al., MLHC 2026).

    MotionAge predicts a binary outcome (five-year mortality in the paper)
    from a window of epoch-level wearable activity. A recurrent encoder reads
    activity intensity with hour-of-day and day-of-week embeddings, and its
    hidden states are averaged over worn epochs only (wear-mask pooling), so
    device non-wear is not mistaken for sedentary behavior. When static
    covariates are given (e.g. age, sex, BMI, blood pressure), they are
    encoded by an MLP and combined with the activity representation by a
    late-fusion MLP (the paper's FA/FRC late-fusion variants). Without them
    the model is the activity-only (F) variant.

    The paper converts predicted risk into an age scale (the MotionAge
    phenotype) with a sex-specific calibration fitted on training
    participants. That calibration is an evaluation step outside this model.

    Expected sample fields (key names are configurable):
        - ``intensity``: ``"tensor"`` of shape ``[seq_len]``, epoch activity.
        - ``hour``: ``"tensor"`` of shape ``[seq_len]``, hour of day (0-23).
        - ``day``: ``"tensor"`` of shape ``[seq_len]``, day of week (0-6).
        - ``wear_mask`` (optional): ``"tensor"`` of shape ``[seq_len]``,
          1 for valid epochs.
        - ``static`` (optional): ``"tensor"`` of shape ``[num_covariates]``.
          NaN values are replaced by 0; use the TensorProcessor options
          ``normalization``, ``missing_indicators`` and ``impute_strategy``
          to scale them and keep missingness explicit.

    Args:
        dataset: The SampleDataset produced by ``dataset.set_task()``. It must
            have exactly one label; the label processor sets the output size,
            activation and loss.
        intensity_key: Input key of the activity intensity. Default is
            ``"intensity"``.
        hour_key: Input key of the hour-of-day index. Default is ``"hour"``.
        day_key: Input key of the day-of-week index. Default is ``"day"``.
        mask_key: Input key of the wear mask, or ``None`` to treat all epochs
            as valid. Default is ``"wear_mask"``.
        static_key: Input key of the static covariate vector, or ``None`` for
            the activity-only model. Default is ``None``.
        hidden_dim: Recurrent hidden size. Default is 64.
        num_layers: Number of recurrent layers. Default is 2.
        dropout: Dropout between recurrent layers. Default is 0.117.
        hour_emb_dim: Hour embedding size. Default is 8.
        day_emb_dim: Day embedding size. Default is 4.
        rnn_type: ``"gru"`` or ``"lstm"``. Default is ``"gru"``.
        pooling: ``"masked"`` (paper), ``"mean"`` or ``"last"``. Default is
            ``"masked"``.
        covariate_hidden_dim: Hidden size of the covariate MLP. Default is 64.
        covariate_output_dim: Output size of the covariate MLP. Default is 32.
        fusion_hidden_dim: Hidden size of the fusion MLP. Default is 64.
        fusion_dropout: Dropout in the covariate and fusion MLPs. Default is
            0.1.

    Raises:
        ValueError: If a configured key is not in the dataset's input schema
            or the dataset does not have exactly one label.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import MotionAge
        >>> samples = [
        ...     {
        ...         "patient_id": f"p{i}",
        ...         "intensity": [0.1 * i] * 12,
        ...         "hour": [h % 24 for h in range(12)],
        ...         "day": [0] * 12,
        ...         "wear_mask": [1.0] * 10 + [0.0] * 2,
        ...         "static": [60.0 + i, float(i % 2)],
        ...         "mortality": i % 2,
        ...     }
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "intensity": "tensor",
        ...         "hour": "tensor",
        ...         "day": "tensor",
        ...         "wear_mask": "tensor",
        ...         "static": "tensor",
        ...     },
        ...     output_schema={"mortality": "binary"},
        ... )
        >>> model = MotionAge(dataset, static_key="static", hidden_dim=16)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> sorted(model(**batch).keys())
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        intensity_key: str = "intensity",
        hour_key: str = "hour",
        day_key: str = "day",
        mask_key: Optional[str] = "wear_mask",
        static_key: Optional[str] = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.117,
        hour_emb_dim: int = 8,
        day_emb_dim: int = 4,
        rnn_type: Literal["gru", "lstm"] = "gru",
        pooling: PoolingMode = "masked",
        covariate_hidden_dim: int = 64,
        covariate_output_dim: int = 32,
        fusion_hidden_dim: int = 64,
        fusion_dropout: float = 0.1,
    ) -> None:
        """Checks the input keys and builds the encoder and prediction head.

        The arguments are described in the class docstring.

        Raises:
            ValueError: If a configured key is not in the input schema, the
                dataset does not have exactly one label, or the size of the
                static input cannot be inferred.
        """
        super().__init__(dataset=dataset)
        if len(self.label_keys) != 1:
            raise ValueError(
                f"MotionAge needs exactly one label key, got {self.label_keys}."
            )
        self.label_key = self.label_keys[0]
        self.intensity_key = intensity_key
        self.hour_key = hour_key
        self.day_key = day_key
        self.mask_key = mask_key
        self.static_key = static_key

        used_keys = [intensity_key, hour_key, day_key, mask_key, static_key]
        used_keys = [key for key in used_keys if key is not None]
        missing = [key for key in used_keys if key not in self.feature_keys]
        if missing:
            raise ValueError(
                f"Keys {missing} are not in the dataset input schema "
                f"{self.feature_keys}."
            )
        unused = [key for key in self.feature_keys if key not in used_keys]
        if unused:
            logger.warning("MotionAge ignores input keys %s.", unused)

        self.encoder = MotionAgeLayer(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            hour_emb_dim=hour_emb_dim,
            day_emb_dim=day_emb_dim,
            rnn_type=rnn_type,
            pooling=pooling,
        )
        output_size = self.label_processor.output_size()

        self.covariate_encoder: Optional[nn.Module] = None
        if static_key is None:
            self.fc = nn.Linear(hidden_dim, output_size)
        else:
            static_dim = self.dataset.input_processors[static_key].size()
            if not static_dim:
                raise ValueError(
                    f"Cannot infer the size of static input '{static_key}'."
                )
            self.static_dim = int(static_dim)
            self.covariate_encoder = _build_mlp(
                self.static_dim,
                covariate_hidden_dim,
                covariate_output_dim,
                fusion_dropout,
            )
            self.fc = _build_mlp(
                hidden_dim + covariate_output_dim,
                fusion_hidden_dim,
                output_size,
                fusion_dropout,
            )

    def _get(self, kwargs: Dict[str, Any], key: str) -> torch.Tensor:
        """Returns the input tensor for ``key`` on the model device.

        Args:
            kwargs: The batch passed to ``forward``.
            key: The input key.

        Returns:
            The tensor for ``key``.
        """
        value = kwargs[key]
        if isinstance(value, (tuple, list)):
            value = value[0]
        return torch.as_tensor(value).to(self.device)

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Runs the model on a batch.

        Args:
            **kwargs: A batch from ``get_dataloader``. It must contain the
                configured input keys; the label key is optional.

        Returns:
            A dictionary with ``logit`` and ``y_prob``, plus ``loss`` and
            ``y_true`` when the label is present, and ``embed`` (the fused
            patient representation) when ``embed=True`` is passed.
        """
        intensity = self._get(kwargs, self.intensity_key).float()
        hour = self._get(kwargs, self.hour_key)
        day = self._get(kwargs, self.day_key)
        mask = None
        if self.mask_key is not None:
            mask = self._get(kwargs, self.mask_key).float()

        emb = self.encoder(intensity, hour, day, mask)
        if self.covariate_encoder is not None:
            static = self._get(kwargs, self.static_key).float()
            static = torch.nan_to_num(static, nan=0.0)
            emb = torch.cat([emb, self.covariate_encoder(static)], dim=-1)

        logits = self.fc(emb)
        results = {
            "logit": logits,
            "y_prob": self.label_processor.activation(logits),
        }
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            results["loss"] = self.label_processor.loss(logits, y_true)
            results["y_true"] = y_true
        if kwargs.get("embed", False):
            results["embed"] = emb
        return results

    def forward_from_embedding(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Runs ``forward``; MotionAge already consumes dense inputs.

        Args:
            **kwargs: Same as ``forward``.

        Returns:
            Same as ``forward``.
        """
        return self.forward(**kwargs)

    @staticmethod
    def aggregate_participant_risk(
        y_prob: torch.Tensor, patient_ids: List[str]
    ) -> Dict[str, float]:
        """Averages window-level probabilities per participant.

        The paper scores each participant by the mean predicted risk over
        their windows.

        Args:
            y_prob: Window probabilities of shape ``[num_windows]`` or
                ``[num_windows, 1]``.
            patient_ids: Participant id for each window.

        Returns:
            Mapping from participant id to mean predicted risk.

        Raises:
            ValueError: If the lengths differ.
        """
        probs = y_prob.detach().reshape(-1).cpu().tolist()
        if len(probs) != len(patient_ids):
            raise ValueError(
                f"Got {len(probs)} probabilities for {len(patient_ids)} ids."
            )
        totals: Dict[str, List[float]] = {}
        for pid, prob in zip(patient_ids, probs):
            totals.setdefault(str(pid), []).append(prob)
        return {pid: sum(vals) / len(vals) for pid, vals in totals.items()}
