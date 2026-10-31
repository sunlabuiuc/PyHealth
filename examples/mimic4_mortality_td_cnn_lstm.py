"""Synthetic TD-CNN-LSTM example: TD training, evaluation and delta ablation.

This script runs the PyHealth ``TDCNNLSTM`` contribution end to end on a small,
deterministic, **invented** paired-state dataset (tasks T39-T41):

1. Build a :class:`~pyhealth.datasets.SampleDataset` whose samples are ICU
   measurement-time states. Each sample has the five aligned event channels
   (``timepoints``, ``values``, ``features``, ``delta_time``,
   ``delta_value``), the five matching ``next_*`` successor channels, and an
   observed 28-day mortality ``label`` with both classes present.
2. Split it by patient (stratified by outcome, so every split contains both
   classes), derive feature-specific scaling from the training split only,
   and create the model.
3. Train with the explicit TD step (zero grads, forward, backward, AdamW step,
   then ``update_target()``), keeping the epoch with the best validation
   AUROC.
4. Evaluate held-out patients for the full model and for the
   ``use_delta_embeddings=False`` ablation, trained under identical settings.

No PhysioNet credentials, MIMIC-IV files or network access are needed. The
synthetic scores only show that the pipeline works. They are **not**
evidence about clinical performance; the MIMIC-IV reproduction is run
privately with the authors' preprocessing.

The standard PyHealth ``Trainer`` does not call ``model.update_target()`` after
its optimizer step, so it must not be used unchanged for TD training; use
:func:`training_step` instead.

Example:
    $ python examples/mimic4_mortality_td_cnn_lstm.py --epochs 3
"""

import argparse
import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from pyhealth.datasets import SampleDataset, create_sample_dataset, get_dataloader
from pyhealth.models.td_cnn_lstm import TDCNNLSTM

STATE_CHANNELS: Tuple[str, ...] = (
    "timepoints",
    "values",
    "features",
    "delta_time",
    "delta_value",
)
"""The five aligned per-event channels of one ICU state."""

NEXT_PREFIX = "next_"
"""Prefix of the successor-state channels used by the TD target."""

LOOKBACK_MINUTES = 168 * 60
"""Reference 168-hour lookback window, in minutes before the state marker."""


# ---------------------------------------------------------------------------
# T34: optimizer and explicit TD training step
# ---------------------------------------------------------------------------
def make_reference_optimizer(
    model: TDCNNLSTM,
    n_training_batches: int,
    *,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> torch.optim.AdamW:
    """Constructs the AdamW optimizer for the online mortality predictor.

    By default, follows the *released train.py* rather than the paper's
    different learning-rate expression: ``lr = 1 / sqrt(n_parameters)`` and
    ``weight_decay = 1 / (lr * n_training_batches)``. The paper appendix
    instead describes ``1 / n_parameters`` for the learning rate.

    The reference loader's ``segments`` is the number of **full training
    batches per epoch**, ``n_samples // batch_size``. Supply that number here,
    not the number of patients or the total number of epochs. The authors
    also discard the final incomplete batch. If a different data-loader policy
    is used, document the difference in the private reproduction protocol.

    Optional overrides permit a documented departure from the authors' code.
    In both modes only the online predictor's trainable parameters are updated,
    excluding the frozen TD target and non-trainable scaling buffers.

    Args:
        model: Configured TD or supervised ``TDCNNLSTM``.
        n_training_batches: Positive count of full batches in one epoch.
        learning_rate: Optional positive learning rate override.
        weight_decay: Optional non-negative AdamW decay override.

    Returns:
        An ``AdamW`` optimizer covering only online trainable parameters.

    Raises:
        TypeError: If the model or batch count has an invalid type.
        ValueError: If a count or optimizer hyperparameter is invalid.
    """
    if not isinstance(model, TDCNNLSTM):
        raise TypeError("model must be a TDCNNLSTM")
    if isinstance(n_training_batches, bool) or not isinstance(
        n_training_batches, int
    ):
        raise TypeError("n_training_batches must be an integer")
    if n_training_batches <= 0:
        raise ValueError("n_training_batches must be positive")

    parameter_count = model.n_online_trainable_parameters
    if parameter_count <= 0:
        raise ValueError("The online predictor has no trainable parameters")

    if learning_rate is None:
        learning_rate = 1.0 / math.sqrt(parameter_count)
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")

    if weight_decay is None:
        weight_decay = 1.0 / (learning_rate * n_training_batches)
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be finite and non-negative")

    online_parameters = [
        parameter for parameter in model.online.parameters() if parameter.requires_grad
    ]
    return torch.optim.AdamW(
        online_parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
    )


def training_step(
    model: TDCNNLSTM,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """Performs one training step, including the TD target update when needed.

    The order is: zero online gradients, run the labeled model forward pass,
    backpropagate the objective-specific loss, update online weights, and then
    perform the target-network soft update for TD training. Supervised mode
    skips the target update because it has no target network.

    Args:
        model: TD or supervised model to optimize.
        optimizer: Optimizer configured for the online predictor.
        batch: Labeled PyHealth batch. TD batches also need matching successor
            state channels. See ``TDCNNLSTM.forward`` for required keys.

    Returns:
        The standard PyHealth output dictionary, with all tensors detached to
        prevent accumulation of autograd graphs in training logs.

    Raises:
        ValueError: If the batch is unlabeled or has invalid inputs.
    """
    model.train()
    optimizer.zero_grad()
    outputs = model(**batch)
    if "loss" not in outputs:
        raise ValueError("training_step requires an observed mortality label")

    outputs["loss"].backward()
    optimizer.step()
    if model.objective == "td":
        model.update_target()
    return {key: value.detach() for key, value in outputs.items()}


# ---------------------------------------------------------------------------
# T39: minimal public synthetic SampleDataset
# ---------------------------------------------------------------------------
def _padded(values: Sequence[float], pad: int, fill: float) -> List[float]:
    """Returns ``values`` as Python floats, right-padded with ``fill``."""
    return [float(value) for value in values] + [fill] * pad


def _empty_state(seq_len: int) -> Dict[str, List[float]]:
    """Returns an all-padded state, meaning *no eligible successor*."""
    return {
        "timepoints": [math.nan] * seq_len,
        "values": [math.nan] * seq_len,
        "features": [-1] * seq_len,
        "delta_time": [math.nan] * seq_len,
        "delta_value": [math.nan] * seq_len,
    }


def _generate_state(
    rng: np.random.Generator,
    n_events: int,
    seq_len: int,
    n_features: int,
    severity: float,
) -> Dict[str, List[float]]:
    """Generates one right-padded state of invented events.

    The first event is the state marker (time 0); the others are distinct
    earlier times within the 168-hour lookback. Even-numbered features drift
    upward toward the marker in proportion to ``severity``, so both the values
    and the same-feature deltas carry a (synthetic) mortality signal.

    Args:
        rng: Generator that owns all randomness of the dataset.
        n_events: Number of valid events, ``1 <= n_events <= seq_len``.
        seq_len: Padded state length ``L``.
        n_features: Feature vocabulary size.
        severity: Latent deterioration of this patient at this state.

    Returns:
        Mapping from each of the five channel names to ``seq_len`` entries.
    """
    earlier = rng.choice(
        np.arange(1, LOOKBACK_MINUTES + 1), size=n_events - 1, replace=False
    )
    timepoints = np.concatenate([[0.0], np.sort(earlier).astype(float)])
    features = rng.integers(0, n_features, size=n_events)

    feature_ids = np.arange(n_features)
    baseline = feature_ids.astype(float)
    effect = np.where(feature_ids % 2 == 0, 1.5, 0.0)
    recency = 1.0 - timepoints / LOOKBACK_MINUTES
    values = (
        baseline[features]
        + severity * effect[features] * recency
        + rng.normal(0.0, 0.5, size=n_events)
    )

    # we define deltas against the previous observation of the same feature
    delta_time = np.full(n_events, math.nan)
    delta_value = np.full(n_events, math.nan)
    last_seen: Dict[int, int] = {}
    for index in np.argsort(-timepoints, kind="stable"):  # oldest event first
        feature = int(features[index])
        if feature in last_seen:
            previous = last_seen[feature]
            delta_time[index] = timepoints[previous] - timepoints[index]
            delta_value[index] = values[index] - values[previous]
        last_seen[feature] = int(index)

    pad = seq_len - n_events
    return {
        "timepoints": _padded(timepoints, pad, math.nan),
        "values": _padded(values, pad, math.nan),
        "features": [int(feature) for feature in features] + [-1] * pad,
        "delta_time": _padded(delta_time, pad, math.nan),
        "delta_value": _padded(delta_value, pad, math.nan),
    }


def _check_positive_int(name: str, value: Any) -> None:
    """Raises if ``value`` is not a positive integer (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def make_synthetic_samples(
    n_patients: int = 48,
    states_per_patient: int = 3,
    seq_len: int = 32,
    n_features: int = 6,
    positive_fraction: float = 0.35,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generates invented paired-state samples for the TD-CNN-LSTM model.

    Each patient has ``states_per_patient`` consecutive states. The successor
    of state ``k`` is state ``k + 1`` of the same patient; the final state has
    an all-padded successor, which the TD loss treats as terminal and replaces
    with the observed label. Lengths vary from a single event to ``seq_len``,
    so samples include short and fully used sequences. Padding uses ``NaN``
    for scalar channels and ``-1`` for feature IDs, as in the T07 contract.

    The 28-day outcome is an invented, patient-level flag: exactly
    ``round(n_patients * positive_fraction)`` patients are positive, which
    guarantees both mortality classes. All randomness comes from one
    ``numpy`` generator seeded by ``seed``; global random state is unchanged.

    Args:
        n_patients: Number of synthetic patients.
        states_per_patient: Number of consecutive states per patient.
        seq_len: Padded length ``L`` of every state.
        n_features: Feature vocabulary size; valid IDs are ``0..n_features-1``.
        positive_fraction: Fraction of patients with the positive outcome.
        seed: Seed for the private random generator.

    Returns:
        A list of raw sample dictionaries with ``patient_id``,
        ``state_index``, the ten state channels, and an integer ``label``.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count is not positive, or the class sizes are too
            small for a stratified train/validation/test split.

    Examples:
        >>> samples = make_synthetic_samples(n_patients=12, seq_len=8)
        >>> len(samples), len(samples[0]["next_values"])
        (36, 8)
    """
    for name, value in (
        ("n_patients", n_patients),
        ("states_per_patient", states_per_patient),
        ("seq_len", seq_len),
        ("n_features", n_features),
    ):
        _check_positive_int(name, value)
    if not 0.0 < float(positive_fraction) < 1.0:
        raise ValueError("positive_fraction must be strictly between 0 and 1")
    n_positive = int(round(n_patients * float(positive_fraction)))
    if n_positive < 3 or n_patients - n_positive < 3:
        raise ValueError(
            "need at least 3 positive and 3 negative patients so that every "
            f"split has both classes; got {n_positive} positive of {n_patients}"
        )

    rng = np.random.default_rng(seed)
    positive = np.zeros(n_patients, dtype=bool)
    positive[rng.permutation(n_patients)[:n_positive]] = True

    samples: List[Dict[str, Any]] = []
    for patient in range(n_patients):
        latent = rng.normal(1.0 if positive[patient] else 0.0, 0.5)
        states = []
        for index in range(states_per_patient):
            n_events = int(rng.integers(1, seq_len + 1))
            severity = latent * (index + 1) / states_per_patient
            states.append(_generate_state(rng, n_events, seq_len, n_features, severity))

        for index, state in enumerate(states):
            successor = (
                states[index + 1]
                if index + 1 < states_per_patient
                else _empty_state(seq_len)
            )
            sample: Dict[str, Any] = {
                "patient_id": f"synthetic-{patient:03d}",
                "state_index": index,
            }
            for key in STATE_CHANNELS:
                sample[key] = list(state[key])
                sample[NEXT_PREFIX + key] = list(successor[key])
            sample["label"] = int(positive[patient])
            samples.append(sample)
    return samples


def synthetic_input_schema() -> Dict[str, Any]:
    """Returns the ten-channel input schema (``int64`` IDs, ``float32`` rest)."""
    schema: Dict[str, Any] = {}
    for prefix in ("", NEXT_PREFIX):
        for key in STATE_CHANNELS:
            schema[prefix + key] = (
                ("tensor", {"dtype": torch.long}) if key == "features" else "tensor"
            )
    return schema


def build_synthetic_dataset(**kwargs: Any) -> SampleDataset:
    """Builds the in-memory synthetic ``SampleDataset``.

    Args:
        **kwargs: Forwarded to :func:`make_synthetic_samples`.

    Returns:
        A ``SampleDataset`` with output schema ``{"label": "binary"}``.

    Examples:
        >>> dataset = build_synthetic_dataset(n_patients=12, seq_len=8)
        >>> sorted(dataset.output_schema)
        ['label']
    """
    return create_sample_dataset(
        samples=make_synthetic_samples(**kwargs),
        input_schema=synthetic_input_schema(),
        output_schema={"label": "binary"},
        dataset_name="synthetic_td_icu",
        task_name="td_icu_mortality_28d",
        in_memory=True,
    )


def split_by_patient_stratified(
    dataset: SampleDataset,
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> Tuple[SampleDataset, SampleDataset, SampleDataset]:
    """Splits by patient, stratified by the patient-level outcome.

    Unlike ``split_by_patient``, every split is guaranteed at least one
    patient of each class, so validation and test AUROC are defined on the
    synthetic data. No patient appears in more than one split.

    Args:
        dataset: Dataset whose samples carry ``patient_id`` and ``label``.
        ratios: Train, validation and test fractions summing to 1.
        seed: Seed of the patient shuffle.

    Returns:
        ``(train, validation, test)`` dataset subsets.

    Raises:
        ValueError: For invalid ratios or a class with fewer than 3 patients.
    """
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must be three positive fractions")
    if not math.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must sum to 1")

    patients_by_label: Dict[int, List[str]] = {0: [], 1: []}
    for patient_id, indices in dataset.patient_to_index.items():
        label = int(torch.as_tensor(dataset[indices[0]]["label"]).reshape(-1)[0])
        patients_by_label[label].append(patient_id)

    rng = np.random.default_rng(seed)
    split_patients: List[List[str]] = [[], [], []]
    for label in (0, 1):
        patients = sorted(patients_by_label[label], key=str)
        if len(patients) < 3:
            raise ValueError(f"class {label} needs at least 3 patients")
        rng.shuffle(patients)
        n_train = max(1, int(len(patients) * ratios[0]))
        n_val = max(1, int(len(patients) * ratios[1]))
        if len(patients) - n_train - n_val < 1:
            n_train = len(patients) - n_val - 1
        split_patients[0] += patients[:n_train]
        split_patients[1] += patients[n_train : n_train + n_val]
        split_patients[2] += patients[n_train + n_val :]

    subsets = []
    for patients in split_patients:
        indices = sorted(
            index for patient in patients for index in dataset.patient_to_index[patient]
        )
        subsets.append(dataset.subset(indices))
    return subsets[0], subsets[1], subsets[2]


def _as_numpy(value: Any) -> np.ndarray:
    """Converts a tensor or array-like value to a flat ``numpy`` array."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1)


def _mean_std(values: np.ndarray) -> Tuple[float, float]:
    """Returns finite-value mean and sample std (``NaN`` std if undefined).

    An unseen feature gets mean ``0`` and an undefined std; the model replaces
    undefined or zero standard deviations with ``1``.
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, math.nan
    std = float(values.std(ddof=1)) if values.size > 1 else math.nan
    return float(values.mean()), std


def compute_training_scaling(
    train_dataset: SampleDataset, n_features: int
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Derives the model's ``scaling`` mapping from training samples only.

    Uses current-state events of the training split. Successor states belong
    to the same training patients, so they add no out-of-split information.
    ``timepoints`` uses one shared mean/std; ``values``, ``delta_time`` and
    ``delta_value`` use per-feature statistics in vocabulary order.

    Args:
        train_dataset: The training split. Never pass validation/test data.
        n_features: Feature vocabulary size.

    Returns:
        ``{"timepoints", "values", "delta_time", "delta_value"}``, each a
        ``{"mean", "std"}`` mapping of ``float32`` tensors.
    """
    channels: Dict[str, List[np.ndarray]] = {key: [] for key in STATE_CHANNELS}
    for sample in train_dataset:
        for key in STATE_CHANNELS:
            channels[key].append(_as_numpy(sample[key]))
    data = {key: np.concatenate(parts) for key, parts in channels.items()}
    features = data["features"].astype(np.int64)
    event = np.isfinite(data["values"]) & np.isfinite(data["timepoints"])

    time_mean, time_std = _mean_std(data["timepoints"][event])
    scaling: Dict[str, Dict[str, torch.Tensor]] = {
        "timepoints": {
            "mean": torch.tensor(time_mean, dtype=torch.float32),
            "std": torch.tensor(time_std, dtype=torch.float32),
        }
    }
    for key in ("values", "delta_time", "delta_value"):
        means, stds = [], []
        for feature in range(n_features):
            mean, std = _mean_std(data[key][event & (features == feature)])
            means.append(mean)
            stds.append(std)
        scaling[key] = {
            "mean": torch.tensor(means, dtype=torch.float32),
            "std": torch.tensor(stds, dtype=torch.float32),
        }
    return scaling


# ---------------------------------------------------------------------------
# T40: training, evaluation and full-versus-no-delta ablation
# ---------------------------------------------------------------------------
def safe_binary_auroc(y_true: Any, y_prob: Any) -> float:
    """Computes AUROC, returning ``NaN`` when it is undefined (T41).

    AUROC is undefined for fewer than two samples or a single observed class;
    ``sklearn`` raises in that case, which would abort an evaluation loop.

    Args:
        y_true: Observed binary labels.
        y_prob: Predicted probabilities with the same number of entries.

    Returns:
        The AUROC as a float, or ``math.nan`` if it is undefined.

    Raises:
        ValueError: If the shapes differ or a probability is not finite.

    Examples:
        >>> safe_binary_auroc([0, 1, 1], [0.2, 0.7, 0.6])
        1.0
        >>> math.isnan(safe_binary_auroc([1, 1], [0.2, 0.7]))
        True
    """
    labels = _as_numpy(y_true).astype(float)
    probs = _as_numpy(y_prob).astype(float)
    if labels.shape != probs.shape:
        raise ValueError(
            f"y_true and y_prob sizes differ: {labels.shape} vs {probs.shape}"
        )
    if not np.isfinite(probs).all():
        raise ValueError("y_prob must be finite")
    if labels.size < 2 or np.unique(labels).size < 2:
        return math.nan
    return float(roc_auc_score(labels, probs))


@torch.no_grad()
def evaluate(model: TDCNNLSTM, dataloader: Any) -> Dict[str, Any]:
    """Scores a split in evaluation mode.

    Args:
        model: Trained model.
        dataloader: Loader of labeled batches.

    Returns:
        ``auroc`` (``NaN`` if undefined), ``n_samples`` and ``n_positive``.
    """
    model.eval()
    y_true, y_prob = [], []
    for batch in dataloader:
        outputs = model(**batch)
        y_true.append(_as_numpy(outputs["y_true"]))
        y_prob.append(_as_numpy(outputs["y_prob"]))
    labels = np.concatenate(y_true) if y_true else np.zeros(0)
    probs = np.concatenate(y_prob) if y_prob else np.zeros(0)
    return {
        "auroc": safe_binary_auroc(labels, probs),
        "n_samples": int(labels.size),
        "n_positive": int(labels.sum()),
    }


def train_one_epoch(
    model: TDCNNLSTM,
    optimizer: torch.optim.Optimizer,
    dataloader: Any,
    batch_size: int,
) -> float:
    """Runs one epoch of explicit TD training steps.

    Like the authors' loader, the final incomplete batch is skipped. This also
    keeps training-mode batch normalization away from one-sample batches.

    Args:
        model: TD model to optimize.
        optimizer: Optimizer from :func:`make_reference_optimizer`.
        dataloader: Shuffled training loader.
        batch_size: Full batch size used by ``dataloader``.

    Returns:
        Mean training loss over the full batches.
    """
    losses = []
    for batch in dataloader:
        if len(batch["label"]) < batch_size:
            continue
        outputs = training_step(model, optimizer, batch)
        losses.append(float(outputs["loss"]))
    return float(np.mean(losses)) if losses else math.nan


@dataclass
class ExampleConfig:
    """Settings of the synthetic example (small enough for CPU tests).

    ``learning_rate`` and ``weight_decay`` default to ``None``, which applies
    the released ``train.py`` formulas through
    :func:`make_reference_optimizer`. The reference settings use
    ``hidden_dim=32``, ``dropout=0.5``, batch size 64 and 10 epochs on MIMIC-IV.
    """

    n_patients: int = 48
    states_per_patient: int = 3
    seq_len: int = 32
    n_features: int = 6
    positive_fraction: float = 0.35
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    epochs: int = 3
    batch_size: int = 8
    hidden_dim: int = 8
    dropout: float = 0.5
    target_alpha: float = 0.99
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    seed: int = 42
    device: str = "cpu"


def fit_model(
    dataset: SampleDataset,
    train_dataset: SampleDataset,
    val_loader: Any,
    scaling: Mapping[str, Mapping[str, Any]],
    use_delta_embeddings: bool,
    config: ExampleConfig,
) -> Tuple[TDCNNLSTM, List[Dict[str, Any]], int]:
    """Trains one TD model and restores its best-validation-AUROC epoch.

    The model seed, training-loader shuffle seed and optimizer settings depend
    only on ``config``, so the full and no-delta models are trained under
    identical conditions.

    Args:
        dataset: Full dataset, used for the model's schema checks.
        train_dataset: Training split.
        val_loader: Validation loader used for checkpoint selection.
        scaling: Training-derived statistics.
        use_delta_embeddings: ``False`` trains the delta ablation.
        config: Example settings.

    Returns:
        ``(model, history, best_epoch)``. If validation AUROC is undefined in
        every epoch, the last epoch is kept.

    Raises:
        ValueError: If the training split has no full batch.
    """
    n_training_batches = len(train_dataset) // config.batch_size
    if n_training_batches < 1:
        raise ValueError(
            f"training split has {len(train_dataset)} samples, fewer than "
            f"batch_size={config.batch_size}"
        )

    torch.manual_seed(config.seed)
    model = TDCNNLSTM(
        dataset,
        n_features=config.n_features,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        use_delta_embeddings=use_delta_embeddings,
        objective="td",
        target_alpha=config.target_alpha,
        scaling=scaling,
    ).to(config.device)
    optimizer = make_reference_optimizer(
        model,
        n_training_batches,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # we build a fresh loader so each model sees the same shuffle order
    train_loader = get_dataloader(
        train_dataset, batch_size=config.batch_size, shuffle=True, seed=config.seed
    )

    history: List[Dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_score, best_epoch = -math.inf, 0
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, config.batch_size)
        validation = evaluate(model, val_loader)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_auroc": validation["auroc"]}
        )
        score = validation["auroc"]
        comparable = -math.inf if math.isnan(score) else score
        # we keep the latest epoch while no epoch has a defined AUROC
        if comparable > best_score or best_score == -math.inf:
            best_score, best_epoch = comparable, epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history, best_epoch


def run_example(config: Optional[ExampleConfig] = None) -> Dict[str, Any]:
    """Runs the full synthetic pipeline and the delta ablation.

    Global PyTorch CPU random state is restored afterwards.

    Args:
        config: Settings; defaults to :class:`ExampleConfig`.

    Returns:
        A dictionary with the configuration, split sizes and, for ``"full"``
        and ``"no_delta"``, the training history, best epoch, and validation
        and test metrics.
    """
    config = ExampleConfig() if config is None else config
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if config.batch_size < 2:
        raise ValueError("batch_size must be at least 2 for batch normalization")

    with torch.random.fork_rng(devices=[]):
        dataset = build_synthetic_dataset(
            n_patients=config.n_patients,
            states_per_patient=config.states_per_patient,
            seq_len=config.seq_len,
            n_features=config.n_features,
            positive_fraction=config.positive_fraction,
            seed=config.seed,
        )
        train, val, test = split_by_patient_stratified(
            dataset, config.ratios, seed=config.seed
        )
        scaling = compute_training_scaling(train, config.n_features)
        val_loader = get_dataloader(val, batch_size=config.batch_size, shuffle=False)
        test_loader = get_dataloader(test, batch_size=config.batch_size, shuffle=False)

        models: Dict[str, Dict[str, Any]] = {}
        for name, use_delta in (("full", True), ("no_delta", False)):
            model, history, best_epoch = fit_model(
                dataset, train, val_loader, scaling, use_delta, config
            )
            models[name] = {
                "use_delta_embeddings": use_delta,
                "best_epoch": best_epoch,
                "history": history,
                "validation": evaluate(model, val_loader),
                "test": evaluate(model, test_loader),
            }

    return {
        "config": asdict(config),
        "n_samples": {"train": len(train), "val": len(val), "test": len(test)},
        "models": models,
    }


def _format_auroc(value: float) -> str:
    """Formats an AUROC, showing undefined values explicitly."""
    return "undefined" if math.isnan(value) else f"{value:.3f}"


def format_results(results: Mapping[str, Any]) -> str:
    """Formats the ablation table printed by :func:`main`."""
    sizes = results["n_samples"]
    lines = [
        "Synthetic TD-CNN-LSTM example (invented data; not a clinical result)",
        f"states: train={sizes['train']} val={sizes['val']} test={sizes['test']}",
        "",
        f"{'model':<10}{'best epoch':>12}{'val AUROC':>12}{'test AUROC':>12}",
    ]
    for name, entry in results["models"].items():
        lines.append(
            f"{name:<10}{entry['best_epoch']:>12}"
            f"{_format_auroc(entry['validation']['auroc']):>12}"
            f"{_format_auroc(entry['test']['auroc']):>12}"
        )
    full = results["models"]["full"]["test"]["auroc"]
    reduced = results["models"]["no_delta"]["test"]["auroc"]
    difference = full - reduced
    lines.append(f"full - no_delta test AUROC: {_format_auroc(difference)}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Command-line entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        The results of :func:`run_example`.
    """
    defaults = ExampleConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-patients", type=int, default=defaults.n_patients)
    parser.add_argument("--seq-len", type=int, default=defaults.seq_len)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    args = parser.parse_args(argv)

    config = ExampleConfig(
        n_patients=args.n_patients,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    results = run_example(config)
    print(format_results(results))
    return results


if __name__ == "__main__":
    main()
