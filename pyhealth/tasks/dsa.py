"""PyHealth tasks for the UCI Daily and Sports Activities (DSA) dataset.

Implements two classification tasks and the Inter-domain Pairwise Distance
(IPD) computation from:

    Zhang et al. "Daily Physical Activity Monitoring — Adaptive Learning
    from Multi-source Motion Sensor Data." CHIL 2024 (PMLR 248:39-54).
    https://proceedings.mlr.press/v248/zhang24a.html

.. note:: Paper vs. Code Discrepancies

    The authors' released implementation (github.com/Oceanjinghai/
    HealthTimeSerial) diverges from the paper in several ways. This module
    replicates the **code's actual behavior** and documents each divergence:

    1. **IPD computation**: The paper describes a K-dimensional KDE with a
       K×K bandwidth matrix and MCMC sampling (Algorithm 1, Eq. 1). The
       code fits a *scalar* 1-D KDE (bandwidth=7.8, hardcoded) on flattened
       pairwise distances and draws exactly 10 deterministic samples
       (``random_state=0``).

    2. **Transfer mechanism**: The paper describes per-epoch learning-rate
       decay ``λ^{j+1} = λ^j · (1 − α_q)`` (Eq. 5). The code instead
       scales the *number of training epochs* per source domain:
       ``epochs = int(30 * 7 * weight / weight_all) + 1``. The learning
       rate stays fixed at 0.005 throughout.

    3. **Fine-tuning**: The paper describes k-fold cross-validation with an
       adaptive LR and R=5 degeneration stopping (Appendix A, Eq. 7-9).
       The code does a single fixed ``model.fit`` for 40 epochs with no
       adaptive stopping.

    4. **Learning rate**: The paper states ``λ₀ = 5×10⁻⁴`` (Appendix B).
       The code uses ``lr=0.005`` — ten times larger.

    5. **Domain ordering**: The paper sorts source domains in descending
       IPD order (Algorithm 2, line 10). The code iterates domains in
       their natural index order without sorting.

    6. **Models**: The paper references PyTorch LSTM and sktime
       Encoder/ResNet/TapNet. The code uses Keras LSTM (64 units,
       dropout=0.2) and a single-block Keras FCN. TapNet is absent.

Author:
    Edward Guan (edwardg2@illinois.edu)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import KernelDensity

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

# =====================================================================
# Constants — matching the author's code, not the paper
# =====================================================================

# Column slices within each 45-column segment row (0-indexed, end exclusive).
# Mirrors DSADataset._domain_cols so tasks are self-contained.
_DOMAIN_COLS: Dict[str, Tuple[int, int]] = {
    "T":  (0,  9),
    "RA": (9,  18),
    "LA": (18, 27),
    "RL": (27, 36),
    "LL": (36, 45),
}

# Distance metrics supported by compute_pairwise_distances
SUPPORTED_METRICS: List[str] = [
    "boss",
    "dtw_classic",
    "dtw_sakoechiba",
    "dtw_itakura",
    "dtw_multiscale",
    "dtw_fast",
    "euclidean",
]

# KDE hyperparameters (hardcoded in author's code, not stated in paper)
DEFAULT_KDE_BANDWIDTH: float = 7.8
DEFAULT_KDE_N_SAMPLES: int = 10
DEFAULT_KDE_RANDOM_STATE: int = 0

# Training hyperparameters (from author's code, NOT from paper Appendix B)
#
#   Code         vs.  Paper (Appendix B)
#   lr=0.005          λ₀ = 5×10⁻⁴
#   30 epochs/src     J = 50
#   40 epochs/tgt     J_target = 100
#   batch=16          (not stated)
#   no k-fold         k = 10
#   no degen stop     R = 5
CODE_LEARNING_RATE: float = 0.005
CODE_SOURCE_EPOCHS: int = 30
CODE_TARGET_EPOCHS_NO_TRANSFER: int = 40
CODE_TARGET_EPOCHS_NAIVE: int = 30
CODE_TARGET_EPOCHS_WEIGHTED: int = 40
CODE_EPOCH_SCALE_FACTOR: int = 7  # the '7' in int(30 * 7 * w / w_all) + 1
CODE_BATCH_SIZE: int = 16


# =====================================================================
# Private helpers (replicate DSADataset internals so tasks are
# self-contained and do not need a dataset reference at call time)
# =====================================================================

def _load_segment(filepath: str) -> np.ndarray:
    """Load one segment file into a (125, 45) float32 array."""
    arr = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
    if arr.shape != (125, 45):
        raise ValueError(
            f"Expected shape (125, 45), got {arr.shape}: {filepath}"
        )
    return arr


def _slice_domain(raw: np.ndarray, domain: str) -> np.ndarray:
    """Extract one domain's channels from a raw (125, 45) array.

    Returns shape (9, 125) — channels × timesteps.
    """
    start, end = _DOMAIN_COLS[domain]
    return raw[:, start:end].T


def _minmax_scale(ts: np.ndarray) -> np.ndarray:
    """Scale each channel of a (K, T) array independently to [-1, 1].

    Channels with zero range (flat signal) are left as zeros.
    """
    scaled = np.zeros_like(ts)
    for k in range(ts.shape[0]):
        mn, mx = ts[k].min(), ts[k].max()
        if mx > mn:
            scaled[k] = 2.0 * (ts[k] - mn) / (mx - mn) - 1.0
    return scaled


def _load_domain_ts(filepath: str, domain: str, scale: bool) -> np.ndarray:
    """Load, slice, and optionally scale one domain from a segment file."""
    raw = _load_segment(filepath)
    ts = _slice_domain(raw, domain)
    if scale:
        ts = _minmax_scale(ts)
    return ts


# =====================================================================
# Task classes
# =====================================================================

class DSAActivityClassification(BaseTask):
    """Full 19-class activity classification task for the DSA dataset.

    This is the standard multiclass formulation used in most activity
    recognition literature. The paper does not evaluate this setup —
    it uses binary classification only — so this task extends the
    paper's scope.

    Attributes:
        task_name (str): ``"dsa_activity_classification"``
        input_schema (Dict[str, str]): ``{"time_series": "tensor"}``
        output_schema (Dict[str, str]): ``{"label": "multiclass"}``
        target_domain (str): Sensor placement to load (e.g. ``"LA"``).
        scale (bool): Whether to apply per-channel min-max scaling.

    Example::

        >>> from pyhealth.datasets import DSADataset
        >>> dataset = DSADataset(root="./data/DSA")
        >>> task = DSAActivityClassification(target_domain="LA")
        >>> samples = dataset.set_task(task)
        >>> samples[0].keys()
        dict_keys(['patient_id', 'visit_id', 'time_series', 'label',
                   'activity_name', 'pair_id'])
    """

    task_name: str = "dsa_activity_classification"
    input_schema: Dict[str, str] = {"time_series": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        target_domain: str = "LA",
        scale: bool = True,
    ) -> None:
        """Initialise the 19-class activity classification task.

        Args:
            target_domain (str): Sensor domain to load. Must be one of
                ``["T", "RA", "LA", "RL", "LL"]``. Defaults to ``"LA"``
                (Left Arm, simulating a wrist wearable).
            scale (bool): If ``True``, apply per-channel min-max scaling
                to ``[-1, 1]``. Defaults to ``True``.

        Raises:
            ValueError: If ``target_domain`` is not a valid domain key.
        """
        if target_domain not in _DOMAIN_COLS:
            raise ValueError(
                f"target_domain must be one of {list(_DOMAIN_COLS)}, "
                f"got '{target_domain}'."
            )
        self.target_domain = target_domain
        self.scale = scale
        super().__init__()

    def __call__(self, patient) -> List[Dict]:
        """Extract multiclass samples from one patient.

        Each segment file in the patient's record becomes one sample.
        Time series data is loaded from disk, sliced to ``target_domain``,
        and optionally scaled.

        Args:
            patient: A PyHealth ``Patient`` object from ``DSADataset``.

        Returns:
            List of sample dicts, each containing:

            - ``patient_id`` (str): Subject identifier.
            - ``visit_id`` (str): Segment identifier.
            - ``time_series`` (np.ndarray): Shape ``(9, 125)``.
            - ``label`` (int): Activity index in ``[0, 18]``.
            - ``activity_name`` (str): Human-readable activity string.
            - ``pair_id`` (str): Pairwise synchronisation key for IPD.
        """
        samples = []
        for event in patient.get_events(event_type="dsa"):
            ts = _load_domain_ts(event.filepath, self.target_domain, self.scale)
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id":   event.visit_id,
                    "time_series": ts,
                    "label":       int(event.label),
                    "activity_name": event.activity_name,
                    "pair_id":     event.pair_id,
                }
            )
        return samples


class DSABinaryActivityClassification(BaseTask):
    """Binary one-vs-rest activity classification for the DSA dataset.

    Replicates the paper's experimental setup: one activity is the
    positive class (label=1), all others are negative (label=0).
    Positive samples are upsampled during training and negative samples
    are downsampled during evaluation to maintain class balance.

    Attributes:
        task_name (str): ``"dsa_binary_classification"``
        input_schema (Dict[str, str]): ``{"time_series": "tensor"}``
        output_schema (Dict[str, str]): ``{"label": "binary"}``
        positive_activity_id (int): 1-indexed activity ID treated as
            the positive class.
        target_domain (str): Sensor placement to load.
        scale (bool): Whether to apply per-channel min-max scaling.

    Example::

        >>> task = DSABinaryClassification(
        ...     positive_activity_id=12,
        ...     target_domain="LA",
        ... )
        >>> samples = dataset.set_task(task)
        >>> samples[0].keys()
        dict_keys(['patient_id', 'visit_id', 'time_series', 'label',
                   'activity_id', 'activity_name', 'pair_id'])
    """

    task_name: str = "dsa_binary_classification"
    input_schema: Dict[str, str] = {"time_series": "tensor"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(
        self,
        positive_activity_id: int,
        target_domain: str = "LA",
        scale: bool = True,
    ) -> None:
        """Initialise the binary activity classification task.

        Args:
            positive_activity_id (int): 1-indexed activity ID treated as
                the positive class (1–19). All other activities are
                negative. In the paper's protocol this is chosen randomly
                for each repetition.
            target_domain (str): Sensor domain to load. Must be one of
                ``["T", "RA", "LA", "RL", "LL"]``. Defaults to ``"LA"``.
            scale (bool): If ``True``, apply per-channel min-max scaling
                to ``[-1, 1]``. Defaults to ``True``.

        Raises:
            ValueError: If ``target_domain`` is not a valid domain key.
            ValueError: If ``positive_activity_id`` is not in 1–19.
        """
        if target_domain not in _DOMAIN_COLS:
            raise ValueError(
                f"target_domain must be one of {list(_DOMAIN_COLS)}, "
                f"got '{target_domain}'."
            )
        if not (1 <= positive_activity_id <= 19):
            raise ValueError(
                f"positive_activity_id must be in [1, 19], "
                f"got {positive_activity_id}."
            )
        self.positive_activity_id = positive_activity_id
        self.target_domain = target_domain
        self.scale = scale
        super().__init__()

    def __call__(self, patient) -> List[Dict]:
        """Extract binary samples from one patient.

        Each segment becomes one sample with ``label=1`` if the
        segment's activity matches ``positive_activity_id``, else
        ``label=0``.

        Args:
            patient: A PyHealth ``Patient`` object from ``DSADataset``.

        Returns:
            List of sample dicts, each containing:

            - ``patient_id`` (str): Subject identifier.
            - ``visit_id`` (str): Segment identifier.
            - ``time_series`` (np.ndarray): Shape ``(9, 125)``.
            - ``label`` (int): ``1`` if positive class, else ``0``.
            - ``activity_id`` (int): 1-indexed activity ID.
            - ``activity_name`` (str): Human-readable activity string.
            - ``pair_id`` (str): Pairwise synchronisation key for IPD.
        """
        samples = []
        for event in patient.get_events(event_type="dsa"):
            ts = _load_domain_ts(event.filepath, self.target_domain, self.scale)
            is_positive = int(event.activity_id) == self.positive_activity_id
            samples.append(
                {
                    "patient_id":    patient.patient_id,
                    "visit_id":      event.visit_id,
                    "time_series":   ts,
                    "label":         1 if is_positive else 0,
                    "activity_id":   int(event.activity_id),
                    "activity_name": event.activity_name,
                    "pair_id":       event.pair_id,
                }
            )
        return samples


# =====================================================================
# Inter-domain Pairwise Distance (IPD)
# =====================================================================

def compute_pairwise_distances(
    source_ts: np.ndarray,
    target_ts: np.ndarray,
    metric: str = "dtw_classic",
) -> np.ndarray:
    """Compute scalar pairwise distances between paired time series.

    Replicates ``cal_similarity`` from the author's ``metrics.py``.
    Each pair of univariate time series produces one scalar distance.

    .. note:: Paper vs. Code

        The paper (Algorithm 1) computes per-channel distances and
        assembles them into K-dimensional vectors. The code squeezes
        each sample to 1-D and computes a single scalar, collapsing
        the multivariate structure entirely.

    Args:
        source_ts (np.ndarray): Source domain array of shape ``(N, T, 1)``
            or ``(N, T)`` where N is the number of paired samples.
        target_ts (np.ndarray): Target domain array with the same shape.
        metric (str): Distance metric name. One of ``SUPPORTED_METRICS``.

    Returns:
        np.ndarray: Shape ``(N,)`` containing one distance per pair.

    Raises:
        ValueError: If ``metric`` is not in ``SUPPORTED_METRICS``.
    """
    from pyts.metrics import boss
    from pyts.metrics import dtw as _dtw

    n_samples = source_ts.shape[0]
    distances = np.zeros(n_samples)

    for i in range(n_samples):
        s = np.squeeze(source_ts[i])
        t = np.squeeze(target_ts[i])

        if metric == "boss":
            distances[i] = boss(s, t)
        elif metric == "dtw_classic":
            distances[i] = _dtw(s, t)
        elif metric == "dtw_sakoechiba":
            distances[i] = _dtw(s, t, method="sakoechiba", options={"window_size": 0.5})
        elif metric == "dtw_itakura":
            distances[i] = _dtw(s, t, method="itakura", options={"max_slope": 1.5})
        elif metric == "dtw_multiscale":
            distances[i] = _dtw(s, t, method="multiscale", options={"resolution": 2})
        elif metric == "dtw_fast":
            distances[i] = _dtw(s, t, method="fast", options={"radius": 1})
        elif metric == "euclidean":
            distances[i] = float(np.linalg.norm(s - t))
        else:
            raise ValueError(
                f"Unknown metric '{metric}'. Supported: {SUPPORTED_METRICS}"
            )

    return distances


def compute_ipd_weight(
    distances: np.ndarray,
    bandwidth: float = DEFAULT_KDE_BANDWIDTH,
    n_samples: int = DEFAULT_KDE_N_SAMPLES,
    random_state: int = DEFAULT_KDE_RANDOM_STATE,
) -> float:
    """Compute a single IPD weight from pairwise distances via KDE.

    Replicates the author's code: fit a 1-D Gaussian KDE on the scalar
    distances, draw ``n_samples`` deterministic samples, and return the
    mean as the domain weight.

    .. note:: Paper vs. Code

        The paper (Section 4.1) describes fitting a multivariate Gaussian
        KDE on K-dimensional difference vectors, then sampling via MCMC
        and computing a matrix norm. The code fits a **scalar** KDE on
        flattened distances, draws exactly 10 samples with a fixed seed,
        and returns the scalar mean. The bandwidth 7.8 is hardcoded with
        no justification or scale-dependence on the distance metric.

    Args:
        distances (np.ndarray): Shape ``(N,)`` from
            ``compute_pairwise_distances``.
        bandwidth (float): KDE bandwidth. Author's code hardcodes ``7.8``.
        n_samples (int): Number of points to sample from the fitted KDE.
        random_state (int): Seed for deterministic KDE sampling.

    Returns:
        float: Scalar weight (higher = less similar to target domain).
    """
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
        distances.flatten().reshape(-1, 1)
    )
    weight = float(
        np.mean(kde.sample(n_samples, random_state=random_state), axis=0)[0]
    )
    return weight


def compute_all_ipd_weights(
    domain_data: Dict[str, np.ndarray],
    target_domain: str,
    metric: str = "dtw_classic",
    bandwidth: float = DEFAULT_KDE_BANDWIDTH,
) -> Dict[str, float]:
    """Compute IPD weights for all source domains relative to a target.

    Top-level function that runs the full IPD pipeline for one choice
    of target domain.

    Args:
        domain_data (Dict[str, np.ndarray]): Mapping of domain key to
            array of shape ``(N, T, 1)`` or ``(N, T)``. All domains must
            have the same number of paired samples N.
        target_domain (str): Key of the target domain in ``domain_data``.
        metric (str): Distance metric for pairwise computation.
        bandwidth (float): KDE bandwidth.

    Returns:
        Dict[str, float]: Mapping of source domain key to its scalar IPD
        weight. The target domain is excluded from the output.

    Example::

        >>> weights = compute_all_ipd_weights(
        ...     domain_data={"T": ts_t, "RA": ts_ra, "LA": ts_la,
        ...                  "RL": ts_rl, "LL": ts_ll},
        ...     target_domain="LA",
        ...     metric="dtw_classic",
        ... )
        >>> for domain, w in sorted(weights.items(), key=lambda x: x[1]):
        ...     print(f"{domain}: {w:.4f}")
    """
    target_ts = domain_data[target_domain]
    weights: Dict[str, float] = {}

    for domain, source_ts in domain_data.items():
        if domain == target_domain:
            continue
        distances = compute_pairwise_distances(source_ts, target_ts, metric)
        weights[domain] = compute_ipd_weight(distances, bandwidth=bandwidth)
        logger.info(
            "IPD weight %s → %s: %.4f (metric=%s)",
            domain, target_domain, weights[domain], metric,
        )

    return weights


# =====================================================================
# Epoch scaling — replicating the author's transfer mechanism
# =====================================================================

def compute_weighted_epochs(
    weights: Dict[str, float],
    base_epochs: int = CODE_SOURCE_EPOCHS,
    scale_factor: int = CODE_EPOCH_SCALE_FACTOR,
) -> Dict[str, int]:
    """Compute per-domain epoch counts from IPD weights.

    Replicates the author's epoch-scaling formula:
    ``epochs = int(base * scale_factor * weight / weight_sum) + 1``

    .. note:: Paper vs. Code

        The paper (Eq. 5) describes learning-rate decay
        ``λ^{j+1} = λ^j · (1 − α_q)`` where ``α_q = g_q / Σg_l``.
        The code does NOT implement LR decay. Instead, it scales the
        number of epochs per source domain proportionally to the IPD
        weight, keeping the learning rate fixed at 0.005.

    Args:
        weights (Dict[str, float]): Mapping of source domain key to IPD
            weight, from ``compute_all_ipd_weights``.
        base_epochs (int): Base epoch count before scaling (code: 30).
        scale_factor (int): Multiplier in the formula (code: 7,
            unexplained in the paper).

    Returns:
        Dict[str, int]: Mapping of source domain key to epoch count.
    """
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        return {d: base_epochs for d in weights}
    return {
        domain: int(base_epochs * scale_factor * w / weight_sum) + 1
        for domain, w in weights.items()
    }


# =====================================================================
# Experiment configuration
# =====================================================================

class ExperimentConfig:
    """Configuration for one run of the DPAM replication.

    Bundles all hyperparameters needed to run the three experimental
    conditions (No Transfer, Naive Transfer, Weighted Transfer) for a
    given target domain and distance metric.

    Code-default values replicate the author's implementation.
    Paper-stated values are provided in comments for reference.

    Args:
        target_domain (str): Sensor domain to classify from.
        metric (str): Distance metric for IPD computation.
        learning_rate (float): Optimizer LR (code: 0.005, paper: 5e-4).
        batch_size (int): Training batch size (code: 16).
        source_epochs (int): Epochs per source domain (code: 30, paper J=50).
        target_epochs_no_transfer (int): Epochs for No Transfer (code: 40).
        target_epochs_naive (int): Epochs for Naive Transfer (code: 30).
        target_epochs_weighted (int): Epochs for Weighted Transfer (code: 40).
        epoch_scale_factor (int): Multiplier in epoch formula (code: 7).
        kde_bandwidth (float): KDE bandwidth (code: 7.8).
        kde_n_samples (int): KDE sample count (code: 10).
        positive_activity_id (Optional[int]): Activity ID for binary
            setup. ``None`` uses 19-class classification.
        n_repeats (int): Number of random subject-split repetitions.
        n_train_subjects (int): Number of subjects in training set.

    Example::

        >>> config = ExperimentConfig(
        ...     target_domain="LA",
        ...     metric="dtw_classic",
        ...     positive_activity_id=12,
        ... )
    """

    def __init__(
        self,
        target_domain: str = "LA",
        metric: str = "dtw_classic",
        learning_rate: float = CODE_LEARNING_RATE,
        batch_size: int = CODE_BATCH_SIZE,
        source_epochs: int = CODE_SOURCE_EPOCHS,
        target_epochs_no_transfer: int = CODE_TARGET_EPOCHS_NO_TRANSFER,
        target_epochs_naive: int = CODE_TARGET_EPOCHS_NAIVE,
        target_epochs_weighted: int = CODE_TARGET_EPOCHS_WEIGHTED,
        epoch_scale_factor: int = CODE_EPOCH_SCALE_FACTOR,
        kde_bandwidth: float = DEFAULT_KDE_BANDWIDTH,
        kde_n_samples: int = DEFAULT_KDE_N_SAMPLES,
        positive_activity_id: Optional[int] = None,
        n_repeats: int = 15,
        n_train_subjects: int = 6,
    ) -> None:
        self.target_domain = target_domain
        self.metric = metric
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.source_epochs = source_epochs
        self.target_epochs_no_transfer = target_epochs_no_transfer
        self.target_epochs_naive = target_epochs_naive
        self.target_epochs_weighted = target_epochs_weighted
        self.epoch_scale_factor = epoch_scale_factor
        self.kde_bandwidth = kde_bandwidth
        self.kde_n_samples = kde_n_samples
        self.positive_activity_id = positive_activity_id
        self.n_repeats = n_repeats
        self.n_train_subjects = n_train_subjects

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig(target_domain={self.target_domain!r}, "
            f"metric={self.metric!r}, "
            f"positive_activity_id={self.positive_activity_id})"
        )


class ExperimentResult:
    """Results from one repetition of the experiment.

    Args:
        repeat_idx (int): Index of this repetition (0-based).
        train_subjects (List[int]): Subject IDs used for training.
        test_subjects (List[int]): Subject IDs used for testing.
        metric (str): Distance metric used for IPD.
        ipd_weights (Dict[str, float]): Per-domain IPD weights.
        weighted_epochs (Dict[str, int]): Per-domain epoch counts.
        accuracy_no_transfer (float): RCC for No Transfer baseline.
        accuracy_naive_transfer (float): RCC for Naive Transfer.
        accuracy_weighted_transfer (float): RCC for Weighted Transfer.
    """

    def __init__(
        self,
        repeat_idx: int = 0,
        train_subjects: Optional[List[int]] = None,
        test_subjects: Optional[List[int]] = None,
        metric: str = "dtw_classic",
        ipd_weights: Optional[Dict[str, float]] = None,
        weighted_epochs: Optional[Dict[str, int]] = None,
        accuracy_no_transfer: float = 0.0,
        accuracy_naive_transfer: float = 0.0,
        accuracy_weighted_transfer: float = 0.0,
    ) -> None:
        self.repeat_idx = repeat_idx
        self.train_subjects = train_subjects or []
        self.test_subjects = test_subjects or []
        self.metric = metric
        self.ipd_weights = ipd_weights or {}
        self.weighted_epochs = weighted_epochs or {}
        self.accuracy_no_transfer = accuracy_no_transfer
        self.accuracy_naive_transfer = accuracy_naive_transfer
        self.accuracy_weighted_transfer = accuracy_weighted_transfer

    def __repr__(self) -> str:
        return (
            f"ExperimentResult(repeat={self.repeat_idx}, "
            f"no_transfer={self.accuracy_no_transfer:.4f}, "
            f"naive={self.accuracy_naive_transfer:.4f}, "
            f"weighted={self.accuracy_weighted_transfer:.4f})"
        )
