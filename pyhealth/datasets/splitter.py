import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from numbers import Real

import numpy as np
import torch

from .sample_dataset import SampleDataset

# TODO: train_dataset.dataset still access the whole dataset which may leak information
# TODO: add more splitting methods


class Split(ABC):
    """Interface for ``BaseDataset.set_task(task, split=...)``.

    ``set_task`` fits every processor on the samples of part 0 and returns one
    ``SampleDataset`` per part, in order, as views over one processed cache.
    It uses the parts exactly as returned: they may overlap or leave samples
    out. :class:`PatientSplit` is the reference implementation.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import Split
        >>> class HoldoutSplit(Split):
        ...     def __init__(self, test_patients):
        ...         self.test_patients = sorted(test_patients)
        ...     def split_indices(self, patient_to_index):
        ...         test = set(self.test_patients)
        ...         parts = ([], [])
        ...         for pid, indices in patient_to_index.items():
        ...             parts[pid in test].extend(indices)
        ...         return [np.array(sorted(p), dtype=np.int64) for p in parts]
        ...     def to_dict(self):
        ...         return {"kind": "holdout", "test_patients": self.test_patients}
        >>> HoldoutSplit(["p2"]).split_indices({"p1": [0, 1], "p2": [2]})
        [array([0, 1]), array([2])]
    """

    @abstractmethod
    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> list[np.ndarray]:
        """Returns the sample indices of each part.

        Args:
            patient_to_index: Patient ID to the indices of its samples.

        Returns:
            One ``int64`` array of sample indices per part, training part
            first. Sorted indices keep reads from the cache sequential.
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """Returns a JSON-serialisable description of the split.

        It goes into the processed-cache key, so it must change whenever the
        parts would, and is saved as ``SampleDataset.fit_split``.
        """


@dataclass(frozen=True)
class PatientSplit(Split):
    """A patient-level split for ``BaseDataset.set_task(task, split=...)``.

    With a split, ``set_task`` fits every processor on the training patients
    only and returns one dataset per part, so validation and test patients
    never shape preprocessing (vocabularies, learned statistics).

    Patients are sorted by ID before shuffling with ``seed``, so the same
    ``PatientSplit`` gives the same patients on every run and machine,
    whatever order the samples are stored in.

    Unlike :func:`split_by_patient`, which splits an already-processed
    ``SampleDataset`` whose processors were fitted on all samples, the parts
    are chosen before fitting. ``split_by_patient`` shuffles patients in the
    order they appear in the dataset, so for the same ratios and seed it
    generally picks different patients.

    Args:
        ratios: Two or more non-negative fractions summing to 1. The first
            part is the training part, e.g. (train, test), (train, val, test)
            or (train, val, cal, test).
        seed: Seed for the shuffle.

    Examples:
        >>> from pyhealth.datasets import PatientSplit
        >>> split = PatientSplit(ratios=(0.7, 0.1, 0.2), seed=42)
        >>> parts = split.split_indices({"p1": [0, 1], "p2": [2], "p3": [3, 4]})
        >>> sorted(int(i) for part in parts for i in part)
        [0, 1, 2, 3, 4]
    """

    ratios: Sequence[float]
    seed: int = 0

    def __post_init__(self):
        ratios = tuple(float(r) for r in self.ratios)
        if len(ratios) < 2:
            raise ValueError(f"PatientSplit needs at least 2 ratios, training part first; got {ratios}.")
        if any(r < 0 for r in ratios):
            raise ValueError(f"PatientSplit ratios must be non-negative, got {ratios}.")
        if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
            raise ValueError(f"PatientSplit ratios must sum to 1, got {ratios} (sum {sum(ratios)}).")
        object.__setattr__(self, "ratios", ratios)

    def to_dict(self) -> dict:
        """JSON-serialisable description, used in cache keys and metadata."""
        return {"kind": "patient", "ratios": list(self.ratios), "seed": int(self.seed)}

    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> list[np.ndarray]:
        """Splits sample indices by patient.

        Patients are sorted, shuffled with ``seed`` and cut by ``ratios``.

        Args:
            patient_to_index: Patient ID to the indices of its samples.

        Returns:
            One sorted ``int64`` array of sample indices per ratio, in order
            (train first). Each patient's samples land in exactly one part.
        """
        patients = sorted(patient_to_index, key=str)
        order = np.random.default_rng(self.seed).permutation(len(patients))
        cuts = [
            int(len(patients) * sum(self.ratios[: k + 1]))
            for k in range(len(self.ratios) - 1)
        ]
        parts = []
        for part in np.split(order, cuts):
            indices = np.fromiter(
                chain.from_iterable(patient_to_index[patients[p]] for p in part),
                dtype=np.int64,
            )
            indices.sort()
            parts.append(indices)
        return parts


def _validate_ratios(ratios: object, expected_length: int) -> None:
    """Validate split proportions without changing the caller's values."""
    if not isinstance(ratios, (list, tuple, np.ndarray)):
        raise TypeError("ratios must be a list, tuple, or 1-D NumPy array")
    if isinstance(ratios, np.ndarray) and ratios.ndim != 1:
        raise ValueError("ratios must be one-dimensional")
    if len(ratios) != expected_length:
        raise ValueError(f"ratios must contain exactly {expected_length} values")

    values = []
    for index, value in enumerate(ratios):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"ratios[{index}] must be a real number, not a boolean")
        try:
            numeric_value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                f"ratios[{index}] must be finite and between 0 and 1"
            ) from exc
        if not math.isfinite(numeric_value):
            raise ValueError(f"ratios[{index}] must be finite")
        if not 0 <= value <= 1:
            raise ValueError(f"ratios[{index}] must be between 0 and 1")
        values.append(numeric_value)

    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"ratios must sum to 1.0 within 1e-6; got {total!r}")


def _label_to_int(label) -> int:
    """Convert a stored label (int/np scalar/torch scalar) to Python int."""
    if torch.is_tensor(label):
        return int(label.item())
    return int(label)


def sample_balanced(
    dataset: SampleDataset,
    ratio: float = 1.0,
    subsample: float = 1.0,
    seed: int | None = None,
) -> SampleDataset:
    """Keep positives and negatives at a target ratio, then cap total size.

    Args:
        dataset: Dataset with ``patient_to_index`` populated.
        ratio: Negatives per positive (e.g., 1.0 -> ~1 neg per pos). Values <=0 keep only positives.
        subsample: Max fraction of the original dataset size to retain. If the ratio-selected set
            exceeds ``len(dataset) * subsample``, both positives and negatives are downsampled
            proportionally while preserving the ratio as closely as possible.
        seed: Optional RNG seed for reproducible negative sampling.

    Returns:
        A new ``SampleDataset`` containing all positives plus sampled negatives,
        with refreshed ``patient_to_index`` and ``record_to_index`` mappings.
    """

    if ratio < 0:
        raise ValueError("ratio must be non-negative")
    if subsample <= 0 or subsample > 1:
        raise ValueError("subsample must be in (0, 1]")

    rng = np.random.default_rng(seed)

    pos_indices: list[int] = []
    neg_indices: list[int] = []

    for idx in range(len(dataset)):
        label = _label_to_int(dataset[idx]["label"])
        if label == 1:
            pos_indices.append(idx)
        else:
            neg_indices.append(idx)

    if not pos_indices:
        return dataset

    desired_pos = len(pos_indices)
    desired_neg = min(len(neg_indices), int(round(desired_pos * ratio)))

    cap = max(1, int(len(dataset) * subsample))
    desired_total = desired_pos + desired_neg

    if desired_total <= cap:
        keep_pos = desired_pos
        keep_neg = desired_neg
    else:
        ratio_effective = desired_neg / desired_pos if desired_pos > 0 else 0.0
        keep_pos = max(1, min(desired_pos, int(cap / (1 + ratio_effective))))
        keep_neg = int(round(keep_pos * ratio_effective)) if ratio_effective > 0 else 0
        keep_neg = min(keep_neg, len(neg_indices))
        if keep_pos + keep_neg > cap:
            keep_neg = max(0, cap - keep_pos)

    if keep_pos < desired_pos:
        pos_keep = list(rng.choice(pos_indices, size=keep_pos, replace=False))
    else:
        pos_keep = pos_indices

    if keep_neg > 0:
        neg_keep = list(rng.choice(neg_indices, size=keep_neg, replace=False))
    else:
        neg_keep = []

    keep_indices = pos_keep + neg_keep

    balanced = dataset.subset(keep_indices)  # type: ignore

    # Rebuild patient_to_index and record_to_index for the reduced set.
    balanced.patient_to_index = {}
    balanced.record_to_index = {}
    for i in range(len(balanced)):
        sample = balanced[i]
        pid = sample.get("patient_id")
        rid = sample.get("record_id", sample.get("visit_id"))
        if pid is not None:
            balanced.patient_to_index.setdefault(pid, []).append(i)
        if rid is not None:
            balanced.record_to_index.setdefault(rid, []).append(i)

    return balanced


def split_by_visit(
    dataset: SampleDataset,
    ratios: tuple[float, float, float] | list[float],
    seed: int | None = None,
):
    """Splits the dataset by visit (i.e., samples).

    Args:
        dataset: a `SampleDataset` object
        ratios: a list/tuple of ratios for train / val / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, test_dataset: ``SampleDataset`` views
            (``dataset.subset``) over the same processed samples.

    Raises:
        TypeError: If ``ratios`` has an unsupported container or element type.
        ValueError: If its length, components, or total is invalid.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [{"patient_id": f"p{i}", "record_id": f"r{i}", "feature": i,
        ...             "label": i % 2} for i in range(8)]
        >>> dataset = create_sample_dataset(samples, {"feature": "raw"}, {"label": "raw"})
        >>> train, val, test = split_by_visit(dataset, [0.5, 0.25, 0.25], seed=42)
        >>> [len(part) for part in (train, val, test)]
        [4, 2, 2]

    Note:
        The processors of ``dataset`` (vocabularies, learned statistics) were
        fitted on all of its samples, so every part shaped them. To fit them on
        the training part only, split inside ``set_task`` instead:
        ``base_dataset.set_task(task, split=...)`` with :class:`PatientSplit`
        or a custom :class:`Split`.
    """
    _validate_ratios(ratios, expected_length=3)
    rng = np.random.default_rng(seed)
    index = np.arange(len(dataset))
    rng.shuffle(index)
    train_index = index[: int(len(dataset) * ratios[0])]
    val_index = index[
        int(len(dataset) * ratios[0]) : int(len(dataset) * (ratios[0] + ratios[1]))
    ]
    test_index = index[int(len(dataset) * (ratios[0] + ratios[1])) :]
    train_dataset = dataset.subset(train_index) # type: ignore
    val_dataset = dataset.subset(val_index) # type: ignore
    test_dataset = dataset.subset(test_index) # type: ignore
    return train_dataset, val_dataset, test_dataset


def split_by_patient(
    dataset: SampleDataset,
    ratios: tuple[float, float, float] | list[float],
    seed: int | None = None,
):
    """Splits the dataset by patient.

    Args:
        dataset: a `SampleDataset` object
        ratios: a list/tuple of ratios for train / val / test
        seed: random seed for shuffling the dataset. Patients are sorted by ID
            before shuffling, so the split depends only on the seed and the
            set of patients, not on the order of the samples.

    Returns:
        train_dataset, val_dataset, test_dataset: ``SampleDataset`` views
            (``dataset.subset``) over the same processed samples.

    Raises:
        TypeError: If ``ratios`` has an unsupported container or element type.
        ValueError: If its length, components, or total is invalid.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [{"patient_id": f"p{i}", "record_id": f"r{i}", "feature": i,
        ...             "label": i % 2} for i in range(8)]
        >>> dataset = create_sample_dataset(samples, {"feature": "raw"}, {"label": "raw"})
        >>> train, val, test = split_by_patient(dataset, [0.5, 0.25, 0.25], seed=42)
        >>> [len(part) for part in (train, val, test)]
        [4, 2, 2]

    Note:
        The processors of ``dataset`` (vocabularies, learned statistics) were
        fitted on all of its samples, so every part shaped them. To fit them on
        the training part only, split inside ``set_task`` instead:
        ``base_dataset.set_task(task, split=...)`` with :class:`PatientSplit`
        or a custom :class:`Split`.
    """
    _validate_ratios(ratios, expected_length=3)
    rng = np.random.default_rng(seed)
    # Sort first: dict order follows sample order, which callers may change.
    patient_indx = sorted(dataset.patient_to_index, key=str)
    num_patients = len(patient_indx)
    rng.shuffle(patient_indx)
    train_patient_indx = patient_indx[: int(num_patients * ratios[0])]
    val_patient_indx = patient_indx[
        int(num_patients * ratios[0]) : int(num_patients * (ratios[0] + ratios[1]))
    ]
    test_patient_indx = patient_indx[int(num_patients * (ratios[0] + ratios[1])) :]
    train_index = list(
        chain(*[dataset.patient_to_index[i] for i in train_patient_indx])
    )
    val_index = list(chain(*[dataset.patient_to_index[i] for i in val_patient_indx]))
    test_index = list(chain(*[dataset.patient_to_index[i] for i in test_patient_indx]))
    train_dataset = dataset.subset(train_index) # type: ignore
    val_dataset = dataset.subset(val_index) # type: ignore
    test_dataset = dataset.subset(test_index) # type: ignore
    return train_dataset, val_dataset, test_dataset


def split_by_sample(
    dataset: SampleDataset,
    ratios: tuple[float, float, float] | list[float],
    seed: int | None = None,
    get_index: bool = False,
):
    """Splits the dataset by sample

    Args:
        dataset: a `SampleDataset` object
        ratios: a list/tuple of ratios for train / val / test
        seed: random seed for shuffling the dataset
        get_index: if True, return three ``torch.Tensor`` index vectors
            instead of datasets

    Returns:
        train_dataset, val_dataset, test_dataset: ``SampleDataset`` views
            (``dataset.subset``) over the same processed samples, or index
            tensors if ``get_index`` is True.

    Raises:
        TypeError: If ``ratios`` has an unsupported container or element type.
        ValueError: If its length, components, or total is invalid.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [{"patient_id": f"p{i}", "record_id": f"r{i}", "feature": i,
        ...             "label": i % 2} for i in range(8)]
        >>> dataset = create_sample_dataset(samples, {"feature": "raw"}, {"label": "raw"})
        >>> train, val, test = split_by_sample(dataset, [0.5, 0.25, 0.25], seed=42)
        >>> [len(part) for part in (train, val, test)]
        [4, 2, 2]

    Note:
        The processors of ``dataset`` (vocabularies, learned statistics) were
        fitted on all of its samples, so every part shaped them. To fit them on
        the training part only, split inside ``set_task`` instead:
        ``base_dataset.set_task(task, split=...)`` with :class:`PatientSplit`
        or a custom :class:`Split`.
    """
    _validate_ratios(ratios, expected_length=3)
    rng = np.random.default_rng(seed)
    index = np.arange(len(dataset))
    rng.shuffle(index)
    train_index = index[: int(len(dataset) * ratios[0])]
    val_index = index[
        int(len(dataset) * ratios[0]) : int(len(dataset) * (ratios[0] + ratios[1]))
    ]
    test_index = index[int(len(dataset) * (ratios[0] + ratios[1])) :]
    train_dataset = dataset.subset(train_index) # type: ignore
    val_dataset = dataset.subset(val_index) # type: ignore
    test_dataset = dataset.subset(test_index) # type: ignore

    if get_index:
        return (
            torch.tensor(train_index),
            torch.tensor(val_index),
            torch.tensor(test_index),
        )
    else:
        return train_dataset, val_dataset, test_dataset


def split_by_visit_conformal(
    dataset: SampleDataset,
    ratios: tuple[float, float, float, float] | list[float],
    seed: int | None = None,
):
    """Splits the dataset by visit (i.e., samples) for conformal prediction.

    Args:
        dataset: a `SampleDataset` object
        ratios: a list/tuple of ratios for train / val / cal / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, cal_dataset, test_dataset:
            ``SampleDataset`` views (``dataset.subset``) over the same
            processed samples.

    Raises:
        TypeError: If ``ratios`` has an unsupported container or element type.
        ValueError: If its length, components, or total is invalid.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [{"patient_id": f"p{i}", "record_id": f"r{i}", "feature": i,
        ...             "label": i % 2} for i in range(8)]
        >>> dataset = create_sample_dataset(samples, {"feature": "raw"}, {"label": "raw"})
        >>> train, val, cal, test = split_by_visit_conformal(
        ...     dataset, [0.5, 0.25, 0.125, 0.125], seed=42)
        >>> [len(part) for part in (train, val, cal, test)]
        [4, 2, 1, 1]

    Note:
        The processors of ``dataset`` (vocabularies, learned statistics) were
        fitted on all of its samples, so every part shaped them. To fit them on
        the training part only, split inside ``set_task`` instead:
        ``base_dataset.set_task(task, split=...)`` with :class:`PatientSplit`
        or a custom :class:`Split`.
    """
    _validate_ratios(ratios, expected_length=4)
    rng = np.random.default_rng(seed)

    index = np.arange(len(dataset))
    rng.shuffle(index)

    # Calculate split points
    train_end = int(len(dataset) * ratios[0])
    val_end = int(len(dataset) * (ratios[0] + ratios[1]))
    cal_end = int(len(dataset) * (ratios[0] + ratios[1] + ratios[2]))

    train_index = index[:train_end]
    val_index = index[train_end:val_end]
    cal_index = index[val_end:cal_end]
    test_index = index[cal_end:]

    train_dataset = dataset.subset(train_index) # type: ignore
    val_dataset = dataset.subset(val_index) # type: ignore
    cal_dataset = dataset.subset(cal_index) # type: ignore
    test_dataset = dataset.subset(test_index) # type: ignore

    return train_dataset, val_dataset, cal_dataset, test_dataset


def split_by_patient_conformal(
    dataset: SampleDataset,
    ratios: tuple[float, float, float, float] | list[float],
    seed: int | None = None,
):
    """Splits the dataset by patient for conformal prediction.

    Args:
        dataset: a `SampleDataset` object
        ratios: a list/tuple of ratios for train / val / cal / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, cal_dataset, test_dataset:
            ``SampleDataset`` views (``dataset.subset``) over the same
            processed samples.

    Raises:
        TypeError: If ``ratios`` has an unsupported container or element type.
        ValueError: If its length, components, or total is invalid.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [{"patient_id": f"p{i}", "record_id": f"r{i}", "feature": i,
        ...             "label": i % 2} for i in range(8)]
        >>> dataset = create_sample_dataset(samples, {"feature": "raw"}, {"label": "raw"})
        >>> train, val, cal, test = split_by_patient_conformal(
        ...     dataset, [0.5, 0.25, 0.125, 0.125], seed=42)
        >>> [len(part) for part in (train, val, cal, test)]
        [4, 2, 1, 1]

    Note:
        The processors of ``dataset`` (vocabularies, learned statistics) were
        fitted on all of its samples, so every part shaped them. To fit them on
        the training part only, split inside ``set_task`` instead:
        ``base_dataset.set_task(task, split=...)`` with :class:`PatientSplit`
        or a custom :class:`Split`.
    """
    _validate_ratios(ratios, expected_length=4)
    rng = np.random.default_rng(seed)

    # Sort first: dict order follows sample order, which callers may change.
    patient_indx = sorted(dataset.patient_to_index, key=str)
    num_patients = len(patient_indx)
    rng.shuffle(patient_indx)

    # Calculate split points
    train_end = int(num_patients * ratios[0])
    val_end = int(num_patients * (ratios[0] + ratios[1]))
    cal_end = int(num_patients * (ratios[0] + ratios[1] + ratios[2]))

    train_patient_indx = patient_indx[:train_end]
    val_patient_indx = patient_indx[train_end:val_end]
    cal_patient_indx = patient_indx[val_end:cal_end]
    test_patient_indx = patient_indx[cal_end:]

    train_index = list(
        chain(*[dataset.patient_to_index[i] for i in train_patient_indx])
    )
    val_index = list(chain(*[dataset.patient_to_index[i] for i in val_patient_indx]))
    cal_index = list(chain(*[dataset.patient_to_index[i] for i in cal_patient_indx]))
    test_index = list(chain(*[dataset.patient_to_index[i] for i in test_patient_indx]))

    train_dataset = dataset.subset(train_index) # type: ignore
    val_dataset = dataset.subset(val_index) # type: ignore
    cal_dataset = dataset.subset(cal_index) # type: ignore
    test_dataset = dataset.subset(test_index) # type: ignore

    return train_dataset, val_dataset, cal_dataset, test_dataset


def split_by_predefined(
    dataset: SampleDataset,
    ratios: tuple[float, ...] | list[float],
    seed: int | None = None,
    get_index: bool = False,
    by_patient: bool = False,
    split_key: str = "split",
    train_value: str = "train",
):
    """Splits a dataset that ships with an official train/test partition.

    Every sample must carry ``split_key``. Samples whose value equals
    ``train_value`` form the pool that is shuffled and cut by ``ratios``;
    all other samples are the held-out test set. This is the one place that
    holds the logic for pre-defined splits: a dataset whose task tags each
    sample with its official partition should call this function (directly or
    through a thin wrapper) instead of adding a new splitter here.

    Args:
        dataset: a ``SampleDataset``; needs ``patient_to_index`` if
            ``by_patient=True``.
        ratios: 2 values (train / val) or 3 values (train / val / cal),
            summing to 1.0. Applied to the train pool only.
        seed: random seed for shuffling the pool. With ``by_patient=True``
            patients are sorted by ID before shuffling, so the split does not
            depend on the order of the samples.
        get_index: if True, return ``torch.Tensor`` index vectors instead of
            datasets.
        by_patient: keep all of a patient's samples in the same split. The
            first sample of each patient decides which partition the patient
            belongs to.
        split_key: sample field holding the official partition label.
        train_value: value of ``split_key`` that marks the train pool.

    Returns:
        ``(train, val, test)`` for 2 ratios, ``(train, val, cal, test)`` for 3
        ratios; ``SampleDataset`` views, or index tensors if ``get_index``.

    Examples:
        >>> from pyhealth.datasets import split_by_predefined
        >>> train, val, test = split_by_predefined(  # doctest: +SKIP
        ...     samples, [0.8, 0.2], seed=42, by_patient=True
        ... )
    """
    assert len(ratios) in (2, 3), (
        "ratios must have 2 (train/val) or 3 (train/val/cal) elements"
    )
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"

    pool: dict = {}  # group key (patient ID or sample index) -> sample indices
    holdout: list[int] = []

    if by_patient:
        # A patient belongs to one partition, so the first sample decides.
        for pid, indices in dataset.patient_to_index.items():
            first = dataset[indices[0]]
            assert split_key in first, (
                f"Patient {pid}: sample missing '{split_key}' field."
            )
            if first[split_key] == train_value:
                pool[pid] = list(indices)
            else:
                holdout.extend(indices)
        # Sort first: dict order follows sample order, which callers may change.
        keys = sorted(pool, key=str)
    else:
        for i in range(len(dataset)):
            sample = dataset[i]  # read once per sample
            assert split_key in sample, (
                f"Sample {i} is missing the '{split_key}' field."
            )
            if sample[split_key] == train_value:
                pool[i] = [i]
            else:
                holdout.append(i)
        keys = list(pool)  # already ascending sample indices

    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n = len(keys)
    bounds = [0] + [int(n * c) for c in np.cumsum(ratios)[:-1]] + [n]
    parts = [
        np.array(list(chain(*[pool[k] for k in keys[a:b]])), dtype=int)
        for a, b in zip(bounds[:-1], bounds[1:])
    ]
    parts.append(np.array(holdout, dtype=int))

    if get_index:
        return tuple(torch.tensor(p) for p in parts)
    return tuple(dataset.subset(p) for p in parts)  # type: ignore


def split_by_patient_conformal_tuh(
    dataset: SampleDataset,
    ratios: tuple[float, float, float] | list[float],
    seed: int | None = None,
    get_index: bool = False,
):
    """Splits a TUH EEG dataset by PATIENT within the TUH train partition.

    Unlike :func:`split_by_sample_conformal_tuh`, which shuffles windows
    independently, this function ensures that **all windows from a given patient
    stay in the same split** (train, val, or cal).  This prevents within-patient
    data leakage between training and calibration — a critical requirement for
    honest evaluation of conformal prediction methods, especially NCP, which
    relies on nearest-neighbor search in embedding space.

    The TUH corpus is designed so that each patient appears exclusively in either
    the training partition (265 patients for TUSZ) or the evaluation partition (50
    patients), never both.  The test set is therefore always the TUH eval
    partition (fixed, fully held-out patients).

    This is a thin wrapper around :func:`split_by_predefined`.

    Args:
        dataset: a ``SampleDataset`` produced by ``EEGEventsTUEV`` or
            ``EEGAbnormalTUAB``.  Each sample must carry a ``"split"`` field
            (``"train"`` or ``"eval"``) and the dataset must have a
            ``patient_to_index`` mapping.
        ratios: fraction of *patients* in the TUH train partition to assign to
            train / val / cal respectively.  Must be a length-3 sequence summing
            to 1.0.
        seed: random seed used to shuffle the patient list.
        get_index: if ``True``, return four :class:`torch.Tensor` index vectors
            instead of datasets.

    Returns:
        ``(train_dataset, val_dataset, cal_dataset, test_dataset)``

    Raises:
        TypeError: If ``ratios`` has an unsupported container or element type.
        ValueError: If its length, components, or total is invalid.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [{"patient_id": f"p{i}", "record_id": f"r{i}", "feature": i,
        ...             "label": i % 2, "split": "train" if i < 6 else "eval"}
        ...            for i in range(8)]
        >>> dataset = create_sample_dataset(samples, {"feature": "raw"}, {"label": "raw"})
        >>> train, val, cal, test = split_by_patient_conformal_tuh(
        ...     dataset, [0.5, 0.25, 0.25], seed=42)
        >>> [len(part) for part in (train, val, cal, test)]
        [3, 1, 2, 2]

    Note:
        The processors of ``dataset`` (vocabularies, learned statistics) were
        fitted on all of its samples, so every part shaped them. To fit them on
        the training part only, split inside ``set_task`` instead:
        ``base_dataset.set_task(task, split=...)`` with :class:`PatientSplit`
        or a custom :class:`Split`.
    """
    _validate_ratios(ratios, expected_length=3)
    return split_by_predefined(dataset, ratios, seed, get_index, by_patient=True)


def split_by_sample_conformal_tuh(
    dataset: SampleDataset,
    ratios: tuple[float, float, float] | list[float],
    seed: int | None = None,
    get_index: bool = False,
):
    """Splits a TUH EEG dataset (TUEV/TUAB) using its pre-defined train/eval split.

    This is a thin wrapper around :func:`split_by_predefined`.

    Args:
        dataset: a ``SampleDataset`` object produced by ``EEGEventsTUEV`` or ``EEGAbnormalTUAB``
        ratios: the fraction of the train pool assigned to train / val / cal respectively
        seed: random seed for shuffling the train pool
        get_index: if True, return four ``torch.Tensor`` index vectors instead
            of datasets

    Returns:
        train_dataset, val_dataset, cal_dataset, test_dataset

    Raises:
        TypeError: If ``ratios`` has an unsupported container or element type.
        ValueError: If its length, components, or total is invalid.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [{"patient_id": f"p{i}", "record_id": f"r{i}", "feature": i,
        ...             "label": i % 2, "split": "train" if i < 6 else "eval"}
        ...            for i in range(8)]
        >>> dataset = create_sample_dataset(samples, {"feature": "raw"}, {"label": "raw"})
        >>> train, val, cal, test = split_by_sample_conformal_tuh(
        ...     dataset, [0.5, 0.25, 0.25], seed=42)
        >>> [len(part) for part in (train, val, cal, test)]
        [3, 1, 2, 2]

    Note:
        The processors of ``dataset`` (vocabularies, learned statistics) were
        fitted on all of its samples, so every part shaped them. To fit them on
        the training part only, split inside ``set_task`` instead:
        ``base_dataset.set_task(task, split=...)`` with :class:`PatientSplit`
        or a custom :class:`Split`.
    """
    _validate_ratios(ratios, expected_length=3)
    return split_by_predefined(dataset, ratios, seed, get_index, by_patient=False)


def split_by_patient_tuh(
    dataset: SampleDataset,
    ratios: tuple[float, float] | list[float],
    seed: int | None = None,
    get_index: bool = False,
):
    """Splits a TUH EEG dataset by PATIENT within the TUH train partition.

    Like :func:`split_by_patient_conformal_tuh` but returns a 3-way split
    (train / val / test) instead of 4-way (train / val / cal / test).  The
    test set is always the TUH eval partition; no calibration set is produced.

    Ensures that **all windows from a given patient stay in the same split**
    (train or val), preventing within-patient data leakage.

    This is a thin wrapper around :func:`split_by_predefined`.

    Args:
        dataset: a ``SampleDataset`` produced by ``EEGEventsTUEV`` or
            ``EEGAbnormalTUAB``.  Each sample must carry a ``"split"`` field
            (``"train"`` or ``"eval"``) and the dataset must have a
            ``patient_to_index`` mapping.
        ratios: fraction of *patients* in the TUH train partition to assign to
            train / val respectively.  Must be a length-2 sequence summing
            to 1.0.
        seed: random seed used to shuffle the patient list.
        get_index: if ``True``, return three :class:`torch.Tensor` index
            vectors instead of datasets.

    Returns:
        ``(train_dataset, val_dataset, test_dataset)``

    Raises:
        TypeError: If ``ratios`` has an unsupported container or element type.
        ValueError: If its length, components, or total is invalid.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [{"patient_id": f"p{i}", "record_id": f"r{i}", "feature": i,
        ...             "label": i % 2, "split": "train" if i < 6 else "eval"}
        ...            for i in range(8)]
        >>> dataset = create_sample_dataset(samples, {"feature": "raw"}, {"label": "raw"})
        >>> train, val, test = split_by_patient_tuh(dataset, [0.5, 0.5], seed=42)
        >>> [len(part) for part in (train, val, test)]
        [3, 3, 2]

    Note:
        The processors of ``dataset`` (vocabularies, learned statistics) were
        fitted on all of its samples, so every part shaped them. To fit them on
        the training part only, split inside ``set_task`` instead:
        ``base_dataset.set_task(task, split=...)`` with :class:`PatientSplit`
        or a custom :class:`Split`.
    """
    _validate_ratios(ratios, expected_length=2)
    return split_by_predefined(dataset, ratios, seed, get_index, by_patient=True)


def split_by_sample_tuh(
    dataset: SampleDataset,
    ratios: tuple[float, float] | list[float],
    seed: int | None = None,
    get_index: bool = False,
):
    """Splits a TUH EEG dataset (TUEV/TUAB) using its pre-defined split.

    Like :func:`split_by_sample_conformal_tuh` but returns a 3-way split
    (train / val / test) instead of 4-way (train / val / cal / test).  The
    test set is always the TUH eval partition; no calibration set is produced.

    This is a thin wrapper around :func:`split_by_predefined`.

    Args:
        dataset: a ``SampleDataset`` object produced by ``EEGEventsTUEV`` or
            ``EEGAbnormalTUAB``
        ratios: the fraction of the train pool assigned to train / val
            respectively.  Must be a length-2 sequence summing to 1.0.
        seed: random seed for shuffling the train pool
        get_index: if True, return three ``torch.Tensor`` index vectors instead
            of datasets

    Returns:
        train_dataset, val_dataset, test_dataset

    Raises:
        TypeError: If ``ratios`` has an unsupported container or element type.
        ValueError: If its length, components, or total is invalid.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [{"patient_id": f"p{i}", "record_id": f"r{i}", "feature": i,
        ...             "label": i % 2, "split": "train" if i < 6 else "eval"}
        ...            for i in range(8)]
        >>> dataset = create_sample_dataset(samples, {"feature": "raw"}, {"label": "raw"})
        >>> train, val, test = split_by_sample_tuh(dataset, [0.5, 0.5], seed=42)
        >>> [len(part) for part in (train, val, test)]
        [3, 3, 2]

    Note:
        The processors of ``dataset`` (vocabularies, learned statistics) were
        fitted on all of its samples, so every part shaped them. To fit them on
        the training part only, split inside ``set_task`` instead:
        ``base_dataset.set_task(task, split=...)`` with :class:`PatientSplit`
        or a custom :class:`Split`.
    """
    _validate_ratios(ratios, expected_length=2)
    return split_by_predefined(dataset, ratios, seed, get_index, by_patient=False)


def split_by_sample_conformal(
    dataset: SampleDataset,
    ratios: tuple[float, float, float, float] | list[float],
    seed: int | None = None,
    get_index: bool = False,
):
    """Splits the dataset by sample for conformal prediction.

    Args:
        dataset: a `SampleDataset` object
        ratios: a list/tuple of ratios for train / val / cal / test
        seed: random seed for shuffling the dataset
        get_index: if True, return indices instead of datasets

    Returns:
        train_dataset, val_dataset, cal_dataset, test_dataset:
            ``SampleDataset`` views (``dataset.subset``) over the same
            processed samples, or four index tensors if ``get_index`` is True.

    Raises:
        TypeError: If ``ratios`` has an unsupported container or element type.
        ValueError: If its length, components, or total is invalid.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [{"patient_id": f"p{i}", "record_id": f"r{i}", "feature": i,
        ...             "label": i % 2} for i in range(8)]
        >>> dataset = create_sample_dataset(samples, {"feature": "raw"}, {"label": "raw"})
        >>> train, val, cal, test = split_by_sample_conformal(
        ...     dataset, [0.5, 0.25, 0.125, 0.125], seed=42)
        >>> [len(part) for part in (train, val, cal, test)]
        [4, 2, 1, 1]

    Note:
        The processors of ``dataset`` (vocabularies, learned statistics) were
        fitted on all of its samples, so every part shaped them. To fit them on
        the training part only, split inside ``set_task`` instead:
        ``base_dataset.set_task(task, split=...)`` with :class:`PatientSplit`
        or a custom :class:`Split`.
    """
    _validate_ratios(ratios, expected_length=4)
    rng = np.random.default_rng(seed)

    index = np.arange(len(dataset))
    rng.shuffle(index)

    # Calculate split points
    train_end = int(len(dataset) * ratios[0])
    val_end = int(len(dataset) * (ratios[0] + ratios[1]))
    cal_end = int(len(dataset) * (ratios[0] + ratios[1] + ratios[2]))

    train_index = index[:train_end]
    val_index = index[train_end:val_end]
    cal_index = index[val_end:cal_end]
    test_index = index[cal_end:]

    if get_index:
        return (
            torch.tensor(train_index),
            torch.tensor(val_index),
            torch.tensor(cal_index),
            torch.tensor(test_index),
        )
    else:
        train_dataset = dataset.subset(train_index) # type: ignore
        val_dataset = dataset.subset(val_index) # type: ignore
        cal_dataset = dataset.subset(cal_index) # type: ignore
        test_dataset = dataset.subset(test_index) # type: ignore
        return train_dataset, val_dataset, cal_dataset, test_dataset
