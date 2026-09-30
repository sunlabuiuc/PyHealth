import math
from itertools import chain
from numbers import Real
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from .sample_dataset import SampleDataset

# TODO: train_dataset.dataset still access the whole dataset which may leak information
# TODO: add more splitting methods


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
    seed: Optional[int] = None,
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

    pos_indices: List[int] = []
    neg_indices: List[int] = []

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
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by visit (i.e., samples).

    Args:
        dataset: a `SampleDataset` object
        ratios: three finite proportions for train / val / test. The values
            must be in [0, 1] and sum to 1 within 1e-6.
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, test_dataset: three ``SampleDataset`` subsets.

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
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by patient.

    Args:
        dataset: a `SampleDataset` object
        ratios: three finite proportions for train / val / test. The values
            must be in [0, 1] and sum to 1 within 1e-6.
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, test_dataset: three ``SampleDataset`` subsets.

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
    """
    _validate_ratios(ratios, expected_length=3)
    rng = np.random.default_rng(seed)
    patient_indx = list(dataset.patient_to_index.keys())
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
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
    get_index: Optional[bool] = False,
):
    """Splits the dataset by sample

    Args:
        dataset: a `SampleDataset` object
        ratios: three finite proportions for train / val / test. The values
            must be in [0, 1] and sum to 1 within 1e-6.
        seed: random seed for shuffling the dataset
        get_index: if True, return three index tensors instead of subsets.

    Returns:
        train_dataset, val_dataset, test_dataset: three ``SampleDataset``
            subsets, or three index tensors if ``get_index=True``.

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
    ratios: Union[Tuple[float, float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by visit (i.e., samples) for conformal prediction.

    Args:
        dataset: a `SampleDataset` object
        ratios: four finite proportions for train / val / cal / test. The
            values must be in [0, 1] and sum to 1 within 1e-6.
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, cal_dataset, test_dataset: four
            ``SampleDataset`` subsets.

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
    ratios: Union[Tuple[float, float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by patient for conformal prediction.

    Args:
        dataset: a `SampleDataset` object
        ratios: four finite proportions for train / val / cal / test. The
            values must be in [0, 1] and sum to 1 within 1e-6.
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, cal_dataset, test_dataset: four
            ``SampleDataset`` subsets.

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
    """
    _validate_ratios(ratios, expected_length=4)
    rng = np.random.default_rng(seed)

    patient_indx = list(dataset.patient_to_index.keys())
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


def split_by_patient_conformal_tuh(
    dataset: SampleDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
    get_index: Optional[bool] = False,
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

    Args:
        dataset: a ``SampleDataset`` produced by ``EEGEventsTUEV`` or
            ``EEGAbnormalTUAB``.  Each sample must carry a ``"split"`` field
            (``"train"`` or ``"eval"``) and the dataset must have a
            ``patient_to_index`` mapping.
        ratios: fraction of *patients* in the TUH train partition to assign to
            train / val / cal respectively. Three finite values in [0, 1]
            must sum to 1 within 1e-6.
        seed: random seed used to shuffle the patient list.
        get_index: if ``True``, return four :class:`torch.Tensor` index vectors
            instead of ``SampleDataset`` subsets.

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
    """
    _validate_ratios(ratios, expected_length=3)

    # Bucket patients by partition using dataset.patient_to_index (fast path).
    # TUH guarantees each patient is in exactly one partition, so inspecting the
    # first sample per patient is sufficient.
    train_patient_to_indices: dict = {}
    test_list: List[int] = []

    for pid, indices in dataset.patient_to_index.items():
        first_sample = dataset[indices[0]]
        assert "split" in first_sample, (
            f"Patient {pid}: sample missing 'split' field. "
            "Use EEGEventsTUEV or EEGAbnormalTUAB to build the dataset."
        )
        if first_sample["split"] == "train":
            train_patient_to_indices[pid] = list(indices)
        else:
            test_list.extend(list(indices))

    # Shuffle patients deterministically
    patient_ids = list(train_patient_to_indices.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))

    train_pids = patient_ids[:train_end]
    val_pids = patient_ids[train_end:val_end]
    cal_pids = patient_ids[val_end:]

    train_index = np.array(list(chain(*[train_patient_to_indices[p] for p in train_pids])))
    val_index = np.array(list(chain(*[train_patient_to_indices[p] for p in val_pids])))
    cal_index = np.array(list(chain(*[train_patient_to_indices[p] for p in cal_pids])))
    test_index = np.array(test_list)

    if get_index:
        return (
            torch.tensor(train_index),
            torch.tensor(val_index),
            torch.tensor(cal_index),
            torch.tensor(test_index),
        )
    else:
        return (
            dataset.subset(train_index),  # type: ignore
            dataset.subset(val_index),    # type: ignore
            dataset.subset(cal_index),    # type: ignore
            dataset.subset(test_index),   # type: ignore
        )


def split_by_sample_conformal_tuh(
    dataset: SampleDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
    get_index: Optional[bool] = False,
):
    """Splits a TUH EEG dataset (TUEV/TUAB) using its pre-defined train/eval split.

    Args:
        dataset: a ``SampleDataset`` object produced by ``EEGEventsTUEV`` or ``EEGAbnormalTUAB``
        ratios: three finite proportions in [0, 1] for train / val / cal
            within the TUH training pool, summing to 1 within 1e-6.
        seed: random seed for shuffling the train pool
        get_index: if True, return four ``torch.Tensor`` index vectors instead
            of ``SampleDataset`` subsets

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
    """
    _validate_ratios(ratios, expected_length=3)

    # verify every sample has the required "split" field
    for i in range(len(dataset)):
        assert "split" in dataset[i], (
            f"Sample {i} is missing the 'split' field. "
            "Make sure you used EEGEventsTUEV or EEGAbnormalTUAB to build the dataset."
        )

    train_pool: List[int] = []
    test_list: List[int] = []
    for i in range(len(dataset)):
        if dataset[i]["split"] == "train":
            train_pool.append(i)
        else:
            test_list.append(i)

    # shuffle only the train pool
    rng = np.random.default_rng(seed)
    train_arr = np.array(train_pool)
    rng.shuffle(train_arr)

    # Slice into train / val / cal.
    n = len(train_arr)
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))

    train_index = train_arr[:train_end]
    val_index = train_arr[train_end:val_end]
    cal_index = train_arr[val_end:]
    test_index = np.array(test_list)

    if get_index:
        return (
            torch.tensor(train_index),
            torch.tensor(val_index),
            torch.tensor(cal_index),
            torch.tensor(test_index),
        )
    else:
        return (
            dataset.subset(train_index),  # type: ignore
            dataset.subset(val_index),    # type: ignore
            dataset.subset(cal_index),    # type: ignore
            dataset.subset(test_index),   # type: ignore
        )


def split_by_patient_tuh(
    dataset: SampleDataset,
    ratios: Union[Tuple[float, float], List[float]],
    seed: Optional[int] = None,
    get_index: Optional[bool] = False,
):
    """Splits a TUH EEG dataset by PATIENT within the TUH train partition.

    Like :func:`split_by_patient_conformal_tuh` but returns a 3-way split
    (train / val / test) instead of 4-way (train / val / cal / test).  The
    test set is always the TUH eval partition; no calibration set is produced.

    Ensures that **all windows from a given patient stay in the same split**
    (train or val), preventing within-patient data leakage.

    Args:
        dataset: a ``SampleDataset`` produced by ``EEGEventsTUEV`` or
            ``EEGAbnormalTUAB``.  Each sample must carry a ``"split"`` field
            (``"train"`` or ``"eval"``) and the dataset must have a
            ``patient_to_index`` mapping.
        ratios: fraction of *patients* in the TUH train partition to assign to
            train / val respectively. Two finite values in [0, 1] must sum
            to 1 within 1e-6.
        seed: random seed used to shuffle the patient list.
        get_index: if ``True``, return three :class:`torch.Tensor` index
            vectors instead of ``SampleDataset`` subsets.

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
    """
    _validate_ratios(ratios, expected_length=2)

    train_patient_to_indices: dict = {}
    test_list: List[int] = []

    for pid, indices in dataset.patient_to_index.items():
        first_sample = dataset[indices[0]]
        assert "split" in first_sample, (
            f"Patient {pid}: sample missing 'split' field. "
            "Use EEGEventsTUEV or EEGAbnormalTUAB to build the dataset."
        )
        if first_sample["split"] == "train":
            train_patient_to_indices[pid] = list(indices)
        else:
            test_list.extend(list(indices))

    patient_ids = list(train_patient_to_indices.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    train_end = int(n * ratios[0])

    train_pids = patient_ids[:train_end]
    val_pids = patient_ids[train_end:]

    train_index = np.array(
        list(chain(*[train_patient_to_indices[p] for p in train_pids]))
    )
    val_index = np.array(
        list(chain(*[train_patient_to_indices[p] for p in val_pids]))
    )
    test_index = np.array(test_list)

    if get_index:
        return (
            torch.tensor(train_index),
            torch.tensor(val_index),
            torch.tensor(test_index),
        )
    else:
        return (
            dataset.subset(train_index),  # type: ignore
            dataset.subset(val_index),    # type: ignore
            dataset.subset(test_index),   # type: ignore
        )


def split_by_sample_tuh(
    dataset: SampleDataset,
    ratios: Union[Tuple[float, float], List[float]],
    seed: Optional[int] = None,
    get_index: Optional[bool] = False,
):
    """Splits a TUH EEG dataset (TUEV/TUAB) using its pre-defined split.

    Like :func:`split_by_sample_conformal_tuh` but returns a 3-way split
    (train / val / test) instead of 4-way (train / val / cal / test).  The
    test set is always the TUH eval partition; no calibration set is produced.

    Args:
        dataset: a ``SampleDataset`` object produced by ``EEGEventsTUEV`` or
            ``EEGAbnormalTUAB``
        ratios: two finite proportions in [0, 1] for train / val within
            the TUH training pool, summing to 1 within 1e-6.
        seed: random seed for shuffling the train pool
        get_index: if True, return three ``torch.Tensor`` index vectors instead
            of ``SampleDataset`` subsets

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
    """
    _validate_ratios(ratios, expected_length=2)

    for i in range(len(dataset)):
        assert "split" in dataset[i], (
            f"Sample {i} is missing the 'split' field. "
            "Use EEGEventsTUEV or EEGAbnormalTUAB to build the dataset."
        )

    train_pool: List[int] = []
    test_list: List[int] = []
    for i in range(len(dataset)):
        if dataset[i]["split"] == "train":
            train_pool.append(i)
        else:
            test_list.append(i)

    rng = np.random.default_rng(seed)
    train_arr = np.array(train_pool)
    rng.shuffle(train_arr)

    n = len(train_arr)
    train_end = int(n * ratios[0])

    train_index = train_arr[:train_end]
    val_index = train_arr[train_end:]
    test_index = np.array(test_list)

    if get_index:
        return (
            torch.tensor(train_index),
            torch.tensor(val_index),
            torch.tensor(test_index),
        )
    else:
        return (
            dataset.subset(train_index),  # type: ignore
            dataset.subset(val_index),    # type: ignore
            dataset.subset(test_index),   # type: ignore
        )


def split_by_sample_conformal(
    dataset: SampleDataset,
    ratios: Union[Tuple[float, float, float, float], List[float]],
    seed: Optional[int] = None,
    get_index: Optional[bool] = False,
):
    """Splits the dataset by sample for conformal prediction.

    Args:
        dataset: a `SampleDataset` object
        ratios: four finite proportions for train / val / cal / test. The
            values must be in [0, 1] and sum to 1 within 1e-6.
        seed: random seed for shuffling the dataset
        get_index: if True, return indices instead of Subset objects

    Returns:
        train_dataset, val_dataset, cal_dataset, test_dataset: four
            ``SampleDataset`` subsets, or four index tensors if
            ``get_index=True``.

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
