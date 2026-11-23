"""Ordered preprocessed PTB-XL data and reproducible paired views for H1.

Contributor: Chia Yang Tsai (UIUC NetID: ct68).
Category-3 order comes from bbj-lab/protoecgnet at author commit
fefbe6166a87a961c9c65d220d6344e8d53f2c49. No ECG preprocessing, statistics
fitting, checkpoint conversion or research evaluation is performed here.
"""

import math
import operator
from collections.abc import Iterator, Sequence
from numbers import Real
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..datasets import SampleDataset, create_sample_dataset, get_dataloader

PTBXL_MORPHOLOGY_LABEL_ORDER: tuple[str, ...] = (
    "NDT",
    "NST_",
    "LNGQT",
    "IMI",
    "ASMI",
    "LVH",
    "LAFB",
    "ISC_",
    "IRBBB",
    "ISCAL",
    "CRBBB",
    "CLBBB",
    "ILMI",
    "LAO/LAE",
    "AMI",
    "ALMI",
    "ISCIN",
    "INJAS",
    "LMI",
    "ISCIL",
    "LPFB",
    "ISCAS",
    "INJAL",
    "ISCLA",
    "RVH",
    "ANEUR",
    "RAO/RAE",
    "WPW",
    "ILBBB",
    "IPLMI",
    "ISCAN",
    "IPMI",
    "SEHYP",
    "INJIN",
    "INJLA",
    "PMI",
    "INJIL",
    "ABQRS",
    "PVC",
    "STD_",
    "VCLVH",
    "QWAVE",
    "LOWT",
    "NT_",
    "PAC",
    "LPR",
    "INVT",
    "LVOLT",
    "HVOLT",
    "TAB_",
    "STE_",
    "PRC(S)",
)


def _integer(value: Any, name: str) -> int:
    """Reject silent truncation, including fractional ECG IDs and seeds."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a nonnegative integer.")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a nonnegative integer.") from error
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return int(result)


def _ptbxl_id(value: Any, name: str) -> int:
    """Canonicalize integral PTB-XL CSV IDs without truncating fractions."""
    if isinstance(value, str):
        if value.isdecimal():
            return int(value)
        try:
            value = float(value)
        except ValueError as error:
            raise ValueError(
                f"{name} must be an integral numeric PTB-XL ID."
            ) from error
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        if not math.isfinite(value) or value < 0 or int(value) != value:
            raise ValueError(f"{name} must be an integral numeric PTB-XL ID.")
        return int(value)
    return _integer(value, name)


def make_ecg_noisy_view(
    clean: Tensor,
    ecg_ids: Sequence[int],
    *,
    stream_seed: int,
    epoch_or_draw: int,
    alpha: float = 0.01,
) -> Tensor:
    """Generate the locked CPU additive-noise rule after ECG preprocessing.

    Matches the research function's float32 centering, float64 AC-RMS
    reduction, float32 scale/noise and NumPy SeedSequence ordering. The seed
    sequence is [stream_seed, ECG ID, epoch_or_draw]; batch order and batch
    size do not affect a record's perturbation. Zero AC RMS gives zero noise.
    This does not change global NumPy/PyTorch RNG states or mutate clean.

    Args:
        clean: Finite CPU float32 ECGs, (N, leads, time) or (N, 1, leads, time).
        ecg_ids: Nonnegative integral ECG IDs, one per row.
        stream_seed: Explicit stream identifier, shared by training arms.
            Evaluation must use the separately locked evaluation stream.
        epoch_or_draw: Training epoch or evaluation draw ID.
        alpha: Nonnegative finite noise scale; 0.01 in the fixed research setup.

    Returns:
        A new CPU float32 tensor in the original shape. As in the research
        data function, clean is detached for NumPy noise generation. H1 model
        gradients still propagate through both clean/noisy model branches.

    Examples:
        >>> clean = torch.zeros(1, 12, 1000)
        >>> noisy = make_ecg_noisy_view(
        ...     clean, [1], stream_seed=2026111801, epoch_or_draw=1)
        >>> torch.equal(noisy, clean)
        True
    """
    if not isinstance(clean, Tensor) or clean.dtype != torch.float32:
        raise TypeError("clean must be a CPU float32 tensor.")
    if clean.device.type != "cpu":
        raise ValueError("Generate ECG noise on CPU before model device transfer.")
    if clean.ndim not in (3, 4) or min(clean.shape) == 0:
        raise ValueError(
            "clean must be nonempty (N, leads, time) or (N, 1, leads, time)."
        )
    if clean.ndim == 4 and clean.shape[1] != 1:
        raise ValueError(
            "Four-dimensional ECG input requires a singleton image channel."
        )
    if not bool(torch.isfinite(clean).all()):
        raise ValueError("clean must contain finite values.")
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a nonnegative real number.")
    if not math.isfinite(alpha) or not 0 <= alpha <= float(np.finfo(np.float32).max):
        raise ValueError(
            "alpha must be finite, nonnegative and representable in float32."
        )
    stream_seed = _integer(stream_seed, "stream_seed")
    epoch_or_draw = _integer(epoch_or_draw, "epoch_or_draw")
    if len(ecg_ids) != len(clean):
        raise ValueError("Exactly one ECG ID is required per clean row.")
    ids = [_integer(value, "ecg_id") for value in ecg_ids]
    x = clean.detach().numpy()
    centered = x - x.mean(axis=-1, keepdims=True)
    scale = np.sqrt(
        np.mean(centered.astype(np.float64) ** 2, axis=-1, keepdims=True)
    ).astype(np.float32)
    unit_noise = np.stack(
        [
            np.random.default_rng(
                np.random.SeedSequence(
                    [
                        stream_seed,
                        ecg_id,
                        epoch_or_draw,
                    ]
                )
            )
            .standard_normal(x.shape[1:])
            .astype(np.float32)
            for ecg_id in ids
        ]
    )
    noisy = x + np.float32(alpha) * scale * unit_noise
    if not np.isfinite(noisy).all():
        raise ValueError("Noise generation produced nonfinite values.")
    return torch.from_numpy(noisy)


def create_h1_ptbxl_dataset(
    signals: Tensor,
    targets: Tensor,
    *,
    ecg_ids: Sequence[int | str],
    patient_ids: Sequence[int | str],
    folds: Sequence[int],
    split: str,
    label_names: Sequence[str] = PTBXL_MORPHOLOGY_LABEL_ORDER,
    in_memory: bool = True,
) -> SampleDataset:
    """Adapt an already-preprocessed PTB-XL split to the native SampleDataset.

    Does not filter/standardize ECGs, drop all-zero targets, infer label order,
    refit training statistics, or choose splits. Explicit folds must agree
    with the declared split. Targets are interpreted in the supplied order.

    Args:
        signals: Finite CPU float32 tensors (N, 12, 1000) or (N, 1, 12, 1000).
        targets: Binary multi-hot CPU tensor (N, len(label_names)).
        ecg_ids: Unique integral numeric PTB-XL ECG IDs.
        patient_ids: Integral numeric patient IDs; repeated patients are allowed.
        folds: Official fold numbers for these records.
        split: train (folds 1-8), validation (9), or test (10).
        label_names: Explicit unique label order; defaults to author Category 3.
            Smaller synthetic label sets are allowed for tests/examples.
        in_memory: Use native in-memory samples; False uses a temporary native
            disk-backed SampleDataset. No Drive paths are read or written.

    Returns:
        SampleDataset with one signal feature, a fixed multilabel processor,
        and raw ecg_id/record_id/patient_id/strat_fold metadata on every row.

    Examples:
        >>> dataset = create_h1_ptbxl_dataset(  # doctest: +SKIP
        ...     signals, targets, ecg_ids=ecg_ids, patient_ids=patient_ids,
        ...     folds=folds, split="train")
    """
    allowed = {"train": set(range(1, 9)), "validation": {9}, "test": {10}}
    if split not in allowed:
        raise ValueError("split must be train, validation or test.")
    if not isinstance(signals, Tensor) or signals.dtype != torch.float32:
        raise TypeError("signals must be preprocessed CPU float32 tensors.")
    if signals.device.type != "cpu":
        raise ValueError("Dataset signals must be on CPU.")
    if signals.ndim == 4 and signals.shape[1] == 1:
        signals = signals.squeeze(1)
    if signals.ndim != 3 or signals.shape[1:] != (12, 1000) or len(signals) == 0:
        raise ValueError("PTB-XL signals must have shape (nonempty N, 12, 1000).")
    if not bool(torch.isfinite(signals).all()):
        raise ValueError("signals must be finite.")
    labels = list(label_names)
    if not labels or any(not isinstance(name, str) or not name for name in labels):
        raise ValueError("label_names must contain nonempty strings.")
    if len(set(labels)) != len(labels):
        raise ValueError("label_names must be unique and ordered explicitly.")
    if not isinstance(targets, Tensor) or targets.device.type != "cpu":
        raise TypeError("targets must be a CPU tensor.")
    if targets.shape != (len(signals), len(labels)) or not bool(
        ((targets == 0) | (targets == 1)).all()
    ):
        raise ValueError("targets must be binary (N, len(label_names)).")
    if any(len(values) != len(signals) for values in (ecg_ids, patient_ids, folds)):
        raise ValueError("IDs, patients and folds must each have N entries.")
    ids = [_ptbxl_id(value, "ecg_id") for value in ecg_ids]
    patients = [_ptbxl_id(value, "patient_id") for value in patient_ids]
    fold_numbers = [_integer(value, "fold") for value in folds]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ECG IDs are not allowed within a split.")
    if not set(fold_numbers) <= allowed[split]:
        raise ValueError("Record folds do not agree with the declared split.")
    samples = [
        {
            "patient_id": str(patients[row]),
            "record_id": str(ids[row]),
            "ecg_id": ids[row],
            "strat_fold": fold_numbers[row],
            "signal": signals[row].detach().clone(),
            "labels": [
                labels[index] for index in targets[row].nonzero().flatten().tolist()
            ],
        }
        for row in range(len(signals))
    ]
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"labels": ("multilabel", {"labels": labels})},
        dataset_name="PTB-XL preprocessed H1 split",
        in_memory=in_memory,
    )
    dataset.h1_split = split
    return dataset


class PairedECGLoader:
    """Wrap a native training loader with reproducible noisy views per full pass.

    One complete loader pass defines one noise epoch. This matches Trainer
    when steps_per_epoch is omitted or equals len(loader): its next exhausted
    iterator advances the epoch without resetting the optimizer. Partial passes,
    overlapping iterators and mid-epoch resume are not supported here. Validation
    should use a separate clean get_dataloader, never this training wrapper.

    Args:
        dataset: Split built by create_h1_ptbxl_dataset with split='train'.
        batch_size: Number of ECG pairs per batch.
        stream_seed: Fixed training noise stream, shared by both arms.
        initial_epoch: First noise epoch, normally 1; explicit for full-pass resume.
        alpha: Noise strength, 0.01 in the fixed research setup.
        shuffle: Shuffle samples using the native PyHealth loader.
        seed: Native sample-order seed; distinct from the noise stream seed.
        **loader_kwargs: Options forwarded to get_dataloader, e.g. num_workers.

    Examples:
        >>> pairs = PairedECGLoader(  # doctest: +SKIP
        ...     dataset, batch_size=32, stream_seed=2026111801)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        batch_size: int,
        *,
        stream_seed: int,
        initial_epoch: int = 1,
        alpha: float = 0.01,
        shuffle: bool = True,
        seed: int = 42,
        **loader_kwargs: Any,
    ) -> None:
        if getattr(dataset, "h1_split", None) != "train":
            raise ValueError(
                "PairedECGLoader requires an explicitly declared train split."
            )
        self.stream_seed = _integer(stream_seed, "stream_seed")
        self.next_epoch = _integer(initial_epoch, "initial_epoch")
        # Validate scalar settings before iteration or starting a training loop.
        make_ecg_noisy_view(
            torch.zeros(1, 1, 1),
            [0],
            stream_seed=self.stream_seed,
            epoch_or_draw=self.next_epoch,
            alpha=alpha,
        )
        self.alpha = float(alpha)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_samples = shuffle
        self._loader = get_dataloader(
            dataset, batch_size=batch_size, shuffle=shuffle, seed=seed, **loader_kwargs
        )
        self.sampler = getattr(self._loader, "sampler", None)

    def __len__(self) -> int:
        """Return the number of batches in one full pass."""
        return len(self._loader)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Advance the noise epoch for the next complete native loader pass."""
        epoch = self.next_epoch
        self.next_epoch += 1
        return self._batches(iter(self._loader), epoch)

    def _batches(
        self,
        iterator: Iterator[dict[str, Any]],
        epoch: int,
    ) -> Iterator[dict[str, Any]]:
        for batch in iterator:
            if not all(
                _integer(value, "fold") in range(1, 9) for value in batch["strat_fold"]
            ):
                raise ValueError("Paired training must contain only folds 1-8.")
            if "noisy_signal" in batch:
                raise ValueError(
                    "Batch already contains noisy_signal; do not overwrite it."
                )
            result = dict(batch)
            result["noisy_signal"] = make_ecg_noisy_view(
                batch["signal"],
                batch["ecg_id"],
                stream_seed=self.stream_seed,
                epoch_or_draw=epoch,
                alpha=self.alpha,
            )
            yield result
