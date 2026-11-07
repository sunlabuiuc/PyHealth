"""Persist output arrays and serve saved base-model outputs through the model API."""

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from .collection import _row_count, collect_predictions

_ROW = "__cpbench_row__"
_ROLE = "__cpbench_role__"
_SAMPLE_HASH = "__cpbench_sample_hash__"

# --- Artifact files ---


def save_predictions(predictions: Mapping[str, np.ndarray], path: Path) -> None:
    """Atomically write named arrays to an explicit NPZ path, replacing that file.

    No Result or Evaluation is read or returned. The caller owns cache naming,
    cache reuse and result construction. Parent directories are created as needed.

    Examples:
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "predictions.npz"
        ...     save_predictions({"y_true": np.array([0, 1])}, path)
        ...     with np.load(path) as saved:
        ...         print(saved["y_true"].tolist())
        [0, 1]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=path.parent, prefix=f".{path.name}.") as temporary:
        pending = Path(temporary) / "predictions.npz"
        np.savez_compressed(pending, **predictions)
        os.replace(pending, path)


def save_model_predictions(
    predictions, path: Path, *, mode, label_key, class_labels=None
) -> None:
    """Save base-model outputs as split-to-array mappings, not conformal sets.

    Each split includes encoded y_true and any outputs needed by its CP methods
    (e.g. y_prob, logit, embed). class_labels lists classes in encoded column order.
    Object arrays are rejected; reading the artifact never enables pickle.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> values = {"test": {"y_true": np.array([0]),
        ...                    "y_prob": np.array([[0.8, 0.2]])}}
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "model.npz"
        ...     save_model_predictions(values, path, mode="multiclass",
        ...                            label_key="label", class_labels=["a", "b"])
        ...     loaded, metadata = load_model_predictions(path)
        ...     print(metadata["class_labels"], loaded["test"]["y_true"].tolist())
        ['a', 'b'] [0]
    """
    metadata = {
        "mode": mode,
        "label_key": label_key,
        "class_labels": None
        if class_labels is None
        else [
            item.item() if isinstance(item, np.generic) else item
            for item in class_labels
        ],
    }
    _validate_predictions(predictions, metadata)
    arrays = {"__metadata__": np.asarray(json.dumps(metadata, allow_nan=False))}
    for role, outputs in predictions.items():
        arrays.update({f"{role}/{name}": values for name, values in outputs.items()})

    save_predictions(arrays, path)


def load_model_predictions(path: Path):
    """Load base-model outputs and their label metadata from one NPZ artifact.

    Missing files, invalid arrays and final-CP prediction files are rejected.
    The returned arrays own their data after the archive is closed.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "model.npz"
        ...     save_model_predictions(
        ...         {"test": {"y_true": np.array([0]),
        ...                   "logit": np.array([[2.0, 1.0]])}}, path,
        ...         mode="multiclass", label_key="label", class_labels=[0, 1],
        ...     )
        ...     predictions, metadata = load_model_predictions(path)
        ...     print(predictions["test"]["logit"].shape)
        (1, 2)
    """
    predictions = {}
    with np.load(path, allow_pickle=False) as archive:
        if "__metadata__" not in archive.files:
            raise ValueError("expected a base-model prediction artifact with metadata")

        metadata = json.loads(str(archive["__metadata__"].item()))
        for key in archive.files:
            if key == "__metadata__":
                continue
            if key.count("/") != 1:
                raise ValueError(f"invalid model prediction field: {key!r}")

            role, name = key.split("/")
            predictions.setdefault(role, {})[name] = archive[key]

    _validate_predictions(predictions, metadata)

    return predictions, metadata


def _validate_predictions(predictions, metadata):
    if not isinstance(metadata, dict):
        raise TypeError("model prediction metadata must be a mapping")
    if not isinstance(metadata.get("mode"), str) or not metadata["mode"]:
        raise ValueError("model prediction metadata must name the model mode")
    label_key = metadata.get("label_key")
    if (
        not isinstance(label_key, str)
        or not label_key
        or label_key in {_ROW, _ROLE, _SAMPLE_HASH}
    ):
        raise ValueError("model prediction metadata must name a valid label key")
    labels = metadata.get("class_labels")
    if labels is not None and (
        not isinstance(labels, list) or not labels or len(set(labels)) != len(labels)
    ):
        raise ValueError("class_labels must contain unique labels in column order")
    if metadata["mode"] in {"multiclass", "multilabel"} and labels is None:
        raise ValueError("classification model predictions require class_labels")
    if not predictions:
        raise ValueError("model predictions must contain at least one split")

    for role, outputs in predictions.items():
        if not isinstance(role, str) or not role or "/" in role:
            raise ValueError(
                "model prediction split names must be nonempty without '/'"
            )
        if "y_true" not in outputs:
            raise ValueError(f"model predictions for {role!r} require y_true")
        _row_count(outputs)
        truth = np.asarray(outputs["y_true"])
        if metadata["mode"] == "multiclass" and (
            truth.ndim != 1
            or not np.issubdtype(truth.dtype, np.integer)
            or np.any((truth < 0) | (truth >= len(labels)))
        ):
            raise ValueError(f"{role}.y_true must contain encoded class indices")
        if metadata["mode"] == "multilabel" and (
            truth.shape != (len(truth), len(labels)) or not np.isin(truth, [0, 1]).all()
        ):
            raise ValueError(f"{role}.y_true must contain binary class masks")
        for name, values in outputs.items():
            if not isinstance(name, str) or not name or "/" in name:
                raise ValueError("model prediction field names must not contain '/'")
            values = np.asarray(values)
            if not np.issubdtype(values.dtype, np.number) and values.dtype != np.bool_:
                raise ValueError(f"{role}.{name} must contain numeric encoded outputs")
            if not np.isfinite(values).all():
                raise ValueError(f"{role}.{name} contains non-finite model outputs")
            if (
                metadata["mode"] in {"multiclass", "multilabel"}
                and name in {"y_prob", "logit"}
                and values.shape != (len(outputs["y_true"]), len(labels))
            ):
                raise ValueError(f"{role}.{name} does not match the class columns")
            if name == "y_prob" and metadata["mode"] in {"multiclass", "multilabel"}:
                if np.any((values < 0) | (values > 1)):
                    raise ValueError(
                        f"{role}.y_prob must contain probabilities in [0, 1]"
                    )
                if metadata["mode"] == "multiclass" and not np.allclose(
                    values.sum(axis=1), 1, atol=1e-5
                ):
                    raise ValueError(f"{role}.y_prob rows must sum to one")


def require_model_outputs(predictions, required, outputs):
    """Check split and field requirements before a CP method starts calibration.

    Examples:
        >>> require_model_outputs(
        ...     {"test": {"y_true": np.array([0]), "y_prob": np.ones((1, 1))}},
        ...     ("test",), {"test": ("y_prob",)},
        ... )
    """
    missing = set(required) - predictions.keys()
    if missing:
        raise ValueError(f"model predictions are missing splits: {sorted(missing)}")
    for role in required:
        if not len(predictions[role]["y_true"]):
            raise ValueError(f"model prediction split {role!r} must be nonempty")
    for role, fields in (outputs or {}).items():
        missing = set(fields) - predictions.get(role, {}).keys()
        if missing:
            raise ValueError(
                f"model predictions are missing {role} outputs: {sorted(missing)}"
            )


# --- Base-model inference ---


def collect_model_predictions(model, splits, outputs, batch_size):
    """Collect base outputs once per split, requesting embeddings only when needed.

    Examples:
        >>> class Model:
        ...     feature_keys, label_keys = ["x"], ["label"]
        ...     def __call__(self, x):
        ...         return {"y_prob": x}
        >>> samples = [{"x": torch.tensor([0.75, 0.25]), "label": torch.tensor(0)}]
        >>> values = collect_model_predictions(
        ...     Model(), {"test": samples}, {"test": ("y_prob",)}, batch_size=1
        ... )
        >>> values["test"]["y_prob"].tolist()
        [[0.75, 0.25]]
    """
    predictions = {}
    label_key = model.label_keys[0]
    for role, requested in outputs.items():
        dataset = splits[role]
        fields = set(requested) - {"y_true"}

        def predict_batch(batch, fields=fields):
            features = {key: batch[key] for key in model.feature_keys}
            arguments = {"embed": True} if "embed" in fields else {}
            sample_hashes = _sample_hashes(features, len(batch[label_key]))
            with torch.no_grad():
                result = model(**features, **arguments)
            missing = fields - result.keys()
            if missing:
                raise ValueError(
                    f"base model does not return outputs: {sorted(missing)}"
                )

            return {
                "y_true": torch.as_tensor(batch[label_key]).detach().cpu().numpy(),
                _SAMPLE_HASH: sample_hashes,
                **{name: result[name].detach().cpu().numpy() for name in fields},
            }

        predictions[role] = collect_predictions(dataset, batch_size, predict_batch)

    return predictions


def _sample_hashes(features, count):
    # Identify input rows independently of the requested outputs and batch size.
    hashes = []
    for index in range(count):
        digest = hashlib.sha256()
        for key, values in sorted(features.items()):
            digest.update(key.encode())
            _hash_value(digest, _sample_value(values, index))
        hashes.append(np.frombuffer(digest.digest(), dtype=np.uint8))

    return np.stack(hashes)


def _sample_value(values, index):
    if values is None:
        return None
    if isinstance(values, tuple):
        return tuple(_sample_value(part, index) for part in values)
    if isinstance(values, dict):
        return {key: _sample_value(part, index) for key, part in values.items()}

    return values[index]


def _hash_value(digest, value):
    if isinstance(value, torch.Tensor):
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    elif isinstance(value, dict):
        for key, item in sorted(value.items()):
            digest.update(str(key).encode())
            _hash_value(digest, item)
    elif isinstance(value, (list, tuple)):
        digest.update(str(len(value)).encode())
        for item in value:
            _hash_value(digest, item)
    else:
        digest.update(repr(value).encode())


def _merge_model_predictions(existing, produced):
    if existing is None:
        return produced

    for field in ("y_true", _SAMPLE_HASH):
        if field not in existing or not np.array_equal(
            existing[field], produced[field]
        ):
            raise ValueError(
                "cached predictions and new outputs have different sample ordering or labels"
            )

    return {**existing, **produced}


# --- Native calls backed by saved rows ---


class _CachedDataset(Dataset):
    def __init__(self, role, outputs, label_key, vocabulary):
        self.role, self.outputs, self.label_key = role, outputs, label_key
        self.output_processors = {label_key: SimpleNamespace(label_vocab=vocabulary)}

    def __len__(self):
        return len(self.outputs["y_true"])

    def __getitem__(self, index):
        return {
            _ROLE: self.role,
            _ROW: torch.tensor(index),
            self.label_key: torch.as_tensor(self.outputs["y_true"][index]),
        }


class _CachedModel(torch.nn.Module):
    def __init__(self, predictions, metadata, path, device):
        super().__init__()
        self.register_buffer("_device", torch.empty(0), persistent=False)
        self.predictions = predictions
        self.predictions_path = Path(path)
        self.mode = metadata["mode"]
        self.feature_keys = [_ROLE, _ROW]
        self.label_keys = [metadata["label_key"]]
        labels = metadata["class_labels"]
        vocabulary = (
            None if labels is None else {label: i for i, label in enumerate(labels)}
        )
        self.splits = {
            role: _CachedDataset(role, values, self.label_keys[0], vocabulary)
            for role, values in predictions.items()
        }
        self.dataset = next(iter(self.splits.values()))
        self.to(device)
        self.eval()

    @property
    def device(self):
        return self._device.device

    def forward(self, __cpbench_role__, __cpbench_row__, embed=False, **kwargs):
        role = __cpbench_role__[0]
        if any(item != role for item in __cpbench_role__):
            raise ValueError("a cached-model batch must belong to one split")
        rows = __cpbench_row__.detach().cpu().numpy()
        values = self.predictions[role]
        if embed and "embed" not in values:
            raise ValueError(f"model predictions are missing {role} outputs: ['embed']")
        result = {
            name: torch.as_tensor(array[rows]).to(self.device)
            for name, array in values.items()
            if (name != "embed" or embed)
            and (name != "y_true" or self.label_keys[0] in kwargs)
        }

        return result
