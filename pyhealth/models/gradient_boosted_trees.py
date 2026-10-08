"""Shared base for gradient-boosted tree models (XGBoost, and later LightGBM).

Tree ensembles are fit once on the full training set with
``model.fit(train_loader, val_loader=None)``, outside ``Trainer.train``. After
fitting, ``forward()`` returns the usual ``{loss, y_prob, y_true, logit}``
dictionary, so ``Trainer.inference``, ``Trainer.evaluate`` and the metrics work
unchanged, and ``Trainer.save_ckpt`` / ``load_ckpt`` round-trip the fitted
trees.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from ..datasets import SampleDataset, get_dataloader
from ..processors import (
    BinaryLabelProcessor,
    DeepNestedSequenceProcessor,
    MultiClassLabelProcessor,
    MultiLabelProcessor,
    NestedMultiHotProcessor,
    NestedSequenceProcessor,
    RegressionLabelProcessor,
    SequenceProcessor,
    TensorProcessor,
    TimeseriesProcessor,
)
from .base_model import BaseModel

logger = logging.getLogger(__name__)

# Bumped when the checkpoint layout of the extra state changes.
_STATE_VERSION = 1
# Vocabulary indices of <pad> and <unk>; excluded from code counts.
_RESERVED_CODES = 2

_DENSE = (
    TensorProcessor,
    TimeseriesProcessor,
    BinaryLabelProcessor,
    MultiClassLabelProcessor,
    RegressionLabelProcessor,
)
_TOKEN_SEQUENCES = (
    SequenceProcessor,
    NestedSequenceProcessor,
    DeepNestedSequenceProcessor,
)


class GradientBoostedTreeModel(BaseModel):
    """Base class for gradient-boosted tree models on fixed-width features.

    Subclasses provide the library-specific parts: building one estimator,
    margins, probabilities, exact TreeSHAP contributions, and converting a
    fitted estimator to and from bytes.

    **Feature matrix.** Fields are concatenated in the order of the dataset's
    ``input_schema`` (``self.feature_keys``). Column order matters: column
    subsampling (e.g. ``colsample_bytree``) depends on it, so the same seed
    with a different order gives different trees. Each field's columns are
    recorded in :attr:`feature_layout` and named in :attr:`feature_names`.

    Supported inputs:

    - ``tensor``, ``timeseries`` and label processors used as inputs
      (``binary``, ``multiclass``, ``regression``): flattened per sample. The
      per-sample shape must be the same in every batch, or a ``ValueError``
      names the field. ``NaN`` is passed through as missing.
    - ``multi_hot``: one column per vocabulary entry.
    - ``nested_multihot``: summed over visits, one column per code (number of
      visits containing the code), excluding ``<pad>`` and ``<unk>``.
    - ``sequence``, ``nested_sequence``, ``deep_nested_sequence``: only with
      ``bag_of_codes=True``, as per-sample counts over the processor's
      vocabulary, excluding ``<pad>`` and ``<unk>``. Their padded width depends
      on the batch, so flattening them would misalign columns between batches;
      without ``bag_of_codes`` they raise a ``ValueError`` naming the field.

    - Custom (non-PyHealth) processors returning numeric tensors: flattened
      like ``tensor``, with the same fixed-shape check.

    Any other PyHealth processor raises a ``ValueError`` naming the field.

    **Memory.** ``fit`` materializes the training (and validation) matrix as
    dense float32: ``n_samples * n_columns * 4`` bytes. With ``bag_of_codes``
    the width is the vocabulary size; trim large vocabularies first.

    Args:
        dataset: The dataset whose schemas and processors define the inputs and
            the label.
        bag_of_codes: Encode padded code sequences as per-sample code counts.
            Default False (they raise an error).
        feature_names: Optional column names for ``tensor``-like fields,
            ``{field: [name, ...]}``, one per flattened column.
        scale_pos_weight: Weight of positive examples for binary and multilabel
            labels: a number, one number per label (multilabel), ``"balanced"``
            (alias ``"auto"``) for negatives / positives in the training
            labels, or None. Weighting inflates predicted probabilities, so
            recalibrate on held-out patients if calibration matters.

    Examples:
        >>> from pyhealth.models import GradientBoostedTreeModel, XGBoostModel
        >>> issubclass(XGBoostModel, GradientBoostedTreeModel)
        True
        >>> model = XGBoostModel(dataset, bag_of_codes=True)  # doctest: +SKIP
        >>> model.fit(train_loader, val_loader)  # doctest: +SKIP
        >>> model.feature_layout[0]["key"], model.feature_names[:2]  # doctest: +SKIP
        ('conditions', ['conditions=4019', 'conditions=2724'])
    """

    #: Tells ``Trainer.train`` that this model is fit with ``model.fit``.
    fit_outside_trainer = True

    def __init__(
        self,
        dataset: SampleDataset,
        bag_of_codes: bool = False,
        feature_names: dict[str, Sequence[str]] | None = None,
        scale_pos_weight: float | Sequence[float] | str | None = None,
    ):
        super().__init__(dataset)
        if len(self.label_keys) != 1:
            raise ValueError(
                f"{type(self).__name__} supports exactly one label, got "
                f"{self.label_keys}."
            )
        self.label_key = self.label_keys[0]
        if self.mode not in ("binary", "multiclass", "multilabel", "regression"):
            raise ValueError(
                f"{type(self).__name__} supports binary, multiclass, multilabel "
                f"and regression labels; the label processor gives mode "
                f"{self.mode!r}."
            )
        if scale_pos_weight is not None and self.mode not in ("binary", "multilabel"):
            raise ValueError(
                "scale_pos_weight applies to binary or multilabel labels; this "
                f"model's mode is {self.mode!r}."
            )
        if isinstance(scale_pos_weight, str) and scale_pos_weight not in ("balanced", "auto"):
            raise ValueError(
                "scale_pos_weight must be a number, a sequence, 'balanced', "
                f"'auto' or None, not {scale_pos_weight!r}."
            )
        self.bag_of_codes = bag_of_codes
        self.scale_pos_weight = scale_pos_weight
        self._custom_names = {k: list(v) for k, v in (feature_names or {}).items()}
        self.feature_layout = self._initial_layout()
        self.estimators_: list[Any] = []
        self.fitted_scale_pos_weight: list[float] | None = None

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def _make_estimator(self, scale_pos_weight: float | None) -> Any:
        """Returns one unfitted estimator for ``self.mode``."""
        raise NotImplementedError

    def _fit_estimator(self, estimator, X, y, eval_set) -> None:
        raise NotImplementedError

    def _predict_margin(self, estimator, X: np.ndarray) -> np.ndarray:
        """Raw scores before the link function, ``(n,)`` or ``(n, n_classes)``."""
        raise NotImplementedError

    def _predict_value(self, estimator, X: np.ndarray) -> np.ndarray:
        """Probabilities (classification) or predictions (regression)."""
        raise NotImplementedError

    def _contributions(self, estimator, X: np.ndarray) -> np.ndarray:
        """Exact TreeSHAP values with the bias as the last column.

        Shape ``(n, n_columns + 1)``, or ``(n, n_classes, n_columns + 1)``.
        """
        raise NotImplementedError

    def _to_bytes(self, estimator) -> bytes:
        raise NotImplementedError

    def _from_bytes(self, raw: bytes) -> Any:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Feature layout
    # ------------------------------------------------------------------
    def _initial_layout(self) -> list[dict[str, Any]]:
        """Field kinds and, where the processor fixes it, the width."""
        layout = []
        for key in self.feature_keys:
            proc = self.dataset.input_processors[key]
            name = type(proc).__name__
            if isinstance(proc, MultiLabelProcessor):  # includes multi_hot
                kind, width = "multi_hot", proc.size()
            elif isinstance(proc, NestedMultiHotProcessor):
                kind, width = "visit_counts", len(proc.code_vocab) - _RESERVED_CODES
            elif isinstance(proc, _TOKEN_SEQUENCES):
                if not self.bag_of_codes:
                    raise ValueError(
                        f"Field {key!r} uses {name}, whose padded width depends "
                        f"on the batch, so its columns would not line up between "
                        f"batches. Pass bag_of_codes=True to encode it as code "
                        f"counts over the vocabulary, or use a fixed-width "
                        f"processor (tensor, multi_hot, nested_multihot)."
                    )
                kind, width = "counts", len(proc.code_vocab) - _RESERVED_CODES
            elif isinstance(proc, _DENSE) or not type(proc).__module__.startswith(
                "pyhealth."
            ):
                # Custom processors are taken as dense; their shape is checked.
                kind, width = "dense", None
            else:
                raise ValueError(
                    f"Field {key!r} uses {name}, which {type(self).__name__} "
                    f"cannot turn into fixed-width columns. Supported: tensor, "
                    f"timeseries (fixed length), multi_hot, nested_multihot, "
                    f"label processors as inputs, and code sequences with "
                    f"bag_of_codes=True."
                )
            layout.append(
                {"key": key, "kind": kind, "processor": name, "width": width, "shape": None}
            )
        self._assign_columns(layout)
        return layout

    @staticmethod
    def _assign_columns(layout: list[dict[str, Any]]) -> None:
        start = 0
        for field in layout:
            if field["width"] is None:
                field["start"] = field["stop"] = None
                start = None
                continue
            if start is not None:
                field["start"], field["stop"] = start, start + field["width"]
                start += field["width"]
            else:
                field["start"] = field["stop"] = None

    @property
    def feature_names(self) -> list[str]:
        """One name per column, in matrix order (available after fit)."""
        return [n for field in self.feature_layout for n in self._field_names(field)]

    def _field_names(self, field: dict[str, Any]) -> list[str]:
        key, proc = field["key"], self.dataset.input_processors[field["key"]]
        if field["kind"] == "dense":
            if field["width"] is None:
                return []
            custom = self._custom_names.get(key)
            if custom is not None:
                return list(custom)
            if field["width"] == 1:
                return [key]
            return [f"{key}[{j}]" for j in range(field["width"])]
        if field["kind"] == "multi_hot":
            by_index = {i: tok for tok, i in proc.label_vocab.items()}
            return [f"{key}={by_index.get(i, i)}" for i in range(field["width"])]
        by_index = {i: tok for tok, i in proc.code_vocab.items()}
        return [f"{key}={by_index[i]}" for i in range(_RESERVED_CODES, len(proc.code_vocab))]

    def _field_block(self, field: dict[str, Any], x: Any) -> np.ndarray:
        """One field of a batch as a float32 ``(n, width)`` block."""
        key = field["key"]
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Field {key!r} is not a tensor in this batch ({type(x).__name__}); "
                f"{type(self).__name__} needs numeric, fixed-width fields."
            )
        x = x.detach().cpu()
        n = x.shape[0]
        if field["kind"] == "dense":
            shape = tuple(x.shape[1:])
            if field["shape"] is None:
                field["shape"], field["width"] = list(shape), int(np.prod(shape, dtype=int))
                if key in self._custom_names and len(self._custom_names[key]) != field["width"]:
                    raise ValueError(
                        f"feature_names[{key!r}] has {len(self._custom_names[key])} "
                        f"names but the field has {field['width']} columns."
                    )
            elif tuple(field["shape"]) != shape:
                raise ValueError(
                    f"Field {key!r} has per-sample shape {shape} in this batch but "
                    f"{tuple(field['shape'])} when the model was fit. Tree models "
                    f"need a fixed width; a varying length is usually padding."
                )
            return x.reshape(n, -1).numpy().astype(np.float32, copy=False)
        if field["kind"] == "multi_hot":
            if tuple(x.shape[1:]) != (field["width"],):
                raise ValueError(
                    f"Field {key!r} has shape {tuple(x.shape[1:])}, expected "
                    f"({field['width']},) from its vocabulary."
                )
            return x.numpy().astype(np.float32, copy=False)
        vocab = field["width"] + _RESERVED_CODES
        if field["kind"] == "visit_counts":
            counts = x.reshape(n, -1, x.shape[-1]).sum(dim=1)
        else:  # "counts": token indices, any padded shape
            idx = x.reshape(n, -1).long()
            counts = torch.zeros(n, vocab, dtype=torch.float64)
            counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float64))
        if counts.shape[1] != vocab:
            raise ValueError(
                f"Field {key!r} has a vocabulary of {counts.shape[1]} entries, "
                f"expected {vocab}."
            )
        return counts[:, _RESERVED_CODES:].numpy().astype(np.float32)

    def build_feature_matrix(self, **batch: Any) -> np.ndarray:
        """Turns a batch into the float32 ``(n, n_columns)`` matrix the trees see.

        Examples:
            >>> X = model.build_feature_matrix(**batch)  # doctest: +SKIP
            >>> X.shape[1] == len(model.feature_names)  # doctest: +SKIP
            True
        """
        blocks = []
        for field in self.feature_layout:
            if field["key"] not in batch:
                raise KeyError(f"Batch is missing input field {field['key']!r}.")
            blocks.append(self._field_block(field, batch[field["key"]]))
        self._assign_columns(self.feature_layout)
        return np.concatenate(blocks, axis=1)

    def _labels(self, y: Any) -> np.ndarray:
        y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        if self.mode == "multilabel":
            return y.reshape(y.shape[0], -1)
        if self.mode == "regression":
            return y.reshape(-1).astype(np.float64)
        return y.reshape(-1).astype(np.int64)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def _collect(self, data, name: str) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(data, DataLoader):
            loader = data
            if loader.drop_last:
                raise ValueError(
                    f"{name} has drop_last=True; tree models must see every "
                    f"sample. Use a loader with drop_last=False."
                )
            if isinstance(loader.sampler, RandomSampler) or getattr(
                loader.dataset, "shuffle", False
            ):
                logger.warning(
                    "%s shuffles; row order changes row subsampling, so fits "
                    "are only reproducible with an unshuffled loader "
                    "(get_dataloader(..., shuffle=False)).",
                    name,
                )
        else:
            loader = get_dataloader(data, batch_size=1024, shuffle=False)
        X, y = [], []
        for batch in loader:
            X.append(self.build_feature_matrix(**batch))
            y.append(self._labels(batch[self.label_key]))
        if not X:
            raise ValueError(f"{name} is empty.")
        return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

    def _resolve_pos_weight(self, y: np.ndarray) -> list[float | None]:
        n_out = y.shape[1] if self.mode == "multilabel" else 1
        w = self.scale_pos_weight
        if w is None:
            return [None] * n_out
        if isinstance(w, str):  # "balanced" / "auto"
            y2 = y.reshape(y.shape[0], n_out).astype(np.float64)
            pos = y2.sum(axis=0)
            neg = y2.shape[0] - pos
            if (pos == 0).any():
                logger.warning(
                    "scale_pos_weight=%r: some labels have no positive training "
                    "examples; their weight is set to 1.", w
                )
            return [float(n / p) if p > 0 else 1.0 for n, p in zip(neg, pos)]
        weights = [float(v) for v in np.atleast_1d(np.asarray(w, dtype=np.float64))]
        if len(weights) == 1:
            return weights * n_out
        if len(weights) != n_out:
            raise ValueError(
                f"scale_pos_weight has {len(weights)} values for {n_out} labels."
            )
        return weights

    def fit(
        self,
        train_data: DataLoader | SampleDataset,
        val_data: DataLoader | SampleDataset | None = None,
    ) -> GradientBoostedTreeModel:
        """Fits the trees on the full training set in one call.

        Batches are concatenated in loader order and no sample is dropped. Pass
        an unshuffled loader (or a dataset, which is read unshuffled) so the
        fit is reproducible.

        Args:
            train_data: Training loader or dataset (the training split only).
            val_data: Optional validation loader or dataset, used for early
                stopping when ``early_stopping_rounds`` is set.

        Returns:
            The fitted model.

        Raises:
            ValueError: If a field is not fixed-width, the loader drops
                samples, or a multiclass training set lacks a class.
        """
        self.feature_layout = self._initial_layout()
        X, y = self._collect(train_data, "train_data")
        eval_X = eval_y = None
        if val_data is not None:
            eval_X, eval_y = self._collect(val_data, "val_data")
        if self.mode == "multiclass":
            n_classes = self.get_output_size()
            missing = sorted(set(range(n_classes)) - set(np.unique(y).tolist()))
            if missing:
                raise ValueError(
                    f"Classes {missing} have no training samples; the trees need "
                    f"every class of the label processor in the training set."
                )
        pos_weights = self._resolve_pos_weight(y)
        self.fitted_scale_pos_weight = (
            None if self.scale_pos_weight is None else [float(w) for w in pos_weights]
        )
        estimators = []
        for j, w in enumerate(pos_weights):
            est = self._make_estimator(w)
            if self.mode == "multilabel":
                yj = y[:, j].astype(np.int64)
                ev = None if eval_X is None else (eval_X, eval_y[:, j].astype(np.int64))
            else:
                yj, ev = y, None if eval_X is None else (eval_X, eval_y)
            self._fit_estimator(est, X, yj, ev)
            estimators.append(est)
        self.estimators_ = estimators
        return self

    @property
    def is_fitted(self) -> bool:
        return bool(self.estimators_)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted. Call "
                f"model.fit(train_loader, val_loader) first (Trainer.train does "
                f"not fit tree models)."
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_numpy(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Margins and probabilities (or predictions) for a feature matrix.

        Returns:
            ``(logit, y_prob)``, each ``(n, 1)`` for binary and regression,
            ``(n, n_classes)`` for multiclass, ``(n, n_labels)`` for
            multilabel.
        """
        self._check_fitted()
        if self.mode == "multilabel":
            logit = np.stack([self._predict_margin(e, X) for e in self.estimators_], 1)
            prob = np.stack([self._predict_value(e, X) for e in self.estimators_], 1)
            return logit, prob
        est = self.estimators_[0]
        logit, value = self._predict_margin(est, X), self._predict_value(est, X)
        if self.mode == "binary":
            return logit.reshape(-1, 1), value[:, 1:2]
        if self.mode == "regression":
            return logit.reshape(-1, 1), value.reshape(-1, 1)
        return logit, value

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Predicts with the fitted trees.

        ``logit`` is the trees' raw margin (not an inverted probability), and
        ``loss`` is ``get_loss_function()`` on it.

        Returns:
            ``{"logit", "y_prob"}``, plus ``"loss"`` and ``"y_true"`` when the
            batch has the label.
        """
        self._check_fitted()
        logit_np, prob_np = self.predict_numpy(self.build_feature_matrix(**kwargs))
        logit = torch.as_tensor(logit_np, dtype=torch.float32, device=self.device)
        y_prob = torch.as_tensor(prob_np, dtype=torch.float32, device=self.device)
        results = {"logit": logit, "y_prob": y_prob}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            results["y_true"] = y_true
            results["loss"] = self.get_loss_function()(logit, y_true)
        return results

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    def explain(self, **batch: Any) -> dict[str, Any]:
        """Exact TreeSHAP attributions for a batch, per field and column.

        Contributions are on the margin (log-odds for classification) and add
        up exactly, up to float32 rounding: for every sample, the sum over all
        columns plus ``bias`` equals ``logit``.

        Returns:
            A dict with ``"attributions"`` (``{field: tensor}``, each
            ``(n, field_width)``, with a trailing class/label dimension for
            multiclass and multilabel), ``"bias"`` (``(n,)``, or ``(n, K)``),
            ``"logit"`` and ``"feature_names"`` (``{field: [name, ...]}``).

        Examples:
            >>> out = model.explain(**batch)  # doctest: +SKIP
            >>> total = sum(a.sum(1) for a in out["attributions"].values())  # doctest: +SKIP
            >>> torch.allclose(total + out["bias"], out["logit"].squeeze(-1), atol=1e-4)  # doctest: +SKIP
            True
        """
        self._check_fitted()
        X = self.build_feature_matrix(**batch)
        if self.mode == "multilabel":
            c = np.stack([self._contributions(e, X) for e in self.estimators_], 1)
        else:
            c = self._contributions(self.estimators_[0], X)
        if c.ndim == 3:  # (n, K, d + 1) -> (n, d + 1, K)
            c = np.transpose(c, (0, 2, 1))
        values = torch.as_tensor(c, dtype=torch.float32, device=self.device)
        attributions = {
            f["key"]: values[:, f["start"]:f["stop"]] for f in self.feature_layout
        }
        logit, _ = self.predict_numpy(X)
        return {
            "attributions": attributions,
            "bias": values[:, -1],
            "logit": torch.as_tensor(logit, dtype=torch.float32, device=self.device),
            "feature_names": {f["key"]: self._field_names(f) for f in self.feature_layout},
        }

    def mean_abs_shap(self, data: Iterable[dict] | SampleDataset) -> dict[str, float]:
        """Global importance: mean |TreeSHAP| per column, largest first.

        For multiclass and multilabel outputs the mean is also taken over
        classes or labels.

        Examples:
            >>> ranking = model.mean_abs_shap(test_loader)  # doctest: +SKIP
            >>> list(ranking)[:3]  # doctest: +SKIP
            ['age', 'fev1', 'conditions=J45.909']
        """
        if isinstance(data, SampleDataset):
            data = get_dataloader(data, batch_size=1024, shuffle=False)
        total, n = None, 0
        for batch in data:
            out = self.explain(**batch)
            vals = torch.cat([out["attributions"][f["key"]] for f in self.feature_layout], 1)
            s = vals.abs().double()
            s = s.sum(0) if s.ndim == 2 else s.mean(-1).sum(0)
            total = s if total is None else total + s
            n += vals.shape[0]
        means = (total / n).cpu().tolist()
        pairs = sorted(zip(self.feature_names, means), key=lambda p: -p[1])
        return dict(pairs)

    # ------------------------------------------------------------------
    # Persistence (state_dict / Trainer.save_ckpt, load_ckpt)
    # ------------------------------------------------------------------
    def _layout_signature(self, layout: list[dict[str, Any]]) -> list[tuple]:
        return [(f["key"], f["kind"], f["processor"]) for f in layout]

    def get_extra_state(self) -> dict[str, Any]:
        """Fitted trees as raw bytes in uint8 tensors, plus the feature layout.

        Only tensors, strings and numbers, so checkpoints load with
        ``torch.load(weights_only=True)``.
        """
        return {
            "version": _STATE_VERSION,
            "mode": self.mode,
            "layout": json.dumps(self.feature_layout),
            "scale_pos_weight": json.dumps(self.fitted_scale_pos_weight),
            "estimators": [
                torch.from_numpy(np.frombuffer(self._to_bytes(e), dtype=np.uint8).copy())
                for e in self.estimators_
            ],
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if not state or not state.get("estimators"):
            self.estimators_ = []
            return
        if state.get("version") != _STATE_VERSION:
            raise ValueError(
                f"Checkpoint has tree-model state version {state.get('version')}, "
                f"expected {_STATE_VERSION}."
            )
        if state["mode"] != self.mode:
            raise ValueError(
                f"Checkpoint was fit for mode {state['mode']!r}, but this model's "
                f"label gives mode {self.mode!r}."
            )
        saved = json.loads(state["layout"])
        expected = self._initial_layout()
        if self._layout_signature(saved) != self._layout_signature(expected):
            raise ValueError(
                "Checkpoint feature layout does not match this model's dataset: "
                f"saved fields {self._layout_signature(saved)}, "
                f"expected {self._layout_signature(expected)}."
            )
        for s, e in zip(saved, expected):
            if e["width"] is not None and s["width"] != e["width"]:
                raise ValueError(
                    f"Field {s['key']!r} had {s['width']} columns when the "
                    f"checkpoint was saved but this dataset's processor gives "
                    f"{e['width']}; the vocabularies differ."
                )
        self.feature_layout = saved
        self.fitted_scale_pos_weight = json.loads(state["scale_pos_weight"])
        self.estimators_ = [
            self._from_bytes(t.cpu().numpy().tobytes()) for t in state["estimators"]
        ]
