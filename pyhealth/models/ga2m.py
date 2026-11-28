"""Generalized Additive Model with pairwise interactions (GA2M).

Torch reimplementation of Hegselmann et al. (2020), "An evaluation of the
doctor-interpretability of generalized additive models with interactions",
PMLR 126. Unlike InterpretML's ``ExplainableBoostingClassifier`` (cyclic
gradient boosting), this model represents each additive term as a lookup
table of learnable weights over quantile bins and trains all tables jointly
by ordinary SGD through :class:`~pyhealth.trainer.Trainer`.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..datasets import SampleDataset, get_dataloader
from .base_model import BaseModel

#: Sentinel used by the task for an unmeasured feature; always binned
#: separately from any observed value.
MISSING_VALUE = -1.0


class GA2M(BaseModel):
    """Doctor-interpretable GA2M: additive 1D terms plus selected 2D terms.

    The model predicts mortality log-odds as::

        logit(x) = bias + sum_j f_j(x_j) + sum_{(j,k) in pairs} f_jk(x_j, x_k)

    Each ``f_j`` / ``f_jk`` is a table of learnable weights indexed by
    quantile bin. A dedicated reserved bin (index 0) represents a missing
    feature (:data:`MISSING_VALUE`, ``-1.0``), so missing and observed values
    never share a bin.

    Bin edges are quantile thresholds of the *observed* (non-missing) values
    of each feature, computed by :meth:`fit_bins`. ``__init__`` calls
    ``fit_bins(dataset)`` once so the model is immediately usable (as every
    other PyHealth model is after construction); call :meth:`fit_bins` again
    explicitly with a train-only loader/dataset before training to avoid
    leaking validation/test statistics into the bin edges.

    Args:
        dataset: The dataset to train the model. Used to read the width of
            the ``"features"`` input (a single ``tensor`` field) and the
            label processor.
        n_bins: Number of quantile bins for each 1D term (plus the reserved
            missing bin). Default 256.
        pairs: Optional list of ``(i, j)`` feature-index pairs to model as 2D
            interaction terms. ``None`` (default) fits 1D terms only.
        pair_bins: Number of quantile bins for each 2D term (plus the
            reserved missing bin). Defaults to ``n_bins`` when ``None``.
        feature_names: Optional names for the ``n_features`` columns of
            ``"features"``, used only to label :meth:`get_shape_functions`
            output. Does not import or depend on any task module.
        excluded_features: Optional feature indices whose 1D terms are
            removed (held at zero and left out of
            :meth:`get_shape_functions`). Pairs may not use them. This
            supports the paper's clinician-pruning ablation: retrain
            without the risk curves doctors flagged as implausible.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "features": [1.0, -1.0, 2.0], "mortality": 0},
        ...     {"patient_id": "p1", "features": [2.0, 3.0, -1.0], "mortality": 1},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"mortality": "binary"},
        ... )
        >>> model = GA2M(dataset=dataset, n_bins=4, pairs=[(0, 1)])
        >>> from pyhealth.datasets import get_dataloader
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> ret = model(**batch)
        >>> sorted(ret.keys())
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        n_bins: int = 256,
        pairs: Sequence[tuple[int, int]] | None = None,
        pair_bins: int | None = None,
        feature_names: Sequence[str] | None = None,
        excluded_features: Sequence[int] | None = None,
    ):
        super().__init__(dataset)
        if "features" not in self.feature_keys:
            raise ValueError(
                f"GA2M expects an input field named 'features'; got "
                f"{self.feature_keys}."
            )
        assert len(self.label_keys) == 1, "GA2M supports exactly one label key"
        self.label_key = self.label_keys[0]
        if self.mode != "binary":
            raise ValueError(
                f"GA2M supports binary labels only; got mode={self.mode!r}."
            )

        n_features = dataset.input_processors["features"].size()
        if n_features is None:
            raise ValueError(
                "The 'features' input processor has not been fit; pass a "
                "dataset whose samples were used to build it."
            )
        self.n_features = int(n_features)
        if feature_names is not None and len(feature_names) != self.n_features:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries, expected "
                f"{self.n_features}."
            )
        self.feature_names = list(feature_names) if feature_names else [
            f"x{i}" for i in range(self.n_features)
        ]

        self.excluded_features: list[int] = sorted(set(excluded_features or ()))
        for j in self.excluded_features:
            if not 0 <= j < self.n_features:
                raise ValueError(f"Invalid excluded feature index {j!r}.")

        self.n_bins = n_bins
        self.pair_bins = pair_bins if pair_bins is not None else n_bins
        self.pairs: list[tuple[int, int]] = (
            [tuple(sorted(p)) for p in pairs] if pairs else []
        )
        seen_pairs = set()
        for i, j in self.pairs:
            if not (0 <= i < self.n_features and 0 <= j < self.n_features) or i == j:
                raise ValueError(f"Invalid feature pair {(i, j)!r}.")
            if i in self.excluded_features or j in self.excluded_features:
                raise ValueError(
                    f"Pair {(i, j)!r} uses an excluded feature."
                )
            if (i, j) in seen_pairs:
                raise ValueError(
                    f"Duplicate feature pair {(i, j)!r} in `pairs` (after "
                    f"sorting each pair ascending); each pair must appear "
                    f"at most once."
                )
            seen_pairs.add((i, j))

        self.bias = nn.Parameter(torch.zeros(()))
        self.terms_1d = nn.Parameter(torch.zeros(self.n_features, n_bins + 1))
        self.terms_2d = nn.ParameterDict(
            {
                self._pair_key(i, j): nn.Parameter(
                    torch.zeros(self.pair_bins + 1, self.pair_bins + 1)
                )
                for i, j in self.pairs
            }
        )

        # 1 for kept 1D terms, 0 for excluded ones (multiplies terms_1d).
        mask = torch.ones(self.n_features, 1)
        mask[self.excluded_features] = 0.0
        self.register_buffer("term_mask", mask)

        # Quantile edges, (n_features, n_bins - 1) / (n_features, pair_bins - 1).
        # Buffers (not Parameters): learned from data once, not by SGD.
        self.register_buffer(
            "bin_edges", torch.zeros(self.n_features, max(n_bins - 1, 0))
        )
        self.register_buffer(
            "pair_bin_edges", torch.zeros(self.n_features, max(self.pair_bins - 1, 0))
        )
        self.fit_bins(dataset)

    @staticmethod
    def _pair_key(i: int, j: int) -> str:
        return f"{i}_{j}"

    def _collect_features(self, data: DataLoader | SampleDataset) -> torch.Tensor:
        """Reads every 'features' row of ``data`` into one ``(n, d)`` tensor."""
        if isinstance(data, DataLoader):
            if data.drop_last:
                raise ValueError(
                    "fit_bins requires drop_last=False so every sample is "
                    "seen; pass a loader built with "
                    "get_dataloader(..., drop_last=False)."
                )
            loader = data
        else:
            loader = get_dataloader(data, batch_size=1024, shuffle=False)
        rows = [batch["features"] for batch in loader]
        if not rows:
            raise ValueError("fit_bins received no samples.")
        return torch.cat(rows, dim=0).to(torch.float32)

    @staticmethod
    def _quantile_edges(column: torch.Tensor, n_bins: int) -> torch.Tensor:
        """``n_bins - 1`` quantile thresholds from the observed values of a
        column, excluding :data:`MISSING_VALUE`. Falls back to zeros (every
        observed value shares one bin) when too few observations remain."""
        observed = column[column != MISSING_VALUE]
        n_edges = max(n_bins - 1, 0)
        if n_edges == 0:
            return torch.zeros(0)
        if observed.numel() < 2:
            return torch.zeros(n_edges)
        quantiles = torch.linspace(0, 1, n_edges + 2)[1:-1]
        return torch.quantile(observed.double(), quantiles.double()).float()

    def fit_bins(self, train_data: DataLoader | SampleDataset) -> GA2M:
        """Recomputes quantile bin edges from ``train_data`` only.

        Call this explicitly with a train-split loader/dataset (e.g. from
        :func:`~pyhealth.datasets.splitter.split_by_patient`) before training,
        to avoid leaking validation/test statistics into the bin edges;
        ``__init__`` already calls this once with the dataset it was given.

        Args:
            train_data: A loader or dataset over the training split only.

        Returns:
            self.
        """
        features = self._collect_features(train_data)
        if features.shape[1] != self.n_features:
            raise ValueError(
                f"'features' has width {features.shape[1]}, expected "
                f"{self.n_features}."
            )
        edges = torch.stack(
            [
                self._quantile_edges(features[:, j], self.n_bins)
                for j in range(self.n_features)
            ]
        )
        pair_edges = torch.stack(
            [
                self._quantile_edges(features[:, j], self.pair_bins)
                for j in range(self.n_features)
            ]
        )
        self.bin_edges = edges.to(self.bin_edges.device)
        self.pair_bin_edges = pair_edges.to(self.pair_bin_edges.device)
        return self

    def _bin_indices(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """``x``: ``(batch, n_features)``; ``edges``: ``(n_features, n_edges)``.

        Returns integer bin indices of the same shape as ``x``: 0 for
        :data:`MISSING_VALUE`, else ``1 + searchsorted(edges[j], x[:, j])``.
        """
        missing = x == MISSING_VALUE
        # torch.searchsorted needs edges and x on matching dtype/device.
        idx = torch.searchsorted(
            edges.to(x.dtype), x.t().contiguous(), right=True
        ).t()
        idx = idx + 1
        idx = idx.masked_fill(missing, 0)
        return idx

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            **kwargs: Must contain ``"features"`` (``(batch, n_features)``
                tensor) and, for training/evaluation, the label field.

        Returns:
            Dict with ``logit``, ``y_prob``, and (when the label is present)
            ``loss`` and ``y_true``.
        """
        x = kwargs["features"]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device=self.device, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        bins_1d = self._bin_indices(x, self.bin_edges)  # (batch, n_features)
        feature_idx = torch.arange(self.n_features, device=x.device).expand_as(
            bins_1d
        )
        terms_1d = self.terms_1d * self.term_mask
        logit = self.bias + terms_1d[feature_idx, bins_1d].sum(dim=1)

        if self.pairs:
            bins_2d = self._bin_indices(x, self.pair_bin_edges)
            for i, j in self.pairs:
                table = self.terms_2d[self._pair_key(i, j)]
                bi, bj = bins_2d[:, i], bins_2d[:, j]
                logit = logit + table[bi, bj]

        logit = logit.unsqueeze(-1)
        y_prob = self.label_processor.activation(logit)
        results = {"logit": logit, "y_prob": y_prob}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            results["loss"] = self.label_processor.loss(logit, y_true)
            results["y_true"] = y_true
        return results

    def smoothness_penalty(self) -> torch.Tensor:
        """Mean squared second difference of every 1D term over its real
        (non-missing) bins, and of every 2D term along both axes.

        Adding ``lam * model.smoothness_penalty()`` to the loss discourages
        jagged risk curves when bins are fine and some bins hold few
        training samples. The reserved missing bin is not smoothed.
        """
        terms_1d = (self.terms_1d * self.term_mask)[:, 1:]
        if terms_1d.shape[1] < 3:
            return self.bias.new_zeros(())
        penalty = terms_1d.diff(n=2, dim=1).pow(2).mean()
        for table in self.terms_2d.values():
            real = table[1:, 1:]
            if real.shape[0] >= 3:
                penalty = penalty + real.diff(n=2, dim=0).pow(2).mean()
                penalty = penalty + real.diff(n=2, dim=1).pow(2).mean()
        return penalty

    def get_shape_functions(
        self,
    ) -> dict[object, tuple[torch.Tensor, torch.Tensor]]:
        """Centered 1D and 2D shape functions, for plotting/inspection.

        Each term is centered by subtracting the mean of its *real* bins
        only (excluding the reserved missing bin at index 0) from every
        entry, including the missing bin's own value. This keeps the
        real-value curve/surface zero-centered over its own clinically
        meaningful domain regardless of how large the learned missing-bin
        weight is; the missing bin's displayed value is relative to that
        same (real-bin) baseline, not folded into its own centering. This
        has no effect on :meth:`forward`, which uses the raw learned
        tables.

        Returns:
            A dict keyed by feature name (1D terms) or a
            ``(name_i, name_j)`` pair (2D terms), each mapping to
            ``(bin_edges, centered_values)``. For a 1D term, ``bin_edges``
            has ``n_bins - 1`` entries (the internal thresholds between the
            ``n_bins`` real-value bins) while ``centered_values`` has
            ``n_bins + 1`` entries: the reserved missing bin (index 0)
            followed by the ``n_bins`` real bins. A 2D term's
            ``centered_values`` is ``(pair_bins + 1, pair_bins + 1)``, with
            row/column index 0 similarly reserved for "that feature was
            missing".
        """
        shapes: dict[object, tuple[torch.Tensor, torch.Tensor]] = {}
        for j, name in enumerate(self.feature_names):
            if j in self.excluded_features:
                continue
            table = self.terms_1d[j].detach()
            shapes[name] = (self.bin_edges[j].detach(), table - table[1:].mean())
        for i, j in self.pairs:
            table = self.terms_2d[self._pair_key(i, j)].detach()
            key = (self.feature_names[i], self.feature_names[j])
            shapes[key] = (
                (self.pair_bin_edges[i].detach(), self.pair_bin_edges[j].detach()),
                table - table[1:, 1:].mean(),
            )
        return shapes


def rank_pairs_by_contribution_variance(
    model: GA2M, data: DataLoader | SampleDataset
) -> list[tuple[tuple[int, int], float]]:
    """Ranks ``model``'s 2D terms by their contribution variance on ``data``.

    Implements the pair-selection half of the paper's procedure: train a
    model with every candidate pair at coarse bins, call this to rank pairs
    by how much they actually vary across the training population, keep the
    top ``k``, then construct and retrain a new :class:`GA2M` with only
    those pairs. The two-phase orchestration lives in the example script,
    not here, so this model class stays a plain, testable additive model.

    Args:
        model: A fitted or in-training :class:`GA2M` whose ``pairs`` are the
            candidates to rank (e.g. all ``C(n_features, 2)`` pairs).
        data: Loader or dataset to compute each pair's contribution variance
            over (typically the training split).

    Returns:
        ``[(pair, variance), ...]`` sorted by descending variance.

    Examples:
        >>> ranked = rank_pairs_by_contribution_variance(
        ...     model, train_loader
        ... )  # doctest: +SKIP
        >>> top_pairs = [pair for pair, _ in ranked[:34]]  # doctest: +SKIP
    """
    if not model.pairs:
        return []
    loader = (
        data
        if isinstance(data, DataLoader)
        else get_dataloader(data, batch_size=1024, shuffle=False)
    )
    model.eval()
    contributions: dict[tuple[int, int], list[torch.Tensor]] = {
        pair: [] for pair in model.pairs
    }
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device=model.device, dtype=torch.float32)
            bins_2d = model._bin_indices(x, model.pair_bin_edges)
            for i, j in model.pairs:
                table = model.terms_2d[model._pair_key(i, j)]
                contributions[(i, j)].append(table[bins_2d[:, i], bins_2d[:, j]])
    variances = {
        pair: torch.cat(values).double().var(unbiased=False).item()
        for pair, values in contributions.items()
    }
    return sorted(variances.items(), key=lambda item: item[1], reverse=True)


def all_pairs(n_features: int) -> list[tuple[int, int]]:
    """Every unordered feature-index pair, e.g. all 561 pairs for 34 features.

    Examples:
        >>> from pyhealth.models.ga2m import all_pairs
        >>> all_pairs(3)
        [(0, 1), (0, 2), (1, 2)]
        >>> len(all_pairs(34))
        561
    """
    return list(itertools.combinations(range(n_features), 2))
