"""MedMod discretizer and normalizer, ported for the PyHealth IHM task.

Source: MedMod repo (github.com/nyuad-cai/MedMod, commit in
``files/MEDMOD_COMMIT``), file ``src/data/preprocessing.py``. Line numbers
below refer to that file. The algorithm is unchanged; the port only

* fixes the settings MedMod calls it with (``ehr_funcs``, lines 247-252):
  ``store_masks=True``, ``impute_strategy='previous'``, ``start_time='zero'``,
  and drops the branches for the other values; the bin width stays a setting
  (default 1.0, the ``--timestep`` default),
* drops the statistics counters (lines 29-32, 147-150), which do not affect output,
* loads the normalizer pickle with a restricted unpickler instead of
  ``pickle.load`` (line 220), so a tampered file cannot run code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

# MedMod's own files (Apache-2.0, see files/NOTICE), bundled with this package.
MEDMOD_FILES = Path(__file__).resolve().parent / "files"
DEFAULT_CONFIG = MEDMOD_FILES / "discretizer_config.json"
# The normalizer MedMod's released code loads for IHM: the *phenotyping* one.
# ehr_funcs tests `args.task is 'mortality'` (preprocessing.py:259), an identity
# check on an argparse string that is False in practice, so it falls through to
# the `else` branch at line 265-266 (ph_ts{timestep}...normalizer).
PH_NORMALIZER = MEDMOD_FILES / "ph_ts1.0.input_str:previous.start_time:zero.normalizer"

EPS = 1e-6  # preprocessing.py:38


def _check_timestep(timestep: float) -> float:
    if not timestep > 0:
        raise ValueError(f"timestep must be > 0 hours, got {timestep}")
    return float(timestep)


class Discretizer:
    """Port of ``Discretizer`` (preprocessing.py:11-172) with the settings
    MedMod calls it with (``ehr_funcs``, lines 247-252): masks stored,
    previous-value imputation, time measured from ICU admission.

    Args:
        timestep: bin width in hours (MedMod's default 1.0).
    """

    def __init__(self, timestep: float = 1.0):
        with open(DEFAULT_CONFIG) as f:  # lines 15-21
            config = json.load(f)
        self._id_to_channel: List[str] = config["id_to_channel"]
        self._channel_to_id = {c: i for i, c in enumerate(self._id_to_channel)}
        self._is_categorical_channel = config["is_categorical_channel"]
        self._possible_values = config["possible_values"]
        self._normal_values = config["normal_values"]

        self._header = ["Hours"] + self._id_to_channel  # line 23
        self._timestep = _check_timestep(timestep)

        # Column layout, lines 59-69: channels in config order, a categorical
        # channel takes len(possible_values) one-hot columns.
        self._begin_pos: List[int] = []
        cur_len = 0
        for channel in self._id_to_channel:
            self._begin_pos.append(cur_len)
            if self._is_categorical_channel[channel]:
                cur_len += len(self._possible_values[channel])
            else:
                cur_len += 1
        self._n_data_cols = cur_len

    @property
    def channels(self) -> List[str]:
        return list(self._id_to_channel)

    @property
    def header(self) -> List[str]:
        """Input header: ``Hours`` + the 17 channels in config order."""
        return list(self._header)

    def output_header(self) -> List[str]:
        """Output column names, built as in lines 156-168."""
        out = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                out += [f"{channel}->{v}" for v in self._possible_values[channel]]
            else:
                out.append(channel)
        return out + [f"mask->{c}" for c in self._id_to_channel]

    def continuous_columns(self) -> List[int]:
        """Columns MedMod z-scores: header names without ``->`` (line 254)."""
        return [i for i, name in enumerate(self.output_header()) if "->" not in name]

    def with_timestep(self, timestep: float) -> "Discretizer":
        """The same discretizer with another bin width (hours). The column
        layout does not depend on it, so the config is not read again."""
        other = copy.copy(self)
        other._timestep = _check_timestep(timestep)
        return other

    def _write(self, data: np.ndarray, bin_id: int, channel: str, value: str) -> None:
        # lines 77-88. `list.index` raises ValueError for a categorical value
        # that is not in possible_values, exactly like MedMod.
        channel_id = self._channel_to_id[channel]
        begin = self._begin_pos[channel_id]
        if self._is_categorical_channel[channel]:
            values = self._possible_values[channel]
            category_id = values.index(value)
            data[bin_id, begin : begin + len(values)] = 0.0
            data[bin_id, begin + category_id] = 1.0
        else:
            data[bin_id, begin] = float(value)

    def transform(
        self,
        X: Sequence[Sequence[str]],
        header: Optional[Sequence[str]] = None,
        end: Optional[float] = None,
    ) -> np.ndarray:
        """Bin rows ``[Hours, v_1, ..., v_17]`` (strings, '' = missing).

        Returns the (N_bins, 76) float64 array (MedMod returns ``(data,
        header)``; use :meth:`output_header` for the names).
        """
        header = list(self._header if header is None else header)
        assert header[0] == "Hours"  # line 37
        n_channels = len(self._id_to_channel)

        ts = [float(row[0]) for row in X]  # lines 41-43
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i + 1] + EPS

        first_time = 0.0  # start_time='zero', lines 45-50

        max_hours = (max(ts) if end is None else end) - first_time  # lines 52-55
        n_bins = int(max_hours / self._timestep + 1.0 - EPS)  # line 57

        data = np.zeros((n_bins, self._n_data_cols), dtype=float)  # line 71
        mask = np.zeros((n_bins, n_channels), dtype=int)  # line 72
        original_value = [["" for _ in range(n_channels)] for _ in range(n_bins)]

        # lines 90-109: rows in time order; a later value in the same bin
        # overwrites an earlier one and the mask is set to 1.
        for row in X:
            t = float(row[0]) - first_time
            if t > max_hours + EPS:
                continue
            bin_id = int(t / self._timestep - EPS)
            assert 0 <= bin_id < n_bins
            for j in range(1, len(row)):
                if row[j] == "":
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]
                mask[bin_id][channel_id] = 1
                self._write(data, bin_id, channel, row[j])
                original_value[bin_id][channel_id] = row[j]

        # impute_strategy='previous', lines 116-131: an unobserved bin takes the
        # channel's last observed value, or its normal value before the first.
        prev_values: List[List[str]] = [[] for _ in range(n_channels)]
        for bin_id in range(n_bins):
            for channel in self._id_to_channel:
                channel_id = self._channel_to_id[channel]
                if mask[bin_id][channel_id] == 1:
                    prev_values[channel_id].append(original_value[bin_id][channel_id])
                    continue
                if prev_values[channel_id]:
                    imputed_value = prev_values[channel_id][-1]
                else:
                    imputed_value = self._normal_values[channel]
                self._write(data, bin_id, channel, imputed_value)

        # store_masks=True, lines 152-153
        return np.hstack([data, mask.astype(np.float32)])


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Allows only the globals a pickled ``{'means', 'stds'}`` dict of numpy
    arrays needs (checked with ``python -m pickletools`` on the MedMod file:
    ``numpy.core.multiarray._reconstruct``, ``numpy.ndarray``, ``numpy.dtype``)."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed")
        if module.startswith("numpy.core") and int(np.__version__.split(".")[0]) >= 2:
            module = module.replace("numpy.core", "numpy._core", 1)
        return super().find_class(module, name)


class Normalizer:
    """Port of ``Normalizer`` (preprocessing.py:181-232)."""

    def __init__(self, fields: Optional[Sequence[int]] = None):
        self._fields = None if fields is None else list(fields)
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        # Stable across runs (no object address): PyHealth puts str() of
        # pre-fitted processors into its cache key (base_dataset.py:1207-1230).
        if self._means is None:
            state = "unfitted"
        else:
            digest = hashlib.sha256(
                np.ascontiguousarray(self._means).tobytes()
                + np.ascontiguousarray(self._stds).tobytes()
            ).hexdigest()[:12]
            state = f"sha256:{digest}"
        n = "all" if self._fields is None else len(self._fields)
        return f"Normalizer(fields={n}, {state})"

    def fit(self, arrays: Iterable[np.ndarray]) -> "Normalizer":
        """Means and stds over every bin of every array, all columns, as
        ``_feed_data`` + ``_save_params`` (lines 193-215) compute them; MedMod's
        create_normalizer_state.py:87-93 feeds each training stay's discretized,
        unnormalized array."""
        count, sum_x, sum_sq_x = 0, None, None
        for x in arrays:
            x = np.asarray(x, dtype=float)
            count += x.shape[0]
            if sum_x is None:
                sum_x, sum_sq_x = x.sum(axis=0), (x**2).sum(axis=0)
            else:
                sum_x += x.sum(axis=0)
                sum_sq_x += (x**2).sum(axis=0)
        if count < 2:
            raise ValueError("need at least 2 rows to fit a normalizer")
        means = 1.0 / count * sum_x
        var = 1.0 / (count - 1) * (sum_sq_x - 2.0 * sum_x * means + count * means**2)
        # For a constant column, rounding can make var a tiny negative number;
        # MedMod's sqrt then gives NaN and the 1e-7 floor misses it. Clamp at 0
        # so it gets the floor; every other column is unchanged.
        stds = np.sqrt(np.maximum(var, 0.0))
        stds[stds < 1e-7] = 1e-7
        self._means, self._stds = means, stds
        return self

    def load_params(self, path: str | Path = PH_NORMALIZER) -> "Normalizer":
        # line 220: pickle.load(..., encoding='latin1')
        with open(path, "rb") as f, warnings.catch_warnings():
            # numpy >= 2.4 warns about the py2-era `align=0` in the pickled dtype.
            warnings.simplefilter(
                "ignore",
                category=getattr(np, "exceptions", np).VisibleDeprecationWarning,
            )
            dct = _NumpyOnlyUnpickler(f, encoding="latin1").load()
        self._means = np.asarray(dct["means"], dtype=float)
        self._stds = np.asarray(dct["stds"], dtype=float)
        return self

    @property
    def means(self) -> np.ndarray:
        assert self._means is not None
        return self._means

    @property
    def stds(self) -> np.ndarray:
        assert self._stds is not None
        return self._stds

    def transform(self, X: np.ndarray) -> np.ndarray:
        # lines 224-232: only the listed columns are z-scored.
        fields = range(X.shape[1]) if self._fields is None else self._fields
        ret = 1.0 * X
        for col in fields:
            ret[:, col] = (X[:, col] - self.means[col]) / self.stds[col]
        return ret


def medmod_ehr_funcs(timestep: float = 1.0) -> tuple[Discretizer, Normalizer]:
    """Equivalent of ``ehr_funcs`` (preprocessing.py:247-269): MedMod's
    discretizer and the normalizer its released code loads (``PH_NORMALIZER``).

    Examples:
        >>> from pyhealth.datasets.medmod.discretizer import medmod_ehr_funcs
        >>> discretizer, normalizer = medmod_ehr_funcs()
        >>> len(discretizer.output_header())
        76
    """
    discretizer = Discretizer(timestep=timestep)
    normalizer = Normalizer(fields=discretizer.continuous_columns())
    normalizer.load_params(PH_NORMALIZER)
    return discretizer, normalizer
