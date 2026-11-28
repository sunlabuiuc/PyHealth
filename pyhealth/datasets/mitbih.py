"""MIT-BIH Arrhythmia dataset and MedTsLLM's record-level anomaly split.

Contributors: Thi Van Trinh Tran and Zexi Li.
Task: reconstruction-based ECG anomaly detection on abnormal beats.
Reference: MedTsLLM, https://arxiv.org/abs/2408.07773.
Data: https://physionet.org/content/mitdb/1.0.0/ (Moody and Mark, IEEE Eng.
Med. Biol. Mag. 20(3):45-50, 2001).

One patient per 30-minute, two-lead recording; a YAML-configured metadata
table; signals are read lazily with ``wfdb`` by the task. The split
reproduces MedTsLLM's 6/4 MLII+V1 record selection exactly.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import dask.dataframe as dd
import numpy as np
import pandas as pd

from .base_dataset import BaseDataset
from .splitter import Split
from .utils import MODULE_CACHE_PATH

if TYPE_CHECKING:
    from pyhealth.tasks.mitbih import MITBIHAnomalyDetection

logger = logging.getLogger(__name__)

MITBIH_EVENT_TYPE = "records"
MITBIH_METADATA_FILE = "mitbih-pyhealth.csv"

# WFDB beat annotation symbols. Everything except "N" counts as abnormal.
# Non-beat annotations (rhythm "+", noise "~", artifact "|", comments, ...)
# describe the recording rather than a heartbeat and are ignored.
MITBIH_BEAT_SYMBOLS = frozenset("NLRBAaJSVrFejnE/fQ?!x\\")
MITBIH_NORMAL_SYMBOL = "N"


def _require_wfdb():
    """Imports ``wfdb`` lazily with an actionable error message.

    Returns:
        module: The imported ``wfdb`` module.

    Raises:
        ImportError: If ``wfdb`` is not installed.

    Examples:
        >>> wfdb = _require_wfdb()  # doctest: +SKIP
    """
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "Reading MIT-BIH records requires the optional 'wfdb' dependency. "
            "Install it with `pip install wfdb` (also provided by "
            "`pip install pyhealth[ptbxl]`)."
        ) from exc
    return wfdb


def beat_anomaly_ratio(symbols: Sequence[str]) -> float:
    """Fraction of beat annotations that are abnormal (not ``"N"``).

    Args:
        symbols: WFDB annotation symbols of one record.

    Returns:
        float: Abnormal beats divided by all beat annotations, or ``0.0`` if
        the record has no beat annotations.

    Examples:
        >>> beat_anomaly_ratio(["N", "N", "V", "+", "N"])
        0.25
    """
    beats = [s for s in symbols if s in MITBIH_BEAT_SYMBOLS]
    if not beats:
        return 0.0
    return sum(s != MITBIH_NORMAL_SYMBOL for s in beats) / len(beats)


@dataclass(frozen=True)
class MITBIHAnomalyRatioSplit(Split):
    """Deterministic record-level train/test split by anomalous-beat ratio.

    Records with the fewest abnormal beats go to training (approximating
    access to mostly-normal, unlabeled data) and the most abnormal go to
    testing. Records whose beats are 100% abnormal are excluded. Ties are
    broken by record ID, so the split is fully deterministic.

    Two modes are supported:

    - ``"paper"`` (default): the selection used by MedTsLLM for anomaly
      detection. Only records whose leads are exactly ``required_leads``
      (MLII + V1) are eligible (33 records), then the ``n_train`` least and
      ``n_test`` most abnormal are used. With the defaults this yields
      train = 101, 112, 115, 121, 122, 230 and test = 208, 212, 217, 231,
      which reproduces the paper's reported sizes exactly (1,354,170 /
      902,780 samples at 125 Hz).
    - ``"fraction"``: all eligible records regardless of leads; the least
      abnormal ``train_frac`` go to training and the rest to testing.

    Construct it with :meth:`MITBIHDataset.anomaly_ratio_split`, which fills
    in the per-record metadata.

    Args:
        anomaly_ratios: Record ID to abnormal-beat ratio.
        leads: Record ID to its lead names, e.g. ``("MLII", "V1")``.
        mode: ``"paper"`` or ``"fraction"``. Defaults to ``"paper"``.
        required_leads: Leads required in ``"paper"`` mode.
        n_train: Number of training records in ``"paper"`` mode.
        n_test: Number of test records in ``"paper"`` mode.
        train_frac: Training fraction in ``"fraction"`` mode.

    Examples:
        >>> split = MITBIHAnomalyRatioSplit(
        ...     anomaly_ratios={"a": 0.0, "b": 0.3, "c": 0.1},
        ...     leads={"a": ("MLII", "V1"), "b": ("MLII", "V1"),
        ...            "c": ("MLII", "V1")},
        ...     n_train=2, n_test=1,
        ... )
        >>> split.select_records()
        (['a', 'c'], ['b'])
    """

    anomaly_ratios: Mapping[str, float] = field(default_factory=dict)
    leads: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    mode: Literal["paper", "fraction"] = "paper"
    required_leads: tuple[str, ...] = ("MLII", "V1")
    n_train: int = 6
    n_test: int = 4
    train_frac: float = 0.8

    def select_records(self) -> tuple[list[str], list[str]]:
        """Computes the training and test record IDs.

        Returns:
            tuple[list[str], list[str]]: Training and test record IDs, each
            ordered by increasing anomaly ratio.

        Raises:
            ValueError: If the mode is unknown or too few records are eligible.

        Examples:
            >>> MITBIHAnomalyRatioSplit(
            ...     anomaly_ratios={"a": 0.0, "b": 0.5, "c": 1.0},
            ...     mode="fraction", train_frac=0.5,
            ... ).select_records()
            (['a'], ['b'])
        """
        eligible = {r: v for r, v in self.anomaly_ratios.items() if v < 1.0}
        if self.mode == "paper":
            required = tuple(self.required_leads)
            eligible = {
                r: v
                for r, v in eligible.items()
                if tuple(self.leads.get(r, ())) == required
            }
        elif self.mode != "fraction":
            raise ValueError(f"Unknown mode {self.mode!r}; use 'paper' or 'fraction'.")

        ordered = sorted(eligible, key=lambda r: (eligible[r], r))
        if self.mode == "paper":
            if len(ordered) < self.n_train + self.n_test:
                raise ValueError(
                    f"Only {len(ordered)} eligible records for n_train="
                    f"{self.n_train} + n_test={self.n_test}."
                )
            return ordered[: self.n_train], ordered[len(ordered) - self.n_test :]
        n_train = max(1, round(len(ordered) * self.train_frac))
        return ordered[:n_train], ordered[n_train:]

    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> list[np.ndarray]:
        """Returns sample indices of the training and test parts.

        Args:
            patient_to_index: Patient (record) ID to the indices of its samples.

        Returns:
            list[np.ndarray]: Sorted ``int64`` sample indices for the training
            part, then the test part. Samples of unselected records are left
            out.

        Examples:
            >>> split = MITBIHAnomalyRatioSplit(
            ...     anomaly_ratios={"a": 0.0, "b": 0.5}, mode="fraction",
            ...     train_frac=0.5,
            ... )
            >>> split.split_indices({"a": [0, 1], "b": [2]})
            [array([0, 1]), array([2])]
        """
        train_ids, test_ids = self.select_records()
        part_of = {r: 0 for r in train_ids} | {r: 1 for r in test_ids}
        parts: tuple[list[int], list[int]] = ([], [])
        for pid, indices in patient_to_index.items():
            part = part_of.get(str(pid))
            if part is not None:
                parts[part].extend(indices)
        return [np.asarray(sorted(p), dtype=np.int64) for p in parts]

    def to_dict(self) -> dict[str, Any]:
        """Serializable description of the split, used in the cache key.

        Returns:
            dict[str, Any]: Mode, parameters, and the selected record IDs.

        Examples:
            >>> MITBIHAnomalyRatioSplit(
            ...     anomaly_ratios={"a": 0.0, "b": 0.5}, mode="fraction",
            ...     train_frac=0.5,
            ... ).to_dict()["test_records"]
            ['b']
        """
        train_ids, test_ids = self.select_records()
        return {
            "kind": "mitbih_anomaly_ratio",
            "mode": self.mode,
            "required_leads": list(self.required_leads),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_frac": self.train_frac,
            "train_records": train_ids,
            "test_records": test_ids,
        }


class MITBIHDataset(BaseDataset):
    """MIT-BIH Arrhythmia Database (48 two-lead, 30-minute ECG recordings).

    Each recording is treated as one patient whose ID is the record name
    (e.g. ``"100"``). On construction, the WFDB headers and beat annotations
    are read once and summarized in a metadata CSV with one row per record,
    loaded through the ``records`` table of ``configs/mitbih.yaml``. Each
    patient therefore has one ``records`` event with the file path, sampling
    rate, lead names, abnormal-beat ratio and header comments. Waveforms are
    loaded lazily by the task.

    The CSV is written to ``metadata_cache_dir``, so the data directory may
    be read-only. A hash of its content is part of the dataset name, so
    changing the records in ``root`` never reuses a stale event cache.

    Args:
        root: Directory containing the ``.hea``, ``.dat`` and ``.atr`` files of
            the PhysioNet ``mitdb`` 1.0.0 release.
        dataset_name: Name of the dataset. Defaults to ``"mitbih"``.
        config_path: YAML config. Defaults to the packaged
            ``configs/mitbih.yaml``.
        metadata_cache_dir: Directory for the derived metadata CSV. Defaults
            to a folder under the PyHealth cache, keyed by ``root``.
        cache_dir: Cache directory, see :class:`BaseDataset`.
        num_workers: Number of workers for task processing.
        dev: Whether to run in dev mode (limits the number of patients).

    Raises:
        FileNotFoundError: If ``root`` does not exist, has no ``.hea`` files,
            or a record has no ``.atr`` beat annotation file.

    Examples:
        >>> from pyhealth.datasets import MITBIHDataset  # doctest: +SKIP
        >>> ds = MITBIHDataset(root="/data/mitdb/1.0.0")  # doctest: +SKIP
        >>> train, test = ds.set_task(split=ds.anomaly_ratio_split())  # doctest: +SKIP
        >>> train[0]["signal"].shape  # doctest: +SKIP
        torch.Size([128, 2])
    """

    def __init__(
        self,
        root: str | Path,
        dataset_name: str | None = None,
        config_path: str | Path | None = None,
        metadata_cache_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            raise FileNotFoundError(f"MIT-BIH directory not found: {root_path}")
        self.record_ids = sorted(p.stem for p in root_path.glob("*.hea"))
        if not self.record_ids:
            raise FileNotFoundError(f"No .hea files found in {root_path}")

        root_key = hashlib.sha1(str(root_path).encode("utf-8")).hexdigest()[:10]
        self.metadata_cache_dir = Path(
            metadata_cache_dir
            if metadata_cache_dir is not None
            else Path(MODULE_CACHE_PATH) / "mitbih" / root_key
        )
        self.record_metadata = self._read_metadata(root_path)
        metadata_key = self.prepare_metadata()
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "mitbih.yaml"

        super().__init__(
            root=str(root_path),
            tables=["records"],
            dataset_name=f"{dataset_name or 'mitbih'}_{metadata_key}",
            config_path=str(config_path),
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def _read_metadata(self, root: Path) -> dict[str, dict[str, Any]]:
        """Reads headers and beat annotations of every record.

        Args:
            root: Dataset directory.

        Returns:
            dict[str, dict[str, Any]]: Record ID to metadata.

        Raises:
            FileNotFoundError: If a record has no ``.atr`` annotation file.

        Examples:
            >>> meta = ds._read_metadata(Path("/data/mitdb"))  # doctest: +SKIP
        """
        wfdb = _require_wfdb()
        metadata = {}
        for rid in self.record_ids:
            path = root / rid
            if not path.with_suffix(".atr").is_file():
                raise FileNotFoundError(
                    f"Record {rid} has no beat annotation file {rid}.atr in {root}"
                )
            header = wfdb.rdheader(str(path))
            ann = wfdb.rdann(str(path), "atr")
            metadata[rid] = {
                "signal_file": str(path),
                "fs": float(header.fs),
                "n_samples": int(header.sig_len),
                "leads": tuple(header.sig_name),
                "anomaly_ratio": beat_anomaly_ratio(ann.symbol),
                "comments": "; ".join(c.strip() for c in header.comments or []),
            }
        return metadata

    def prepare_metadata(self) -> str:
        """Writes the ``records`` table CSV to ``metadata_cache_dir``.

        The file is rewritten only when its content changes.

        Returns:
            str: A 10-character content hash of the CSV, used in the cache key.

        Examples:
            >>> ds.prepare_metadata()  # doctest: +SKIP
            '3f1c0a9b2e'
        """
        rows = [
            {
                "patient_id": rid,
                "record_id": rid,
                "signal_file": meta["signal_file"],
                "fs": meta["fs"],
                "n_samples": meta["n_samples"],
                "leads": ",".join(meta["leads"]),
                "anomaly_ratio": meta["anomaly_ratio"],
                "comments": meta["comments"],
            }
            for rid, meta in self.record_metadata.items()
        ]
        content = pd.DataFrame(rows).to_csv(index=False, lineterminator="\n")
        self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.metadata_cache_dir / MITBIH_METADATA_FILE
        if not csv_path.is_file() or csv_path.read_text(encoding="utf-8") != content:
            tmp_path = csv_path.with_name(csv_path.name + ".tmp")
            tmp_path.write_text(content, encoding="utf-8", newline="")
            os.replace(tmp_path, csv_path)
        return hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]

    def load_table(self, table_name: str) -> dd.DataFrame:
        """Loads a table from ``metadata_cache_dir`` instead of ``root``.

        Args:
            table_name: Table name from the YAML config (``"records"``).

        Returns:
            dd.DataFrame: The table loaded by :meth:`BaseDataset.load_table`.

        Examples:
            >>> ds.load_table("records").compute()  # doctest: +SKIP
        """
        data_root = self.root
        self.root = str(self.metadata_cache_dir)
        try:
            return super().load_table(table_name)
        finally:
            self.root = data_root

    @property
    def default_task(self) -> MITBIHAnomalyDetection:
        """Default task: :class:`~pyhealth.tasks.MITBIHAnomalyDetection`.

        Returns:
            MITBIHAnomalyDetection: Task with MedTsLLM defaults (125 Hz,
            128-sample windows).

        Examples:
            >>> task = ds.default_task  # doctest: +SKIP
        """
        from pyhealth.tasks.mitbih import MITBIHAnomalyDetection

        return MITBIHAnomalyDetection()

    def anomaly_ratio_split(
        self,
        mode: Literal["paper", "fraction"] = "paper",
        **kwargs: Any,
    ) -> MITBIHAnomalyRatioSplit:
        """Creates a :class:`MITBIHAnomalyRatioSplit` for this dataset.

        Args:
            mode: ``"paper"`` (MedTsLLM 6/4 MLII+V1 selection) or
                ``"fraction"``.
            **kwargs: Other :class:`MITBIHAnomalyRatioSplit` parameters, e.g.
                ``train_frac=0.8``.

        Returns:
            MITBIHAnomalyRatioSplit: Split with this dataset's record metadata.

        Examples:
            >>> split = ds.anomaly_ratio_split()  # doctest: +SKIP
            >>> split.select_records()  # doctest: +SKIP
            (['115', '122', '230', '112', '121', '101'], ['208', '212', '231', '217'])
        """
        return MITBIHAnomalyRatioSplit(
            anomaly_ratios={
                r: m["anomaly_ratio"] for r, m in self.record_metadata.items()
            },
            leads={r: m["leads"] for r, m in self.record_metadata.items()},
            mode=mode,
            **kwargs,
        )
