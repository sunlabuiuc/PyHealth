"""METABRIC breast cancer survival dataset.

The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC)
cohort, in the nine-feature survival version popularised by DeepSurv
(Katzman et al., 2018) and used by SODEN (Tang et al., 2022) and Survival
Mixture Density Networks (Han et al., 2022).
"""

from __future__ import annotations

import io
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset
from .splitter import SubjectSplit

logger = logging.getLogger(__name__)

#: Feature columns of ``metabric.csv``, in the order of the source ``.npz``
#: arrays: four gene-expression values, age at diagnosis, then four binary
#: clinical indicators. :class:`~pyhealth.tasks.MetabricSurvivalPrediction`
#: keeps its own copy of these names, so the dataset and the task do not
#: import from each other.
METABRIC_FEATURES: tuple[str, ...] = (
    "mki67",
    "egfr",
    "pgr",
    "erbb2",
    "age_at_diagnosis",
    "hormone_treatment",
    "radiotherapy",
    "chemotherapy",
    "er_positive",
)
_NUM_CONTINUOUS = 5

#: Number of published train/validation/test splits.
NUM_PAPER_SPLITS = 10
_PARTS: tuple[str, ...] = ("train", "valid", "test")
# The source files store time as months / 365.25 (see ``prepare_metabric``).
_SOURCE_TIME_SCALE = 365.25
#: Byte-identical copies of the released split files, tried in order. Both are
#: pinned to a commit so the download cannot change: the Survival MDN
#: repository, then SODEN's (MIT licensed), whose files the paper reuses.
_SOURCE_URLS: tuple[str, ...] = (
    (
        "https://raw.githubusercontent.com/XintianHan/Survival-MDN/"
        "71c61f649f3c8784fe13b092b38e0cbb3a184dbd/data/metabric"
    ),
    (
        "https://raw.githubusercontent.com/jiaqima/SODEN/"
        "3d10ba6c8f4526ecf8ed2e04937518d1820332aa/data/metabric"
    ),
)
_CSV_NAME = "metabric.csv"


def _npz_name(part: str, split_id: int) -> str:
    return f"{part}_{split_id}.npz"


def _read_npz(source: Path | str, name: str, download: bool) -> dict[str, np.ndarray]:
    if download:
        errors = []
        for base in _SOURCE_URLS:
            url = f"{base}/{name}"
            try:
                logger.info("Downloading %s", url)
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = io.BytesIO(response.read())
            except OSError as error:  # URLError and HTTPError are OSErrors
                logger.warning("Could not download %s: %s", url, error)
                errors.append(f"{url}: {error}")
                continue
            with np.load(payload) as data:
                return {key: data[key] for key in data.files}
        raise OSError(
            f"Could not download {name} from any source:\n  "
            + "\n  ".join(errors)
            + "\nDownload the METABRIC split files manually and put them in root."
        )
    path = Path(source) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put the Survival MDN METABRIC files "
            f"({{train,valid,test}}_{{1..{NUM_PAPER_SPLITS}}}.npz) there, or pass "
            "download=True."
        )
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _row_keys(raw: np.ndarray, labels: np.ndarray) -> list[bytes]:
    """Byte keys identifying a patient by its raw features and labels."""
    rows = np.ascontiguousarray(np.hstack([raw, labels]).astype(np.float32))
    return [row.tobytes() for row in rows]


def _validate_values(name: str, raw: np.ndarray, labels: np.ndarray) -> None:
    """Checks one split file's values before they are converted.

    Raises:
        ValueError: If a value is not finite, a binary feature or an event is
            not exactly 0 or 1, or a time is negative.
    """
    if not (np.isfinite(raw).all() and np.isfinite(labels).all()):
        raise ValueError(f"{name}: features and labels must be finite")
    binary = raw[:, _NUM_CONTINUOUS:]
    if not np.isin(binary, (0.0, 1.0)).all():
        raise ValueError(
            f"{name}: {', '.join(METABRIC_FEATURES[_NUM_CONTINUOUS:])} must be "
            "exactly 0 or 1"
        )
    if not np.isin(labels[:, 1], (0.0, 1.0)).all():
        raise ValueError(f"{name}: event values must be exactly 0 or 1")
    if (labels[:, 0] < 0).any():
        raise ValueError(f"{name}: times must be non-negative")


def prepare_metabric(
    root: str,
    source_dir: str | None = None,
    download: bool = False,
) -> Path:
    """Builds ``metabric.csv`` from the Survival MDN / SODEN split files.

    The Survival MDN repository (https://github.com/XintianHan/Survival-MDN)
    ships METABRIC as ten train/validation/test splits,
    ``data/metabric/{train,valid,test}_{1..10}.npz``, the same files SODEN
    released. Each file holds three unnamed arrays: ``arr_0`` (features
    standardised with the training split's statistics), ``arr_1`` (time,
    event) and ``arr_2`` (raw features). The splits partition the same 1,904
    patients in different ways.

    This function keeps the raw features, so that PyHealth fits any scaling
    on the training patients itself, and gives each patient a stable ID
    (``MB0001`` ...) in the order of split 1 (train, then validation, then
    test). Patients are matched across splits by their raw features and
    labels, which are unique in the released files. The split membership of
    every patient is kept as columns ``split_1`` ... ``split_10``.

    Values are not rounded. The continuous features are written with the
    shortest decimal that reads back as the same ``float32`` value of the
    source files, and the binary indicators and events as integers. Time is
    stored in months, the unit of the DeepSurv release: the source files
    store months divided by 365.25, so ``duration_months`` is that value
    times 365.25, in full ``float64`` precision.

    Args:
        root: Directory to write ``metabric.csv`` to.
        source_dir: Directory holding the 30 ``.npz`` files. Defaults to
            ``root``. Ignored when ``download`` is True.
        download: Fetch the ``.npz`` files from the Survival MDN repository
            instead of reading them from disk.

    Returns:
        Path to the written ``metabric.csv``.

    Raises:
        FileNotFoundError: If a split file is missing.
        ValueError: If the files are malformed, a value is invalid (a
            non-finite value, a binary feature or event other than exactly 0
            or 1, a negative time), two patients cannot be told
            apart, or a split does not cover the same patients as split 1.

    Examples:
        >>> from pyhealth.datasets.metabric import prepare_metabric
        >>> prepare_metabric("/data/metabric", download=True)  # doctest: +SKIP
        PosixPath('/data/metabric/metabric.csv')
    """
    root_path = Path(root)
    source = Path(source_dir) if source_dir is not None else root_path
    num_features = len(METABRIC_FEATURES)

    raw_blocks, label_blocks, ids = [], [], {}
    membership: dict[int, dict[bytes, str]] = {}
    for split_id in range(1, NUM_PAPER_SPLITS + 1):
        membership[split_id] = {}
        for part in _PARTS:
            data = _read_npz(source, _npz_name(part, split_id), download)
            if not {"arr_1", "arr_2"} <= set(data):
                raise ValueError(
                    f"{_npz_name(part, split_id)} must contain arr_1 and arr_2"
                )
            raw, labels = data["arr_2"], data["arr_1"]
            if raw.ndim != 2 or raw.shape[1] != num_features:
                raise ValueError(
                    f"{_npz_name(part, split_id)}: expected {num_features} "
                    f"features, got shape {raw.shape}"
                )
            if labels.ndim != 2 or labels.shape != (raw.shape[0], 2):
                raise ValueError(
                    f"{_npz_name(part, split_id)}: labels must have shape "
                    f"({raw.shape[0]}, 2), got {labels.shape}"
                )
            _validate_values(_npz_name(part, split_id), raw, labels)
            keys = _row_keys(raw, labels)
            for key in keys:
                if key in membership[split_id]:
                    raise ValueError(
                        f"split {split_id}: two patients share the same features "
                        "and labels, so they cannot be matched across splits"
                    )
                membership[split_id][key] = part
            if split_id == 1:
                for key in keys:
                    ids[key] = f"MB{len(ids) + 1:04d}"
                raw_blocks.append(raw)
                label_blocks.append(labels)
        if set(membership[split_id]) != set(ids):
            raise ValueError(
                f"split {split_id} does not cover the same patients as split 1"
            )

    raw = np.vstack(raw_blocks)
    labels = np.vstack(label_blocks)
    keys = _row_keys(raw, labels)
    raw = raw.astype(np.float32)
    table = pd.DataFrame({"patient_id": [ids[key] for key in keys]})
    for index, column in enumerate(METABRIC_FEATURES):
        if index < _NUM_CONTINUOUS:
            # Shortest decimal that round-trips the float32 source value.
            table[column] = [
                float(np.format_float_positional(value, unique=True))
                for value in raw[:, index]
            ]
        else:
            table[column] = raw[:, index].astype(int)
    table["duration_months"] = labels[:, 0].astype(np.float64) * _SOURCE_TIME_SCALE
    table["event"] = labels[:, 1].astype(int)
    for split_id in range(1, NUM_PAPER_SPLITS + 1):
        table[f"split_{split_id}"] = [membership[split_id][key] for key in keys]

    root_path.mkdir(parents=True, exist_ok=True)
    out = root_path / _CSV_NAME
    table.to_csv(out, index=False)
    logger.info("Wrote %d METABRIC patients to %s", len(table), out)
    return out


class MetabricDataset(BaseDataset):
    """METABRIC breast cancer survival dataset (1,904 patients).

    Each patient is one row with four gene-expression values (MKI67, EGFR,
    PGR, ERBB2), age at diagnosis, four binary clinical indicators (hormone
    treatment, radiotherapy, chemotherapy, ER positive), the follow-up time
    in months and an event indicator (1 death observed,
    0 censored). About 42% of patients are censored.

    This is the nine-feature version released by DeepSurv (Katzman et al.,
    2018) and used, with ten fixed train/validation/test splits, by SODEN
    (Tang et al., 2022) and Survival Mixture Density Networks (Han et al.,
    2022). The dataset reads ``metabric.csv`` from ``root``. If it is
    missing, the dataset builds it with :func:`prepare_metabric` from the
    Survival MDN split files in ``root``, or downloads them first when
    ``download=True``. :meth:`paper_split` gives one of the ten published
    splits.

    Papers:
        Curtis C, Shah SP, Chin SF, et al. The genomic and transcriptomic
        architecture of 2,000 breast tumours reveals novel subgroups.
        Nature. 2012;486:346-352.

        Han X, Goldstein M, Ranganath R. Survival Mixture Density Networks.
        MLHC 2022, PMLR 182:224-248.
        https://proceedings.mlr.press/v182/han22a.html

    Getting the data:
        - Source: the Survival MDN split files,
          https://github.com/XintianHan/Survival-MDN
        - Access: open, no account needed.
        - Download: pass ``download=True``.
        - ``root``: the folder holding ``metabric.csv``, or the split files that
          :func:`prepare_metabric` builds it from.

    Args:
        root: Directory holding ``metabric.csv`` or the split files.
        tables: Tables to load. Defaults to ``["metabric"]``, the only one.
        dataset_name: Name of the dataset. Defaults to ``"metabric"``.
        config_path: Path to the YAML config. Defaults to the bundled
            ``configs/metabric.yaml``.
        download: Download the split files if ``metabric.csv`` is missing.
        **kwargs: Passed to :class:`~pyhealth.datasets.BaseDataset`.

    Raises:
        FileNotFoundError: If neither ``metabric.csv`` nor the split files are
            in ``root`` and ``download`` is False.

    Examples:
        >>> from pyhealth.datasets import MetabricDataset
        >>> from pyhealth.tasks import MetabricSurvivalPrediction
        >>> dataset = MetabricDataset(root="/data/metabric", download=True)
        ... # doctest: +SKIP
        >>> train, valid, test = dataset.set_task(  # doctest: +SKIP
        ...     MetabricSurvivalPrediction(), split=dataset.paper_split(1)
        ... )
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        csv_path = Path(root) / _CSV_NAME
        if not csv_path.exists():
            logger.info("%s not found, building it from the split files", csv_path)
            prepare_metabric(root, download=download)
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "metabric.yaml")
        super().__init__(
            root=root,
            tables=tables or ["metabric"],
            dataset_name=dataset_name or "metabric",
            config_path=config_path,
            **kwargs,
        )

    def paper_split(self, split_id: int = 1) -> SubjectSplit:
        """Returns one of the ten published train/validation/test splits.

        The split is a :class:`~pyhealth.datasets.SubjectSplit` holding the
        patient IDs of the training, validation and test parts of split
        ``split_id``, as released with SODEN and Survival MDN, so results are
        comparable with the papers' tables. With it, ``set_task`` fits every
        processor (for example feature standardisation) on that split's
        training patients only, as the authors did.

        Args:
            split_id: The split number, 1 to 10.

        Returns:
            A :class:`~pyhealth.datasets.SubjectSplit` for
            ``set_task(task, split=...)``.

        Raises:
            ValueError: If ``split_id`` is out of range, or ``metabric.csv``
                has no column for it or an unknown part name in it.
        """
        if not 1 <= int(split_id) <= NUM_PAPER_SPLITS:
            raise ValueError(
                f"split_id must be between 1 and {NUM_PAPER_SPLITS}, got {split_id}"
            )
        column = f"split_{int(split_id)}"
        table = pd.read_csv(
            Path(self.root) / _CSV_NAME,
            usecols=lambda c: c in {"patient_id", column},
            dtype=str,
        )
        if column not in table.columns:
            raise ValueError(f"metabric.csv has no column {column!r}")
        unknown = set(table[column]) - set(_PARTS)
        if unknown:
            raise ValueError(f"{column} has unknown parts {sorted(unknown)}")
        return SubjectSplit(
            subjects=tuple(
                tuple(table.loc[table[column] == part, "patient_id"]) for part in _PARTS
            )
        )
