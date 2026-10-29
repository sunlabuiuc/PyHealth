# Authors: Aritra Saha and Ugochukwu Njoku.
# Survival Mixture Density Networks, Han et al., MLHC 2022.
# https://proceedings.mlr.press/v182/han22a.html
"""Local Rotterdam/GBSG benchmark ingestion and explicit HDF5 preparation."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from pyhealth.tasks.survival_time_to_event import GBSG_FEATURES, _gbsg_values

from .base_dataset import BaseDataset

_COLUMNS = ("source_id", "source_group", "source_row", *GBSG_FEATURES, "time", "event")

# The Survival MDN authors' released GBSG files store time / max_time + 0.001,
# and their data loader adds another 0.001 before training and evaluation.
GBSG_PAPER_TIME_OFFSET = 0.002
_RELEASED_FILE_OFFSET = 0.001
_SPLITS = ("train", "valid", "test")


def _read_gbsg_csv(root: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return source IDs, features [N, 7] and outcomes [N, 2] from gbsg.csv."""
    path = Path(root).expanduser() / "gbsg.csv"
    ids, features, outcomes = [], [], []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            x, time, event = _gbsg_values(row, row["source_id"])
            ids.append(row["source_id"])
            features.append(x)
            outcomes.append((time, event))
    if not ids:
        raise ValueError("GBSG CSV must contain at least one source row")
    if any(not source_id.strip() for source_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("GBSG source IDs must be nonempty and unique")
    return ids, np.asarray(features, dtype=float), np.asarray(outcomes, dtype=float)


def gbsg_paper_time_transform(root: str | Path) -> dict:
    """Time conversion used by the Survival MDN authors for GBSG.

    Their released split files store each time divided by the largest time
    in the full 2,232-record cohort, plus 0.001, and their loader adds a
    further 0.001. Passing the result to
    :class:`~pyhealth.tasks.SurvivalTimeToEventGBSG` reproduces those times.
    The divisor uses every record, including held-out ones, exactly as the
    authors did; use it only for the paper-reproduction protocol.

    Args:
        root: Directory containing gbsg.csv from :func:`prepare_gbsg_csv`.

    Returns:
        ``{"time_divisor": max_time_in_months, "time_offset": 0.002}``.

    Examples:
        >>> from pyhealth.datasets import gbsg_paper_time_transform
        >>> transform = gbsg_paper_time_transform("/data/gbsg_csv")  # doctest: +SKIP
        >>> transform["time_offset"]  # doctest: +SKIP
        0.002
    """
    _, _, outcomes = _read_gbsg_csv(root)
    return {
        "time_divisor": float(outcomes[:, 0].max()),
        "time_offset": GBSG_PAPER_TIME_OFFSET,
    }


def gbsg_released_split_ids(
    root: str | Path,
    released_dir: str | Path,
    repeat: int,
    tolerance: float = 1e-6,
) -> dict[str, list[str]]:
    """Map one of the authors' released GBSG splits to source IDs.

    The Survival MDN authors released ten train/validation/test splits as
    ``{train,valid,test}_{repeat}.npz`` files (``arr_0``: features,
    ``arr_1``: ``[time, event]`` in their time units). Those files carry no
    record identifiers, so each released row is matched to a source row by
    its seven features, its event and its converted time. Every source row
    must be used exactly once across the three files.

    Two pairs of GBSG source records are identical in every recorded value.
    Each such pair is assigned in source order; because the records are
    identical, the choice does not change any split's data.

    The released files come from the authors' repository, which has no
    license; download them yourself. This function only reads local files.

    Args:
        root: Directory containing gbsg.csv from :func:`prepare_gbsg_csv`.
        released_dir: Directory containing the released ``.npz`` files.
        repeat: Split number, 1 to 10 in the authors' release.
        tolerance: Largest allowed absolute difference for features and
            converted times. Default 1e-6.

    Returns:
        ``{"train": [...], "valid": [...], "test": [...]}`` source IDs, in
        the row order of each released file.

    Raises:
        ValueError: If a released row matches no unused source row, or the
            three files do not cover every source row exactly once.

    Examples:
        >>> from pyhealth.datasets import gbsg_released_split_ids
        >>> ids = gbsg_released_split_ids(  # doctest: +SKIP
        ...     "/data/gbsg_csv", "/data/Survival-MDN/data/gbsg", repeat=1
        ... )
        >>> [len(ids[s]) for s in ("train", "valid", "test")]  # doctest: +SKIP
        [1785, 223, 224]
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if (
        isinstance(repeat, bool)
        or not isinstance(repeat, (int, np.integer))
        or not 1 <= repeat <= 10
    ):
        raise ValueError("repeat must be an integer from 1 through 10")
    ids, features, outcomes = _read_gbsg_csv(root)
    released_times = outcomes[:, 0] / outcomes[:, 0].max() + _RELEASED_FILE_OFFSET
    used = np.zeros(len(ids), dtype=bool)
    memberships = {}
    for split in _SPLITS:
        path = Path(released_dir).expanduser() / f"{split}_{repeat}.npz"
        with np.load(path, allow_pickle=False) as data:
            x, y = np.asarray(data["arr_0"], float), np.asarray(data["arr_1"], float)
        if x.ndim != 2 or x.shape[1] != 7 or y.shape != (len(x), 2):
            raise ValueError(f"{path.name}: expected arr_0 [N, 7] and arr_1 [N, 2]")
        members = []
        for row in range(len(x)):
            match = (
                ~used
                & (np.abs(features - x[row]).max(axis=1) <= tolerance)
                & (outcomes[:, 1] == y[row, 1])
                & (np.abs(released_times - y[row, 0]) <= tolerance)
            )
            candidates = np.flatnonzero(match)
            if candidates.size == 0:
                raise ValueError(f"{path.name} row {row}: no matching source record")
            used[candidates[0]] = True
            members.append(ids[candidates[0]])
        memberships[split] = members
    if not used.all():
        raise ValueError(
            f"Released repeat {repeat} leaves {int((~used).sum())} source rows unused"
        )
    return memberships


def prepare_gbsg_csv(source: str | Path, output_dir: str | Path) -> dict:
    """Convert a local DeepSurv GBSG HDF5 release to CSV and provenance JSON.

    Preserves the source's train/test groups as metadata, concatenates their
    rows without deduplication, and creates stable IDs such as ``train_42``.
    Features keep their original column order, x0 through x6. Times stay in
    source months; no split, normalization, offset or imputation is applied.
    The full public release has 2,232 records, but smaller fixtures are allowed.
    This function performs no download and refuses to overwrite existing files.
    HDF5 support requires the optional ``h5py`` package.

    Args:
        source: Local HDF5 file with train/test groups, each containing x, t, e.
        output_dir: Destination directory for gbsg.csv and gbsg-source.json.

    Returns:
        JSON-serializable provenance, also written beside the CSV.

    Examples:
        >>> from pyhealth.datasets import prepare_gbsg_csv
        >>> metadata = prepare_gbsg_csv(  # doctest: +SKIP
        ...     "/data/gbsg_cancer_train_test.h5", "/data/gbsg_csv"
        ... )
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("GBSG HDF5 preparation requires h5py; install h5py") from exc

    source, output_dir = Path(source).expanduser(), Path(output_dir).expanduser()
    csv_path, metadata_path = output_dir / "gbsg.csv", output_dir / "gbsg-source.json"
    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError("Use a new output directory; GBSG output already exists")
    rows, counts = [], {}
    with h5py.File(source, "r") as handle:
        for group in ("train", "test"):
            if group not in handle or not all(
                k in handle[group] for k in ("x", "t", "e")
            ):
                raise ValueError(
                    f"GBSG source needs {group}/x, {group}/t and {group}/e"
                )
            x, times, events = (np.asarray(handle[group][k]) for k in ("x", "t", "e"))
            if x.ndim != 2:
                raise ValueError(f"GBSG {group}: expected x[N,7]")
            n = len(x)
            if (
                n == 0
                or x.shape != (n, 7)
                or times.shape != (n,)
                or events.shape != (n,)
            ):
                raise ValueError(f"GBSG {group}: expected nonempty x[N,7], t[N], e[N]")
            counts[group] = n
            for i in range(n):
                source_id = f"{group}_{i}"
                row = dict(zip(GBSG_FEATURES, x[i]))
                row.update(time=times[i], event=events[i])
                features, time, event = _gbsg_values(row, source_id)
                rows.append(
                    dict(
                        zip(
                            _COLUMNS, (source_id, group, i, *features, time, int(event))
                        )
                    )
                )
    metadata = {
        "format_version": 1,
        "dataset": "DeepSurv Rotterdam/GBSG benchmark",
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_groups": counts,
        "rows": len(rows),
        "feature_order": list(GBSG_FEATURES),
        "time_unit": "months",
        "event_definition": "1 = recurrence-free-survival event; 0 = censored",
        "identity": "Surrogate source group/row IDs; original patient IDs unavailable",
        "protocol": "Unscaled source rows; source groups are not the MDN released splits",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    metadata["csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


class GBSGDataset(BaseDataset):
    """Read a local CSV prepared from the pooled Rotterdam/GBSG benchmark.

    The full release contains 2,232 records and seven features, unlike the
    stand-alone 686-person GBSG2 dataset. Source group/row identifiers are
    stable surrogate IDs. Equal feature/outcome values are retained, while
    duplicate IDs are rejected. Clinical values and times are not modified.
    No network request is made by this loader.

    Args:
        root: Directory containing gbsg.csv, produced by prepare_gbsg_csv.
        tables: Optional table selection; only ["gbsg"] is supported.
        dataset_name: Optional cache/display name, default "gbsg".
        config_path: Optional YAML configuration using the same column contract.
        **kwargs: BaseDataset options, including cache_dir and num_workers.

    Examples:
        >>> from pyhealth.datasets import GBSGDataset
        >>> dataset = GBSGDataset(  # doctest: +SKIP
        ...     root="/data/gbsg_csv", cache_dir="/data/cache"
        ... )
        >>> samples = dataset.set_task()  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        tables = ["gbsg"] if tables is None else tables
        if tables != ["gbsg"]:
            raise ValueError('GBSGDataset supports only tables=["gbsg"]')
        super().__init__(
            root=str(Path(root).expanduser().resolve()),
            tables=tables,
            dataset_name=dataset_name or "gbsg",
            config_path=config_path or str(Path(__file__).parent / "configs/gbsg.yaml"),
            **kwargs,
        )
        table = self.config.tables.get("gbsg")
        if (
            table is None
            or table.patient_id != "source_id"
            or table.timestamp is not None
            or not set(_COLUMNS[1:]).issubset(table.attributes)
        ):
            raise ValueError(
                "GBSG config must use source_id, no timestamp, and all source attributes"
            )
        path = Path(self.root) / table.file_path
        seen = set()
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"GBSG CSV missing columns: {sorted(missing)}")
            for row in reader:
                source_id = row["source_id"]
                if not source_id or source_id in seen:
                    raise ValueError(
                        f"GBSG source_id must be nonempty and unique: {source_id!r}"
                    )
                group = row["source_group"]
                try:
                    index = int(row["source_row"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"GBSG {source_id}: invalid source_row") from exc
                if (
                    group not in ("train", "test")
                    or index < 0
                    or source_id != f"{group}_{index}"
                ):
                    raise ValueError(f"GBSG {source_id}: source group/row ID mismatch")
                _gbsg_values(row, source_id)
                seen.add(source_id)
        if not seen:
            raise ValueError("GBSG CSV must contain at least one source row")

    @property
    def default_task(self):
        """Duration/event task, retaining source features and months."""
        from pyhealth.tasks import SurvivalTimeToEventGBSG

        return SurvivalTimeToEventGBSG()
