"""VSAC value set dataset for PyHealth.

Author: Bryan Chance (NetID: bchance2)
Paper: Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value
    Set Authoring (Mukherjee et al., 2026)
Paper link: https://arxiv.org/abs/2604.14616
Description: Loads value sets downloaded from the NLM Value Set Authority
    Center (VSAC) FHIR API. Each value set is a "patient" keyed by its OID,
    with one ``valuesets`` event (title, publisher, split) and one ``codes``
    event per member code. Expansion parsing and type inference are ported
    from the authors' code (https://github.com/mukhes3/RASC).
"""

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..tasks.value_set_completion import normalize_code_system
from .base_dataset import BaseDataset
from .splitter import Split

logger = logging.getLogger(__name__)

VALUESETS_FILE = "vsac-valuesets.csv"
CODES_FILE = "vsac-codes.csv"
MANIFEST_FILE = "split_manifest.jsonl"
VALUESET_COLUMNS = [
    "oid",
    "title",
    "description",
    "publisher",
    "type",
    "version",
    "status",
    "split",
]
CODE_COLUMNS = ["oid", "code", "system", "display", "rank"]

# From create_dataset/build_dataset.py in the RASC repository.
_SYSTEM_TYPES = {
    "SNOMED-CT": "Condition/Clinical",
    "ICD-10-CM": "Condition/Diagnosis",
    "RxNorm": "Medication",
    "LOINC": "Lab/Observation",
    "CPT": "Procedure",
    "ICD-10-PCS": "Procedure",
    "HCPCS": "Administrative",
}


def infer_value_set_type(systems: Sequence[str]) -> str:
    """Infers a value set's clinical type from its first code system.

    Args:
        systems: Code system URIs of the value set, sorted.

    Returns:
        A type label such as ``"Medication"``, or ``"Other/Unknown"``.

    Examples:
        >>> infer_value_set_type(["http://www.nlm.nih.gov/research/umls/rxnorm"])
        'Medication'
    """
    primary = normalize_code_system(systems[0]) if systems else "OTHER"
    return _SYSTEM_TYPES.get(primary, "Other/Unknown")


@dataclass(frozen=True)
class ValueSetSplit(Split):
    """Splits value sets by a fixed OID-to-split assignment.

    Used with ``BaseDataset.set_task(task, split=...)`` to keep the published
    benchmark splits instead of drawing random ones. Value sets whose split is
    not one of ``parts`` are left out.

    Args:
        assignments: Value set OID to split name.
        parts: Split names in output order, training part first.

    Examples:
        >>> split = ValueSetSplit({"a": "train", "b": "test"}, parts=("train", "test"))
        >>> split.split_indices({"a": [0, 1], "b": [2]})
        [array([0, 1]), array([2])]
    """

    assignments: Mapping[str, str]
    parts: Sequence[str] = ("train", "val", "test")
    _digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        payload = json.dumps(sorted(self.assignments.items())).encode()
        object.__setattr__(self, "_digest", hashlib.sha256(payload).hexdigest())

    def to_dict(self) -> dict:
        """JSON-serialisable description, used in cache keys and metadata."""
        return {
            "kind": "value_set",
            "parts": list(self.parts),
            "assignments_sha256": self._digest,
        }

    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> list[np.ndarray]:
        """Groups sample indices by each value set's assigned split.

        Args:
            patient_to_index: Value set OID to the indices of its samples.

        Returns:
            One sorted ``int64`` array of sample indices per part.
        """
        position = {name: i for i, name in enumerate(self.parts)}
        buckets: list[list[int]] = [[] for _ in self.parts]
        for oid, indices in patient_to_index.items():
            part = position.get(self.assignments.get(oid, ""))
            if part is not None:
                buckets[part].extend(indices)
        return [np.array(sorted(b), dtype=np.int64) for b in buckets]


def _read_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {row["oid"]: row for row in rows}


def _parse_expansion(resource: dict, manifest_row: dict) -> tuple[dict, list[dict]]:
    oid = resource["id"]
    expansion = resource.get("expansion", {})
    contains = [c for c in expansion.get("contains", []) if c.get("code")]
    systems = sorted({c["system"] for c in contains if c.get("system")})
    valueset = {
        "oid": oid,
        "title": (resource.get("title") or resource.get("name") or "").strip(),
        "description": (resource.get("description") or "").strip(),
        "publisher": manifest_row.get("publisher") or resource.get("publisher", ""),
        "type": manifest_row.get("type") or infer_value_set_type(systems),
        "version": resource.get("version", ""),
        "status": resource.get("status", ""),
        "split": manifest_row.get("split", ""),
    }
    codes = [
        {
            "oid": oid,
            "code": c["code"],
            "system": c.get("system", ""),
            "display": c.get("display") or c["code"],
            "rank": rank,
        }
        for rank, c in enumerate(contains)
    ]
    return valueset, codes


class VSACDataset(BaseDataset):
    """Clinical value sets from the NLM Value Set Authority Center (VSAC).

    A value set is a named list of codes (SNOMED-CT, ICD-10-CM, RxNorm, LOINC,
    and others) that defines a clinical concept for quality measures. VSAC
    hosts over 10,000 of them under a free UMLS license, which forbids
    redistribution, so each user downloads their own copy.

    The dataset treats each value set as a patient keyed by OID, with two
    event tables:

    - ``valuesets``: one row per value set (title, description, publisher,
      type, version, status, split).
    - ``codes``: one row per member code (code, system, display, rank).
      ``rank`` is the position in the VSAC expansion, which lets a task
      reproduce the 1,000-code page limit of an unpaged ``$expand`` call.

    ``root`` must hold ``expansions/<oid>.json`` files, each the ValueSet
    resource returned by ``ValueSet/<oid>/$expand`` (the layout written by the
    RASC ``download_vsac.py`` script), and optionally ``split_manifest.jsonl``
    with ``oid``, ``split``, ``type``, and ``publisher`` per line. With a
    manifest, only its OIDs are loaded and its fields take precedence.
    The first run flattens these into two CSV files under ``root``.

    Args:
        root: Directory with ``expansions/`` and an optional manifest.
        tables: Additional tables to load beyond ``valuesets`` and ``codes``.
        dataset_name: Name of the dataset. Defaults to ``"vsac"``.
        config_path: Path to the YAML config. Defaults to the bundled
            ``configs/vsac.yaml``.
        **kwargs: Passed to :class:`~pyhealth.datasets.BaseDataset`.

    Raises:
        FileNotFoundError: If ``root`` has neither the flattened CSV files nor
            an ``expansions/`` directory.

    Examples:
        >>> from pyhealth.datasets import VSACDataset
        >>> dataset = VSACDataset(root="/path/to/vsac_data")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> split = dataset.manifest_split()  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "vsac.yaml")
        if not all((Path(root) / f).exists() for f in (VALUESETS_FILE, CODES_FILE)):
            self.prepare_metadata(root)
        super().__init__(
            root=root,
            tables=["valuesets", "codes"] + (tables or []),
            dataset_name=dataset_name or "vsac",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Flattens VSAC expansion files into the two CSV tables.

        Expansions with fewer codes than ``expansion.total`` were saved from a
        single ``$expand`` page; they are kept and logged, since the authors'
        download script stores them that way.

        Args:
            root: Directory with ``expansions/`` and an optional manifest.

        Raises:
            FileNotFoundError: If ``root/expansions`` does not exist.

        Examples:
            >>> VSACDataset.prepare_metadata("/path/to/vsac_data")  # doctest: +SKIP
        """
        root_path = Path(root)
        exp_dir = root_path / "expansions"
        if not exp_dir.is_dir():
            raise FileNotFoundError(
                f"{exp_dir} not found. Download VSAC expansions first, e.g. with "
                "create_dataset/download_vsac.py from "
                "https://github.com/mukhes3/RASC."
            )
        manifest = _read_manifest(root_path / MANIFEST_FILE)
        paths = (
            [exp_dir / f"{oid}.json" for oid in manifest]
            if manifest
            else sorted(exp_dir.glob("*.json"))
        )

        valuesets, codes, missing, truncated = [], [], 0, 0
        for path in paths:
            if not path.exists():
                missing += 1
                continue
            resource = json.loads(path.read_text())
            valueset, members = _parse_expansion(
                resource, manifest.get(resource["id"], {})
            )
            total = resource.get("expansion", {}).get("total", len(members))
            truncated += total > len(members)
            valuesets.append(valueset)
            codes.extend(members)

        if missing:
            logger.warning(f"{missing} manifest OIDs have no expansion file.")
        if truncated:
            logger.warning(
                f"{truncated} expansions hold fewer codes than expansion.total "
                "(single $expand page)."
            )
        pd.DataFrame(valuesets, columns=VALUESET_COLUMNS).to_csv(
            root_path / VALUESETS_FILE, index=False
        )
        pd.DataFrame(codes, columns=CODE_COLUMNS).to_csv(
            root_path / CODES_FILE, index=False
        )
        logger.info(f"Wrote {len(valuesets)} value sets and {len(codes)} codes.")

    def manifest_split(
        self, parts: Sequence[str] = ("train", "val", "test")
    ) -> ValueSetSplit:
        """Returns the split recorded in the ``valuesets`` table.

        Args:
            parts: Split names to keep, training part first.

        Returns:
            A :class:`ValueSetSplit` for ``set_task(task, split=...)``.

        Examples:
            >>> train, val, test = dataset.set_task(  # doctest: +SKIP
            ...     task, split=dataset.manifest_split()
            ... )
        """
        frame = pd.read_csv(
            Path(self.root) / VALUESETS_FILE, dtype=str, keep_default_na=False
        )
        return ValueSetSplit(dict(zip(frame["oid"], frame["split"])), tuple(parts))
