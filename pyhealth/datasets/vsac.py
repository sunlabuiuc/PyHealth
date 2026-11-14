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

import json
import logging
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..tasks.value_set_completion import normalize_code_system
from .base_dataset import BaseDataset
from .splitter import SubjectSplit

logger = logging.getLogger(__name__)

VALUESETS_FILE = "vsac-valuesets.csv"
CODES_FILE = "vsac-codes.csv"
MANIFEST_FILE = "split_manifest.jsonl"
RESOURCES_DIR = "resources"
FHIR_BASE = "https://cts.nlm.nih.gov/fhir"
PAGE_SIZE = 1000  # VSAC's maximum codes per $expand response
_REQUEST_DELAY = 0.12  # seconds between requests, as in the RASC download script
_TIMEOUT = 60
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
# Dotted digits with an optional -<yyyymmdd> version. OIDs become file names.
_OID_PATTERN = re.compile(r"\d+(?:\.\d+)+(?:-\d{8})?")

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


def _get_json(
    session: requests.Session, path: str, params: dict[str, int] | None = None
) -> dict[str, Any]:
    """GETs one VSAC FHIR endpoint and returns its JSON body.

    Args:
        session: Session carrying the UMLS API key.
        path: Path under ``FHIR_BASE``.
        params: Query parameters.

    Returns:
        The decoded response.

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    response = session.get(f"{FHIR_BASE}/{path}", params=params, timeout=_TIMEOUT)
    response.raise_for_status()
    time.sleep(_REQUEST_DELAY)
    return response.json()


def _fetch_expansion(
    session: requests.Session, oid: str, paginate: bool
) -> dict[str, Any]:
    """Fetches a value set's ``$expand`` response, optionally every page.

    Args:
        session: Session carrying the UMLS API key.
        oid: Value set id; a dated id pins that version.
        paginate: Request further pages until ``expansion.total`` codes are
            held. ``False`` keeps the first page only, as the RASC download
            script does.

    Returns:
        The ValueSet resource, with all fetched codes in
        ``expansion.contains``.

    Raises:
        requests.HTTPError: On a non-2xx response.
        RuntimeError: If a page comes back empty before ``total`` is reached.
    """
    path = f"ValueSet/{oid}/$expand"
    resource = _get_json(session, path, {"offset": 0, "count": PAGE_SIZE})
    expansion = resource.setdefault("expansion", {})
    contains = expansion.setdefault("contains", [])
    total = expansion.get("total", len(contains))
    while paginate and len(contains) < total:
        page = _get_json(session, path, {"offset": len(contains), "count": PAGE_SIZE})
        batch = page.get("expansion", {}).get("contains", [])
        if not batch:
            raise RuntimeError(f"{oid}: empty page at offset {len(contains)}")
        contains.extend(batch)
    expansion["offset"] = 0
    return resource


def _write_json(path: Path, resource: dict[str, Any]) -> None:
    """Writes JSON by renaming a temporary file into place.

    An interrupted write leaves no partial file for a resumed download to skip.

    Args:
        path: Destination file.
        resource: JSON-serializable object.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(resource))
    os.replace(tmp, path)


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Reads ``split_manifest.jsonl`` into OID -> row; empty if the file is absent.

    Args:
        path: Manifest path.

    Returns:
        One row per OID with ``split`` and optionally ``type`` and ``publisher``.

    Raises:
        ValueError: If a line is not JSON or has no valid ``oid``.
    """
    if not path.exists():
        return {}
    rows = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not _OID_PATTERN.fullmatch(row["oid"]):
                raise ValueError(row["oid"])
            rows[row["oid"]] = row
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{number}: invalid manifest row") from error
    return rows


def _read_resource(path: Path) -> dict[str, Any]:
    """Loads one ValueSet JSON file, naming the file on failure.

    Args:
        path: File holding a FHIR ValueSet resource.

    Returns:
        The parsed resource.

    Raises:
        ValueError: If the file is not JSON or not a ValueSet object.
    """
    try:
        resource = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(resource, dict) or resource.get("resourceType") != "ValueSet":
        raise ValueError(f"{path}: not a FHIR ValueSet resource")
    return resource


def _parse_expansion(
    oid: str,
    resource: dict[str, Any],
    manifest_row: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Flattens one ``$expand`` response into a valuesets row and code rows.

    Args:
        oid: Value set id, taken from the file name.
        resource: The ``$expand`` response.
        manifest_row: The value set's manifest row, or ``{}``.
        metadata: The plain ``ValueSet/<oid>`` read, or ``{}``. VSAC omits
            ``purpose`` from ``$expand`` responses, so the description falls
            back to it.

    Returns:
        The ``valuesets`` row and one ``codes`` row per member code, with
        ``rank`` the position in the expansion.
    """
    expansion = resource.get("expansion", {})
    contains = [c for c in expansion.get("contains", []) if c.get("code")]
    systems = sorted({c["system"] for c in contains if c.get("system")})
    valueset = {
        "oid": oid,
        "title": (resource.get("title") or resource.get("name") or "").strip(),
        "description": (
            resource.get("description")
            or metadata.get("description")
            or metadata.get("purpose")
            or ""
        ).strip(),
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
    resource returned by ``ValueSet/<oid>/$expand``, as written by
    :meth:`download_expansions` or the RASC ``download_vsac.py`` script (which
    keeps only the first 1,000 codes). Optional files:

    - ``resources/<oid>.json``: the plain ValueSet read. ``description``
      falls back to its ``purpose`` text, which ``$expand`` omits.
    - ``split_manifest.jsonl``: ``oid``, ``split``, ``type``, and
      ``publisher`` per line. With a manifest, only its OIDs are loaded and
      its fields take precedence.

    The file name is the OID. The first run flattens these into two CSV files
    under ``root``. They are not refreshed, so delete them to rebuild after
    the expansions or the manifest change.

    Getting the data:
        - Source: https://vsac.nlm.nih.gov
        - Access: free UMLS license. Sign up at
          https://uts.nlm.nih.gov/uts/signup-login, then copy your API key from
          your UTS profile.
        - Download: :meth:`download_expansions` with ``api_key`` (or the
          ``UMLS_API_KEY`` environment variable set) writes the files ``root``
          needs.
        - ``root``: the folder holding ``expansions/``.

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
    def download_expansions(
        root: str,
        oids: Sequence[str],
        api_key: str | None = None,
        paginate: bool = True,
        metadata: bool = True,
        session: requests.Session | None = None,
    ) -> list[str]:
        """Downloads value sets from the VSAC FHIR API into ``root``.

        Writes ``expansions/<oid>.json`` and, with ``metadata``,
        ``resources/<oid>.json`` (the plain ValueSet read, which carries the
        ``purpose`` text that ``$expand`` omits). Files already on disk are
        skipped, so an interrupted download resumes.

        Args:
            root: Dataset directory.
            oids: Value set ids. Dated ids (``<oid>-<yyyymmdd>``) pin versions,
                e.g. the RASC release manifest's.
            api_key: UMLS API key. Defaults to the ``UMLS_API_KEY``
                environment variable.
            paginate: Fetch every page. ``False`` keeps only the first 1,000
                codes, reproducing the RASC benchmark's labels.
            metadata: Also fetch each value set's ValueSet read.
            session: HTTP session to use instead of one built from
                ``api_key``.

        Returns:
            The OIDs that failed.

        Raises:
            ValueError: If an OID is malformed, or no API key is given or set
                and no session is passed.

        Examples:
            >>> failed = VSACDataset.download_expansions(  # doctest: +SKIP
            ...     "vsac_data", ["2.16.840.1.113883.3.464.1003.103.12.1001"]
            ... )
        """
        if session is None:
            api_key = api_key or os.environ.get("UMLS_API_KEY")
            if not api_key:
                raise ValueError("Pass api_key or set UMLS_API_KEY.")
            session = requests.Session()
            session.auth = ("apikey", api_key)
            session.headers["Accept"] = "application/json"
        invalid = [oid for oid in oids if not _OID_PATTERN.fullmatch(oid)]
        if invalid:
            raise ValueError(f"Not VSAC OIDs: {invalid}")
        exp_dir = Path(root) / "expansions"
        meta_dir = Path(root) / RESOURCES_DIR
        exp_dir.mkdir(parents=True, exist_ok=True)
        if metadata:
            meta_dir.mkdir(parents=True, exist_ok=True)
        failed = []
        for oid in oids:
            exp_path, meta_path = exp_dir / f"{oid}.json", meta_dir / f"{oid}.json"
            try:
                if not exp_path.exists():
                    _write_json(exp_path, _fetch_expansion(session, oid, paginate))
                if metadata and not meta_path.exists():
                    resource = _get_json(session, f"ValueSet/{oid}")
                    resource.pop("compose", None)
                    _write_json(meta_path, resource)
            except (requests.RequestException, RuntimeError, ValueError) as error:
                logger.warning(f"{oid}: download failed ({error})")
                failed.append(oid)
        logger.info(f"Downloaded {len(oids) - len(failed)} of {len(oids)} value sets.")
        return failed

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Flattens VSAC expansion files into the two CSV tables.

        Expansions with fewer codes than ``expansion.total`` were saved from a
        single ``$expand`` page; they are kept and logged, since the authors'
        download script stores them that way.

        Args:
            root: Directory with ``expansions/`` and an optional manifest.

        Raises:
            FileNotFoundError: If ``root/expansions`` does not exist or holds
                no expansion to load.
            ValueError: If an expansion, resource, or manifest line is not
                valid JSON of the expected shape.

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
        meta_dir = root_path / RESOURCES_DIR

        valuesets, codes, missing, truncated, renamed = [], [], 0, 0, 0
        for path in paths:
            if not path.exists():
                missing += 1
                continue
            oid = path.stem
            resource = _read_resource(path)
            renamed += resource.get("id", oid) != oid
            meta_path = meta_dir / path.name
            metadata = _read_resource(meta_path) if meta_path.exists() else {}
            valueset, members = _parse_expansion(
                oid, resource, manifest.get(oid, {}), metadata
            )
            total = resource.get("expansion", {}).get("total", len(members))
            truncated += total > len(members)
            valuesets.append(valueset)
            codes.extend(members)

        if not valuesets:
            raise FileNotFoundError(f"No expansion files found in {exp_dir}.")
        if missing:
            logger.warning(f"{missing} manifest OIDs have no expansion file.")
        if renamed:
            logger.warning(
                f"{renamed} expansions have a resource id that differs from "
                "their file name; the file name is used as the OID."
            )
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
    ) -> SubjectSplit:
        """Returns the split recorded in the ``valuesets`` table.

        Value sets whose split is not in ``parts`` are left out.

        Args:
            parts: Split names to keep, training part first.

        Returns:
            A :class:`~pyhealth.datasets.SubjectSplit` of OIDs for
            ``set_task(task, split=...)``.

        Raises:
            ValueError: If no value set has a split, i.e. the CSV files were
                built without ``split_manifest.jsonl``.

        Examples:
            >>> train, val, test = dataset.set_task(  # doctest: +SKIP
            ...     task, split=dataset.manifest_split()
            ... )
        """
        frame = pd.read_csv(
            Path(self.root) / VALUESETS_FILE, dtype=str, keep_default_na=False
        )
        if not frame["split"].astype(bool).any():
            raise ValueError(
                f"No value set in {VALUESETS_FILE} has a split. Add "
                f"{MANIFEST_FILE} to root and delete the CSV files to rebuild."
            )
        return SubjectSplit([frame.loc[frame["split"] == part, "oid"] for part in parts])
