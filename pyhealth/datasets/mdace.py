"""
PyHealth dataset for MDACE, the MIMIC Documents Annotated with Code Evidence
corpus.

Dataset link:
    https://github.com/3mcloud/MDACE

Dataset paper: (please cite if you use this dataset)
    Cheng, Hua, et al. "MDACE: MIMIC Documents Annotated with Code
    Evidence." Proceedings of the 61st Annual Meeting of the Association
    for Computational Linguistics (ACL), 2023.

Span trimming follows the preprocessing of Edin et al. (``prepare_mdace.py``
in https://github.com/JoakimEdin/explainable-medical-coding):
    Edin, Joakim, et al. "An Unsupervised Approach to Achieve Supervised-Level
    Explainability in Healthcare Records." EMNLP 2024.

Contributors:
    Juri Moon (jmoon326)

Paper link: https://arxiv.org/abs/2406.08958
"""
import json
import logging
import os
import shutil
import string
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..tasks import MDACEEvidenceCoding
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

_PUNCTUATION = set(string.punctuation + "\n\t ")


def trim_span(span: tuple[int, int], text: str) -> tuple[int, int]:
    """Trims one punctuation/whitespace character at each end of a span.

    Reproduces ``trim_annotations`` from the authors' preprocessing exactly:
    the character *at* ``end`` and *at* ``start`` are checked once each.

    Args:
        span: ``(begin, end)`` character offsets from the MDACE annotation.
        text: Full note text.

    Returns:
        The trimmed ``(begin, end)`` pair.

    Example:
        >>> trim_span((4, 12), "Dx: (sepsis), fever")
        (5, 11)
    """
    start, end = span
    # The authors index text[end] unguarded; guard so a span ending at the
    # last character cannot raise.
    if end < len(text) and text[end] in _PUNCTUATION:
        end -= 1
    if start < len(text) and text[start] in _PUNCTUATION:
        start += 1
    return start, end


def load_annotations(ann_dir: Path) -> dict[tuple[str, str], dict]:
    """Reads MDACE JSON files and groups spans per (note, code).

    Args:
        ann_dir: Folder holding one ``<hadm_id>-ICD-<v>.json`` per admission.

    Returns:
        ``{(hadm_id, note_id): {"category", "description", "codes": {code:
        {"system", "spans"}}}}``, codes kept in first-seen order.

    Example:
        >>> from pathlib import Path
        >>> ann = load_annotations(
        ...     Path("test-resources/core/mdace/MDace/Inpatient/ICD-9/1.0")
        ... )
        >>> ann["900001", "11"]["codes"]["401.9"]
        {'system': 'ICD-9-CM', 'spans': [(19, 32)]}
    """
    notes: dict[tuple[str, str], dict] = {}
    for path in sorted(ann_dir.glob("*.json")):
        with open(path, encoding="utf8") as f:
            case = json.load(f)
        hadm_id = str(case["hadm_id"])
        for note in case["notes"]:
            codes: dict[str, dict] = defaultdict(
                lambda: {"system": None, "spans": []}
            )
            for ann in note["annotations"]:
                entry = codes[ann["code"]]
                entry["system"] = ann["code_system"]
                entry["spans"].append((ann["begin"], ann["end"]))
            notes[(hadm_id, str(note["note_id"]))] = {
                "category": note["category"],
                "description": note["description"],
                "codes": dict(codes),
            }
    return notes


def read_splits(splits_dir: str) -> dict[str, str]:
    """Maps hadm_id to split name from ``MDace-ev-{train,val,test}.csv``.

    Args:
        splits_dir: Folder with the official MDACE split files (one hadm_id
            per line, no header).

    Returns:
        ``{hadm_id: "train" | "val" | "test"}``.

    Example:
        >>> read_splits("test-resources/core/mdace/splits")["900001"]
        'train'
    """
    out: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = Path(splits_dir) / f"MDace-ev-{split}.csv"
        for line in path.read_text().split():
            out[line.strip()] = split
    return out


class MDACEDataset(BaseDataset):
    """MDACE: MIMIC-III notes annotated with ICD code evidence spans.

    MDACE annotates 302 inpatient admissions (ICD-9 version) with character
    spans in MIMIC-III notes that support each billed code. The annotation
    files are public but contain no note text; text is joined from MIMIC-III
    ``NOTEEVENTS.csv[.gz]``, which requires PhysioNet credentialing.

    On construction the dataset streams NOTEEVENTS in chunks, keeps only the
    annotated notes, trims each span by one punctuation/whitespace character
    at either end (as in Edin et al.), and writes one Parquet table with one
    row per annotated note into a temporary directory. The data folders are
    only read.

    Per-note attributes: ``hadm_id``, ``note_id``, ``category``,
    ``description``, ``text``, ``codes`` / ``code_systems`` (JSON lists) and
    ``spans`` (JSON list aligned with ``codes``; each entry is a list of
    ``[begin, end]`` pairs), and ``split`` (``train``/``val``/``test`` from
    ``splits_dir``, else null).

    Args:
        root: MDACE folder containing ``<setting>/ICD-<v>/1.0/*.json``.
        mimic3_root: Folder containing ``NOTEEVENTS.csv`` or
            ``NOTEEVENTS.csv.gz``.
        setting: ``"Inpatient"`` or ``"Profee"``.
        icd_version: 9 or 10.
        splits_dir: Optional folder with ``MDace-ev-{train,val,test}.csv``.
        config_path: Path to the YAML config. Defaults to ``mdace.yaml``.
        **kwargs: Passed to :class:`BaseDataset` (e.g. ``cache_dir``).

    Examples:
        >>> from pyhealth.datasets import MDACEDataset
        >>> dataset = MDACEDataset(
        ...     root="/path/to/MDace",
        ...     mimic3_root="/path/to/mimic-iii/1.4",
        ...     splits_dir="/path/to/splits/mdace/inpatient",
        ... )
        >>> samples = dataset.set_task()
    """

    def __init__(
        self,
        root: str,
        mimic3_root: str,
        setting: str = "Inpatient",
        icd_version: int = 9,
        splits_dir: str | None = None,
        config_path: str | None = str(
            Path(__file__).parent / "configs" / "mdace.yaml"
        ),
        **kwargs,
    ) -> None:
        ann_dir = Path(root) / setting / f"ICD-{icd_version}" / "1.0"
        if not ann_dir.is_dir():
            raise FileNotFoundError(f"MDACE annotation folder not found: {ann_dir}")
        notes_path = next(
            (
                Path(mimic3_root) / name
                for name in ("NOTEEVENTS.csv.gz", "NOTEEVENTS.csv")
                if (Path(mimic3_root) / name).is_file()
            ),
            None,
        )
        if notes_path is None:
            raise FileNotFoundError(f"NOTEEVENTS.csv[.gz] not found in {mimic3_root}")

        self._tmp_dir = tempfile.mkdtemp(prefix="pyhealth_mdace_")
        self._index_data(ann_dir, notes_path, splits_dir, self._tmp_dir)
        super().__init__(
            root=self._tmp_dir,
            tables=["mdace"],
            dataset_name="MDACE",
            config_path=config_path,
            **kwargs,
        )

    def __del__(self):
        shutil.rmtree(getattr(self, "_tmp_dir", ""), ignore_errors=True)

    @staticmethod
    def _index_data(
        ann_dir: Path,
        notes_path: Path,
        splits_dir: str | None,
        output_dir: str,
    ) -> pd.DataFrame:
        """Joins annotations with note text and writes ``mdace.parquet``.

        Args:
            ann_dir: Folder of MDACE JSON files for one setting/version.
            notes_path: MIMIC-III NOTEEVENTS file.
            splits_dir: Optional folder with the official split files.
            output_dir: Folder to write the Parquet table to.

        Returns:
            The written DataFrame (one row per annotated note).
        """
        annotations = load_annotations(ann_dir)
        wanted = {int(note_id) for _, note_id in annotations}
        splits = read_splits(splits_dir) if splits_dir else {}

        found: dict[str, dict] = {}
        # Note: full NOTEEVENTS scan per build (~minutes); cache the
        # Parquet outside the temp dir if rebuilds become frequent.
        for chunk in pd.read_csv(
            notes_path,
            usecols=["ROW_ID", "SUBJECT_ID", "CHARTDATE", "TEXT"],
            dtype={"ROW_ID": "int64", "SUBJECT_ID": "int64", "CHARTDATE": str},
            chunksize=100_000,
        ):
            chunk = chunk[chunk["ROW_ID"].isin(wanted)]
            for row in chunk.itertuples(index=False):
                found[str(row.ROW_ID)] = {
                    "patient_id": str(row.SUBJECT_ID),
                    "chartdate": row.CHARTDATE,
                    "text": row.TEXT,
                }

        rows: list[dict] = []
        for (hadm_id, note_id), note in annotations.items():
            src = found.get(note_id)
            if src is None:
                logger.warning("MDACE note %s missing from NOTEEVENTS", note_id)
                continue
            text = src["text"]
            codes = list(note["codes"])
            rows.append(
                {
                    **src,
                    "hadm_id": hadm_id,
                    "note_id": note_id,
                    "category": note["category"],
                    "description": note["description"],
                    "codes": json.dumps(codes),
                    "code_systems": json.dumps(
                        [note["codes"][c]["system"] for c in codes]
                    ),
                    "spans": json.dumps(
                        [
                            [
                                list(trim_span(s, text))
                                for s in note["codes"][c]["spans"]
                            ]
                            for c in codes
                        ]
                    ),
                    "split": splits.get(hadm_id),
                }
            )
        df = pd.DataFrame(rows, dtype="string")
        df.to_parquet(os.path.join(output_dir, "mdace.parquet"), index=False)
        return df

    @property
    def default_task(self) -> MDACEEvidenceCoding:
        """Returns the default task for this dataset."""
        return MDACEEvidenceCoding()
