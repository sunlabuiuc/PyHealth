"""
PyHealth dataset for the PhysioNet De-Identification dataset.

Dataset link:
    https://physionet.org/content/deidentifiedmedicaltext/1.0/

Dataset paper: (please cite if you use this dataset)
    Neamatullah, Ishna, et al. "Automated de-identification of free-text
    medical records." BMC Medical Informatics and Decision Making 8.1 (2008).

Paper link:
    https://doi.org/10.1186/1472-6947-8-32

PHI category mapping in classify_phi() inspired by the label groupings
in the bert-deid reference implementation by Johnson et al.:
    https://github.com/alistairewj/bert-deid/blob/master/bert_deid/label.py

Task paper:
    Johnson, Alistair E.W., et al. "Deidentification of free-text medical
    records using pre-trained bidirectional transformers." Proceedings of
    the ACM Conference on Health, Inference, and Learning (CHIL), 2020.

Author:
    Matt McKenna (mtm16@illinois.edu)
"""
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

_UNKNOWN_PHI_TAGS: set = set()

# -- Record parsing regexes --

_RECORD_START = re.compile(r"START_OF_RECORD=(\d+)\|\|\|\|(\d+)\|\|\|\|")
_RECORD_END = re.compile(r"\|\|\|\|END_OF_RECORD")
_PHI_TAG = re.compile(r"\[\*\*(.+?)\*\*\]", re.DOTALL)
_PHI_SPLIT = re.compile(r"\[\*\*(?:.+?)\*\*\]", re.DOTALL)


def _parse_file(path: Path) -> Dict[Tuple[str, str], str]:
    """Parse a PhysioNet record file into {(patient_id, note_id): body}.

    Args:
        path: Path to id.text or id.res file.

    Returns:
        Dictionary mapping (patient_id, note_id) to note body text.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    out: Dict[Tuple[str, str], str] = {}
    for m in _RECORD_START.finditer(raw):
        pid, nid = m.group(1), m.group(2)
        body_start = m.end()
        end_m = _RECORD_END.search(raw, body_start)
        body = raw[body_start : end_m.start() if end_m else len(raw)]
        out[(pid, nid)] = body.strip()
    return out


def classify_phi(raw: str) -> str:
    """Map raw [**...**] tag text to one of the 7 PHI categories.

    Args:
        raw: The text inside a [**...**] tag.

    Returns:
        One of: AGE, DATE, CONTACT, LOCATION, ID, PROFESSION, NAME.
    """
    t = re.sub(r"[^a-z0-9 ]+", " ", raw.strip().lower()).strip()

    if any(k in t for k in ("year old", " yo ", " age ")):
        return "AGE"
    if any(k in t for k in ("date", "month", "day", "year", "holiday")):
        return "DATE"
    if re.fullmatch(r"[\d]{1,2}[ \-/][\d]{1,2}([ \-/][\d]{2,4})?", t):
        return "DATE"
    if re.fullmatch(r"[\d]+", raw.strip()):
        return "DATE"
    if any(k in t for k in ("phone", "fax", "email", "pager", "contact")):
        return "CONTACT"
    if any(
        k in t
        for k in (
            "hospital",
            "location",
            "street",
            "county",
            "state",
            "country",
            "zip",
            "address",
            "ward",
            "room",
        )
    ):
        return "LOCATION"
    if any(
        k in t
        for k in (
            "mrn",
            "medical record",
            "record number",
            "ssn",
            "account",
            "serial",
            "unit no",
            "unit number",
            "identifier",
        )
    ):
        return "ID"
    if " id " in f" {t} ":
        return "ID"
    if any(
        k in t
        for k in (
            "doctor",
            " dr ",
            " md ",
            "nurse",
            "attending",
            "resident",
            "profession",
            "service",
            "provider",
        )
    ):
        return "PROFESSION"
    if any(
        k in t
        for k in (
            "name",
            "initial",
            "alias",
            "patient",
            "first name",
            "last name",
        )
    ):
        return "NAME"
    if raw not in _UNKNOWN_PHI_TAGS:
        _UNKNOWN_PHI_TAGS.add(raw)
        logger.warning(
            "classify_phi: no keyword match for tag '%s', defaulting to NAME",
            raw,
        )
    return "NAME"


def phi_spans_in_original(
    orig: str, deid: str
) -> List[Tuple[int, int, str]]:
    """Find PHI character spans in orig by anchoring on non-PHI chunks.

    Uses non-PHI text from the de-identified version as anchors to locate
    where the original PHI text appears in the original note.

    Args:
        orig: Original note text (with real PHI).
        deid: De-identified note text (PHI replaced with [**...**] tags).

    Returns:
        List of (char_start, char_end, phi_category) tuples.
    """
    parts = _PHI_SPLIT.split(deid)
    tags = _PHI_TAG.findall(deid)

    spans: List[Tuple[int, int, str]] = []
    pos = 0

    for i, tag_inner in enumerate(tags):
        before = parts[i]
        if before:
            idx = orig.find(before, pos)
            pos = (idx + len(before)) if idx != -1 else (pos + len(before))

        phi_start = pos

        after = parts[i + 1]
        if after:
            idx = orig.find(after, phi_start)
            phi_end = idx if idx != -1 else phi_start
        else:
            phi_end = len(orig)

        if phi_end > phi_start:
            spans.append((phi_start, phi_end, classify_phi(tag_inner)))

        pos = phi_end

    return spans


def bio_tag(
    text: str, spans: List[Tuple[int, int, str]]
) -> List[Tuple[str, str]]:
    """Whitespace-tokenize text and assign BIO labels from char-level spans.

    Args:
        text: Original note text.
        spans: List of (char_start, char_end, phi_category) tuples.

    Returns:
        List of (word, label) tuples.
    """
    char_label = ["O"] * len(text)
    for start, end, cat in spans:
        for i in range(start, min(end, len(text))):
            char_label[i] = cat

    result: List[Tuple[str, str]] = []
    for m in re.finditer(r"\S+", text):
        w_start, w_end = m.start(), m.end()
        word = m.group()
        # Collect PHI categories for this token's characters, ignoring O's.
        # If empty, the token has no PHI and we label it O.
        cats = [c for c in char_label[w_start:w_end] if c != "O"]
        if not cats:
            result.append((word, "O"))
            continue
        # Pick the most common category. Handles rare cases where a token
        # spans two PHI types, e.g. "Smith01/15" -> chars are NAME+DATE,
        # majority wins (DATE).
        cat = max(set(cats), key=cats.count)
        # B = beginning of a new entity, I = continuation of the same one.
        # Use "I" only if the previous token was the same category,
        # e.g. "Tom"=B-NAME "Garcia"=I-NAME. Otherwise start a new "B".
        prev_label = result[-1][1] if result else "O"
        prefix = (
            "I"
            if prev_label not in ("O",) and prev_label.endswith(cat)
            else "B"
        )
        result.append((word, f"{prefix}-{cat}"))

    return result


class PhysioNetDeIDDataset(BaseDataset):
    """Dataset class for the PhysioNet De-Identification dataset.

    This dataset contains 2,434 nursing notes from 163 patients.
    Each note has original text with PHI (protected health information)
    and a de-identified version with [**...**] tags marking PHI spans.

    The dataset parses both files to produce token-level BIO labels
    for 7 PHI categories: AGE, CONTACT, DATE, ID, LOCATION, NAME,
    PROFESSION.

    Data access requires PhysioNet credentialing:
        1. Create a PhysioNet account at https://physionet.org
        2. Complete the required CITI training
        3. Sign the data use agreement
        4. Download from
           https://physionet.org/content/deidentifiedmedicaltext/1.0/

    Attributes:
        root (str): Root directory containing id.text and id.res files.
        dataset_name (str): Name of the dataset.

    Example::
        >>> dataset = PhysioNetDeIDDataset(root="./data/physionet_deid")
    """

    def __init__(
        self,
        root: str = ".",
        config_path: Optional[str] = str(
            Path(__file__).parent / "configs" / "physionet_deid.yaml"
        ),
        **kwargs,
    ) -> None:
        """Initializes the PhysioNet De-Identification dataset.

        Args:
            root: Root directory containing id.text and id.res files.
            config_path: Path to the configuration file.

        Raises:
            FileNotFoundError: If id.text or id.res not found in root.

        Example::
            >>> dataset = PhysioNetDeIDDataset(root="./data")
        """
        self._verify_data(root)
        self._tmp_dir = tempfile.mkdtemp(prefix="pyhealth_deid_")
        self._index_data(root, self._tmp_dir)

        super().__init__(
            root=self._tmp_dir,
            tables=["physionet_deid"],
            dataset_name="PhysioNetDeID",
            config_path=config_path,
            **kwargs,
        )

    def __del__(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _verify_data(self, root: str) -> None:
        """Verify that required data files exist.

        Args:
            root: Root directory to check.

        Raises:
            FileNotFoundError: If id.text or id.res is missing.
        """
        for fname in ("id.text", "id.res"):
            path = os.path.join(root, fname)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Required file '{fname}' not found in {root}"
                )

    def _index_data(self, root: str, output_dir: str) -> pd.DataFrame:
        """Parse id.text and id.res into a CSV for BaseDataset to load.

        Reads data files from root but writes the metadata CSV to
        output_dir so the data directory can be read-only.

        Args:
            root: Root directory containing the data files.
            output_dir: Directory to write the metadata CSV to.

        Returns:
            DataFrame with columns: patient_id, note_id, text, labels.
        """
        root_path = Path(root)
        orig_records = _parse_file(root_path / "id.text")
        deid_records = _parse_file(root_path / "id.res")

        rows = []
        for key in sorted(
            orig_records, key=lambda k: (int(k[0]), int(k[1]))
        ):
            pid, nid = key
            orig = orig_records[key]
            # Missing key yields empty string (no deid version).
            deid = deid_records.get(key, "")
            spans = phi_spans_in_original(orig, deid)
            tagged = bio_tag(orig, spans)

            tokens = " ".join(w for w, _ in tagged)
            labels = " ".join(lbl for _, lbl in tagged)

            rows.append(
                {
                    "patient_id": pid,
                    "note_id": nid,
                    "text": tokens,
                    "labels": labels,
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(
            os.path.join(output_dir, "physionet_deid_metadata.csv"),
            index=False,
        )
        return df
