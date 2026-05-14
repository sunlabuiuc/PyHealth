"""DOSSIER: EHR Fact Checking via Structured Biomedical Knowledge Grounding.

Implements the full evaluation pipeline from:
    Zhang et al., "Dossier: Fact Checking in Electronic Health Records
    while Preserving Patient Privacy", MLHC 2024.

The pipeline translates natural language claims about a patient's ICU stay
into SQL queries that are executed against four evidence tables built from
MIMIC-III (Admission, Lab, Vital, Input), optionally joined with a global
biomedical knowledge graph (SemMedDB).  An LLM generates both the SQL and
the final veracity stance (True / False / Not-Enough-Information).

Classes:
    SQLExecutor: Manages a persistent SQLite database for query execution.
    DOSSIERPromptGenerator: Builds LLM prompts for SQL-based EHR fact-checking.
    DOSSIERPipeline: End-to-end DOSSIER EHR fact-checking pipeline.

Design note
-----------
DOSSIER is a *zero-shot inference pipeline* rather than a gradient-trained
model, so :class:`DOSSIERPipeline` does **not** subclass ``nn.Module`` or
``BaseModel``.  Instead it provides a scikit-learn-style ``predict`` /
``evaluate`` interface that fits naturally into PyHealth's evaluation scripts.

Example:
    >>> pipeline = DOSSIERPipeline(
    ...     mimic3_root="/data/mimic3",
    ...     claims_path="/data/claims.csv",
    ...     llm="claude-haiku-4-5",
    ...     prompt_variant="full",
    ...     semmeddb_path="/data/semmeddb_processed_10.csv",
    ...     umls_dir="/data/umls",
    ...     cui_mapping_path="/data/umls/mimic3_cui_mapping.csv",
    ... )
    >>> results = pipeline.run(output_dir="./output")
    >>> metrics = pipeline.evaluate(results)
    >>> print(metrics["accuracy"])
    0.72
"""

from __future__ import annotations

import logging
import os
import re
import threading
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from textwrap import dedent
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stance utilities
# ---------------------------------------------------------------------------

STANCE_TO_INT: Dict[str, int] = {"T": 0, "F": 1, "N": 2}
INT_TO_STANCE: Dict[int, str] = {0: "T", 1: "F", 2: "N"}
_NEGATE = {"T": "F", "F": "T", "N": "N"}


def _negate_stance(s: str) -> str:
    """Return the logical negation of stance *s* (T↔F; N stays N)."""
    return _NEGATE.get(s, "N")


# ---------------------------------------------------------------------------
# Prompt templates  (faithful to the original source prompts)
# ---------------------------------------------------------------------------

# -- Full (UMLS + Global KG) -------------------------------------------------

_FULL_HEADER = dedent("""\
Given the following SQL tables, your job is to output a valid SQL query which \
can be used to validate a user's natural language claim.
Your query should return a table containing the clinical record(s) which act \
as supporting evidence, and which may be used to prove or disprove the claim.
You should also output non-negative scalar values in <lower></lower> and \
<upper></upper> tags, and a stance character in the <stance></stance> tags.
The stance value should be a single character, either T (indicating true) or \
F (indicating false).
When the number of rows in the returned table is between the lower and upper \
bounds (inclusive), the claim should have veracity equal to the stance.
If the upper bound is positive infinity, you can leave the <upper></upper> \
value blank.
Output a SQL query only if the claim is verifiable and you are confident in \
the generated query; otherwise tell me you don't know. Do not hallucinate any \
clauses.""")

_FULL_SCHEMA = dedent("""\
CREATE TABLE Admission ( t REAL, CUI TEXT, str_label TEXT );
CREATE TABLE Vital ( t REAL, CUI TEXT, Value REAL, Units TEXT, str_label TEXT );
CREATE TABLE Lab ( t REAL, CUI TEXT, Value REAL, Units TEXT, str_label TEXT );
CREATE TABLE Input ( t REAL, CUI TEXT, Amount REAL, Units TEXT, str_label TEXT );
CREATE TABLE Global_KG ( Subject_CUI TEXT, Predicate TEXT, Object_CUI TEXT );""")

_FULL_PROBLEM_SPECS = dedent("""\
Here are some more details about the problem:
- t is given in hours.
- The patient was admitted to the hospital at t=0.
- Rows may not be sorted.
- The Input table contains medication and IV inputs.
- The Admission table has one row for each admission diagnosis, measured at t=0.
- The Vital table contains vital measurements; the Lab table contains laboratory \
measurements.
- The Global_KG table corresponds to triplets from a large biomedical knowledge \
graph. The triplets have the form (Subject_CUI, Predicate, Object_CUI).
- Always specify a predicate when querying Global_KG.
- The Predicate column of Global_KG has the following possible values: {}
- Be very careful about whether an entity is a Subject_CUI or an Object_CUI in \
Global_KG, particularly for the ISA predicate.
- Match on CUI (Concept Unique Identifier) whenever possible instead of str_label.
- Due to varying levels of granularity, always use Global_KG with the ISA \
predicate to check for the presence or value of any entity. Global_KG contains \
self loops with the ISA predicate.
- Your query should always start with "SELECT *". Do not SELECT COUNT.
- Use <thinking></thinking> XML tags for intermediate steps.
- Put your SQL query in <sql></sql> XML tags.
- Put the veracity in <stance></stance> tags (T or F).
- Put bounds in <lower></lower> and <upper></upper> tags.""")

_FULL_EXAMPLES = dedent("""\
Here are some examples:
<example>
H: You are given the following prior knowledge:
- Potentially relevant CUIs found in the claim: ('Anticoagulants', 'C0003280'), \
('Treatment given', 'C0580351')
- The following appear in Subject_CUI column of Global_KG: C0003280
- The following appear in Object_CUI column of Global_KG: C0003280

Claim made at t=70: pt was given a blood thinner in the past 24 hours.

A: <thinking>
- 'Blood thinner' refers to anticoagulants (CUI C0003280).
- 'In the past 24 hours' means t between 70-24 and 70.
- We need rows in Input where the item ISA C0003280, within the time window.
</thinking>
<sql>
SELECT *
FROM Input
JOIN Global_KG ON Input.CUI = Global_KG.Subject_CUI
WHERE Global_KG.Predicate = 'ISA'
  AND Global_KG.Object_CUI = 'C0003280'
  AND Input.t BETWEEN 70-24 AND 70
</sql>
<lower>1</lower>
<upper></upper>
<stance>T</stance>
</example>

<example>
H: You are given the following prior knowledge:
- Potentially relevant CUIs found in the claim: ('Systolic blood pressure', 'C0871470')
- The following appear in Subject_CUI column of Global_KG: C0871470

Claim made at t=100: pt did not have a systolic blood pressure above 140 since t=20.

A: <thinking>
- CUI for systolic BP is C0871470.
- Query Vital for SBP values > 140 since t=20.
- If any row is returned, the claim is false.
</thinking>
<sql>
SELECT *
FROM Vital
JOIN Global_KG ON Vital.CUI = Global_KG.Subject_CUI
WHERE Global_KG.Predicate = 'ISA'
  AND Global_KG.Object_CUI = 'C0871470'
  AND Vital.t >= 20
  AND Vital.Value > 140
</sql>
<lower>1</lower>
<upper></upper>
<stance>F</stance>
</example>""")

# ── No Global KG (UMLS only) ───────────────────────────────────────────────

_NO_GKG_HEADER = dedent("""\
Given the following SQL tables, your job is to output a valid SQL query which \
can be used to validate a user's natural language claim.
Your query should return a table containing the clinical record(s) which act \
as supporting evidence, and which may be used to prove or disprove the claim.
You should also output non-negative scalar values in <lower></lower> and \
<upper></upper> tags, and a stance character in the <stance></stance> tags.
The stance value should be a single character, either T (indicating true) or \
F (indicating false).
When the number of rows in the returned table is between the lower and upper \
bounds (inclusive), the claim should have veracity equal to the stance.
If the upper bound is positive infinity, you can leave the <upper></upper> \
value blank.
Output a SQL query only if the claim is verifiable and you are confident; \
otherwise tell me you don't know.""")

_NO_GKG_SCHEMA = dedent("""\
CREATE TABLE Admission ( t REAL, CUI TEXT, str_label TEXT );
CREATE TABLE Vital ( t REAL, CUI TEXT, Value REAL, Units TEXT, str_label TEXT );
CREATE TABLE Lab ( t REAL, CUI TEXT, Value REAL, Units TEXT, str_label TEXT );
CREATE TABLE Input ( t REAL, CUI TEXT, Amount REAL, Units TEXT, str_label TEXT );""")

_NO_GKG_PROBLEM_SPECS = dedent("""\
Here are some more details about the problem:
- t is given in hours; the patient was admitted at t=0.
- The Input table contains medication and IV inputs.
- Match on CUI whenever possible instead of str_label.
- Your query should always start with "SELECT *". Do not SELECT COUNT.
- Put your SQL query in <sql></sql> tags and veracity in <stance></stance> tags.
- Put bounds in <lower></lower> and <upper></upper> tags.""")

_NO_GKG_EXAMPLES = dedent("""\
Here are some examples:
<example>
H: You are given the following prior knowledge:
- Potentially relevant CUIs found in the claim: ('Anticoagulants', 'C0003280')

Claim made at t=70: pt was given a blood thinner in the past 24 hours.

A: <sql>
SELECT *
FROM Input
WHERE CUI = 'C0003280'
  AND t BETWEEN 70-24 AND 70
</sql>
<lower>1</lower>
<upper></upper>
<stance>T</stance>
</example>""")

# ── No UMLS (Global KG only, name-based) ──────────────────────────────────

_NO_UMLS_HEADER = dedent("""\
Given the following SQL tables, your job is to output a valid SQL query which \
can be used to validate a user's natural language claim.
Your query should return a table containing the clinical record(s) which act \
as supporting evidence, and which may be used to prove or disprove the claim.
You should also output non-negative scalar values in <lower></lower> and \
<upper></upper> tags, and a stance character in the <stance></stance> tags.
The stance value should be a single character, either T (indicating true) or \
F (indicating false).
When the number of rows in the returned table is between the lower and upper \
bounds (inclusive), the claim should have veracity equal to the stance.
If the upper bound is positive infinity, you can leave the <upper></upper> \
value blank.
Output a SQL query only if the claim is verifiable and you are confident in \
the generated query; otherwise tell me you don't know. Do not hallucinate any \
clauses.""")

_NO_UMLS_SCHEMA = dedent("""\
CREATE TABLE Admission ( t REAL, str_label TEXT );
CREATE TABLE Vital ( t REAL, Value REAL, Units TEXT, str_label TEXT );
CREATE TABLE Lab ( t REAL, Value REAL, Units TEXT, str_label TEXT );
CREATE TABLE Input ( t REAL, Amount REAL, Units TEXT, str_label TEXT );
CREATE TABLE Global_KG (Subject_Name TEXT, Predicate TEXT, Object_Name TEXT );""")

_NO_UMLS_PROBLEM_SPECS = dedent("""\
Here are some more details about the problem:
- t is given in hours.
- The patient was admitted to the hospital at t=0.
- Rows may not be sorted.
- The Input table contains medication and IV inputs.
- The Admission table has one row for each admission diagnosis, measured at t=0.
- The Vital table contains vital measurements; the Lab table contains laboratory measurements.
- The Global_KG table corresponds to triplets from a biomedical knowledge graph \
with the form (Subject_Name, Predicate, Object_Name). You should almost always use this table.
- Always specify a predicate when querying Global_KG.
- The Predicate column of Global_KG has the following possible values: {}
- Be careful about whether an entity is a Subject or Object in Global_KG.
- Join patient tables to Global_KG on str_label = Subject_Name (e.g. \
Input.str_label = Global_KG.Subject_Name).
- Your query should always start with "SELECT *". Do not SELECT COUNT.
- Use <thinking></thinking> XML tags for intermediate steps.
- Put your SQL in <sql></sql> and veracity in <stance></stance> tags.
- Put bounds in <lower></lower> and <upper></upper> tags.""")

_NO_UMLS_EXAMPLES = dedent("""\
Here are some examples:
<example>
H: Claim made at t=70: pt was given a blood thinner in the past 24 hours.

A: <thinking>
- 'In the past 24 hours' means between t=46 and t=70.
- Check Input for anticoagulant medications using Global_KG ISA predicate.
- Join on Input.str_label = Global_KG.Subject_Name.
</thinking>
<sql>
SELECT *
FROM Input
JOIN Global_KG ON Input.str_label = Global_KG.Subject_Name
WHERE Global_KG.Predicate = 'ISA'
  AND UPPER(Global_KG.Object_Name) LIKE '%ANTICOAGULANT%'
  AND Input.t BETWEEN 46 AND 70
</sql>
<lower>1</lower>
<upper></upper>
<stance>T</stance>
</example>

<example>
H: Claim made at t=100: pt did not have a systolic blood pressure above 140 since t=20.

A: <thinking>
- Check Vital for systolic BP values > 140 since t=20.
- No KG join needed; search str_label directly.
</thinking>
<sql>
SELECT *
FROM Vital
WHERE UPPER(str_label) LIKE '%SYSTOLIC%'
  AND t >= 20
  AND Value > 140
</sql>
<lower>1</lower>
<upper></upper>
<stance>F</stance>
</example>

<example>
H: Claim made at t=50: pt was administered warfarin at most three times.

A: <thinking>
- Find warfarin or subtypes in Input via Global_KG ISA predicate.
- Rows between 0 and 3 means true.
</thinking>
<sql>
SELECT *
FROM Input
JOIN Global_KG ON Input.str_label = Global_KG.Subject_Name
WHERE Global_KG.Predicate = 'ISA'
  AND UPPER(Global_KG.Object_Name) LIKE '%WARFARIN%'
</sql>
<lower>0</lower>
<upper>3</upper>
<stance>T</stance>
</example>""")

# ── Neither (no UMLS, no KG) ───────────────────────────────────────────────

_NEITHER_HEADER = dedent("""\
Given the following SQL tables, output a valid SQL query to validate a \
natural language claim.
Your query should return the clinical record(s) that prove or disprove the claim.
Also output <lower></lower>, <upper></upper> bounds and <stance></stance> (T or F).
When the row count is between lower and upper (inclusive), the stance is correct.
If the upper bound is infinity, leave <upper></upper> blank.
Only output SQL if you are confident; otherwise say you don't know.""")

_NEITHER_SCHEMA = dedent("""\
CREATE TABLE Admission ( t REAL, str_label TEXT );
CREATE TABLE Vital ( t REAL, Value REAL, Units TEXT, str_label TEXT );
CREATE TABLE Lab ( t REAL, Value REAL, Units TEXT, str_label TEXT );
CREATE TABLE Input ( t REAL, Amount REAL, Units TEXT, str_label TEXT );""")

_NEITHER_PROBLEM_SPECS = dedent("""\
- t is in hours; the patient was admitted at t=0.
- Match on str_label (human-readable name) when CUI is unavailable.
- Your query should always start with "SELECT *". Do not SELECT COUNT.
- Put SQL in <sql></sql> and veracity in <stance></stance> tags.
- Put bounds in <lower></lower> and <upper></upper> tags.""")

_NEITHER_EXAMPLES = dedent("""\
Here are some examples:
<example>
Claim made at t=70: pt was given a blood thinner in the past 24 hours.

A: <sql>
SELECT *
FROM Input
WHERE LOWER(str_label) LIKE '%heparin%'
   OR LOWER(str_label) LIKE '%warfarin%'
   AND t BETWEEN 70-24 AND 70
</sql>
<lower>1</lower>
<upper></upper>
<stance>T</stance>
</example>""")

# ---------------------------------------------------------------------------
# V2 absence-as-evidence examples  (one per prompt variant)
# ---------------------------------------------------------------------------
# WHY THESE EXIST — the paper's original few-shot examples teach two patterns:
#
#   Pattern A (existence): stance=T, lower=1, upper=∞
#     "Find rows that confirm the claim. ≥1 rows → True; 0 rows → NEI."
#     Example: "pt was given a blood thinner" → SELECT from Input WHERE ...
#
#   Pattern B (contradiction): stance=F, lower=1, upper=∞
#     "Find rows that contradict the claim. ≥1 rows → False; 0 rows → NEI."
#     Example: "pt did NOT have SBP above 140" → SELECT WHERE SBP > 140
#     NOTE: this pattern is ONLY taught for negated claims ("did NOT", "never").
#
# Missing pattern — absence-as-evidence: stance=F, lower=0, upper=0
#     "Find rows for the claimed event. If found → claim is True (¬stance);
#      if 0 rows → claim is False (stance)."
#     Correct for MEDICATION claims because MIMIC INPUTEVENTS is a complete
#     record: the absence of a drug entry means the drug was definitively
#     not given (not merely unrecorded), so 0 rows = proof of falsity.
#
# Because the original examples never show this pattern for positively-
# phrased claims ("The patient received X" where gold=F), the LLM always
# applies Pattern A and predicts NEI when 0 rows are found, missing all
# False medication claims (F-recall = 0%).
#
# These V2 examples add one medication-absence demonstration per variant.
# They are appended after the original examples (which are kept unchanged)
# when use_paper_prompt=False (the new default).
# Set use_paper_prompt=True to reproduce the exact original paper prompts.

_NEITHER_ABSENCE_EXAMPLE_V2 = dedent("""\
<example>
Claim made at t=72: The patient received furosemide (Lasix) during this admission.

A: <thinking>
- Search Input for furosemide / Lasix entries.
- MIMIC INPUTEVENTS records every administered drug. If no rows are found
  the drug was definitively not given — the claim is False, not NEI.
- Use lower=0, upper=0, stance=F:
    rows >= 1  →  negate(F) = True   (drug was given; claim is confirmed)
    rows  = 0  →  stance  = False    (drug absent; claim is disproved)
</thinking>
<sql>
SELECT *
FROM Input
WHERE LOWER(str_label) LIKE '%furosemide%'
   OR LOWER(str_label) LIKE '%lasix%'
</sql>
<lower>0</lower>
<upper>0</upper>
<stance>F</stance>
</example>""")

_FULL_ABSENCE_EXAMPLE_V2 = dedent("""\
<example>
H: You are given the following prior knowledge:
- Potentially relevant CUIs found in the claim: ('Furosemide', 'C0016860')
- The following appear in Object_CUI column of Global_KG: C0016860

Claim made at t=72: The patient received furosemide (Lasix) during this admission.

A: <thinking>
- Furosemide CUI: C0016860. This is a medication — check the Input table.
- MIMIC INPUTEVENTS records every administered drug. If no rows are found
  the drug was definitively not given — the claim is False, not NEI.
- Use lower=0, upper=0, stance=F:
    rows >= 1  →  negate(F) = True   (drug was given; claim is confirmed)
    rows  = 0  →  stance  = False    (drug absent; claim is disproved)
- Query Input directly by CUI — no Global_KG join needed for absence checks.
  NOTE: This absence pattern applies ONLY to medication (Input table) claims.
  For Lab, Vital, or Admission claims use Pattern A (stance=T, lower=1).
</thinking>
<sql>
SELECT *
FROM Input
WHERE CUI = 'C0016860'
</sql>
<lower>0</lower>
<upper>0</upper>
<stance>F</stance>
</example>""")

_NO_GKG_ABSENCE_EXAMPLE_V2 = dedent("""\
<example>
H: You are given the following prior knowledge:
- Potentially relevant CUIs found in the claim: ('Furosemide', 'C0016860')

Claim made at t=72: The patient received furosemide (Lasix) during this admission.

A: <thinking>
- Furosemide maps to CUI C0016860. Search Input directly by CUI.
- MIMIC INPUTEVENTS records every administered drug. If no rows are found
  the drug was definitively not given — the claim is False, not NEI.
- Use lower=0, upper=0, stance=F:
    rows >= 1  →  negate(F) = True   (drug was given; claim is confirmed)
    rows  = 0  →  stance  = False    (drug absent; claim is disproved)
</thinking>
<sql>
SELECT *
FROM Input
WHERE CUI = 'C0016860'
</sql>
<lower>0</lower>
<upper>0</upper>
<stance>F</stance>
</example>""")

_NO_UMLS_ABSENCE_EXAMPLE_V2 = dedent("""\
<example>
H: Claim made at t=72: The patient received furosemide (Lasix) during this admission.

A: <thinking>
- Search Input for furosemide via Global_KG name expansion.
- MIMIC INPUTEVENTS records every administered drug. If no rows are found
  the drug was definitively not given — the claim is False, not NEI.
- Use lower=0, upper=0, stance=F:
    rows >= 1  →  negate(F) = True   (drug was given; claim is confirmed)
    rows  = 0  →  stance  = False    (drug absent; claim is disproved)
</thinking>
<sql>
SELECT *
FROM Input
JOIN Global_KG ON Input.str_label = Global_KG.Subject_Name
WHERE Global_KG.Predicate = 'ISA'
  AND UPPER(Global_KG.Object_Name) LIKE '%FUROSEMIDE%'
</sql>
<lower>0</lower>
<upper>0</upper>
<stance>F</stance>
</example>""")

# ---------------------------------------------------------------------------
# SQL schema helpers
# ---------------------------------------------------------------------------

_RENAME_TABLES = {
    "adm": "Admission",
    "lab": "Lab",
    "vit": "Vital",
    "input": "Input",
    "global_kg": "Global_KG",
}

_RENAME_COLS: Dict[str, Dict[str, str]] = {
    "adm": {"DIAGNOSIS": "str_label", "ADMIT_CUI": "CUI", "rel_t": "t"},
    "lab": {"value": "Value", "units": "Units", "rel_t": "t"},
    "vit": {"value": "Value", "units": "Units", "rel_t": "t"},
    "input": {"AMOUNT": "Amount", "units": "Units", "rel_t": "t"},
    "global_kg": {
        "SUBJECT_CUI": "Subject_CUI",
        "SUBJECT_NAME": "Subject_Name",
        "PREDICATE": "Predicate",
        "OBJECT_CUI": "Object_CUI",
        "OBJECT_NAME": "Object_Name",
    },
}

_KEEP_COLS: Dict[str, List[str]] = {
    "Admission": ["t", "str_label", "CUI"],
    "Lab": ["t", "CUI", "Value", "Units", "str_label"],
    "Vital": ["t", "CUI", "Value", "Units", "str_label"],
    "Input": ["t", "CUI", "Amount", "Units", "str_label"],
    "Global_KG": [
        "Subject_CUI",
        "Subject_Name",
        "Object_CUI",
        "Object_Name",
        "Predicate",
    ],
}

# Patient table columns when CUI is excluded (neither / no_umls variants)
_NO_CUI_KEEP_COLS: Dict[str, List[str]] = {
    "Admission": ["t", "str_label"],
    "Lab": ["t", "Value", "Units", "str_label"],
    "Vital": ["t", "Value", "Units", "str_label"],
    "Input": ["t", "Amount", "Units", "str_label"],
}

# Global_KG columns for no_umls (name-based only, no CUIs)
_NO_UMLS_GKG_COLS: List[str] = ["Subject_Name", "Predicate", "Object_Name"]


def _to_sql_schema(
    tables: Dict[str, pd.DataFrame],
    prompt_variant: str = "full",
) -> Dict[str, pd.DataFrame]:
    """Rename and filter columns to match the SQL schema the LLM expects.

    Args:
        tables: Raw patient tables keyed by short names (``"adm"``,
            ``"lab"``, ``"vit"``, ``"input"``, ``"global_kg"``).
        prompt_variant: One of ``"full"``, ``"no_umls"``, ``"no_gkg"``,
            ``"neither"``.  Controls which columns are retained.

    Returns:
        Dict mapping canonical SQL table names (e.g. ``"Admission"``,
        ``"Lab"``) to renamed and filtered DataFrames.
    """
    out: Dict[str, pd.DataFrame] = {}
    for key, df in tables.items():
        if key not in _RENAME_TABLES:
            continue
        new_name = _RENAME_TABLES[key]
        # neither: no Global_KG at all
        if prompt_variant == "neither" and new_name == "Global_KG":
            continue
        renamed = (
            df.drop(columns=["ROW_ID", "t"], errors="ignore")
            .reset_index(drop=True)
            .reset_index()
            .rename(columns=_RENAME_COLS.get(key, {}))
            .rename(columns={"index": "ROW_ID"})
        )
        # Determine which columns to keep based on variant
        if new_name == "Global_KG":
            if prompt_variant == "no_umls":
                # Name-based KG only; no CUI columns
                keep_cols = _NO_UMLS_GKG_COLS
            else:
                keep_cols = _KEEP_COLS["Global_KG"]
        elif prompt_variant in ("neither", "no_umls"):
            # Strip CUI from patient tables
            keep_cols = _NO_CUI_KEEP_COLS.get(new_name, [])
        else:
            keep_cols = _KEEP_COLS[new_name]
        keep = [c for c in ["ROW_ID"] + keep_cols if c in renamed.columns]
        out[new_name] = renamed[keep]
        if "t" in out[new_name].columns:
            out[new_name] = out[new_name].sort_values("t")
    return out


def _add_identity_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Add self-loop ISA edges to ``Global_KG`` so every CUI can match itself.

    Skipped for name-based ``Global_KG`` (``no_umls`` variant) which has no
    ``Subject_CUI`` / ``Object_CUI`` columns.

    Args:
        df: The ``Global_KG`` DataFrame (must have a ``ROW_ID`` column).

    Returns:
        DataFrame with additional self-loop rows appended and duplicates
        removed.
    """
    if "Subject_CUI" not in df.columns or "Object_CUI" not in df.columns:
        return df
    nodes = set(df["Subject_CUI"]).union(set(df["Object_CUI"]))
    loops = pd.DataFrame(
        {"Subject_CUI": list(nodes), "Object_CUI": list(nodes), "Predicate": "ISA"}
    )
    start = int(df["ROW_ID"].max()) + 1
    loops["ROW_ID"] = np.arange(start, start + len(loops))
    merged = pd.concat([df, loops], ignore_index=True).drop_duplicates(
        subset=["Subject_CUI", "Object_CUI", "Predicate"]
    )
    return merged


# ---------------------------------------------------------------------------
# SQL executor with timeout
# ---------------------------------------------------------------------------

class SQLExecutor:
    """Manages a persistent SQLite database for DOSSIER query execution.

    Args:
        in_memory: If True, use an in-memory database (faster but no
            persistence between admissions).  Default False (temp file).

    Examples:
        >>> executor = SQLExecutor(in_memory=True)
        >>> import pandas as pd
        >>> tables = {"Lab": pd.DataFrame({"t": [0.0], "str_label": ["glucose"]})}
        >>> executor.load_tables(tables)
        >>> result = executor.run_query("SELECT * FROM Lab")
    """

    def __init__(self, in_memory: bool = False) -> None:
        """Initialise the executor and create (or open) the backing database.

        Args:
            in_memory: If True, use an in-memory SQLite database (faster but
                tables are lost when the object is garbage-collected).
                Default False (temp file on disk).
        """
        self._in_memory = in_memory
        if in_memory:
            # Persist the connection for the lifetime of the executor so that
            # tables loaded via load_tables() are visible to run_query().
            self._db_path = ":memory:"
            self._conn: Optional[sqlite3.Connection] = sqlite3.connect(":memory:")
        else:
            self._conn = None
            self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            self._db_path = self._tmp.name
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=wal")
            conn.close()

    @property
    def db_path(self) -> str:
        """Return the filesystem path (or ``:memory:``) of the SQLite database."""
        return self._db_path

    def load_tables(
        self,
        tables: Dict[str, pd.DataFrame],
        add_kg_identity_edges: bool = True,
        skip: Optional[List[str]] = None,
    ) -> None:
        """Write tables into the SQLite database.

        Args:
            tables: Dict mapping table name to DataFrame.
            add_kg_identity_edges: Add self-loop ISA edges to ``Global_KG``.
            skip: Table names to skip (e.g. Global_KG if already loaded).
        """
        skip = skip or []
        conn = self._conn if self._in_memory else sqlite3.connect(self._db_path)
        try:
            for name, df in tables.items():
                if name in skip:
                    continue
                add_edges = name == "Global_KG" and add_kg_identity_edges
                to_write = _add_identity_edges(df) if add_edges else df
                to_write.to_sql(name, conn, if_exists="replace", index=False)
                if name == "Global_KG" and "Object_CUI" in to_write.columns:
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_kg_obj"
                        " ON Global_KG(Object_CUI)"
                    )
            conn.commit()
        finally:
            if not self._in_memory:
                conn.close()

    def run_query(self, sql: str, timeout: int = 120) -> pd.DataFrame:
        """Execute SQL with a hard timeout (seconds).

        Uses a daemon thread so the timeout works on all platforms including
        Windows (where multiprocessing spawn would fail to re-import the module).

        Args:
            sql: SQL query string (must begin with SELECT *).
            timeout: Maximum execution time in seconds.

        Returns:
            DataFrame of query results (at most 1 000 rows).

        Raises:
            TimeoutError: If the query exceeds *timeout* seconds.
            Exception: Any SQLite execution error.
        """
        if self._in_memory:
            return pd.read_sql_query(sql, self._conn).iloc[:1000]

        result: list = [None]
        exc: list = [None]

        def _run() -> None:
            """Execute the query in a daemon thread and store the result."""
            try:
                conn = sqlite3.connect(self._db_path)
                result[0] = pd.read_sql_query(sql, conn).iloc[:1000]
            except Exception as e:
                exc[0] = e
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            raise TimeoutError(f"Query timed out after {timeout}s.")
        if exc[0] is not None:
            raise exc[0]
        return result[0]

    def close(self) -> None:
        """Close the database connection and delete any temporary files."""
        if self._in_memory and self._conn:
            self._conn.close()
            self._conn = None
        elif hasattr(self, "_tmp"):
            try:
                self._tmp.close()
            except OSError:
                pass
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.unlink(self._db_path + suffix)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Prompt generator
# ---------------------------------------------------------------------------

class DOSSIERPromptGenerator:
    """Generates LLM prompts for SQL-based EHR fact-checking.

    Args:
        tag_fn: Callable ``(claim_str) -> (entities, cui_list)`` where
            *entities* is a list of dicts with keys ``pretty_name``,
            ``cui``, ``type`` and *cui_list* is a list of CUI strings.
            Pass ``None`` to disable entity tagging (equivalent to the
            *neither* ablation).
        cuis_in_ehr: List of CUI strings present in the MIMIC-III data.
            Used to annotate which claim CUIs can actually be looked up.
        prompt_variant: One of ``"full"``, ``"no_umls"``, ``"no_gkg"``,
            ``"neither"``.  Defaults to ``"full"``.
        add_examples: Whether to prepend few-shot examples.  Default True.
        add_sem_types: Whether to include UMLS semantic types in the
            prior knowledge section.  Default True.
        use_paper_prompt: When ``True``, use the original paper's few-shot
            examples verbatim (no absence-as-evidence pattern).  When
            ``False`` (default), append a V2 absence example after the
            original examples so the LLM learns to use ``lower=0, upper=0,
            stance=F`` for positively-phrased medication claims.

            **Why the default changed:** The paper's examples only teach
            existence queries (``stance=T, lower=1``) and contradiction
            queries for *negated* claims ("did NOT have X").  For positively-
            phrased False claims ("patient received X" where X was not given),
            the LLM always writes an existence query, finds 0 rows, and
            predicts NEI instead of False — so F-recall = 0%.  The V2
            example explicitly demonstrates the ``lower=0, upper=0, stance=F``
            pattern for medication absence, which is unambiguous in MIMIC
            because INPUTEVENTS is a complete medication record.

            Set ``use_paper_prompt=True`` to reproduce the original paper
            results exactly.

    Examples:
        >>> gen = DOSSIERPromptGenerator(tag_fn=None, cuis_in_ehr=[])
        >>> prompt = gen.get_prompt("pt had fever", 24.0, tables={})
    """

    # Appended to any variant's problem_specs when enable_absence_fix=True
    _ABSENCE_HINT: ClassVar[str] = dedent("""\
- For claims that something did NOT happen (e.g. "patient never received X", \
"patient did not have Y"), query for the event and use stance=F with \
lower=0 and upper=0: if the query returns exactly 0 rows, the claim is \
verified as False (the event truly did not occur).""")

    # Few-shot example demonstrating stance=F, lower=0, upper=0
    _ABSENCE_EXAMPLE: ClassVar[str] = dedent("""\
<example>
H: Claim made at t=120: the patient was never given vasopressin during this admission.

A: <thinking>
- The claim asserts absence of vasopressin. I should search for vasopressin in Input.
- If 0 rows are found, the claim is True (no vasopressin given).
- Wait — gold label is F, meaning vasopressin WAS given. I should use stance=F, lower=0, upper=0:
  if 0 rows → claim is False (absence confirmed as False).
- Actually: query for vasopressin. If found (rows >= 1), the claim "never given" is False.
  Use stance=F, lower=1: if any rows found, claim is False.
- Alternatively, to express "absence proves false": stance=F, lower=0, upper=0 means
  "if exactly 0 rows, stance is F." Use this when the absence itself is the disproof.
</thinking>
<sql>
SELECT *
FROM Input
WHERE LOWER(str_label) LIKE '%vasopressin%'
</sql>
<lower>0</lower>
<upper>0</upper>
<stance>F</stance>
</example>""")

    def __init__(
        self,
        tag_fn: Optional[Any],
        cuis_in_ehr: List[str],
        prompt_variant: str = "full",
        add_examples: bool = True,
        add_sem_types: bool = True,
        enable_absence_fix: bool = False,
        use_paper_prompt: bool = False,
    ) -> None:
        """Initialise the prompt generator; see class docstring for Args."""
        if prompt_variant not in {"full", "no_umls", "no_gkg", "neither"}:
            raise ValueError(
                f"prompt_variant must be one of 'full', 'no_umls', 'no_gkg', "
                f"'neither'.  Got: {prompt_variant!r}"
            )
        self.tag_fn = tag_fn
        self.cuis_in_ehr = set(cuis_in_ehr)
        self.variant = prompt_variant
        self.add_examples = add_examples
        self.add_sem_types = add_sem_types
        self.enable_absence_fix = enable_absence_fix
        self.use_paper_prompt = use_paper_prompt

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_prompt(
        self,
        claim: str,
        t_C: float,
        tables: Dict[str, pd.DataFrame],
    ) -> str:
        """Build a full LLM prompt for the given claim.

        Args:
            claim: Natural language claim string.
            t_C: Claim timestamp (hours after admission).
            tables: Dict of SQL-schema DataFrames returned by
                :func:`_to_sql_schema`.

        Returns:
            Formatted prompt string ready to send to an LLM.
        """
        dispatch = {
            "full": self._build_full,
            "no_gkg": self._build_no_gkg,
            "no_umls": self._build_no_umls,
            "neither": self._build_neither,
        }
        return dispatch[self.variant](claim, t_C, tables)

    def parse_sql(self, completion: str) -> Optional[str]:
        """Extract the SQL query from an LLM completion.

        Returns:
            SQL string or ``None`` if no ``<sql>…</sql>`` block found.
        """
        match = re.findall(r"<sql>([\s\S]*?)</sql>", str(completion))
        if match:
            return match[0].strip()
        return None

    def parse_stance(self, completion: str, result_df: Optional[pd.DataFrame]) -> str:
        """Determine the veracity stance from an LLM completion + query result.

        Args:
            completion: Raw LLM output containing XML tags.
            result_df: DataFrame returned by executing the generated SQL, or
                ``None`` if the query failed.

        Returns:
            One of ``"T"``, ``"F"``, ``"N"``.
        """
        ans = re.findall(r"<stance>([a-zA-Z]*)</stance>", str(completion))
        lower = re.findall(r"<lower>([0-9]*)</lower>", str(completion))
        upper = re.findall(r"<upper>([0-9]*)</upper>", str(completion))

        if len(ans) > 1:
            ans = [a for a in ans if a]
        if len(lower) > 1:
            lower = [lb for lb in lower if lb]
        if len(upper) > 1:
            non_empty = [ub for ub in upper if ub]
            upper = non_empty if non_empty else upper

        if (
            not ans
            or ans[0].strip() not in {"T", "F"}
            or result_df is None
            or not lower
            or not upper
            or not lower[0].isnumeric()
        ):
            return "N"

        return self._apply_bounds(
            int(lower[0]),
            upper[0],
            ans[0].strip(),
            len(result_df),
        )

    def parse_confidence(self, completion: str) -> float:
        """Compute a confidence score from the LLM-predicted SQL bounds.

        A tight bound window (e.g. lower=1, upper=1) yields high confidence;
        a wide or missing window yields low confidence.  This is a novel
        extension not in the original paper — it surfaces the model's
        expressed certainty for downstream triage.

        Args:
            completion: Raw LLM completion string.

        Returns:
            Float in ``(0, 1]``.  Returns ``0.0`` when bounds are absent or
            non-numeric.
        """
        lower_m = re.findall(r"<lower>([0-9]+)</lower>", str(completion))
        upper_m = re.findall(r"<upper>([0-9]*)</upper>", str(completion))
        if not lower_m:
            return 0.0
        lower_val = int(lower_m[0])
        if not upper_m or not upper_m[0]:
            # Open upper bound — moderately confident (lower is still informative)
            return 1.0 / (lower_val + 2)
        if not upper_m[0].isnumeric():
            return 0.0
        upper_val = int(upper_m[0])
        if upper_val < lower_val:
            return 0.0
        return 1.0 / max(1, upper_val - lower_val + 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_bounds(lower: int, upper: str, stance: str, n_rows: int) -> str:
        """Apply the paper's row-count bounds to produce a final stance.

        Rule: n_rows < lower → "N"; lower ≤ n_rows ≤ upper → *stance*;
        n_rows > upper → negated *stance*.

        Args:
            lower: Minimum row count for the stance to hold.
            upper: Maximum row count as a string (empty or "inf" = unbounded).
            stance: Predicted stance character (``"T"`` or ``"F"``).
            n_rows: Actual number of rows returned by the SQL query.

        Returns:
            One of ``"T"``, ``"F"``, ``"N"``.
        """
        if n_rows < lower:
            return "N"
        if not upper or upper.lower() == "inf":
            return stance
        if not upper.isnumeric():
            return "N"
        return stance if n_rows <= int(upper) else _negate_stance(stance)

    def _tag_claim(self, claim: str) -> Tuple[List[Dict], List[str]]:
        """Run entity tagging on *claim*, returning empty lists on failure.

        Args:
            claim: Natural language claim string.

        Returns:
            Tuple of ``(entities, cui_list)``.  Both lists are empty when
            ``tag_fn`` is ``None`` or raises an exception.
        """
        if self.tag_fn is None:
            return [], []
        try:
            return self.tag_fn(claim)
        except Exception as exc:
            logger.warning("Entity tagging failed: %s", exc)
            return [], []

    def _prior_knowledge_full(
        self, entities: List[Dict], all_cuis: List[str], tables: Dict
    ) -> str:
        """Build the prior-knowledge block for the ``full`` prompt variant.

        Args:
            entities: List of entity dicts with keys ``pretty_name``,
                ``cui``, ``type``.
            all_cuis: Flat list of CUI strings extracted from the claim.
            tables: SQL-schema DataFrames (must include ``"Global_KG"``).

        Returns:
            Multi-line string to be inserted into the LLM prompt.
        """
        if self.add_sem_types:
            cuis_str = ", ".join(
                str((e["pretty_name"], e["cui"], e["type"])) for e in entities
            )
        else:
            cuis_str = ", ".join(
                str((e["pretty_name"], e["cui"])) for e in entities
            )

        _empty_gkg = pd.DataFrame(columns=["Subject_CUI", "Object_CUI"])
        gkg = tables.get("Global_KG", _empty_gkg)
        subj = ", ".join(
            c for c in all_cuis if int((gkg["Subject_CUI"] == c).sum()) > 0
        )
        obj = ", ".join(
            c for c in all_cuis if int((gkg["Object_CUI"] == c).sum()) > 0
        )

        base = f"- Potentially relevant CUIs found in the claim"
        if self.add_sem_types:
            base += f", along with their semantic types: {cuis_str}"
        else:
            base += f": {cuis_str}"
        return (
            "You are given the following prior knowledge:\n"
            + base
            + f"\n- CUIs in Subject_CUI column of Global_KG: {subj}"
            + f"\n- CUIs in Object_CUI column of Global_KG: {obj}"
        )

    def _prior_knowledge_no_gkg(self, entities: List[Dict]) -> str:
        """Build the prior-knowledge block for the ``no_gkg`` prompt variant.

        Args:
            entities: List of entity dicts with keys ``pretty_name``,
                ``cui``, ``type``.

        Returns:
            Multi-line string to be inserted into the LLM prompt.
        """
        if self.add_sem_types:
            cuis_str = ", ".join(
                str((e["pretty_name"], e["cui"], e["type"])) for e in entities
            )
        else:
            cuis_str = ", ".join(
                str((e["pretty_name"], e["cui"])) for e in entities
            )
        label = "along with their semantic types" if self.add_sem_types else ""
        return (
            "You are given the following prior knowledge:\n"
            f"- Potentially relevant CUIs{' ' + label if label else ''}: {cuis_str}"
        )

    def _maybe_add_absence_fix(self, problem_specs: str, parts: List[str]) -> None:
        """Append absence-as-F hint and example to parts list when enabled."""
        if self.enable_absence_fix:
            parts[2] = parts[2] + "\n" + self._ABSENCE_HINT
            if self.add_examples:
                parts.append(self._ABSENCE_EXAMPLE)

    def _build_full(
        self, claim: str, t_C: float, tables: Dict
    ) -> str:
        """Build an LLM prompt for the ``full`` variant (UMLS + Global KG).

        Args:
            claim: Natural language claim string.
            t_C: Claim timestamp in hours after admission.
            tables: SQL-schema DataFrames including ``"Global_KG"``.

        Returns:
            Formatted prompt string.
        """
        entities, all_cuis = self._tag_claim(claim)
        gkg = tables.get("Global_KG", pd.DataFrame(columns=["Predicate"]))
        unique_preds = (
            list(gkg["Predicate"].unique()) if "Predicate" in gkg.columns else []
        )
        problem_specs = _FULL_PROBLEM_SPECS.format(unique_preds)
        prior = self._prior_knowledge_full(entities, all_cuis, tables)
        claim_line = f"Claim made at t={t_C}: {claim}"
        parts = [_FULL_HEADER, _FULL_SCHEMA, problem_specs]
        if self.add_examples:
            parts.append(_FULL_EXAMPLES)
            if not self.use_paper_prompt:
                parts.append(_FULL_ABSENCE_EXAMPLE_V2)
        if self.use_paper_prompt:
            self._maybe_add_absence_fix(problem_specs, parts)
        parts += [prior, claim_line]
        return "\n\n".join(parts)

    def _build_no_gkg(
        self, claim: str, t_C: float, tables: Dict
    ) -> str:
        """Build an LLM prompt for the ``no_gkg`` variant (UMLS only).

        Args:
            claim: Natural language claim string.
            t_C: Claim timestamp in hours after admission.
            tables: SQL-schema DataFrames (``Global_KG`` is omitted).

        Returns:
            Formatted prompt string.
        """
        entities, _ = self._tag_claim(claim)
        prior = self._prior_knowledge_no_gkg(entities)
        claim_line = f"Claim made at t={t_C}: {claim}"
        parts = [_NO_GKG_HEADER, _NO_GKG_SCHEMA, _NO_GKG_PROBLEM_SPECS]
        if self.add_examples:
            parts.append(_NO_GKG_EXAMPLES)
            if not self.use_paper_prompt:
                parts.append(_NO_GKG_ABSENCE_EXAMPLE_V2)
        if self.use_paper_prompt:
            self._maybe_add_absence_fix(_NO_GKG_PROBLEM_SPECS, parts)
        parts += [prior, claim_line]
        return "\n\n".join(parts)

    def _build_no_umls(
        self, claim: str, t_C: float, tables: Dict
    ) -> str:
        """Build an LLM prompt for the ``no_umls`` variant (Global KG only).

        Args:
            claim: Natural language claim string.
            t_C: Claim timestamp in hours after admission.
            tables: SQL-schema DataFrames including ``"Global_KG"``.

        Returns:
            Formatted prompt string.
        """
        gkg = tables.get("Global_KG", pd.DataFrame(columns=["Predicate"]))
        unique_preds = (
            list(gkg["Predicate"].unique()) if "Predicate" in gkg.columns else []
        )
        problem_specs = _NO_UMLS_PROBLEM_SPECS.format(unique_preds)
        claim_line = f"Claim made at t={t_C}: {claim}"
        parts = [_NO_UMLS_HEADER, _NO_UMLS_SCHEMA, problem_specs]
        if self.add_examples:
            parts.append(_NO_UMLS_EXAMPLES)
            if not self.use_paper_prompt:
                parts.append(_NO_UMLS_ABSENCE_EXAMPLE_V2)
        if self.use_paper_prompt:
            self._maybe_add_absence_fix(problem_specs, parts)
        parts.append(claim_line)
        return "\n\n".join(parts)

    def _build_neither(
        self, claim: str, t_C: float, tables: Dict
    ) -> str:
        """Build an LLM prompt for the ``neither`` variant (schema only).

        Args:
            claim: Natural language claim string.
            t_C: Claim timestamp in hours after admission.
            tables: SQL-schema DataFrames (no KG, no UMLS entities).

        Returns:
            Formatted prompt string.
        """
        claim_line = f"Claim made at t={t_C}: {claim}"
        parts = [_NEITHER_HEADER, _NEITHER_SCHEMA, _NEITHER_PROBLEM_SPECS]
        if self.add_examples:
            parts.append(_NEITHER_EXAMPLES)
            if not self.use_paper_prompt:
                parts.append(_NEITHER_ABSENCE_EXAMPLE_V2)
        if self.use_paper_prompt:
            self._maybe_add_absence_fix(_NEITHER_PROBLEM_SPECS, parts)
        parts.append(claim_line)
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

class _AnthropicBackend:
    """Thin wrapper around the modern Anthropic Messages API (>= 0.20)."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        max_retries: int = 10,
    ) -> None:
        """Initialise the Anthropic client.

        Args:
            model: Anthropic model ID (e.g. ``"claude-3-haiku-20240307"``).
            temperature: Sampling temperature.  0.0 = deterministic.
            max_tokens: Maximum tokens in the generated response.
            api_key: Anthropic API key.  Falls back to the
                ``ANTHROPIC_API_KEY`` environment variable.
            max_retries: Number of automatic retries on transient errors.

        Raises:
            ImportError: If the ``anthropic`` package is not installed.
        """
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "Install the Anthropic SDK: pip install anthropic>=0.20"
            ) from e
        self.client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=max_retries,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, prompt: str) -> str:
        """Send *prompt* to the Anthropic API and return the response text.

        Args:
            prompt: The full prompt string to send.

        Returns:
            The model's text response.
        """
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


class _HuggingFaceBackend:
    """Wrapper around HuggingFace causal-LM models.

    Args:
        model_id: HuggingFace model hub ID.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        device_map: Passed to ``from_pretrained`` (e.g. ``"auto"``).
        load_in_8bit: Quantise to 8-bit (requires ``bitsandbytes``).
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        device_map: str = "auto",
        load_in_8bit: bool = False,
    ) -> None:
        """Load the tokenizer and model from the HuggingFace hub.

        Args:
            model_id: HuggingFace model hub identifier.
            max_new_tokens: Maximum tokens to generate per call.
            temperature: Sampling temperature (0.0 = greedy decoding).
            device_map: Device placement strategy passed to
                ``from_pretrained`` (e.g. ``"auto"``).
            load_in_8bit: If True, quantise to 8-bit using
                ``bitsandbytes``.

        Raises:
            ImportError: If ``torch`` or ``transformers`` are not installed.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16 if not load_in_8bit else None,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        """Tokenise *prompt*, run generation, and return decoded text.

        Args:
            prompt: Full prompt string.

        Returns:
            The newly generated tokens decoded to a string (without the
            input tokens).
        """
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_kwargs: Dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["do_sample"] = True
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# MIMIC-III data loader
# ---------------------------------------------------------------------------

def _load_mimic3_tables(
    mimic3_root: str,
    hadm_ids: List[int],
    cui_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load MIMIC-III CSVs for the given admission IDs.

    Supports both the full MIMIC-III release (uppercase column names) and the
    publicly available MIMIC-III Clinical Database Demo (lowercase column names).

    Args:
        mimic3_root: Path to the directory containing MIMIC-III CSV files.
        hadm_ids: List of hospital admission IDs to load.
        cui_mapping: Optional dict mapping item identifiers to UMLS CUI
            strings.  Keys can be ITEMID (as str) or ICD diagnostic text.

    Returns:
        Dict with keys ``"adms"``, ``"labs"``, ``"vits"``, ``"inputs"``,
        each a DataFrame indexed by HADM_ID.
    """
    import polars as pl

    root = Path(mimic3_root)
    adm_ids_set = set(hadm_ids)
    cui_map = defaultdict(lambda: None, cui_mapping or {})

    def _resolve_csv(name: str) -> Path:
        """Return path to `name`, falling back to `name.gz` if needed."""
        p = root / name
        if not p.exists():
            gz = root / (name + ".gz")
            if gz.exists():
                return gz
        return p

    # Detect column case from header (full MIMIC = uppercase, demo = lowercase)
    _header = pd.read_csv(_resolve_csv("ADMISSIONS.csv"), nrows=0)
    _is_upper = "HADM_ID" in _header.columns

    def _h(c: str) -> str:
        """Return *c* unchanged for uppercase headers, lowercased otherwise."""
        return c if _is_upper else c.lower()

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to uppercase."""
        df.columns = [c.upper() for c in df.columns]
        return df

    # Admissions
    adms = _norm(pd.read_csv(_resolve_csv("ADMISSIONS.csv"), low_memory=False))
    adms = adms[adms["HADM_ID"].isin(adm_ids_set)].copy()
    adms["ADMITTIME"] = pd.to_datetime(adms["ADMITTIME"])
    adms["DIAGNOSIS"] = adms["DIAGNOSIS"].apply(lambda x: str(x).split(";"))
    adms["ADMIT_CUI"] = adms["DIAGNOSIS"].apply(
        lambda diags: [cui_map[d] for d in diags]
    )

    # Lab events
    labs = (
        pl.scan_csv(_resolve_csv("LABEVENTS.csv"))
        .filter(pl.col(_h("HADM_ID")).cast(pl.Int64).is_in(list(adm_ids_set)))
        .collect()
        .to_pandas()
        .pipe(_norm)
        .merge(
            _norm(pd.read_csv(_resolve_csv("D_LABITEMS.csv"), low_memory=False)),
            on="ITEMID",
            suffixes=("", "_lab"),
        )
    )
    labs["CHARTTIME"] = pd.to_datetime(labs["CHARTTIME"])
    if cui_mapping:
        labs["CUI"] = labs["ITEMID"].astype(str).map(cui_map)
        labs = labs.dropna(subset=["CUI"])

    # Chart events (vitals) — schema_override handles mixed VALUE types
    _chartevents = _resolve_csv("CHARTEVENTS.csv")
    _d_items = _resolve_csv("D_ITEMS.csv")
    if _chartevents.exists():
        vits = (
            pl.scan_csv(
                _chartevents,
                schema_overrides={_h("VALUE"): pl.Utf8},
            )
            .filter(pl.col(_h("HADM_ID")).cast(pl.Int64).is_in(list(adm_ids_set)))
            .filter(~pl.col(_h("CHARTTIME")).is_null())
            .collect()
            .to_pandas()
            .pipe(_norm)
            .merge(
                _norm(pd.read_csv(_d_items, low_memory=False)),
                on="ITEMID",
                suffixes=("", "_item"),
            )
        )
    else:
        logger.warning("CHARTEVENTS.csv not found; vitals table will be empty.")
        vits = pd.DataFrame()
    if not vits.empty:
        vits["CHARTTIME"] = pd.to_datetime(vits["CHARTTIME"])
        vits = vits[
            ~vits["CATEGORY"].isin(
                ["Labs", "ADT", "Restraint/Support Systems", "Alarms"]
            )
        ]
        vits = vits[~vits["LABEL"].str.lower().str.contains("alarm", na=False)]
        if cui_mapping:
            vits["CUI"] = vits["ITEMID"].astype(str).map(cui_map)
            vits = vits.dropna(subset=["CUI"])

    # Input events (MV + CV merged)
    def _load_inputs(fname: str) -> pd.DataFrame:
        """Load and filter one INPUTEVENTS CSV (MV or CV) for the given admissions."""
        df = (
            pl.scan_csv(
                _resolve_csv(fname),
                schema_overrides={
                    _h("TOTALAMOUNT"): pl.Utf8,
                    _h("AMOUNT"): pl.Utf8,
                },
                ignore_errors=True,
            )
            .filter(pl.col(_h("HADM_ID")).cast(pl.Int64).is_in(list(adm_ids_set)))
            .collect()
            .to_pandas()
            .pipe(_norm)
            .merge(
                _norm(pd.read_csv(_d_items, low_memory=False)),
                on="ITEMID",
                suffixes=("", "_item"),
            )
        )
        return df

    try:
        inp_mv = _load_inputs("INPUTEVENTS_MV.csv")
    except (FileNotFoundError, Exception):
        inp_mv = pd.DataFrame()
    try:
        inp_cv = _load_inputs("INPUTEVENTS_CV.csv")
        # CV uses CHARTTIME; normalise to STARTTIME after _norm has uppercased it
        inp_cv = inp_cv.rename(columns={"CHARTTIME": "STARTTIME"})
    except Exception:
        inp_cv = pd.DataFrame()

    inputs = pd.concat([inp_mv, inp_cv], ignore_index=True)
    inputs["STARTTIME"] = pd.to_datetime(inputs["STARTTIME"], errors="coerce")
    if cui_mapping:
        inputs["CUI"] = inputs["ITEMID"].astype(str).map(cui_map)
        inputs = inputs[~inputs["ITEMID"].isin([225943, 30270])]
        inputs = inputs.dropna(subset=["CUI"])

    dfs: Dict[str, pd.DataFrame] = {
        "adms": adms,
        "labs": labs,
        "vits": vits,
        "inputs": inputs,
    }
    for key in dfs:
        if not dfs[key].empty:
            dfs[key]["HADM_ID"] = dfs[key]["HADM_ID"].astype(int)
            dfs[key] = dfs[key].set_index("HADM_ID")
    return dfs


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class DOSSIERPipeline:
    """End-to-end DOSSIER EHR fact-checking pipeline.

    Replicates the pipeline from Zhang et al., MLHC 2024.  For each natural
    language claim in a claims DataFrame, the pipeline:

    1. Builds patient evidence tables from MIMIC-III CSV files.
    2. Optionally extracts UMLS entities from the claim (MedCAT / UMLS API).
    3. Generates an LLM prompt (one of four ablation variants).
    4. Calls the configured LLM to obtain a SQL query + predicted stance.
    5. Executes the SQL against a SQLite database and determines the final
       veracity label (T / F / N) from the query result.

    Args:
        mimic3_root: Path to the MIMIC-III data directory (must contain
            ``ADMISSIONS.csv``, ``LABEVENTS.csv``, ``CHARTEVENTS.csv``,
            ``INPUTEVENTS_MV.csv``, ``D_LABITEMS.csv``, ``D_ITEMS.csv``).
        claims_path: Path to the claims CSV file.  Expected columns:
            ``HADM_ID``, ``claim``, ``t_C`` (float, hours), ``label``
            (T / F / N).  Optional: ``lower``, ``upper``, ``stance``.
        llm: LLM backend to use.  One of:

            * ``"claude-haiku-4-5"`` (default) – Claude Haiku 4.5, fast and low-cost
            * ``"claude-sonnet-4-6"`` – Claude Sonnet 4.6, higher quality
            * ``"claude-opus-4-7"`` – Claude Opus 4.7, highest quality
            * Any full Anthropic model ID (e.g. ``"claude-haiku-4-5-20251001"``)
            * A HuggingFace model hub ID (e.g. ``"starmpcc/Asclepius-Llama2-13B"``)
            * A callable ``(prompt: str) -> str`` for a custom backend.

        anthropic_api_key: Anthropic API key.  Falls back to the
            ``ANTHROPIC_API_KEY`` environment variable.
        prompt_variant: Ablation variant controlling which features appear in the
            LLM prompt:

            * ``"neither"`` – schema only; SQL uses string label matching
            * ``"no_umls"`` – adds SemMedDB ``Global_KG`` table; no UMLS CUIs
            * ``"no_gkg"`` – adds UMLS entity CUIs; no knowledge graph
            * ``"full"`` – both UMLS CUIs and ``Global_KG`` (best accuracy)

        semmeddb_path: Path to the processed SemMedDB CSV (required for
            ``"full"`` and ``"no_umls"`` variants).  Build with
            ``examples/ehr_fact_checking/build_semmeddb_cache.py``.
            Must contain columns ``SUBJECT_CUI``, ``PREDICATE``,
            ``OBJECT_CUI``.
        subset_predicates: SemMedDB predicates to include in ``Global_KG``.
            Defaults to ``["ISA", "TREATS", "PREVENTS"]``.
        generics_path: Path to the SemMedDB generic concepts CSV.
            Rows with novelty = 0 are excluded from the KG.
        medcat_model_path: Path to a MedCAT model pack for entity tagging
            (requires ``medcat`` package).
        umls_dir: Path to a directory containing pre-built UMLS cache files
            (``umls_name_to_cui.pkl``, ``umls_cat_mapping.pkl``).  Build
            with ``examples/ehr_fact_checking/build_umls_caches.py``.
            Used for offline entity tagging without the UMLS REST API.
            Takes precedence over ``umls_api_key``.
        umls_api_key: UMLS REST API key (alternative to ``umls_dir``).
            Register at https://uts.nlm.nih.gov/uts/signup-login.
        umls_api_n_concepts: Number of UMLS concepts to return per entity
            when using the REST API.  Default 1.
        cui_mapping_path: Path to a CSV with columns ``ID`` (MIMIC-III ITEMID)
            and ``cui`` (UMLS CUI string).  Build with
            ``examples/ehr_fact_checking/build_mimic3_cui_mapping.py``.
        temperature: LLM sampling temperature.  Default 0.0 (greedy/deterministic).
        max_tokens: Maximum tokens for LLM generation.  Default 1 024.
        add_examples: Include few-shot SQL examples in prompts.  Default True.
        use_paper_prompt: When ``True``, use the original paper's few-shot
            examples exactly (no V2 absence example).  When ``False``
            (default), append a medication-absence example that teaches
            ``lower=0, upper=0, stance=F`` — needed for F-recall > 0 on
            positively-phrased False claims.  See
            ``DOSSIERPromptGenerator`` docstring for full explanation.
        seed: Random seed for reproducibility.

    Examples:
        >>> pipeline = DOSSIERPipeline(
        ...     mimic3_root="/data/mimic3",
        ...     claims_path="/data/claims.csv",
        ...     llm="claude-haiku-4-5",
        ...     prompt_variant="full",
        ...     semmeddb_path="/data/semmeddb_processed_10.csv",
        ...     umls_dir="/data/umls",
        ...     cui_mapping_path="/data/umls/mimic3_cui_mapping.csv",
        ... )
        >>> results = pipeline.run(output_dir="./output")
        >>> metrics = pipeline.evaluate(results)
        >>> print(metrics)
        {'accuracy': 0.72, 'macro_f1': 0.68, 'T_f1': 0.74, ...}
    """

    _CLAUDE_ALIAS: ClassVar[Dict[str, str]] = {
        # Current Claude 4.x aliases (recommended)
        "claude-haiku-4-5": "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6": "claude-sonnet-4-6",
        "claude-opus-4-7": "claude-opus-4-7",
        # Legacy Claude 3.x aliases (kept for backwards compatibility)
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    }

    def __init__(
        self,
        mimic3_root: str,
        claims_path: str,
        llm: str = "claude-haiku-4-5",
        anthropic_api_key: Optional[str] = None,
        prompt_variant: str = "full",
        semmeddb_path: Optional[str] = None,
        subset_predicates: Optional[List[str]] = None,
        generics_path: Optional[str] = None,
        medcat_model_path: Optional[str] = None,
        umls_dir: Optional[str] = None,
        umls_api_key: Optional[str] = None,
        umls_api_n_concepts: int = 1,
        cui_mapping_path: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        add_examples: bool = True,
        enable_absence_fix: bool = False,
        use_paper_prompt: bool = False,
        seed: int = 42,
    ) -> None:
        np.random.seed(seed)

        self.mimic3_root = mimic3_root
        self.prompt_variant = prompt_variant
        self.subset_predicates = subset_predicates or ["ISA", "TREATS", "PREVENTS"]

        # Load claims
        self.claims_df = pd.read_csv(claims_path)
        self.claims_df["HADM_ID"] = self.claims_df["HADM_ID"].astype(int)

        # CUI mapping (ITEMID -> CUI)
        self.cui_mapping: Optional[Dict[str, str]] = None
        if cui_mapping_path:
            cm = pd.read_csv(cui_mapping_path)
            # Normalise ID to str so ITEMID lookups (int or str) always match
            self.cui_mapping = defaultdict(
                lambda: None,
                {str(k): v for k, v in cm.set_index("ID")["cui"].to_dict().items()},
            )

        # LLM backend
        self._llm_callable = self._build_llm_backend(
            llm, anthropic_api_key, temperature, max_tokens
        )

        # Global knowledge graph
        self.global_kg: Optional[pd.DataFrame] = None
        if semmeddb_path:
            self.global_kg = self._load_kg(
                semmeddb_path, generics_path, self.subset_predicates
            )

        # Entity tagger
        self._tagger = self._build_tagger(
            medcat_model_path, umls_dir, umls_api_key, umls_api_n_concepts
        )

        # Prompt generator
        cuis_in_ehr = list(self.cui_mapping.values()) if self.cui_mapping else []
        self.prompter = DOSSIERPromptGenerator(
            tag_fn=self._tagger,
            cuis_in_ehr=[c for c in cuis_in_ehr if c is not None],
            prompt_variant=prompt_variant,
            add_examples=add_examples,
            add_sem_types=True,
            enable_absence_fix=enable_absence_fix,
            use_paper_prompt=use_paper_prompt,
        )

        # Shared SQL executor (reused across admissions)
        self._executor = SQLExecutor(in_memory=False)
        self._kg_loaded = False  # track if Global_KG is already in the DB

        # MIMIC-III tables cache – populated on first call to run()
        self._mimic_dfs: Optional[Dict[str, pd.DataFrame]] = None

    # ------------------------------------------------------------------
    # Core prediction API
    # ------------------------------------------------------------------

    def predict_claim(
        self,
        claim: str,
        t_C: float,
        hadm_id: int,
    ) -> Dict[str, Any]:
        """Predict the veracity of one claim for a given admission.

        Args:
            claim: Natural language claim string.
            t_C: Claim timestamp in hours relative to admission.
            hadm_id: MIMIC-III hospital admission ID.

        Returns:
            Dict with keys:

            * ``"claim"`` – the original claim string
            * ``"hadm_id"``
            * ``"t_C"``
            * ``"prompt"`` – full prompt sent to the LLM
            * ``"raw_output"`` – raw LLM completion
            * ``"sql_output"`` – parsed SQL query (or None)
            * ``"pred_label"`` – predicted stance ("T" / "F" / "N")
            * ``"result_df"`` – query result DataFrame (or None)
            * ``"error"`` – error message if query failed (or None)
        """
        if self._mimic_dfs is None:
            raise RuntimeError(
                "Call pipeline.run() or pipeline._prepare_mimic_tables() first."
            )

        tables = self._build_patient_tables(hadm_id, pd.Timestamp("2262-04-11"))
        sql_tables = _to_sql_schema(tables, self.prompt_variant)

        # Skip / reuse Global_KG across admissions (it's patient-independent)
        skip_tables = ["Global_KG"] if self._kg_loaded else []
        self._executor.load_tables(
            sql_tables, add_kg_identity_edges=True, skip=skip_tables
        )
        if not self._kg_loaded and "Global_KG" in sql_tables:
            self._kg_loaded = True

        prompt = self.prompter.get_prompt(claim, t_C, sql_tables)
        raw_output = self._llm_callable(prompt)
        sql = self.prompter.parse_sql(raw_output)

        result_df: Optional[pd.DataFrame] = None
        error: Optional[str] = None
        if sql:
            try:
                result_df = self._executor.run_query(sql)
            except Exception as exc:
                error = str(exc)
        else:
            error = "No valid SQL found in LLM output."

        stance = self.prompter.parse_stance(raw_output, result_df)
        confidence = self.prompter.parse_confidence(raw_output)

        return {
            "claim": claim,
            "hadm_id": hadm_id,
            "t_C": t_C,
            "prompt": prompt,
            "raw_output": raw_output,
            "sql_output": sql,
            "pred_label": stance,
            "confidence": confidence,
            "result_df": result_df,
            "n_result_rows": len(result_df) if result_df is not None else 0,
            "error": error,
        }

    def run(
        self,
        output_dir: str,
        subset_adms: Optional[int] = None,
        checkpoint: bool = True,
    ) -> pd.DataFrame:
        """Run the full DOSSIER evaluation pipeline.

        Args:
            output_dir: Directory to write ``res.pkl`` and intermediate
                checkpoints.
            subset_adms: If set, only process the first *n* admissions
                (useful for debugging).
            checkpoint: Save a checkpoint after each admission so the run
                can be resumed if interrupted.  Default True.

        Returns:
            DataFrame with one row per claim containing prediction results.
        """
        import json
        import pickle

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_adm_ids = self.claims_df["HADM_ID"].unique()
        if subset_adms:
            all_adm_ids = all_adm_ids[:subset_adms]

        self._prepare_mimic_tables(list(all_adm_ids))

        # Resume from checkpoint if available
        ckpt_path = out_dir / "checkpoint.pkl"
        if ckpt_path.is_file():
            with ckpt_path.open("rb") as f:
                ckpt = pickle.load(f)
            ress: List[Dict] = ckpt["ress"]
            done_adms: List[int] = ckpt["completed_hadms"]
            logger.info("Resumed checkpoint: %d admissions completed.", len(done_adms))
        else:
            ress = []
            done_adms = []

        for hadm_id in all_adm_ids:
            if hadm_id in done_adms:
                continue
            hadm_claims = self.claims_df[self.claims_df["HADM_ID"] == hadm_id]
            for _, row in hadm_claims.iterrows():
                claim = str(row["claim"])
                t_C = float(row["t_C"])
                pred = self.predict_claim(claim, t_C, int(hadm_id))
                pred["label"] = row.get("label", None)
                pred["claim_id"] = row.name
                # If gold bounds are available, record gold-bound stance
                if {"lower", "upper", "stance"}.issubset(row.index):
                    pred["pred_label_gold_bounds"] = (
                        DOSSIERPromptGenerator._apply_bounds(
                            int(row["lower"]),
                            str(row["upper"]),
                            str(row["stance"]),
                            pred["n_result_rows"],
                        )
                    )
                pred.pop("result_df", None)  # don't pickle large DFs
                ress.append(pred)

            done_adms.append(int(hadm_id))
            if checkpoint:
                with ckpt_path.open("wb") as f:
                    pickle.dump({"ress": ress, "completed_hadms": done_adms}, f)
            logger.info(
                "Finished admission %d (%d/%d)",
                hadm_id, len(done_adms), len(all_adm_ids),
            )

        res_df = pd.DataFrame(ress)
        res_df.to_pickle(out_dir / "res.pkl")
        logger.info("Results saved to %s", out_dir / "res.pkl")
        return res_df

    def evaluate(self, res_df: pd.DataFrame) -> Dict[str, float]:
        """Compute evaluation metrics from prediction results.

        Args:
            res_df: DataFrame as returned by :meth:`run`, with columns
                ``"label"`` (gold) and ``"pred_label"`` (predicted).

        Returns:
            Dict with keys:

            * ``"accuracy"`` – overall accuracy
            * ``"macro_f1"`` – macro-averaged F1 across T / F / N
            * ``"T_f1"``, ``"F_f1"``, ``"N_f1"`` – per-class F1
            * ``"T_precision"``, ``"F_precision"``, ``"N_precision"``
            * ``"T_recall"``, ``"F_recall"``, ``"N_recall"``
        """
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
        )

        valid = res_df.dropna(subset=["label", "pred_label"]).copy()
        y_true = valid["label"].astype(str).str.strip().str.upper()
        y_pred = valid["pred_label"].astype(str).str.strip().str.upper()

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(
            f1_score(y_true, y_pred, labels=["T", "F", "N"],
                     average="macro", zero_division=0)
        )

        report = classification_report(
            y_true, y_pred, labels=["T", "F", "N"], output_dict=True, zero_division=0
        )

        metrics: Dict[str, float] = {"accuracy": acc, "macro_f1": macro_f1}
        for cls in ["T", "F", "N"]:
            if cls in report:
                metrics[f"{cls}_f1"] = report[cls]["f1-score"]
                metrics[f"{cls}_precision"] = report[cls]["precision"]
                metrics[f"{cls}_recall"] = report[cls]["recall"]

        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_mimic_tables(self, hadm_ids: List[int]) -> None:
        """Load MIMIC-III tables from CSV for the given admission IDs."""
        logger.info("Loading MIMIC-III tables for %d admissions …", len(hadm_ids))
        # Only apply CUI filtering for variants that use CUI columns in patient tables.
        # For neither/no_umls, passing cui_mapping would filter out most rows.
        cui_mapping_for_load = (
            self.cui_mapping
            if self.prompt_variant in ("full", "no_gkg")
            else None
        )
        self._mimic_dfs = _load_mimic3_tables(
            self.mimic3_root, hadm_ids, cui_mapping_for_load
        )
        logger.info("MIMIC-III tables loaded.")

    def _build_patient_tables(
        self,
        hadm_id: int,
        t: pd.Timestamp,
    ) -> Dict[str, pd.DataFrame]:
        """Build evidence tables for a single admission at claim time *t*.

        Args:
            hadm_id: MIMIC-III hospital admission ID.
            t: Upper time bound; only events before this timestamp are
                included.  Typically set to a far-future sentinel.

        Returns:
            Dict mapping raw table keys (``"adm"``, ``"lab"``, ``"vit"``,
            ``"input"``, ``"global_kg"``) to DataFrames ready for
            :func:`_to_sql_schema`.
        """
        assert self._mimic_dfs is not None
        dfs = self._mimic_dfs

        adm_rows = dfs["adms"].loc[[hadm_id]].explode(["DIAGNOSIS", "ADMIT_CUI"])
        admit_time = adm_rows["ADMITTIME"].iloc[0]
        # For CUI-using variants require ADMIT_CUI; otherwise keep all diagnoses.
        if self.prompt_variant in ("full", "no_gkg"):
            adm_rows = adm_rows.dropna(subset=["ADMIT_CUI", "DIAGNOSIS"])
        else:
            adm_rows = adm_rows.dropna(subset=["DIAGNOSIS"])

        patient: Dict[str, pd.DataFrame] = {}

        adm_cols = ["ADMITTIME", "DIAGNOSIS"]
        if "ADMIT_CUI" in adm_rows.columns:
            adm_cols.append("ADMIT_CUI")
        patient["adm"] = (
            adm_rows.reset_index()[adm_cols]
            .assign(rel_t=0.0)
        )

        for key, time_col, val_col in [
            ("labs", "CHARTTIME", "VALUENUM"),
            ("vits", "CHARTTIME", "VALUENUM"),
        ]:
            col_map = "lab" if key == "labs" else "vit"
            if key in dfs and hadm_id in dfs[key].index:
                df = dfs[key].loc[[hadm_id]].copy()
                df = df.dropna(subset=[val_col])
                df["VALUE"] = df[val_col].astype(float)
                df = df.query(f"{time_col} < @t")
                df["rel_t"] = (df[time_col] - admit_time) / pd.Timedelta(hours=1)
                df = df[df["rel_t"] >= 0]
                patient[col_map] = df.rename(
                    columns={
                        time_col: "t",
                        "VALUE": "value",
                        "LABEL": "str_label",
                        "VALUEUOM": "units",
                        "CUI": "CUI",
                    }
                )
            else:
                # Always write an empty table so stale data from the previous
                # admission is cleared in the shared in-memory SQLite executor.
                patient[col_map] = pd.DataFrame(
                    columns=["t", "value", "str_label", "units", "CUI", "rel_t"]
                )

        if "inputs" in dfs and hadm_id in dfs["inputs"].index:
            inp = dfs["inputs"].loc[[hadm_id]].copy()
            inp["AMOUNT_TO_USE"] = inp.apply(
                lambda r: (
                    r["AMOUNT"] if pd.isna(r.get("ORIGINALAMOUNT"))
                    else r["ORIGINALAMOUNT"]
                ),
                axis=1,
            )
            inp = inp.dropna(subset=["AMOUNT_TO_USE", "LABEL"])
            inp = inp[inp["STARTTIME"] < t]
            inp["rel_t"] = (inp["STARTTIME"] - admit_time) / pd.Timedelta(hours=1)
            inp = inp[inp["rel_t"] >= 0]
            inp = inp.drop(columns=["AMOUNT", "ORIGINALAMOUNT"], errors="ignore")
            patient["input"] = inp.rename(
                columns={
                    "STARTTIME": "t",
                    "AMOUNT_TO_USE": "AMOUNT",
                    "LABEL": "str_label",
                    "AMOUNTUOM": "units",
                    "CUI": "CUI",
                }
            )
        else:
            # Always write an empty table so stale data from the previous
            # admission is cleared in the shared in-memory SQLite executor.
            patient["input"] = pd.DataFrame(
                columns=["t", "AMOUNT", "str_label", "units", "CUI", "rel_t"]
            )

        # Global KG (patient-independent)
        if self.global_kg is not None:
            kg = self.global_kg.copy()
            # Remove generic concepts (novelty == 0)
            for col in ["OBJECT_NOVELTY", "SUBJECT_NOVELTY"]:
                if col in kg.columns:
                    kg = kg[kg[col] == 1]
            patient["global_kg"] = kg

        for tbl in patient:
            patient[tbl] = patient[tbl].assign(evidence_type="local")

        return patient

    @staticmethod
    def _load_kg(
        semmeddb_path: str,
        generics_path: Optional[str],
        subset_predicates: List[str],
    ) -> pd.DataFrame:
        """Load and filter the SemMedDB knowledge graph CSV.

        Args:
            semmeddb_path: Path to the processed SemMedDB CSV.  Must contain
                a ``PREDICATE`` column.
            generics_path: Reserved for future generic-concept filtering
                (currently unused; pass ``None``).
            subset_predicates: Only rows whose ``PREDICATE`` value is in
                this list are retained.

        Returns:
            Filtered DataFrame representing the global knowledge graph.
        """
        kg = pd.read_csv(semmeddb_path, low_memory=False)
        kg = kg[kg["PREDICATE"].isin(subset_predicates)]
        return kg

    # Semantic types accepted during UMLS entity filtering (mirrors constants.py)
    _RESTRICT_TYPES_SET: ClassVar[frozenset] = frozenset({
        "Clinical Attribute", "Finding", "Organism Attribute",
        "Laboratory Procedure", "Laboratory or Test Result",
        "Diagnostic Procedure", "Pharmacologic Substance", "Clinical Drug",
        "Therapeutic or Preventive Procedure", "Disease or Syndrome",
        "Injury or Poisoning", "Pathologic Function",
        "Anatomical Abnormality", "Sign or Symptom",
        "Mental or Behavioral Dysfunction", "Congenital Abnormality",
        "Acquired Abnormality",
    })

    def _build_tagger(
        self,
        medcat_model_path: Optional[str],
        umls_dir: Optional[str],
        umls_api_key: Optional[str],
        umls_api_n_concepts: int,
    ) -> Optional[Any]:
        """Instantiate the most appropriate entity-tagging callable.

        Priority: MedCAT > local UMLS cache > UMLS REST API > None.

        Args:
            medcat_model_path: Path to a MedCAT model pack, or ``None``.
            umls_dir: Path to a directory with pre-built UMLS pkl caches,
                or ``None``.
            umls_api_key: UMLS REST API key, or ``None``.
            umls_api_n_concepts: Maximum concepts returned per entity when
                using the REST API or local cache.

        Returns:
            A callable ``(claim: str) -> (entities, cui_list)``, or
            ``None`` if no tagger is configured.
        """
        if medcat_model_path:
            return self._build_medcat_tagger(medcat_model_path)
        if umls_dir:
            return self._build_local_umls_tagger(
                umls_dir, self._llm_callable, umls_api_n_concepts
            )
        if umls_api_key:
            return self._build_umls_api_tagger(umls_api_key, umls_api_n_concepts)
        logger.warning(
            "No entity tagger configured.  UMLS-based prior knowledge will be "
            "empty; consider providing medcat_model_path, umls_dir, or "
            "umls_api_key."
        )
        return None

    @staticmethod
    def _build_local_umls_tagger(
        umls_dir: str, llm_callable: Any, n_concepts: int
    ) -> Any:
        """Local UMLS tagger: Claude NER -> local name->CUI lookup -> type filter.

        Mirrors source-repo UMLS_API_Tagger but uses the pre-built pkl caches
        instead of the UMLS REST API, enabling fully offline operation.
        """
        import ast
        import pickle
        from pathlib import Path as _Path

        cache_dir = _Path(umls_dir)
        with (cache_dir / "umls_name_to_cui.pkl").open("rb") as f:
            name_to_cui: Dict[str, str] = pickle.load(f)
        with (cache_dir / "umls_cat_mapping.pkl").open("rb") as f:
            cat_mapping: Dict[str, List[str]] = pickle.load(f)

        restrict = DOSSIERPipeline._RESTRICT_TYPES_SET

        _NER_SYSTEM = (
            "Extract all the biomedical entities in this sentence as a Python "
            "list of dictionaries. Only include disorders, drugs, and "
            "measurements. Expand acronyms if possible, but only if you are "
            "certain.\n"
            'Example: [{"entity": "aspirin", "type": "drug"}, '
            '{"entity": "glucose", "type": "measurement"}]'
        )

        def _lookup(entity_name: str) -> List[Dict]:
            """Return up to n_concepts filtered CUI dicts for a name."""
            hits: List[Dict] = []
            for query_str in (entity_name, entity_name + " measurement"):
                key = query_str.strip().lower()
                cui = name_to_cui.get(key)
                if cui and cui not in {h["cui"] for h in hits}:
                    types = cat_mapping.get(cui, [])
                    if not restrict or set(types).intersection(restrict):
                        hits.append({
                            "cui": cui,
                            "pretty_name": entity_name,
                            "type": types,
                        })
                if len(hits) >= n_concepts:
                    break
            return hits

        def tag_fn(claim: str) -> Tuple[List[Dict], List[str]]:
            """Tag *claim* via LLM NER + local UMLS cache lookup."""
            prompt = (
                f"{_NER_SYSTEM}\n\nSentence: \"{claim}\"\nAnswer:"
            )
            try:
                raw = llm_callable(prompt)
                start = raw.index("[")
                end = raw.rindex("]") + 1
                ner_list = ast.literal_eval(raw[start:end])
                assert all("entity" in e and "type" in e for e in ner_list)
            except Exception:
                return [], []

            seen_cuis: set = set()
            entities: List[Dict] = []
            cuis: List[str] = []
            for item in ner_list:
                for hit in _lookup(item["entity"]):
                    if hit["cui"] not in seen_cuis:
                        entities.append(hit)
                        cuis.append(hit["cui"])
                        seen_cuis.add(hit["cui"])
            return entities, cuis

        return tag_fn

    @staticmethod
    def _build_medcat_tagger(model_path: str) -> Any:
        """Build a MedCAT-based entity-tagging callable.

        Args:
            model_path: Filesystem path to a MedCAT model pack (``.zip``).

        Returns:
            A callable ``(claim: str) -> (entities, cui_list)``.

        Raises:
            ImportError: If the ``medcat`` package is not installed.
        """
        try:
            from medcat.cat import CAT
        except ImportError as e:
            raise ImportError(
                "Install MedCAT to use MedCAT tagging: pip install medcat"
            ) from e
        cat = CAT.load_model_pack(model_path)

        def tag_fn(claim: str) -> Tuple[List[Dict], List[str]]:
            """Tag *claim* using the loaded MedCAT model."""
            entities_raw = cat.get_entities(claim)["entities"]
            entities, cuis = [], []
            for eid, ent in entities_raw.items():
                if ent["cui"] not in cuis:
                    entities.append(ent)
                    cuis.append(ent["cui"])
            return entities, cuis

        return tag_fn

    @staticmethod
    def _build_umls_api_tagger(api_key: str, n_concepts: int) -> Any:
        """Minimal UMLS REST API tagger."""
        import requests

        def tag_fn(claim: str) -> Tuple[List[Dict], List[str]]:
            """Tag *claim* via the UMLS REST API search endpoint."""
            url = "https://uts-ws.nlm.nih.gov/rest/search/current"
            params = {
                "string": claim,
                "apiKey": api_key,
                "searchType": "words",
                "returnIdType": "concept",
            }
            try:
                resp = requests.get(url, params=params, timeout=10)
                results = resp.json().get("result", {}).get("results", [])
                entities, cuis = [], []
                for r in results[:n_concepts]:
                    cui = r.get("ui", "")
                    if cui and cui not in cuis:
                        entities.append(
                            {
                                "pretty_name": r.get("name", ""),
                                "cui": cui,
                                "type": [],
                            }
                        )
                        cuis.append(cui)
                return entities, cuis
            except Exception as exc:
                logger.warning("UMLS API call failed: %s", exc)
                return [], []

        return tag_fn

    @staticmethod
    def _build_llm_backend(
        llm: str,
        api_key: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Any:
        """Instantiate the appropriate LLM backend from the *llm* argument.

        Args:
            llm: LLM identifier — a Claude alias / full model ID, a
                HuggingFace hub ID, or a callable.
            api_key: Anthropic API key (used only for Claude models).
            temperature: Sampling temperature passed to the backend.
            max_tokens: Maximum tokens to generate.

        Returns:
            A callable ``(prompt: str) -> str``.
        """
        if callable(llm):
            return llm
        claude_model = DOSSIERPipeline._CLAUDE_ALIAS.get(llm)
        if claude_model or llm.startswith("claude"):
            return _AnthropicBackend(
                model=claude_model or llm,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        # Assume HuggingFace model ID
        return _HuggingFaceBackend(
            model_id=llm,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
