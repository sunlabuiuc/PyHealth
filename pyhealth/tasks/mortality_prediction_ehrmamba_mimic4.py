"""
MIMIC-IV PyHealth tasks for EHR Mamba (V3).

Provides:
    MIMIC4EHRMambaTask          — base class: builds the token sequence for a patient; 
                                   no label.
    MIMIC4EHRMambaMortalityTask — subclass: adds in-hospital mortality label
                                   and minimum-age filter.

Both classes inherit from pyhealth.tasks.base_task.BaseTask and are intended
to be used with MIMIC4EHRDataset.set_task().

Sequence layout (paper §2.1 / Appx.E):
    [CLS]
    [VS] PR:xxx RX:xxx LB:xxx_bin2 [VE] [REG]
    [W2]
    [VS] LB:xxx_bin0 PR:xxx [VE] [REG]
    [M1]
    [VS] … [VE] [REG]

The embedding model (ehrmamba_embedding.py) is responsible for transforming the integer 
token indices and metadata produced here into dense vector representations.  
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base_task import BaseTask


# ── Token type IDs (must match MIMIC4_TOKEN_TYPES in the embedding module) ───
# Structural special tokens occupy indices 0–5; clinical types start at 6.
MIMIC4_TOKEN_TYPES: Dict[str, int] = {
    "PAD":            0,
    "CLS":            1,
    "VS":             2,
    "VE":             3,
    "REG":            4,
    "TIME_INTERVAL":  5,
    "procedures_icd": 6,
    "prescriptions":  7,
    "labevents":      8,
    "other":          9,
}

NUM_TOKEN_TYPES: int = len(MIMIC4_TOKEN_TYPES)  # 10


# ── Special token strings (paper §2.1 + Appx.E) ──────────────────────────────
_CLS_TOKEN = "[CLS]"
_VS_TOKEN  = "[VS]"
_VE_TOKEN  = "[VE]"
_REG_TOKEN = "[REG]"

# Cap on visit_order values (must match MAX_NUM_VISITS in the embedding module)
MAX_NUM_VISITS: int = 512


# ── Lab quantizer (paper Appx. B: 5-bin tokenization) ────────────────────────

class LabQuantizer:
    """Per-itemid 5-bin quantile tokenizer for MIMIC-IV lab results.

    The EHR Mamba paper (Appx. B) bins continuous lab test values into 5
    discrete tokens per itemid.  Quintile boundaries are computed from the
    training set so each bin covers ~20 % of training observations.

    Fit on training patients before calling dataset.set_task(), then pass
    the fitted instance to the task constructor.

    Workflow:
        1. Instantiate: quantizer = LabQuantizer()
        2. Fit on training patients:
               quantizer.fit_from_patients(train_patients)
           or from pre-collected records:
               quantizer.fit_from_records([(itemid, valuenum), ...])
        3. Pass to task: task = MIMIC4EHRMambaMortalityTask(lab_quantizer=quantizer)

    Token format:
        "LB:<itemid>_bin<0-4>"  when valuenum is available.
        "LB:<itemid>"           when valuenum is NULL (fallback).
    """

    def __init__(self, n_bins: int = 5) -> None:
        self.n_bins = n_bins
        # itemid (str) → sorted list of (n_bins - 1) boundary values
        self.boundaries: Dict[str, List[float]] = {}

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit_from_patients(self, patients: List[Any]) -> "LabQuantizer":
        """Compute per-itemid quantile boundaries from a list of Patient objects.

        Args:
            patients: List of PyHealth Patient objects that include a
                      "labevents" event table with "labevents/itemid" and
                      "labevents/valuenum" columns.

        Returns:
            self (for chaining)
        """
        from collections import defaultdict
        import polars as pl

        values_by_item: Dict[str, List[float]] = defaultdict(list)
        for patient in patients:
            try:
                df = patient.get_events(event_type="labevents", return_df=True)
                if df is None or df.height == 0:
                    continue
                itemid_col   = "labevents/itemid"
                valuenum_col = "labevents/valuenum"
                if itemid_col not in df.columns or valuenum_col not in df.columns:
                    continue
                rows = (
                    df.select([
                        pl.col(itemid_col).cast(pl.Utf8),
                        pl.col(valuenum_col).cast(pl.Float64),
                    ])
                    .drop_nulls()
                    .rows()
                )
                for itemid, valuenum in rows:
                    if itemid:
                        values_by_item[itemid].append(valuenum)
            except Exception:
                continue
        self._compute_boundaries(values_by_item)
        return self

    def fit_from_records(
        self, records: List[Tuple[Any, Any]]
    ) -> "LabQuantizer":
        """Compute boundaries from a pre-collected list of (itemid, valuenum) pairs.

        Useful when lab values have already been extracted into a flat list
        (e.g. from a Polars/Pandas DataFrame) without iterating Patient objects.

        Args:
            records: Iterable of (itemid, valuenum) pairs.  Non-numeric valuenum
                     entries are silently skipped.

        Returns:
            self (for chaining)
        """
        from collections import defaultdict

        values_by_item: Dict[str, List[float]] = defaultdict(list)
        for itemid, valuenum in records:
            try:
                values_by_item[str(itemid)].append(float(valuenum))
            except (TypeError, ValueError):
                pass
        self._compute_boundaries(values_by_item)
        return self

    def _compute_boundaries(
        self, values_by_item: Dict[str, List[float]]
    ) -> None:
        """Store (n_bins - 1) linearly-interpolated quantile boundaries per itemid."""
        for itemid, vals in values_by_item.items():
            if not vals:
                continue
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            boundaries: List[float] = []
            for i in range(1, self.n_bins):
                frac = i / self.n_bins
                idx_f = frac * (n - 1)
                lo = int(idx_f)
                hi = min(lo + 1, n - 1)
                interp = vals_sorted[lo] + (idx_f - lo) * (
                    vals_sorted[hi] - vals_sorted[lo]
                )
                boundaries.append(interp)
            self.boundaries[itemid] = boundaries

    # ── Inference ─────────────────────────────────────────────────────────────

    def bin_index(self, itemid: str, valuenum: float) -> int:
        """Return the bin index (0 to n_bins - 1) for a (itemid, valuenum) pair.

        Uses a linear scan of the stored boundaries (4 comparisons at most).
        If itemid was not seen during fit, returns 0.
        """
        boundaries = self.boundaries.get(str(itemid))
        if not boundaries:
            return 0
        for i, boundary in enumerate(boundaries):
            if valuenum < boundary:
                return i
        return self.n_bins - 1

    def token(self, itemid: Any, valuenum: Optional[float]) -> str:
        """Return the binned token string for a single lab result.

        Args:
            itemid:   Lab item identifier (cast to str internally).
            valuenum: Numeric lab value.  Pass None to get an un-binned token.

        Returns:
            "LB:<itemid>_bin<0-4>"  when valuenum is a valid number.
            "LB:<itemid>"           when valuenum is None or non-numeric.
        """
        itemid_str = str(itemid)
        if valuenum is None:
            return f"LB:{itemid_str}"
        try:
            bin_idx = self.bin_index(itemid_str, float(valuenum))
        except (TypeError, ValueError):
            return f"LB:{itemid_str}"
        return f"LB:{itemid_str}_bin{bin_idx}"


# ── Module-level helpers ──────────────────────────────────────────────────────

def _patient_age_at(anchor_age: int, anchor_year: int, event_year: int) -> float:
    """Estimate patient age (years) at the time of a clinical event."""
    return float(anchor_age + (event_year - anchor_year))


def _time_interval_token(weeks_gap: float) -> str:
    """Map an inter-visit gap (weeks) to a paper special token (§2.1, Appx.E).

    Mapping:
        0 ≤ gap < 1  week  → [W0]
        1 ≤ gap < 2  weeks → [W1]
        2 ≤ gap < 3  weeks → [W2]
        3 ≤ gap < ~1 month → [W3]
        1–12 months        → [M{n}]
        > 12 months        → [LT]
    """
    if weeks_gap < 1.0:
        return "[W0]"
    if weeks_gap < 2.0:
        return "[W1]"
    if weeks_gap < 3.0:
        return "[W2]"
    if weeks_gap < 4.345:
        return "[W3]"
    # Note: 1 month ≈ 4.345 weeks (365.25 days / 12 months / 7 days)
    # months = int(weeks_gap / 4.345)
    months = round(weeks_gap / 4.345)  # round to nearest month for cleaner tokenization

    if months <= 12:
        return f"[M{max(months, 1)}]"
    return "[LT]"


# ── Base task ─────────────────────────────────────────────────────────────────

class MIMIC4EHRMambaTask(BaseTask):
    """Base PyHealth task — builds the EHR Mamba sequence.

    Produces ONE sample per patient containing their complete admission history
    as a flat token sequence.  Does NOT assign a prediction label; subclasses
    add task-specific labels and patient filters via the hook methods
    _passes_patient_filter() and _make_label().

    Input features produced:
        input_ids       — token strings (SequenceProcessor converts to int vocab indices)
        token_type_ids  — MIMIC4_TOKEN_TYPES index per token  (List[int] → LongTensor)
        time_stamps     — weeks since first admission per token (List[float] → FloatTensor)
        ages            — patient age in years per token       (List[float] → FloatTensor)
        visit_orders    — 0-based admission index per token    (List[int] → LongTensor)
        visit_segments  — alternating 1/2 per token, 0 for    (List[int] → LongTensor)
                          structural special tokens

    Args:
        tables:        MIMIC-IV tables to include (paper Appx.B):
                       "procedures_icd", "prescriptions", "labevents".
        min_visits:    Minimum number of admissions required (default 1).
        lab_quantizer: Fitted LabQuantizer for 5-bin lab tokenization.
                       If None, lab itemids are used without binning.
    """

    task_name: str = "MIMIC4EHRMamba"
    input_schema: Dict[str, str] = {"input_ids": "sequence"}
    output_schema: Dict[str, str] = {}

    def __init__(
        self,
        tables: Tuple[str, ...] = ("procedures_icd", "prescriptions", "labevents"),
        min_visits: int = 1,
        lab_quantizer: Optional[LabQuantizer] = None,
    ) -> None:
        self.tables = tables
        self.min_visits = min_visits
        self._lab_quantizer = lab_quantizer

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_codes(
        patient: Any,
        event_type: str,
        hadm_id: Any,
        prefix: str,
        code_col: str,
    ) -> List[str]:
        """Return prefixed clinical codes for one admission from one table."""
        try:
            import polars as pl
            df = patient.get_events(
                event_type=event_type,
                filters=[("hadm_id", "==", hadm_id)],
                return_df=True,
            )
            col = f"{event_type}/{code_col}"
            if df is None or df.height == 0 or col not in df.columns:
                return []
            codes = (
                df.select(pl.col(col).cast(pl.Utf8).drop_nulls())
                .to_series()
                .to_list()
            )
            return [f"{prefix}:{c}" for c in codes if c]
        except Exception:
            return []

    @staticmethod
    def _get_lab_codes(
        patient: Any,
        hadm_id: Any,
        quantizer: LabQuantizer,
    ) -> List[str]:
        """Return 5-bin quantized lab tokens for one admission."""
        try:
            import polars as pl
            df = patient.get_events(
                event_type="labevents",
                filters=[("hadm_id", "==", hadm_id)],
                return_df=True,
            )
            if df is None or df.height == 0:
                return []
            itemid_col   = "labevents/itemid"
            valuenum_col = "labevents/valuenum"
            if itemid_col not in df.columns:
                return []
            if valuenum_col in df.columns:
                rows = (
                    df.select([
                        pl.col(itemid_col).cast(pl.Utf8),
                        pl.col(valuenum_col).cast(pl.Float64),
                    ])
                    .rows()
                )
                return [
                    quantizer.token(itemid, valuenum)
                    for itemid, valuenum in rows
                    if itemid
                ]
            else:
                items = (
                    df.select(pl.col(itemid_col).cast(pl.Utf8).drop_nulls())
                    .to_series()
                    .to_list()
                )
                return [f"LB:{c}" for c in items if c]
        except Exception:
            return []

    def _collect_visit_tokens(
        self,
        patient: Any,
        admission: Any,
    ) -> Tuple[List[str], List[int]]:
        """Return (code_strings, type_ids) for a single admission's events."""
        codes: List[str] = []
        type_ids: List[int] = []

        table_cfg: Dict[str, Tuple[str, int, str]] = {
            "procedures_icd": ("PR", MIMIC4_TOKEN_TYPES["procedures_icd"], "icd_code"),
            "prescriptions":  ("RX", MIMIC4_TOKEN_TYPES["prescriptions"],  "drug"),
            "labevents":      ("LB", MIMIC4_TOKEN_TYPES["labevents"],      "itemid"),
        }

        for table in self.tables:
            if table not in table_cfg:
                continue
            prefix, tid, code_col = table_cfg[table]
            if table == "labevents" and self._lab_quantizer is not None:
                batch = self._get_lab_codes(
                    patient, admission.hadm_id, self._lab_quantizer
                )
            else:
                batch = self._get_codes(
                    patient, table, admission.hadm_id, prefix, code_col
                )
            codes.extend(batch)
            type_ids.extend([tid] * len(batch))

        return codes, type_ids

    def _append_special(
        self,
        token: str,
        type_id: int,
        weeks: float,
        age: float,
        input_ids: List[str],
        token_type_ids: List[int],
        time_stamps: List[float],
        ages: List[float],
        visit_orders: List[int],
        visit_segments: List[int],
    ) -> None:
        """Append one structural special token with zero visit_order/segment."""
        input_ids.append(token)
        token_type_ids.append(type_id)
        time_stamps.append(weeks)
        ages.append(age)
        visit_orders.append(0)
        visit_segments.append(0)

    # ── Subclass hooks ────────────────────────────────────────────────────────

    def _passes_patient_filter(self, demographics: Any) -> bool:
        """Return False to drop this patient before sequence building.

        Override in subclasses to add task-specific demographic filters.
        """
        return True

    def _make_label(
        self, patient: Any, admissions: List[Any]
    ) -> Optional[Dict[str, Any]]:
        """Return a dict of label fields to merge into the sample.

        Return None to drop the patient entirely (e.g. missing label data).
        The base implementation returns an empty dict (no label fields).
        Override in subclasses to add task-specific prediction targets.
        """
        return {}

    # ── BaseTask entry point ──────────────────────────────────────────────────

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process one MIMIC-IV patient into an EHR Mamba sample.

        Returns at most one sample per patient (full history).
        Returns an empty list when the patient fails quality filters.
        """
        # ── Demographics ──────────────────────────────────────────────────────
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []
        demo = demographics[0]
        anchor_age: int  = int(getattr(demo, "anchor_age",  0))
        anchor_year: int = int(getattr(demo, "anchor_year", 0))

        if not self._passes_patient_filter(demo):
            return []

        # ── Admissions (chronological) ────────────────────────────────────────
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < self.min_visits:
            return []

        try:
            admissions = sorted(admissions, key=lambda a: a.timestamp)
        except Exception:
            pass

        try:
            first_admit: datetime = admissions[0].timestamp
        except Exception:
            return []

        # ── Build flattened token sequence (paper §2.1 structure) ─────────────
        input_ids:      List[str]   = []
        token_type_ids: List[int]   = []
        time_stamps:    List[float] = []
        ages:           List[float] = []
        visit_orders:   List[int]   = []
        visit_segments: List[int]   = []

        # Single global [CLS] at the start of the full patient sequence (§2.1)
        self._append_special(
            _CLS_TOKEN, MIMIC4_TOKEN_TYPES["CLS"], 0.0, float(anchor_age),
            input_ids, token_type_ids, time_stamps, ages, visit_orders, visit_segments,
        )

        prev_weeks_offset: Optional[float] = None
        visits_added: int = 0

        for admission in admissions:
            try:
                weeks_offset = float(
                    (admission.timestamp - first_admit).total_seconds() / 604_800.0
                )
            except Exception:
                weeks_offset = 0.0

            try:
                age = _patient_age_at(anchor_age, anchor_year, admission.timestamp.year)
            except Exception:
                age = float(anchor_age)

            # Visit segment alternates 1/2 across non-empty visits (§2.2)
            segment = (visits_added % 2) + 1

            codes, type_ids = self._collect_visit_tokens(patient, admission)
            if not codes:
                continue  # skip admissions with no events in requested tables

            v_order = min(visits_added, MAX_NUM_VISITS - 1)

            # ── Inter-visit time-interval token (§2.1, Appx.E) ───────────────
            if prev_weeks_offset is not None:
                gap = weeks_offset - prev_weeks_offset
                self._append_special(
                    _time_interval_token(gap), MIMIC4_TOKEN_TYPES["TIME_INTERVAL"],
                    weeks_offset, age,
                    input_ids, token_type_ids, time_stamps, ages, visit_orders, visit_segments,
                )

            # ── [VS] visit start ──────────────────────────────────────────────
            self._append_special(
                _VS_TOKEN, MIMIC4_TOKEN_TYPES["VS"],
                weeks_offset, age,
                input_ids, token_type_ids, time_stamps, ages, visit_orders, visit_segments,
            )

            # ── Medical event tokens ──────────────────────────────────────────
            n = len(codes)
            input_ids.extend(codes)
            token_type_ids.extend(type_ids)
            time_stamps.extend([weeks_offset] * n)
            ages.extend([age] * n)
            visit_orders.extend([v_order] * n)
            visit_segments.extend([segment] * n)

            # ── [VE] visit end ────────────────────────────────────────────────
            self._append_special(
                _VE_TOKEN, MIMIC4_TOKEN_TYPES["VE"],
                weeks_offset, age,
                input_ids, token_type_ids, time_stamps, ages, visit_orders, visit_segments,
            )

            # ── [REG] register token — follows each [VE] (§2.1) ──────────────
            self._append_special(
                _REG_TOKEN, MIMIC4_TOKEN_TYPES["REG"],
                weeks_offset, age,
                input_ids, token_type_ids, time_stamps, ages, visit_orders, visit_segments,
            )

            prev_weeks_offset = weeks_offset
            visits_added += 1

        if visits_added == 0:
            return []

        # ── Label (subclass hook) ─────────────────────────────────────────────
        label_fields = self._make_label(patient, admissions)
        if label_fields is None:
            return []

        return [
            {
                "patient_id":     patient.patient_id,
                "visit_id":       getattr(admissions[-1], "hadm_id", ""),
                "input_ids":      input_ids,
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "time_stamps":    torch.tensor(time_stamps,    dtype=torch.float),
                "ages":           torch.tensor(ages,           dtype=torch.float),
                "visit_orders":   torch.tensor(visit_orders,   dtype=torch.long),
                "visit_segments": torch.tensor(visit_segments, dtype=torch.long),
                **label_fields,
            }
        ]


# ── Mortality prediction subclass ─────────────────────────────────────────────

class MIMIC4EHRMambaMortalityTask(MIMIC4EHRMambaTask):
    """EHR Mamba task for in-hospital mortality prediction.

    Extends MIMIC4EHRMambaTask with:
      - In-hospital mortality label from the patient's last recorded admission
        (hospital_expire_flag = 1 means the patient died during that stay).
      - Minimum patient age filter (default ≥ 18 years).

    The prediction target is always the LAST admission in the patient's
    history — consistent with the EHR Mamba paper evaluation setup.

    Args:
        min_age:       Minimum anchor_age to include a patient (default 18).
        tables:        Passed through to MIMIC4EHRMambaTask.
        min_visits:    Passed through to MIMIC4EHRMambaTask.
        lab_quantizer: Passed through to MIMIC4EHRMambaTask.

    Example:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from mimic4_ehr_mamba_task import MIMIC4EHRMambaMortalityTask
        >>> ds = MIMIC4EHRDataset(root="/data/mimic-iv/2.2",
        ...                       tables=["patients", "admissions",
        ...                               "procedures_icd", "prescriptions",
        ...                               "labevents"])
        >>> task = MIMIC4EHRMambaMortalityTask()
        >>> sample_ds = ds.set_task(task)
    """

    task_name: str = "MIMIC4EHRMambaMortality"
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, min_age: int = 18, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_age = min_age

    def _passes_patient_filter(self, demographics: Any) -> bool:
        """Return True if the patient meets the minimum age requirement.

        Args:
            demographics: First row of the ``patients`` event table for
                this patient, expected to have an ``anchor_age`` attribute.

        Returns:
            ``True`` when ``anchor_age`` is at least :attr:`min_age`.
        """
        return int(getattr(demographics, "anchor_age", 0)) >= self.min_age

    def _make_label(
        self, patient: Any, admissions: List[Any]
    ) -> Optional[Dict[str, Any]]:
        """Return ``{"label": 0_or_1}`` from the patient's last admission.

        Uses ``hospital_expire_flag`` from the last recorded admission as a
        proxy for in-hospital mortality.

        Args:
            patient: PyHealth Patient object (unused; kept for interface parity).
            admissions: Chronologically sorted list of admission Event objects.

        Returns:
            ``{"label": 1}`` if the patient died during the last admission,
            ``{"label": 0}`` otherwise.  Never returns ``None``.

        Note:
            The paper (Appx. B.2) defines mortality as death within 32 days
            of the final recorded event, computed from event and death
            timestamps.  We use hospital_expire_flag instead — a MIMIC-IV
            field indicating in-hospital death — which is a simpler proxy
            available directly on the admissions row.
        """
        # Note: the paper (Appx. B.2) defines mortality as death within 32 days
        # of the final recorded event, computed by comparing event and death
        # timestamps.  We use hospital_expire_flag instead — a MIMIC-IV field
        # indicating in-hospital death — which is a simpler proxy available
        # directly on the admissions row.  In-hospital mortality and the paper's
        # 32-day post-event mortality are closely related but not identical.
        try:
            label = int(getattr(admissions[-1], "hospital_expire_flag", 0))
        except Exception:
            label = 0
        return {"label": label}


# ── Collation helper ──────────────────────────────────────────────────────────

def collate_ehr_mamba_batch(
    samples: List[Dict[str, Any]],
    pad_token_idx: int = 0,
) -> Dict[str, Any]:
    """Custom collate function for DataLoader with EHR Mamba samples.

    Handles both PyHealth-processed integer tensors (input_ids) and raw Python
    lists for auxiliary metadata fields.  All sequences are right-padded to the
    length of the longest sequence in the batch.

    Args:
        samples:       List of sample dicts as returned by a MIMIC4EHRMambaTask
                       subclass after PyHealth processing (input_ids already
                       converted to int tensors by SequenceProcessor).
        pad_token_idx: Padding index for integer sequence fields (default 0).

    Returns:
        Dict with keys:
            "input_ids"       : (B, L_max) LongTensor
            "token_type_ids"  : (B, L_max) LongTensor  (padded with 0 = PAD type)
            "time_stamps"     : (B, L_max) FloatTensor (padded with 0.0)
            "ages"            : (B, L_max) FloatTensor (padded with 0.0)
            "visit_orders"    : (B, L_max) LongTensor  (padded with 0)
            "visit_segments"  : (B, L_max) LongTensor  (padded with 0)
            "label"           : (B,) LongTensor        (omitted if not in samples)
            "patient_id"      : List[str]
            "visit_id"        : List[str]
    """
    from torch.nn.utils.rnn import pad_sequence

    def _to_long(val: Any) -> torch.Tensor:
        if isinstance(val, torch.Tensor):
            return val.long()
        return torch.tensor(val, dtype=torch.long)

    def _to_float(val: Any) -> torch.Tensor:
        if isinstance(val, torch.Tensor):
            return val.float()
        return torch.tensor(val, dtype=torch.float)

    input_ids = pad_sequence(
        [_to_long(s["input_ids"]) for s in samples],
        batch_first=True,
        padding_value=pad_token_idx,
    )
    token_type_ids = pad_sequence(
        [_to_long(s["token_type_ids"]) for s in samples],
        batch_first=True,
        padding_value=MIMIC4_TOKEN_TYPES["PAD"],
    )
    visit_orders = pad_sequence(
        [_to_long(s["visit_orders"]) for s in samples],
        batch_first=True,
        padding_value=0,
    )
    visit_segments = pad_sequence(
        [_to_long(s["visit_segments"]) for s in samples],
        batch_first=True,
        padding_value=0,
    )
    time_stamps = pad_sequence(
        [_to_float(s["time_stamps"]) for s in samples],
        batch_first=True,
        padding_value=0.0,
    )
    ages = pad_sequence(
        [_to_float(s["ages"]) for s in samples],
        batch_first=True,
        padding_value=0.0,
    )

    batch: Dict[str, Any] = {
        "input_ids":      input_ids,
        "token_type_ids": token_type_ids,
        "time_stamps":    time_stamps,
        "ages":           ages,
        "visit_orders":   visit_orders,
        "visit_segments": visit_segments,
        "patient_id":     [s.get("patient_id", "") for s in samples],
        "visit_id":       [s.get("visit_id", "")   for s in samples],
    }

    if "label" in samples[0]:
        batch["label"] = torch.tensor(
            [int(s["label"]) for s in samples], dtype=torch.long
        )

    return batch
