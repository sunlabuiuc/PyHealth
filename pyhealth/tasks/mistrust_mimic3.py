"""
Medical Mistrust Tasks for MIMIC-III
=====================================
Implements two binary classification tasks that serve as computational proxies
for medical mistrust, as described in:

    Boag et al. "Racial Disparities and Mistrust in End-of-Life Care."
    MLHC 2018. https://arxiv.org/abs/1808.03827

Both tasks extract interpersonal interaction features from CHARTEVENTS and
derive binary labels from free-text NOTEEVENTS.  The resulting samples are
intended for use with ``pyhealth.models.LogisticRegression`` (with
``l1_lambda > 0`` for the paper-equivalent L1 regularisation).

Tasks
-----
MistrustNoncomplianceMIMIC3
    Predicts whether a hospital admission contains documented patient
    noncompliance (search string: "noncompliant").
    Label 1 = noncompliant (mistrustful), 0 = compliant (trusting).

MistrustAutopsyMIMIC3
    Predicts whether the family consented to a post-mortem autopsy.
    Autopsy consent is treated as a signal of distrust in the quality of
    care received.
    Label 1 = consented (mistrustful), 0 = declined (trusting).
    Admissions with ambiguous signals (both consent and decline) are excluded.

Input features
--------------
Both tasks produce ``interpersonal_features``: a *list of feature-key strings*
extracted from CHARTEVENTS (schema type ``"sequence"``).  Each key has the
form ``"<category>||<normalised_value>"``, mirroring the normalisation rules
in trust.ipynb / script 02_chartevents_features.py.

The vocabulary is learned automatically by the PyHealth tokeniser during
``dataset.set_task()``, so no external DictVectorizer is required.

Usage
-----
    >>> from pyhealth.datasets import MIMIC3Dataset
    >>> from pyhealth.tasks import MistrustNoncomplianceMIMIC3
    >>> from pyhealth.models import LogisticRegression
    >>> from pyhealth.trainer import Trainer
    >>>
    >>> base_dataset = MIMIC3Dataset(
    ...     root="/path/to/mimic-iii/1.4",
    ...     tables=["CHARTEVENTS", "NOTEEVENTS"],
    ... )
    >>> task = MistrustNoncomplianceMIMIC3(
    ...     itemid_to_label={720: "ventilator mode", ...}   # from D_ITEMS
    ... )
    >>> sample_dataset = base_dataset.set_task(task)
    >>> model = LogisticRegression(dataset=sample_dataset, l1_lambda=1e-4)
    >>> trainer = Trainer(model=model)
    >>> trainer.train(train_dataloader=..., val_dataloader=..., epochs=50)

Helper
------
    ``build_interpersonal_itemids(d_items_path)`` — reads D_ITEMS.csv.gz and
    returns a ``{itemid: label}`` dict filtered to interpersonal keywords,
    ready to pass to either task.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pyhealth.tasks.base_task import BaseTask


# ---------------------------------------------------------------------------
# Keywords that define "interpersonal" CHARTEVENTS items (trust.ipynb cell 4)
# ---------------------------------------------------------------------------
_INTERPERSONAL_KEYWORDS = [
    "family communication", "follows commands", "education barrier",
    "education learner", "education method", "education readiness",
    "education topic", "pain", "pain level", "pain level (rest)",
    "pain assess method", "restraint", "spiritual support", "support systems",
    "state", "behavior", "behavioral state", "reason for restraint",
    "stress", "safety", "safety measures", "family", "patient/family informed",
    "pt./family informed", "health care proxy", "bath", "bed bath", "bedbath",
    "chg bath", "skin care", "judgement", "family meeting",
    "emotional / physical / sexual harm", "verbal response", "side rails",
    "orientation", "rsbi deferred", "richmond-ras scale", "riker-sas scale",
    "status and comfort", "teaching directed toward", "consults",
    "social work consult", "sitter", "security", "observer", "informed",
]

# Autopsy keyword sets
_AUTOPSY_CONSENT_WORDS = ("consent", "agree", "request")
_AUTOPSY_DECLINE_WORDS = ("decline", "not consent", "refuse", "denied")


# ---------------------------------------------------------------------------
# Public helper: build itemid→label dict from D_ITEMS.csv.gz
# ---------------------------------------------------------------------------

def build_interpersonal_itemids(d_items_path: str) -> Dict[int, str]:
    """Build an ``{itemid: label}`` dict for interpersonal CHARTEVENTS items.

    Reads ``D_ITEMS.csv.gz`` (or uncompressed ``D_ITEMS.csv``) and filters rows
    whose ``LABEL`` contains any of the interpersonal keywords used in
    Boag et al. 2018.  Pass the result to ``MistrustNoncomplianceMIMIC3`` or
    ``MistrustAutopsyMIMIC3`` as ``itemid_to_label``.

    Args:
        d_items_path: Path to ``D_ITEMS.csv.gz`` (or ``.csv``).

    Returns:
        Dict mapping ``itemid (int)`` to ``label (str)`` for all matched rows
        where ``LINKSTO == 'chartevents'``.

    Example:
        >>> from pyhealth.tasks import build_interpersonal_itemids
        >>> itemid_to_label = build_interpersonal_itemids(
        ...     "/path/to/mimic-iii/1.4/D_ITEMS.csv.gz"
        ... )
        >>> len(itemid_to_label)
        168
    """
    import pandas as pd

    df = pd.read_csv(d_items_path, usecols=["ITEMID", "LABEL", "LINKSTO"])
    df = df[df["LINKSTO"] == "chartevents"].copy()

    def _matches(label: str) -> bool:
        lo = str(label).lower()
        return any(k in lo for k in _INTERPERSONAL_KEYWORDS)

    df = df[df["LABEL"].apply(_matches)]
    return dict(zip(df["ITEMID"].astype(int), df["LABEL"].astype(str)))


# ---------------------------------------------------------------------------
# Feature normalisation — mirrors trust.ipynb cell 7
# ---------------------------------------------------------------------------

def _restraint_reason(v: str) -> str:
    if v in ("not applicable", "none", ""):
        return "none"
    if "threat" in v or "acute risk of" in v:
        return "threat of harm"
    if "confusion" in v or "delirium" in v or v == "impaired judgment" or v == "sundowning":
        return "confusion/delirium"
    if "occurence" in v or v == "severe physical agitation" or v == "violent/self des":
        return "presence of violence"
    if v in ("ext/txinterfere", "protection of lines and tubes", "treatment interference"):
        return "treatment interference"
    if "risk for fall" in v or "risk for falling" in v:
        return "risk for falls"
    return v


def _restraint_location(v: str) -> str:
    if v in ("none", ""):
        return "none"
    if "4 point" in v or "4point" in v:
        return "4 point restraint"
    return "some restraint"


def _restraint_device(v: str) -> str:
    if "sitter" in v:
        return "sitter"
    if "limb" in v:
        return "limb"
    return v


def _bath(label: str, v: str) -> str:
    if "part" in label:
        return "partial"
    if "self" in v:
        return "self"
    if "refused" in v:
        return "refused"
    if "shave" in v:
        return "shave"
    if "hair" in v:
        return "hair"
    if "none" in v:
        return "none"
    return "done"


def _normalise_feature(label: str, value: str) -> Optional[str]:
    """Normalise a (label, value) pair into a feature key string.

    Returns ``"<category>||<normalised_value>"`` or ``None`` to skip the row.
    Mirrors the normalisation in ``02_chartevents_features.py`` / trust.ipynb.
    """
    lo = label.lower()
    v = (value or "none").lower().strip()

    if "reason for restraint" in lo:
        return f"reason for restraint||{_restraint_reason(v)}"
    if "restraint location" in lo:
        return f"restraint location||{_restraint_location(v)}"
    if "restraint device" in lo:
        return f"restraint device||{_restraint_device(v)}"
    if "bath" in lo:
        return f"bath||{_bath(lo, v)}"

    # Skipped categories
    if lo in ("behavior", "behavioral state"):
        return None
    if lo.startswith("pain management") or lo.startswith("pain type") \
            or lo.startswith("pain cause") or lo.startswith("pain location"):
        return None

    # Categories kept as-is
    for prefix in ("pain level", "education topic", "safety measures",
                   "side rails", "status and comfort"):
        if lo.startswith(prefix):
            return f"{prefix}||{v}"

    if "informed" in lo:
        return f"informed||{v}"

    return f"{lo}||{v}"


# ---------------------------------------------------------------------------
# Shared extraction helpers
# ---------------------------------------------------------------------------

def _extract_interpersonal_features(
    chartevents: List[Any],
    itemid_to_label: Dict[int, str],
) -> List[str]:
    """Return a deduplicated list of interpersonal feature-key strings.

    Args:
        chartevents: list of chartevents Event objects for one admission.
        itemid_to_label: ``{itemid: label}`` dict (from ``build_interpersonal_itemids``).

    Returns:
        Sorted list of unique ``"category||value"`` strings.
    """
    seen = set()
    for ev in chartevents:
        itemid = getattr(ev, "itemid", None)
        if itemid is None:
            continue
        try:
            itemid = int(itemid)
        except (ValueError, TypeError):
            continue
        if itemid not in itemid_to_label:
            continue
        label = itemid_to_label[itemid]
        value = str(getattr(ev, "value", "") or "")
        fkey = _normalise_feature(label, value)
        if fkey is not None:
            seen.add(fkey)
    return sorted(seen)


def _extract_noncompliance_label(noteevents: List[Any]) -> int:
    """Return 1 if any note contains 'noncompliant', else 0."""
    for ev in noteevents:
        text = str(getattr(ev, "text", "") or "").lower()
        if "noncompliant" in text:
            return 1
    return 0


def _extract_autopsy_label(noteevents: List[Any]) -> Optional[int]:
    """Return 1 (consent/mistrust), 0 (decline/trust), or None (ambiguous/absent)."""
    consented = False
    declined = False
    for ev in noteevents:
        text = str(getattr(ev, "text", "") or "").lower()
        if "autopsy" not in text:
            continue
        for line in text.split("\n"):
            if "autopsy" not in line:
                continue
            if any(w in line for w in _AUTOPSY_DECLINE_WORDS):
                declined = True
            if any(w in line for w in _AUTOPSY_CONSENT_WORDS):
                consented = True
    if consented and declined:
        return None       # ambiguous — exclude
    if consented:
        return 1
    if declined:
        return 0
    return None           # no autopsy mention


# ---------------------------------------------------------------------------
# Task 1: Noncompliance mistrust
# ---------------------------------------------------------------------------

class MistrustNoncomplianceMIMIC3(BaseTask):
    """Predict documented noncompliance as a proxy for medical mistrust.

    For each hospital admission the task produces one sample:

    - ``interpersonal_features``: deduplicated list of normalised CHARTEVENTS
      feature-key strings (schema: ``"sequence"``).
    - ``noncompliance``: ``1`` if any note for this admission contains the
      string ``"noncompliant"``, else ``0`` (schema: ``"binary"``).

    All admissions that appear in the chartevents interpersonal-feature set
    receive a label (default 0 / trust).  Base rate ≈ 0.88 % in MIMIC-III v1.4.

    Args:
        itemid_to_label: ``{itemid (int): label (str)}`` mapping from
            ``build_interpersonal_itemids()``.  Required to identify which
            CHARTEVENTS rows correspond to interpersonal interaction features.
        min_features: minimum number of interpersonal feature keys required
            for a sample to be included.  Defaults to 1.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import MistrustNoncomplianceMIMIC3, build_interpersonal_itemids
        >>> itemid_to_label = build_interpersonal_itemids(
        ...     "/path/to/mimic-iii/1.4/D_ITEMS.csv.gz"
        ... )
        >>> base_dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["CHARTEVENTS", "NOTEEVENTS"],
        ... )
        >>> task = MistrustNoncomplianceMIMIC3(itemid_to_label=itemid_to_label)
        >>> sample_dataset = base_dataset.set_task(task)
    """

    task_name: str = "MistrustNoncomplianceMIMIC3"
    input_schema: Dict[str, str] = {"interpersonal_features": "sequence"}
    output_schema: Dict[str, str] = {"noncompliance": "binary"}

    def __init__(
        self,
        itemid_to_label: Dict[int, str],
        min_features: int = 1,
    ) -> None:
        self.itemid_to_label = itemid_to_label
        self.min_features = min_features

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient into noncompliance classification samples.

        Args:
            patient: a PyHealth Patient object with ``chartevents`` and
                ``noteevents`` event types loaded.

        Returns:
            List of dicts, one per admission, each containing:
                - ``patient_id``
                - ``visit_id`` (hadm_id)
                - ``interpersonal_features`` (list of str)
                - ``noncompliance`` (int 0/1)
        """
        samples = []
        admissions = patient.get_events(event_type="admissions")

        for admission in admissions:
            hadm_id = admission.hadm_id

            chartevents = patient.get_events(
                event_type="chartevents",
                filters=[("hadm_id", "==", hadm_id)],
            )
            features = _extract_interpersonal_features(chartevents, self.itemid_to_label)

            if len(features) < self.min_features:
                continue

            noteevents = patient.get_events(
                event_type="noteevents",
                filters=[("hadm_id", "==", hadm_id)],
            )
            label = _extract_noncompliance_label(noteevents)

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": hadm_id,
                    "interpersonal_features": features,
                    "noncompliance": label,
                }
            )

        return samples


# ---------------------------------------------------------------------------
# Task 2: Autopsy-consent mistrust
# ---------------------------------------------------------------------------

class MistrustAutopsyMIMIC3(BaseTask):
    """Predict autopsy consent as a proxy for medical mistrust.

    Autopsy consent signals post-mortem distrust of the care received.
    Black patients in MIMIC-III v1.4 consent to autopsies at ~39 % vs ~26 %
    for white patients (Boag et al. 2018).

    Only admissions with an explicit, unambiguous autopsy mention in
    NOTEEVENTS receive a label (consent=1 / decline=0).  Admissions where
    both signals appear are excluded.

    Args:
        itemid_to_label: ``{itemid (int): label (str)}`` mapping from
            ``build_interpersonal_itemids()``.
        min_features: minimum interpersonal features required per sample.
            Defaults to 1.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import MistrustAutopsyMIMIC3, build_interpersonal_itemids
        >>> itemid_to_label = build_interpersonal_itemids(
        ...     "/path/to/mimic-iii/1.4/D_ITEMS.csv.gz"
        ... )
        >>> base_dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["CHARTEVENTS", "NOTEEVENTS"],
        ... )
        >>> task = MistrustAutopsyMIMIC3(itemid_to_label=itemid_to_label)
        >>> sample_dataset = base_dataset.set_task(task)
    """

    task_name: str = "MistrustAutopsyMIMIC3"
    input_schema: Dict[str, str] = {"interpersonal_features": "sequence"}
    output_schema: Dict[str, str] = {"autopsy_consent": "binary"}

    def __init__(
        self,
        itemid_to_label: Dict[int, str],
        min_features: int = 1,
    ) -> None:
        self.itemid_to_label = itemid_to_label
        self.min_features = min_features

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient into autopsy-consent classification samples.

        Args:
            patient: a PyHealth Patient object with ``chartevents`` and
                ``noteevents`` event types loaded.

        Returns:
            List of dicts, one per admission with an explicit autopsy signal:
                - ``patient_id``
                - ``visit_id`` (hadm_id)
                - ``interpersonal_features`` (list of str)
                - ``autopsy_consent`` (int 0/1)
        """
        samples = []
        admissions = patient.get_events(event_type="admissions")

        for admission in admissions:
            hadm_id = admission.hadm_id

            noteevents = patient.get_events(
                event_type="noteevents",
                filters=[("hadm_id", "==", hadm_id)],
            )
            label = _extract_autopsy_label(noteevents)
            if label is None:
                continue    # no explicit or ambiguous signal — skip

            chartevents = patient.get_events(
                event_type="chartevents",
                filters=[("hadm_id", "==", hadm_id)],
            )
            features = _extract_interpersonal_features(chartevents, self.itemid_to_label)

            if len(features) < self.min_features:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": hadm_id,
                    "interpersonal_features": features,
                    "autopsy_consent": label,
                }
            )

        return samples
