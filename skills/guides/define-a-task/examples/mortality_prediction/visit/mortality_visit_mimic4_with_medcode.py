"""Next-Visit Mortality Prediction on MIMIC-IV with MedCode Conversions.

This example demonstrates using pyhealth.medcode.CrossMap to normalize raw
clinical codes to higher-level category systems before use as features.

Conversions applied:
- ICD-9/10-CM diagnosis codes  → CCS-CM  categories (~15k codes → ~285)
- ICD-9/10-PCS procedure codes → CCS-PROC categories (~4k codes  → ~231)
- NDC drug codes               → ATC level-3 codes  (~100k NDCs  → ~800)

Why normalize?
- Reduces vocabulary size, improving model generalization
- Groups clinically related codes (e.g., all heart failure variants → one CCS)
- ATC level-3 captures therapeutic subgroup (4-char, e.g., "C09A" = ACE inhibitors)
- Handles the ICD-9 vs ICD-10 split transparently via per-version CrossMaps

⚠️ TEMPORAL SAFETY:
Features from CURRENT visit (i) → Label from NEXT visit (i+1). No leakage.

CrossMap usage:
    from pyhealth.medcode import CrossMap
    cm = CrossMap.load("ICD9CM", "CCSCM")   # loads/downloads once, cached to disk
    cm.map("428.0")                          # ['108']

    cm = CrossMap.load("NDC", "ATC")
    cm.map("00527051210", target_kwargs={"level": 3})  # ['A11C']
"""

import logging
import os
from typing import Any, Dict, List, Optional

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.medcode import CrossMap
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER
# =============================================================================
def _safe_map(
    crossmap: CrossMap,
    code: str,
    target_kwargs: Optional[Dict] = None,
) -> Optional[str]:
    """Map a single code via CrossMap, returning the first result or None.

    CrossMap.map() returns a list (one source code can map to multiple targets).
    We take the first match. Both failure modes emit a logger.warning so the
    agent can diagnose problems without crashing the pipeline:

    - Empty result: code not found in mapping dict. Most commonly caused by
      a formatting mismatch — e.g. ICD code missing its decimal dot, NDC with
      the wrong number of digits, or leading/trailing whitespace. CrossMap
      relies on exact string keys after standardize()/convert(), so even a
      minor format difference produces a silent miss.
    - Exception: unexpected error in standardize(), convert(), or the mapping
      lookup itself (e.g. wrong vocab pair, bad target_kwargs).
    """
    try:
        results = crossmap.map(code, target_kwargs=target_kwargs or {})
    except Exception as e:
        logger.warning(
            "CrossMap(%s->%s) raised %s for code %r: %s",
            crossmap.s_vocab,
            crossmap.t_vocab,
            type(e).__name__,
            code,
            e,
        )
        return None

    if not results:
        # Most common cause: code string is not in the expected format for
        # standardize()/convert() (e.g. ICD code missing dot, NDC wrong digit
        # count). The CrossMap silently returns [] when the looked-up key is
        # absent from the mapping dict, so a bad format is indistinguishable
        # from a genuinely unmapped code at this level.
        logger.warning(
            "CrossMap(%s->%s): no mapping found for code %r "
            "(check that the code string matches the expected format for %s)",
            crossmap.s_vocab,
            crossmap.t_vocab,
            code,
            crossmap.s_vocab,
        )
        return None

    return results[0]


# =============================================================================
# TASK CLASS DEFINITION
# =============================================================================
class NextVisitMortalityWithMedCodeTask(BaseTask):
    """Next-visit mortality prediction with normalized code features.

    Raw ICD and NDC codes are converted to CCS-CM, CCS-PROC, and ATC categories
    using CrossMap before being stored in the sample dict.  The smaller, grouped
    vocabularies are easier for downstream models to learn from.

    CrossMaps are loaded once in __init__ and cached to disk by PyHealth so
    subsequent runs skip the download/CSV-parse step.
    """

    task_name = "NextVisitMortalityWithMedCodeTask"

    input_schema = {
        "conditions": "sequence",   # CCS-CM category strings, e.g. "108"
        "procedures": "sequence",   # CCS-PROC category strings, e.g. "43"
        "drugs": "sequence",        # ATC level-3 strings,  e.g. "C09A"
    }

    output_schema = {"mortality": "binary"}

    def __init__(self):
        super().__init__()

        # ── Diagnosis maps (ICD version-aware) ──────────────────────────────
        # MIMIC-IV has both ICD-9 and ICD-10 codes; route by icd_version field
        self.icd9cm_to_ccscm = CrossMap.load("ICD9CM", "CCSCM")
        self.icd10cm_to_ccscm = CrossMap.load("ICD10CM", "CCSCM")

        # ── Procedure maps (ICD version-aware) ──────────────────────────────
        self.icd9proc_to_ccsproc = CrossMap.load("ICD9PROC", "CCSPROC")
        self.icd10proc_to_ccsproc = CrossMap.load("ICD10PROC", "CCSPROC")

        # ── Drug map ────────────────────────────────────────────────────────
        # NDC → ATC level 3 (4-character therapeutic subgroup)
        # level=1 → 1 char (anatomical), level=3 → 4 chars (therapeutic subgroup),
        # level=5 → full 7-char ATC code (chemical substance)
        self.ndc_to_atc = CrossMap.load("NDC", "ATC")

    # ── Per-code conversion helpers ──────────────────────────────────────────

    def _diagnosis_to_ccs(self, icd_code: str, icd_version: Any) -> Optional[str]:
        """Convert an ICD diagnosis code to its CCS-CM category."""
        version = str(icd_version)
        if version == "9":
            return _safe_map(self.icd9cm_to_ccscm, icd_code)
        if version == "10":
            return _safe_map(self.icd10cm_to_ccscm, icd_code)
        return None

    def _procedure_to_ccs(self, icd_code: str, icd_version: Any) -> Optional[str]:
        """Convert an ICD procedure code to its CCS-PROC category."""
        version = str(icd_version)
        if version == "9":
            return _safe_map(self.icd9proc_to_ccsproc, icd_code)
        if version == "10":
            return _safe_map(self.icd10proc_to_ccsproc, icd_code)
        return None

    def _ndc_to_atc3(self, ndc: str) -> Optional[str]:
        """Convert an NDC code to ATC level-3 (4-char therapeutic subgroup)."""
        return _safe_map(self.ndc_to_atc, ndc, target_kwargs={"level": 3})

    # ── Main task logic ──────────────────────────────────────────────────────

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient and return next-visit mortality samples.

        TEMPORAL PATTERN (No Leakage):
        - Features from CURRENT visit (visit i)   — converted via CrossMap
        - Label   from NEXT    visit (visit i+1)
        - Predicts: Will the patient die in their next hospitalization?

        CODE CONVERSION NOTES:
        - Codes that fail to map (unknown NDC, non-standard ICD) are silently
          dropped; the sequence may be empty if no codes convert successfully.
        - Duplicate converted codes within a visit are preserved (keeps counts).
        """
        samples = []
        visits = patient.get_events(event_type="admissions")

        # Need at least 2 visits for next-visit prediction
        if len(visits) <= 1:
            return samples

        for i in range(len(visits) - 1):
            current_visit = visits[i]
            next_visit = visits[i + 1]

            # Label: mortality from NEXT visit
            if next_visit.hospital_expire_flag not in [0, 1, "0", "1"]:
                continue
            mortality_label = 1 if next_visit.hospital_expire_flag in [1, "1"] else 0

            # ── Get raw events from CURRENT visit ───────────────────────────
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", current_visit.hadm_id)],
            )
            procedures = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", current_visit.hadm_id)],
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", current_visit.hadm_id)],
            )

            # ── Convert ICD diagnosis codes → CCS-CM ───────────────────────
            # icd_version is "9" or "10" in MIMIC-IV; route to the right CrossMap
            conditions = [
                ccs
                for d in diagnoses
                if getattr(d, "icd_code", None) and getattr(d, "icd_version", None)
                for ccs in [self._diagnosis_to_ccs(str(d.icd_code), d.icd_version)]
                if ccs is not None
            ]

            # ── Convert ICD procedure codes → CCS-PROC ─────────────────────
            procs = [
                ccs
                for p in procedures
                if getattr(p, "icd_code", None) and getattr(p, "icd_version", None)
                for ccs in [self._procedure_to_ccs(str(p.icd_code), p.icd_version)]
                if ccs is not None
            ]

            # ── Convert NDC drug codes → ATC level-3 ───────────────────────
            # Prescriptions may have an NDC or just a drug name; only NDC converts.
            drugs = [
                atc
                for rx in prescriptions
                if rx.ndc not in [None, "", "0"]
                for atc in [self._ndc_to_atc3(str(rx.ndc))]
                if atc is not None
            ]

            # Skip visits with no converted features at all
            if not conditions and not procs and not drugs:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": current_visit.hadm_id,
                    "prediction_for_visit": next_visit.hadm_id,
                    "conditions": conditions,   # CCS-CM categories
                    "procedures": procs,        # CCS-PROC categories
                    "drugs": drugs,             # ATC level-3 codes
                    "mortality": mortality_label,
                }
            )

        return samples


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    EHR_ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.2"
    CACHE_DIR = "./cache"

    base_dataset = MIMIC4Dataset(
        ehr_root=EHR_ROOT,
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        cache_dir=CACHE_DIR,
        dev=False,
    )

    # CrossMaps are loaded inside __init__; they download CSV files on first run
    # and cache them to disk (MODULE_CACHE_PATH). Subsequent runs are fast.
    num_workers = min(8, max(1, (os.cpu_count() or 1) - 1))
    task = NextVisitMortalityWithMedCodeTask()
    sample_dataset = base_dataset.set_task(task, num_workers=num_workers)

    print(f"Total samples: {len(sample_dataset)}")
    if len(sample_dataset) > 0:
        s = sample_dataset[0]
        print(f"Keys:       {list(s.keys())}")
        print(f"Conditions: {s['conditions'][:5]}  (CCS-CM categories)")
        print(f"Procedures: {s['procedures'][:5]}  (CCS-PROC categories)")
        print(f"Drugs:      {s['drugs'][:5]}  (ATC level-3)")
        print(f"Mortality:  {s['mortality']}")
