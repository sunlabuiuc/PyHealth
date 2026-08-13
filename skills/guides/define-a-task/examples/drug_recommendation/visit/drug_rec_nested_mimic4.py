"""Visit-Level Drug Recommendation on MIMIC-IV using CUMULATIVE NESTED HISTORY.

This example demonstrates the nested_sequence accumulation pattern used for
drug recommendation — the canonical pattern for any task that needs a model
to see a patient's FULL MULTI-VISIT history of codes, not just the current visit.

Key patterns:
- MIMIC4Dataset with lowercase table names
- Features are 2D nested lists: [[visit_0_codes], [visit_1_codes], ..., [visit_N_codes]]
- Two-pass approach: collect per-visit flat lists first, then run accumulation loop
- Label-leakage guard: zero out current visit slot in drugs_hist
- Requires ≥ 2 valid visits per patient
- Uses "nested_sequence" processor in input_schema (NOT "sequence")
- Label: current visit's drugs (multilabel)

Source (canonical implementation):
  PyHealth/pyhealth/tasks/drug_recommendation.py → DrugRecommendationMIMIC4
"""

import os
from typing import Any, Dict, List

import polars as pl

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.medcode import CrossMap
from pyhealth.tasks import BaseTask

# NOTE: CrossMap is loaded in the task's __init__, not here. At module scope it
# downloads on import and re-runs in every set_task worker process.


# =============================================================================
# TASK CLASS DEFINITION
# =============================================================================
class DrugRecommendationMIMIC4Task(BaseTask):
    """Visit-level drug recommendation using cumulative nested history for MIMIC-IV.

    Each sample represents a patient visit and includes the FULL history of
    conditions, procedures, and prior drug prescriptions across ALL previous
    visits (plus the current visit's conditions/procedures).

    The "nested_sequence" processor receives a 2D list of code lists
    [[v0_codes], [v1_codes], ..., [vN_codes]] and converts it into a padded
    2D tensor — every inner list is one visit's codes.
    """

    task_name: str = "DrugRecommendationMIMIC4Task"

    # ⚠️ Use "nested_sequence", NOT "sequence" — the values are 2D lists!
    input_schema: Dict[str, str] = {
        "conditions": "nested_sequence",  # [[v0_diag], [v1_diag], ..., [vN_diag]]
        "procedures": "nested_sequence",  # [[v0_proc], [v1_proc], ..., [vN_proc]]
        "drugs_hist": "nested_sequence",  # [[v0_drugs], ..., [] (current, blanked)]
    }

    output_schema: Dict[str, str] = {"drugs": "multilabel"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ndc_to_atc = CrossMap.load("NDC", "ATC")

    def _to_atc3(self, ndc: str):
        mapped = self.ndc_to_atc.map(ndc, target_kwargs={"level": 3})
        return mapped[0] if mapped else None

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Two-pass approach:
        Pass 1 — collect per-visit flat code lists.
        Pass 2 — accumulate into 2D nested lists; apply leakage guard.
        """
        raw = []  # per-visit flat dicts

        # Anchor: one row per admission
        admissions = patient.get_events(event_type="admissions")

        for admission in admissions:
            # ── Per-visit flat code lists ──────────────────────────────────
            diagnoses_df = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            conditions = (
                diagnoses_df.select(
                    pl.concat_str(
                        ["diagnoses_icd/icd_version", "diagnoses_icd/icd_code"],
                        separator="_",
                    )
                )
                .to_series()
                .to_list()
            )

            procedures_df = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            procedures = (
                procedures_df.select(
                    pl.concat_str(
                        ["procedures_icd/icd_version", "procedures_icd/icd_code"],
                        separator="_",
                    )
                )
                .to_series()
                .to_list()
            )

            prescriptions_df = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            ndc_list = (
                prescriptions_df.select(pl.col("prescriptions/ndc"))
                .to_series()
                .to_list()
            )
            drugs = [
                a for a in (self._to_atc3(d) for d in ndc_list if d) if a
            ]

            # Skip incomplete visits
            if not conditions or not procedures or not drugs:
                continue

            raw.append(
                {
                    "visit_id": admission.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,  # flat for this visit
                    "procedures": procedures,  # flat for this visit
                    "drugs": drugs,  # label (current visit)
                    "drugs_hist": drugs,  # will be accumulated; current zeroed out
                }
            )

        # Need at least 2 valid visits for history-based prediction
        if len(raw) < 2:
            return []

        # ── Pass 2: historical accumulation ───────────────────────────────
        # After this block, raw[i]["conditions"] = [[v0], [v1], ..., [vi]]
        raw[0]["conditions"] = [raw[0]["conditions"]]
        raw[0]["procedures"] = [raw[0]["procedures"]]
        raw[0]["drugs_hist"] = [raw[0]["drugs_hist"]]

        for i in range(1, len(raw)):
            raw[i]["conditions"] = raw[i - 1]["conditions"] + [raw[i]["conditions"]]
            raw[i]["procedures"] = raw[i - 1]["procedures"] + [raw[i]["procedures"]]
            raw[i]["drugs_hist"] = raw[i - 1]["drugs_hist"] + [raw[i]["drugs_hist"]]

        # ── Leakage guard: blank current-visit slot in drugs_hist ─────────
        # The model must predict drugs for visit i — don't let it see them in history.
        # ⚠️ Use ["<pad>"] NOT [] — empty inner lists cause pack_padded_sequence to
        #    crash with "length <= 0" at training time. The nested_sequence processor
        #    treats ["<pad>"] as a single-token placeholder that gets masked out.
        for i in range(len(raw)):
            raw[i]["drugs_hist"][i] = ["<pad>"]

        # ── Optional: history windowing ────────────────────────────────────
        # To limit history to the last N visits (reduces memory, may improve performance),
        # apply a slice AFTER accumulation and BEFORE the leakage guard:
        #
        #   WINDOW = 3  # or 5
        #   for i in range(len(raw)):
        #       raw[i]["conditions"] = raw[i]["conditions"][-WINDOW:]
        #       raw[i]["procedures"] = raw[i]["procedures"][-WINDOW:]
        #       raw[i]["drugs_hist"] = raw[i]["drugs_hist"][-WINDOW:]
        #
        # ⚠️ Apply the leakage guard AFTER windowing, since the current-visit index
        #    in the sliced list is now [-1] (the last slot), not [i]:
        #
        #   for i in range(len(raw)):
        #       raw[i]["drugs_hist"][-1] = ["<pad>"]  # ⚠️ ["<pad>"] not [] — empty inner lists crash pack_padded_sequence
        #
        # The processor still receives a 2D list — just shorter. No other changes needed.

        return raw


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    DATA_ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.2"
    CACHE_DIR = os.path.expanduser("~/.cache/pyhealth")

    base_dataset = MIMIC4Dataset(
        ehr_root=DATA_ROOT,
        ehr_tables=["admissions", "diagnoses_icd", "procedures_icd", "prescriptions"],
        cache_dir=CACHE_DIR,
        dev=False,
    )

    task = DrugRecommendationMIMIC4Task()
    num_workers = min(8, max(1, (os.cpu_count() or 1) - 1))
    sample_dataset = base_dataset.set_task(task, num_workers=num_workers)

    print(f"Total samples: {len(sample_dataset)}")
    if len(sample_dataset) > 0:
        s = sample_dataset[0]
        print(f"Sample keys: {list(s.keys())}")
        print(f"conditions type: {type(s['conditions'])} — len={len(s['conditions'])}")
        print(f"  visit 0 codes (first 3): {s['conditions'][0][:3]}")
        print(f"drugs (label): {s['drugs'][:5]}")

        # ── Also available in PyHealth as a built-in task ──────────────────
        # from pyhealth.tasks import DrugRecommendationMIMIC4
        # sample_dataset = base_dataset.set_task(DrugRecommendationMIMIC4())
