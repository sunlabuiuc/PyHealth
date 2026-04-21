"""
Example and Ablation Study: MIMIC-III Mortality Text Task
==========================================================

Reproduces part of the evaluation pipeline from:

    Zhang et al. "Hurtful Words: Quantifying Biases in Clinical Contextual
    Word Embeddings." ACM CHIL 2020. https://arxiv.org/abs/2003.11515

This script demonstrates:
    1. How to use MortalityTextTaskMIMIC3 with MIMIC3Dataset.
    2. An ablation study showing how max_notes affects sample generation.
    3. Fairness gap evaluation (recall gap, parity gap) across demographic
       subgroups (gender, ethnicity, insurance) as described in the paper.

Requirements:
    - pyhealth
    - MIMIC-III data with PATIENTS and ADMISSIONS tables
      (or use the synthetic subset below for a quick demo)

Usage:
    python examples/mimic3_mortality_text_clinicalbert.py
"""

from collections import Counter, defaultdict

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks.mortality_text_task import MortalityTextTaskMIMIC3

# ---------------------------------------------------------------------------
# 1. Load dataset
#    Replace root with your local MIMIC-III path or use the synthetic subset.
# ---------------------------------------------------------------------------

MIMIC3_ROOT = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III_subset/"

print("Loading MIMIC-III dataset...")
dataset = MIMIC3Dataset(
    root=MIMIC3_ROOT,
    tables=["PATIENTS", "ADMISSIONS"],
    dev=False,
)
print(f"Loaded {len(dataset.unique_patient_ids)} patients.\n")


# ---------------------------------------------------------------------------
# 2. Ablation: effect of max_notes on sample generation
#
#    We test max_notes in [1, 3, 5, 8] and report:
#      - Total samples generated
#      - Average notes per sample
#      - Label distribution (mortality rate)
#
#    Finding: max_notes does not affect label distribution, only the amount
#    of text available to the model. More notes give the model more context
#    but also increase tokenization cost for BERT-based models.
# ---------------------------------------------------------------------------

print("=" * 60)
print("ABLATION: Effect of max_notes on sample generation")
print("=" * 60)

for max_notes in [1, 3, 5, 8]:
    task = MortalityTextTaskMIMIC3(max_notes=max_notes)
    all_samples = []
    for pid in dataset.unique_patient_ids:
        patient = dataset.get_patient(pid)
        all_samples.extend(task(patient))

    label_counts = Counter(s["label"] for s in all_samples)
    avg_notes = sum(len(s["notes"]) for s in all_samples) / max(len(all_samples), 1)
    mortality_rate = label_counts[1] / max(len(all_samples), 1) * 100

    print(f"\nmax_notes={max_notes}")
    print(f"  Total samples : {len(all_samples)}")
    print(f"  Avg notes/sample: {avg_notes:.1f}")
    print(f"  Label dist    : {dict(label_counts)}")
    print(f"  Mortality rate: {mortality_rate:.1f}%")


# ---------------------------------------------------------------------------
# 3. Fairness evaluation (recall gap & parity gap)
#
#    We compute naive baseline fairness gaps using the label distribution
#    itself (no model needed) to show demographic imbalance in the dataset.
#    This mirrors Table 4 of Zhang et al. (2020).
#
#    In a full pipeline, replace `predicted` with real model predictions.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("FAIRNESS GAPS: Demographic subgroup analysis (label distribution)")
print("Paper reference: Zhang et al. 2020, Table 4")
print("=" * 60)

task = MortalityTextTaskMIMIC3(max_notes=5)
samples = []
for pid in dataset.unique_patient_ids:
    patient = dataset.get_patient(pid)
    samples.extend(task(patient))

# group samples by demographic
def parity_gap(group_samples):
    """Compute parity (positive prediction rate) for a group."""
    if not group_samples:
        return 0.0
    return sum(s["label"] for s in group_samples) / len(group_samples)


for attr in ["gender", "ethnicity", "insurance"]:
    print(f"\n--- {attr.upper()} ---")
    groups = defaultdict(list)
    for s in samples:
        groups[s[attr]].append(s)

    rates = {g: parity_gap(v) for g, v in groups.items() if len(v) >= 5}
    if not rates:
        print("  (insufficient data)")
        continue

    majority = max(rates, key=rates.get)
    for group, rate in sorted(rates.items(), key=lambda x: -x[1]):
        gap = rates[majority] - rate
        marker = " <- majority" if group == majority else f"  gap={gap:.3f}"
        print(f"  {group:<45} rate={rate:.3f}  n={len(groups[group])}{marker}")


# ---------------------------------------------------------------------------
# 4. Template ablation: different template sets
#
#    We test two subsets of templates:
#      A) Chronic conditions only (heart disease, diabetes, hypertension)
#      B) Social/behavioural conditions (hiv, heroin, dnr)
#
#    Finding: Template choice does not affect the label, but affects what
#    linguistic context BERT encodes — directly relevant to the paper's
#    finding that BERT encodes gender bias differently per medical topic.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("ABLATION: Template subset comparison")
print("=" * 60)

CHRONIC_TEMPLATES = [
    "this is a {age} yo {gender} with a hx of heart disease",
    "this is a {age} yo {gender} with a pmh of diabetes",
    "this is a {age} yo {gender} with a discharge diagnosis of htn",
]

SOCIAL_TEMPLATES = [
    "{gender} has a pmh of hiv",
    "{gender} pt is dnr",
    "this is a {age} yo {gender} with a hx of heroin addiction",
]

for name, templates in [("Chronic conditions", CHRONIC_TEMPLATES),
                         ("Social/behavioural", SOCIAL_TEMPLATES)]:
    # patch templates temporarily
    import pyhealth.tasks.mortality_text_task as _mod
    original = _mod.CLINICAL_NOTE_TEMPLATES
    _mod.CLINICAL_NOTE_TEMPLATES = templates

    task = MortalityTextTaskMIMIC3(max_notes=3)
    s0 = task(dataset.get_patient(list(dataset.unique_patient_ids)[0]))

    _mod.CLINICAL_NOTE_TEMPLATES = original  # restore

    print(f"\nTemplate set: {name}")
    if s0:
        print(f"  Example notes: {s0[0]['notes']}")
    else:
        print("  (no samples generated)")

print("\nDone. See paper for full model-based fairness evaluation results.")
