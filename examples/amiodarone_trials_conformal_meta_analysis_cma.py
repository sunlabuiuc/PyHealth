"""PubMedBERT encoder ablation for the amiodarone case study.


This script extends Kaul & Gordon (2024) by testing whether a
domain-specific pretrained language model (PubMedBERT) produces
better priors for conformal meta-analysis than hand-crafted numeric
features on the 21 amiodarone trials from Letelier et al. (2003).

Pipeline:
    1. Load the amiodarone dataset (21 trials, 10 placebo-controlled
       "trusted" + 11 other "untrusted").
    2. For each trial, obtain text = real abstract (if extracted)
       or generated clinical prose as a fallback.
    3. Embed text with PubMedBERT (CLS token, frozen weights).
    4. Train a CMAPriorEncoder MLP head on the untrusted embeddings
       using PyHealth's Trainer.
    5. Run ConformalMetaAnalysisModel on the trusted trials with
       the learned prior.
    6. Compare against the hand-crafted feature baseline and HKSJ.

Ablation dimensions (produces a 5-row result table):
    - Encoder input:  13 hand-crafted features vs 768-dim PubMedBERT
    - MLP head depth: default [64, 32] vs shallow [32] vs deep [128, 64]
    - HKSJ baseline (no encoder at all)

Fallback: if the ``transformers`` package is not installed, the script
skips the BERT rows and runs only the hand-crafted baseline + HKSJ.
This keeps the example runnable without a 440 MB model download.

Expected findings (what the ablation is designed to reveal):
    - If PubMedBERT embeddings encode trial-level information
      that the 13 hand-crafted features miss, the BERT rows should
      show lower Prior MSE than the hand-crafted baseline.
    - Because CMA uses the prior only to set interval centers (the
      kernel is fixed to the hand-crafted features), a better
      prior should translate to narrower ``CMA Width`` with
      coverage staying near ``1 - alpha``.
    - HKSJ ignores priors entirely, so its width is the "no
      learned prior" ceiling: CMA rows should be narrower than
      HKSJ whenever a prior is at least weakly informative.
    - MLP head depth ([32] vs [64, 32] vs [128, 64]) is secondary
      to the input representation on only 11 untrusted training
      trials; expect small differences across rows 2-4.

Usage:
    python amiodarone_trials_conformal_meta_analysis_cma.py

Optional (for the full ablation):
    pip install transformers

Reference:
    Kaul, S. and Gordon, G. J. 2024. "Meta-Analysis with Untrusted
    Data." Proceedings of Machine Learning Research, 259:563-593.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as t_dist
from torch.utils.data import DataLoader

from pyhealth.datasets import AmiodaroneTrialDataset, get_dataloader
from pyhealth.datasets.amiodarone_trial_dataset import FEATURE_COLUMNS
from pyhealth.models.cma_prior_encoder import CMAPriorEncoder
from pyhealth.models.conformal_meta_analysis_krr import (
    ConformalMetaAnalysisModel,
)
from pyhealth.tasks.conformal_meta_analysis import ConformalMetaAnalysisTask
from pyhealth.trainer import Trainer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATASET_ROOT = "./data/amiodarone"
PUBMEDBERT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


# ---------------------------------------------------------------------
# Embedded abstract corpus
# ---------------------------------------------------------------------
# Real published abstracts for 17 of the 21 amiodarone trials, keyed
# by the ``trial_name`` that appears in the dataset CSV. Trials with
# an empty string fall back to ``generate_clinical_prose``.
#
# Source: PubMed-indexed journal abstracts from the original trial
# publications. Quoted verbatim for research reproduction under fair
# use. Not redistributed as a standalone corpus.
AMIODARONE_ABSTRACTS: Dict[str, str] = {
    "Cowan et al.16 (England) 1986": (
        "Thirty-four patients with atrial fibrillation complicating "
        "suspected acute myocardial infarction were randomised to "
        "treatment with intravenous amiodarone (n = 18) or intravenous "
        "digoxin (n = 16). After 24 h, similar proportions of patients "
        "in each group had reverted to sinus rhythm. However, there "
        "was a tendency towards earlier reversion with amiodarone. At "
        "4 h, 72% of the amiodarone group had reverted to sinus rhythm, "
        "compared with 31% of the digoxin group (p < 0.1). This "
        "tendency was more marked in patients with definite infarction "
        "(at 4 h, amiodarone 75% reversion, digoxin 10% reversion). "
        "Neither drug had a significant effect on blood pressure. "
        "Atrial fibrillation may cause serious haemodynamic "
        "deterioration in acute myocardial infarction. In comparison "
        "with digoxin, amiodarone offers more rapid control of the "
        "ventricular response rate and may, in addition, restore sinus "
        "rhythm more rapidly."
    ),
    "Noc et al.17 (Slovenia) 1990": (
        "Amiodarone and verapamil are well-known antiarrhythmic drugs "
        "used for treatment of ventricular and supraventricular "
        "arrhythmias. We compared the efficacy of intravenous "
        "amiodarone versus verapamil for conversion of paroxysmal "
        "atrial fibrillation to sinus rhythm in a single-blind "
        "randomized study. The patient population consisted of 24 "
        "consecutive patients with paroxysmal atrial fibrillation (15 "
        "men and 9 women aged 71 +/- 9.6 years, range 51 to 85). The "
        "duration of arrhythmia ranged from 20 minutes to 48 hours at "
        "entry. Patients were treated with either amiodarone (5 mg/kg "
        "body weight intravenously over 3 minutes) or verapamil "
        "(0.075 mg/kg intravenously over 1 minute, repeated after 10 "
        "minutes). Treatment was considered successful if conversion "
        "occurred within 3 hours. One of the 11 patients initially "
        "given verapamil converted to sinus rhythm. However, 77% (10 "
        "of 13) of patients who initially received amiodarone "
        "converted (p < 0.001). The conversion occurred 10 to 175 "
        "minutes after administration of amiodarone. In conclusion, "
        "intravenous amiodarone is an effective and safe "
        "antiarrhythmic agent in promoting conversion of paroxysmal "
        "atrial fibrillation to sinus rhythm."
    ),
    "Capucci et al.18 (Italy) 1992": (
        "Sixty-two patients with recent-onset (less than or equal to "
        "1 week) atrial fibrillation (NYHA class 1 and 2) were "
        "randomized in a single-blind study to one of the following "
        "treatment groups: (1) flecainide (300 mg) as a single oral "
        "loading dose; or (2) amiodarone (5 mg/kg) as an intravenous "
        "bolus, followed by 1.8 g/day; or (3) placebo for the first "
        "8 hours. Twenty-four-hour Holter recording was performed, "
        "and conversion to sinus rhythm at 3, 8, 12 and 24 hours was "
        "considered as the criterion of efficacy. Conversion to sinus "
        "rhythm was achieved within 8 hours (placebo-controlled "
        "period) in 20 of 22 patients (91%) treated with flecainide, "
        "7 of 19 (37%) treated with amiodarone (p < 0.001 vs "
        "flecainide), and 10 of 21 (48%) treated with placebo (p < "
        "0.01 vs flecainide). Resumption of sinus rhythm within 24 "
        "hours occurred in 21 of 22 patients (95%) with flecainide "
        "and in 17 of 19 (89%) with amiodarone (p = not significant). "
        "Mean conversion times were shorter for flecainide (190 +/- "
        "147 minutes) than for amiodarone (705 +/- 418; p < 0.001)."
    ),
    "Cochrane et al.19 (Australia) 1994": (
        "Despite the widespread use of amiodarone in non-surgical "
        "patients, its role in the management of supraventricular "
        "tachyarrhythmias after cardiac surgery is not clear. We set "
        "out to compare the relative efficacy of amiodarone and "
        "digoxin in the management of atrial fibrillation and flutter "
        "in the early postoperative period. This prospective "
        "randomised trial comprised 30 patients, previously in sinus "
        "rhythm, who developed sustained atrial fibrillation or "
        "flutter following myocardial revascularisation, valve "
        "surgery or combined procedures. Amiodarone was administered "
        "as an intravenous loading dose followed by a continuous "
        "infusion. Digoxin was given as an intravenous loading dose "
        "followed by oral maintenance therapy. There was a marked "
        "reduction in heart rate in both groups, mainly in the first "
        "6 h, from 146 to 89 beats per minute in the amiodarone "
        "group and from 144 to 95 in the digoxin group. At the end "
        "of the 24 h, one of the 15 patients in the amiodarone group "
        "and 3 of the 15 patients in the digoxin group remained in "
        "atrial fibrillation. We conclude that intravenous "
        "amiodarone therapy is safe and at least as effective as "
        "digoxin in the initial management of arrhythmias after "
        "cardiac surgery."
    ),
    "Donovan et al.20 (Australia) 1995": (
        "After 8 hours there were no significant differences in "
        "reversion between the treatment groups: flecainide "
        "(n = 23, 68%), amiodarone (n = 19, 58%) and placebo "
        "(n = 18, 56%). Amiodarone promptly reduced the ventricular "
        "rate and this effect was maintained for 8 hours in those "
        "whose reversion to stable sinus rhythm was unsuccessful; "
        "flecainide was no more effective than placebo in "
        "controlling ventricular rate. Adverse effects were not "
        "significantly different in the 3 groups. Thus, intravenous "
        "flecainide results in earlier reversion of atrial "
        "fibrillation than does intravenous amiodarone or placebo."
    ),
    "Hou et al.21 (Taiwan) 1995": (
        "A 24 h intravenous dosing regimen of amiodarone was designed "
        "to reach a peak plasma concentration at 1 h and to maintain "
        "the concentration above a certain level during the infusion "
        "period. A randomized, open-label, digoxin-controlled study "
        "was undertaken to observe the efficacy and safety of the "
        "dosing regimen of amiodarone in treating recent-onset, "
        "persistent, atrial fibrillation and flutter with ventricular "
        "rates above 130 beats/min. Fifty patients with a mean age "
        "of 70 +/- 7 (SD) years were enrolled and randomly assigned "
        "to receive either amiodarone intravenously (n = 26) or "
        "digoxin (n = 24). The mean heart rates in the amiodarone "
        "group decreased significantly from 157 +/- 20 beats/min to "
        "122 +/- 25 beats/min after 1 h. Overall, 24 of 26 patients "
        "(92%) in the amiodarone group and 17 of 24 (71%) in the "
        "digoxin group were restored to sinus rhythm within 24 h. "
        "The accumulated rates of conversion over 24 h were "
        "significantly different between the two groups (p = "
        "0.0048). Digoxin, while not as effective as amiodarone in "
        "the treatment of recent-onset atrial fibrillation and "
        "flutter, appears to be safer."
    ),
    "Kondili et al.22 (Albania) 1995": "",
    "Galve et al.23 (Spain) 1996": (
        "This study was designed to determine the efficacy of "
        "intravenous amiodarone in the management of recent-onset "
        "atrial fibrillation. One hundred consecutive patients with "
        "recent-onset (< 1 week) atrial fibrillation and not taking "
        "antiarrhythmic agents were randomized to receive either "
        "intravenous amiodarone, 5 mg/kg body weight in 30 min "
        "followed by 1,200 mg over 24 h, or an identical amount of "
        "saline. Both groups received intravenous digoxin. By the "
        "end of the 24-h treatment period, 34 patients (68%, 95% CI "
        "53% to 80%) in the amiodarone group and 30 (60%, 95% CI "
        "45% to 74%) in the control group had returned to sinus "
        "rhythm (p = 0.532). Mean times of conversion were 328 +/- "
        "335 and 332 +/- 359 min, respectively (p = 0.957). Among "
        "patients who did not convert to sinus rhythm, treatment "
        "with amiodarone was associated with a slower ventricular "
        "rate (82 +/- 15 beats/min vs 91 +/- 23 beats/min, p = "
        "0.022). Intravenous amiodarone, at the doses used in this "
        "study, produces a modest but not significant benefit in "
        "converting acute atrial fibrillation to sinus rhythm."
    ),
    "Kontoyannis et al.24 (Greece) 2001": (
        "Atrial fibrillation is a fairly common complication of "
        "acute myocardial infarction (AMI). The aim of this study "
        "was to examine the safety and efficacy of intravenous "
        "amiodarone in converting AF associated with AMI. Seventy "
        "patients with AMI complicated with AF were prospectively "
        "divided into 3 groups: (a) group D (n = 26), 0.75 mg "
        "digoxin was administered intravenously and thereafter as "
        "needed; (b) group AM (n = 16), 300 mg of amiodarone was "
        "infused over 2 hours followed by 44 mg/hour for up to 60 "
        "hours or until sinus rhythm was restored; (c) group D+AM "
        "(n = 28), 0.75 mg of digoxin was administered for the "
        "initial 2 hours followed by amiodarone infusion as in "
        "group AM. Sinus rhythm was restored by the end of the 96th "
        "hour in 18/26 patients from group D, and in all patients "
        "from group AM and group D+AM. The corresponding duration "
        "of AF was 51 +/- 34 hours, 17 +/- 15 hours and 9 +/- 13 "
        "hours, respectively (F = 15.4, p < 0.001). Intravenous "
        "amiodarone was well tolerated and was effective in "
        "decreasing the duration of AF."
    ),
    "Bellandi et al.26 (Italy) 1999": "",
    "Cotter et al.27 (Israel) 1999": (
        "Spontaneous conversion of recent onset paroxysmal atrial "
        "fibrillation to normal sinus rhythm occurs commonly and is "
        "not affected by low-dose amiodarone treatment. In a "
        "randomized, placebo-controlled trial of 100 patients with "
        "paroxysmal atrial fibrillation of recent onset (< 48 h) we "
        "compared the effects of treatment with continuous "
        "intravenous amiodarone 125 mg per hour (total 3 g) and "
        "intravenous placebo. Conversion to normal sinus rhythm "
        "occurred within 24 h in 32 of 50 patients (64%) in the "
        "placebo group, most of whom converted within 8 h. The "
        "conversion rate during 24 h of treatment in the amiodarone "
        "group was 92% (p = 0.0017). In patients still in atrial "
        "fibrillation after 8 h of treatment, the pulse rate "
        "decreased significantly more in the amiodarone as compared "
        "to the placebo group (83 +/- 15 vs 114 +/- 20 beats/min, "
        "p = 0.0014). Intravenous high-dose amiodarone safely "
        "facilitates conversion of paroxysmal atrial fibrillation."
    ),
    "Kochiadakis et al.12 (Greece) 1999": "",
    "Peuhkurinen et al.30 (Finland) 2000": (
        "The present study evaluates the efficacy and safety of a "
        "single oral dose of amiodarone in patients with "
        "recent-onset atrial fibrillation (< 48 hours). Seventy-two "
        "patients were randomized to receive 30 mg/kg of either "
        "amiodarone or placebo. Conversion to sinus rhythm was "
        "verified by 24-hour Holter monitoring. At 8 hours, "
        "approximately 50% of patients in the amiodarone group and "
        "20% in the placebo group had converted to sinus rhythm, "
        "whereas after 24 hours the corresponding figures were 87% "
        "and 35%, respectively. The median time for conversion "
        "(8.7 hours for amiodarone and 7.9 hours for placebo) did "
        "not differ in the groups. Amiodarone was hemodynamically "
        "well tolerated, and the number of adverse events in the "
        "study groups was similar. Amiodarone as a single oral dose "
        "of 30 mg/kg appears to be effective and safe in patients "
        "with recent-onset atrial fibrillation."
    ),
    "Vardas et al.31 (Greece) 2000": (
        "To investigate the efficacy and safety of amiodarone "
        "administered as the drug of first choice in the conversion "
        "of atrial fibrillation, regardless of its duration. "
        "Two-hundred eight consecutive patients (102 men; mean age "
        "65 +/- 10 years) with atrial fibrillation were enrolled. "
        "One-hundred eight patients received amiodarone, and 100 "
        "patients received placebo treatment. Patients randomized to "
        "amiodarone received 300 mg IV for 1 h, then 20 mg/kg for "
        "24 h, followed by 600 mg/d orally for 1 week and 400 mg/d "
        "for 3 weeks. Conversion to sinus rhythm was achieved in 87 "
        "of 108 patients (80.05%) who received amiodarone, and in "
        "40 of 100 patients (40%) in the placebo group (p < "
        "0.0001). Statistical analysis showed that the duration of "
        "the arrhythmia and the size of the left atrium affected "
        "both the likelihood of conversion to sinus rhythm and the "
        "time to conversion in both groups. Amiodarone appears to "
        "be safe and effective in the termination of atrial "
        "fibrillation."
    ),
    "Joseph and Ward32 (Australia) 2000": (
        "A prospective, randomized controlled trial of new-onset "
        "atrial fibrillation was conducted to compare the efficacy "
        "and safety of sotalol and amiodarone (active treatment) "
        "with rate control by digoxin alone for successful "
        "reversion to sinus rhythm at 48 hours. We prospectively "
        "randomly assigned 120 patients with atrial fibrillation of "
        "less than 24 hours duration to treatment with sotalol, "
        "amiodarone, or digoxin using a single intravenous dose "
        "followed by 48 hours of oral treatment. There was a "
        "significant reduction in the time to reversion with both "
        "sotalol (13.0 +/- 2.5 hours, p < 0.01) and amiodarone "
        "(18.1 +/- 2.9 hours, p < 0.05) treatment compared with "
        "digoxin only (26.9 +/- 3.4 hours). By 48 hours, the active "
        "treatment group was significantly more likely to have "
        "reverted to sinus rhythm than the rate control group (95% "
        "vs 78%, p < 0.05). Immediate pharmacologic therapy for "
        "new-onset atrial fibrillation with class III antiarrhythmic "
        "drugs improves complication-free 48-hour reversion rates "
        "compared with rate control with digoxin."
    ),
    "Cybulski et al.33 (Poland) 2001": "",
    "Natale et al.25 (United States) 1998": "",
    "Bianconi et al.28 (Italy) 2000": (
        "This study compared the efficacy and safety of intravenous "
        "dofetilide with amiodarone and placebo in converting "
        "atrial fibrillation or flutter to sinus rhythm. One "
        "hundred and fifty patients with atrial fibrillation or "
        "flutter (duration range 2 h to 6 months) were given 15-min "
        "intravenous infusions of 8 ug/kg of dofetilide (n = 48), "
        "5 mg/kg of amiodarone (n = 50), or placebo (n = 52) and "
        "monitored continuously for 3 h. Sinus rhythm was restored "
        "in 35%, 4%, and 4% of patients, respectively (p < 0.001, "
        "dofetilide vs placebo; p = ns, amiodarone vs placebo). "
        "Dofetilide was more effective in atrial flutter than in "
        "atrial fibrillation (cardioversion rates 75% and 22%, "
        "respectively; p = 0.004). Intravenous dofetilide is "
        "significantly more effective than amiodarone or placebo "
        "in restoring sinus rhythm in patients with atrial "
        "fibrillation or flutter."
    ),
    "Galperin et al.29 (Argentina) 2000": (
        "We sought to assess the efficacy and safety of amiodarone "
        "for restoration and maintenance of sinus rhythm in "
        "patients with chronic atrial fibrillation in a "
        "prospective, randomized, double blind trial. Ninety-five "
        "patients with chronic atrial fibrillation, lasting an "
        "average of 35.6 months, were randomized to either "
        "amiodarone (600 mg/d) (47 patients) or placebo (48 "
        "patients) during four weeks. Nonresponders underwent "
        "electric cardioversion. Sixteen patients (34.04%) in the "
        "amiodarone group reverted within 27.28 +/- 8.85 days in "
        "comparison with 0% in the placebo group (p < 0.000009). "
        "Altogether, conversion was obtained in 79.54% of the "
        "amiodarone group patients and in 38.46% of the placebo "
        "group patients (p < 0.0001). During follow-up, atrial "
        "fibrillation relapsed in 13 (37.14%) of 35 patients of the "
        "amiodarone group within 8.84 +/- 8.57 months and in 12 "
        "(80%) of 15 patients of the placebo group within 2.74 "
        "+/- 3.41 months. Oral amiodarone restored sinus rhythm in "
        "one third of patients with chronic atrial fibrillation, "
        "increased the success rate of electric cardioversion, "
        "decreased the number of relapses and delayed their "
        "occurrence."
    ),
    "Hohnloser et al.3 (Germany) 2000": (
        "Atrial fibrillation is the most commonly encountered "
        "sustained cardiac arrhythmia. Restoration and maintenance "
        "of sinus rhythm is believed by many physicians to be "
        "superior to rate control only. The Pharmacological "
        "Intervention in Atrial Fibrillation (PIAF) trial was a "
        "randomised trial in 252 patients with atrial fibrillation "
        "of between 7 days and 360 days duration, which compared "
        "rate (group A, 125 patients) with rhythm control (group "
        "B, 127 patients). In group A, diltiazem was used as "
        "first-line therapy and amiodarone was used in group B. "
        "Over the entire observation period of 1 year, a similar "
        "proportion of patients reported improvement in symptoms "
        "in both groups (76 vs 70 responders, p = 0.317). "
        "Amiodarone administration resulted in pharmacological "
        "restoration of sinus rhythm in 23% of patients. With "
        "respect to symptomatic improvement, rate versus rhythm "
        "control yielded similar clinical results. Exercise "
        "tolerance is better with rhythm control, although hospital "
        "admission is more frequent."
    ),
    "Villani et al.11 (Italy) 2000": (
        "Pretreatment with calcium channel blocker may improve the "
        "efficacy of electric cardioversion by reversing the "
        "so-called 'electric remodeling' phenomenon. The efficacy "
        "of diltiazem or amiodarone pretreatment (oral, 1 month "
        "before and 1 month after conversion) on direct-current "
        "conversion of persistent atrial fibrillation was assessed "
        "in 120 patients, randomly assigned to 3 matched groups: "
        "A (n = 44, diltiazem); B (n = 46, amiodarone), and C "
        "(n = 30, digoxin). Spontaneous conversion to sinus rhythm "
        "was achieved in 6% of patients of group A, 25% of group "
        "B and 3% of group C (A/C vs B, p < 0.005). Current "
        "conversion was more successful in group B (91%) compared "
        "with group A (76%) and group C (67%) (B vs A/C, p < "
        "0.05). At 1 month the recurrence rate was lower in group "
        "B (28%) versus groups A (56%) and C (78%) (B vs A/C, p < "
        "0.01). Diltiazem is less effective than amiodarone in "
        "determining spontaneous or electric conversion, with a "
        "higher recurrence rate."
    ),
}


# ---------------------------------------------------------------------
# Text rendering (fallback for trials without real abstracts)
# ---------------------------------------------------------------------
def generate_clinical_prose(row: pd.Series) -> str:
    """Render a trial's structured features as clinical prose.

    Used when a real abstract is not available. Output mimics the
    style of a clinical trial abstract so PubMedBERT processes it
    similarly to a real abstract.
    """
    name = row.get("trial_name", "trial")
    dose = row.get("amiodarone_total_24h_mg", 0) or 0
    comp_intensity = int(row.get("comparison_intensity", 0) or 0)
    af_long = bool(row.get("af_duration_gt_48h", 0) or 0)
    outcome_long = bool(row.get("outcome_time_gt_48h", 0) or 0)
    mean_age = row.get("mean_age", 0) or 0
    la_size = row.get("mean_la_size", 0) or 0
    male_frac = row.get("fraction_male", 0) or 0
    cv_frac = row.get("fraction_cv_disease", 0) or 0
    followup = row.get("followup_fraction", 1.0) or 1.0
    masked_pt = bool(row.get("masked_patients", 0) or 0)
    masked_cg = bool(row.get("masked_caregiver", 0) or 0)
    adequate_concealment = bool(row.get("adequate_concealment", 0) or 0)

    comparison = {
        0: "placebo control",
        1: "low-intensity active comparator",
        2: "high-intensity active comparator",
    }.get(comp_intensity, "active comparator")

    af_str = "persistent" if af_long else "recent-onset"
    outcome_str = (
        "long-term conversion (greater than 48 hours)"
        if outcome_long
        else "short-term conversion (within 48 hours)"
    )
    blinding = (
        "double-blinded"
        if (masked_pt and masked_cg)
        else "single-blinded"
        if masked_pt
        else "open-label"
    )

    return (
        f"Randomized controlled trial ({name}) evaluating amiodarone "
        f"versus {comparison} for conversion of atrial fibrillation to "
        f"sinus rhythm. Patients had {af_str} atrial fibrillation. "
        f"Total amiodarone dose over 24 hours: {dose:.0f} mg. "
        f"Outcome assessed as {outcome_str}. "
        f"Mean patient age {mean_age:.0f} years; "
        f"mean left atrial size {la_size:.1f} cm. "
        f"Male fraction {male_frac:.2f}; "
        f"cardiovascular disease prevalence {cv_frac:.2f}. "
        f"Follow-up completion {followup:.2f}. "
        f"Study was {blinding} with "
        f"{'adequate' if adequate_concealment else 'unclear'} "
        f"allocation concealment."
    )


def get_trial_text(
    row: pd.Series,
    abstracts: Dict[str, str],
) -> Tuple[str, str]:
    """Return ``(text, source)`` where source is 'real' or 'generated'."""
    name = row.get("trial_name", "")
    real = abstracts.get(name, "").strip() if name else ""
    if real and len(real) > 100:
        return real, "real"
    return generate_clinical_prose(row), "generated"


# ---------------------------------------------------------------------
# PubMedBERT embedding (deferred import)
# ---------------------------------------------------------------------
def embed_with_bert(
    texts: List[str],
    model_name: str = PUBMEDBERT,
    device: str = "cpu",
) -> torch.Tensor:
    """Return ``[n, 768]`` CLS embeddings, one per text.

    Deferred-imports the ``transformers`` package so the rest of the
    script works without it.
    """
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_emb = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :].squeeze(0).cpu()
            all_emb.append(cls)
            if (i + 1) % 5 == 0:
                print(f"  embedded {i + 1}/{len(texts)}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.stack(all_emb)


# ---------------------------------------------------------------------
# HKSJ baseline (Proposition 10)
# ---------------------------------------------------------------------
def hksj_interval(
    Y: np.ndarray,
    V: np.ndarray,
    alpha: float = 0.1,
) -> Tuple[float, float]:
    """Hartung-Knapp-Sidik-Jonkman prediction interval."""
    n = len(Y)
    nu = 0.0
    for _ in range(1000):
        w = 1.0 / (V + nu)
        ate = np.sum(w * Y) / np.sum(w)
        nu_new = max(
            0.0,
            np.sum(w ** 2 * ((Y - ate) ** 2 - V)) / np.sum(w ** 2)
            + 1.0 / np.sum(w),
        )
        if abs(nu_new - nu) < 1e-8:
            break
        nu = nu_new
    w = 1.0 / (V + nu)
    ate = np.sum(w * Y) / np.sum(w)
    var_ate = np.sum((Y - ate) ** 2 * w) / ((n - 1) * np.sum(w))
    half = t_dist.ppf(1 - alpha / 2, df=n - 1) * np.sqrt(nu + var_ate)
    return float(ate - half), float(ate + half)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _batch_from_samples(sample_dataset) -> Dict[str, torch.Tensor]:
    """Stack all samples into a single batch dict via ``get_dataloader``.

    Thin wrapper around PyHealth's ``get_dataloader`` that pulls
    every sample into a single batch. Using the library's loader
    (instead of hand-rolled stacking) keeps this example aligned
    with PyHealth conventions and automatically inherits the
    default collate's handling of tensor / scalar keys.
    """
    loader = get_dataloader(
        sample_dataset,
        batch_size=len(sample_dataset),
        shuffle=False,
    )
    return next(iter(loader))


class _FeatureDataset(torch.utils.data.Dataset):
    """Minimal Dataset wrapper exposing (features, true_effect) samples.

    Used both for initializing ``CMAPriorEncoder`` (to infer input_dim
    via ``__getitem__(0)``) and as the source for a ``DataLoader``
    consumed by PyHealth's ``Trainer``.
    """

    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self._feats = features.float()
        self._tgts = targets.view(-1).float()
        self.input_schema = {"features": "tensor"}
        self.output_schema = {"true_effect": "regression"}

    def __len__(self) -> int:
        return len(self._feats)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self._feats[i],
            "true_effect": self._tgts[i].unsqueeze(-1),
        }


def _collate_encoder_batch(
    items: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Collate function that stacks per-sample tensors into a batch.

    PyHealth's ``Trainer`` passes the collated dict straight into
    ``model(**batch)``, so keys must match the encoder's ``forward``
    signature (``features``, ``true_effect``).
    """
    return {
        "features": torch.stack([item["features"] for item in items]),
        "true_effect": torch.stack([item["true_effect"] for item in items]),
    }


def train_encoder_with_trainer(
    encoder: CMAPriorEncoder,
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 200,
    batch_size: int = 8,
    lr: float = 1e-3,
) -> CMAPriorEncoder:
    """Train the encoder via PyHealth's ``Trainer``.

    ``CMAPriorEncoder.forward`` already returns the ``{y_pred, y_true,
    loss}`` dict that ``Trainer`` expects, so no wrapper is needed.
    """
    dataset = _FeatureDataset(features, targets)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_encoder_batch,
    )

    trainer = Trainer(
        model=encoder,
        metrics=["mse"],
        enable_logging=False,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=train_loader,
        epochs=epochs,
        optimizer_params={"lr": lr},
        monitor="mse",
        monitor_criterion="min",
        load_best_model_at_last=False,
    )
    encoder.eval()
    return encoder


# ---------------------------------------------------------------------
# Core: run CMA with a given feature representation
# ---------------------------------------------------------------------
def run_cma_with_features(
    u_features: torch.Tensor,
    t_features: torch.Tensor,
    u_batch: Dict[str, torch.Tensor],
    t_batch: Dict[str, torch.Tensor],
    trusted_dataset,
    label: str,
    input_desc: str,
    hidden_dims: Optional[List[int]] = None,
    embed_dim: int = 16,
    alpha: float = 0.1,
) -> Dict:
    """Train encoder on untrusted features, run CMA on trusted.

    The CMA model uses the ORIGINAL hand-crafted features for its
    KRR kernel; the learned encoder only drives the prior mean M.
    This isolates the effect of prior quality from any change in
    the kernel.
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]

    fake_u = _FeatureDataset(u_features, u_batch["true_effect"])

    encoder = CMAPriorEncoder(
        dataset=fake_u,
        hidden_dims=hidden_dims,
        embed_dim=embed_dim,
    )
    train_encoder_with_trainer(
        encoder,
        features=u_features.float(),
        targets=u_batch["true_effect"],
    )

    # Predict M for trusted trials
    with torch.no_grad():
        M = encoder.predict_prior_mean(t_features.float())

    t_with_prior = dict(t_batch)
    t_with_prior["prior_mean"] = M.unsqueeze(-1)

    cma = ConformalMetaAnalysisModel(
        dataset=trusted_dataset,
        alpha=alpha,
    )
    with torch.no_grad():
        out = cma(**t_with_prior)

    lo = out["interval_lower"].cpu().numpy().ravel()
    hi = out["interval_upper"].cpu().numpy().ravel()
    u_true = t_batch["true_effect"].cpu().numpy().ravel()

    finite = np.isfinite(lo) & np.isfinite(hi)
    width = (
        float(np.mean(hi[finite] - lo[finite]))
        if finite.any()
        else np.nan
    )
    coverage = float(np.mean((u_true >= lo) & (u_true <= hi)))
    mse_prior = float(
        torch.mean((M - t_batch["true_effect"].squeeze(-1)) ** 2)
    )

    return {
        "Encoder": label,
        "Input": input_desc,
        "Feature Dim": u_features.shape[1],
        "Prior MSE": round(mse_prior, 4),
        "CMA Width": round(width, 4),
        "CMA Coverage": round(coverage, 3),
    }


# ---------------------------------------------------------------------
# Main ablation
# ---------------------------------------------------------------------
def run_bert_ablation(
    seed: int = 0,
    alpha: float = 0.1,
) -> pd.DataFrame:
    """Run the full ablation and return a results DataFrame."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Detect whether transformers is installed; skip BERT rows if not.
    try:
        import transformers  # noqa: F401

        bert_available = True
    except ImportError:
        print(
            "[INFO] `transformers` not installed. Skipping PubMedBERT "
            "rows; running hand-crafted baseline + HKSJ only. "
            "Install `transformers` to enable the full ablation."
        )
        bert_available = False

    # Abstract corpus is embedded at the top of this file.
    abstracts = AMIODARONE_ABSTRACTS

    # Load dataset and build lookup
    dataset = AmiodaroneTrialDataset(root=DATASET_ROOT)
    csv_path = os.path.join(
        DATASET_ROOT,
        "amiodarone_trials-metadata-pyhealth.csv",
    )
    df = pd.read_csv(csv_path)

    # Build text inputs
    texts: List[str] = []
    n_real, n_generated = 0, 0
    for _, row in df.iterrows():
        text, source = get_trial_text(row, abstracts)
        texts.append(text)
        if source == "real":
            n_real += 1
        else:
            n_generated += 1

    print(
        f"\nText inputs: {n_real} real abstracts, "
        f"{n_generated} generated prose, {len(df)} total\n"
    )

    # Align indices to splits
    untrusted_idx = df.index[df["split"] == "untrusted"].tolist()
    trusted_idx = df.index[df["split"] == "trusted"].tolist()

    # Hand-crafted baseline via the task pipeline
    task_u = ConformalMetaAnalysisTask(
        target_column="log_relative_risk",
        feature_columns=FEATURE_COLUMNS,
        split_column="split",
        split_value="untrusted",
        observed_column=None,
        variance_column=None,
        prior_column=None,
    )
    untrusted = dataset.set_task(task_u)
    u_batch = _batch_from_samples(untrusted)

    task_t = ConformalMetaAnalysisTask(
        target_column="log_relative_risk",
        feature_columns=FEATURE_COLUMNS,
        split_column="split",
        split_value="trusted",
        observed_column="log_relative_risk",
        variance_column="variance",
        prior_column=None,
    )
    trusted = dataset.set_task(task_t)
    t_batch = _batch_from_samples(trusted)

    rows: List[Dict] = []

    # Row 1: hand-crafted 13 features (baseline)
    print("Running hand-crafted baseline...")
    rows.append(
        run_cma_with_features(
            u_features=u_batch["features"],
            t_features=t_batch["features"],
            u_batch=u_batch,
            t_batch=t_batch,
            trusted_dataset=trusted,
            label="MLP",
            input_desc=f"{len(FEATURE_COLUMNS)} hand-crafted features",
            alpha=alpha,
        )
    )

    # Rows 2-4: PubMedBERT variants (only if transformers installed)
    if bert_available:
        print("\nEmbedding trials with PubMedBERT...")
        bert_emb = embed_with_bert(texts, PUBMEDBERT)
        print(f"BERT embeddings shape: {bert_emb.shape}\n")

        u_bert = bert_emb[untrusted_idx]
        t_bert = bert_emb[trusted_idx]

        print("Running PubMedBERT + default MLP...")
        rows.append(
            run_cma_with_features(
                u_features=u_bert,
                t_features=t_bert,
                u_batch=u_batch,
                t_batch=t_batch,
                trusted_dataset=trusted,
                label="PubMedBERT + MLP",
                input_desc=f"{n_real} real / {n_generated} gen",
                hidden_dims=[64, 32],
                embed_dim=16,
                alpha=alpha,
            )
        )

        for arch_name, hd, ed in [
            ("Shallow", [32], 8),
            ("Deep", [128, 64], 16),
        ]:
            print(f"Running PubMedBERT + {arch_name} MLP...")
            rows.append(
                run_cma_with_features(
                    u_features=u_bert,
                    t_features=t_bert,
                    u_batch=u_batch,
                    t_batch=t_batch,
                    trusted_dataset=trusted,
                    label=f"PubMedBERT + {arch_name}",
                    input_desc=f"{n_real} real / {n_generated} gen",
                    hidden_dims=hd,
                    embed_dim=ed,
                    alpha=alpha,
                )
            )

    # Final row: HKSJ baseline
    Y = t_batch["observed_effect"].cpu().numpy().ravel()
    V = t_batch["variance"].cpu().numpy().ravel()
    u_true = t_batch["true_effect"].cpu().numpy().ravel()
    hlo, hhi = hksj_interval(Y, V, alpha=alpha)
    hksj_cov = float(np.mean((u_true >= hlo) & (u_true <= hhi)))
    rows.append(
        {
            "Encoder": "HKSJ (baseline)",
            "Input": "observed Y, V only",
            "Feature Dim": 0,
            "Prior MSE": np.nan,
            "CMA Width": round(hhi - hlo, 4),
            "CMA Coverage": round(hksj_cov, 3),
        }
    )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 80)
    print("PubMedBERT Encoder Ablation for Conformal Meta-Analysis")
    print("=" * 80)

    results = run_bert_ablation()

    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(results.to_string(index=False))

    # Save for the report
    results.to_csv("bert_ablation_results.csv", index=False)
    print("\nSaved bert_ablation_results.csv")