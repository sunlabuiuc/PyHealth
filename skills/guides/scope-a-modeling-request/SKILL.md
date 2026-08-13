---
name: pyhealth-scope-a-modeling-request
description: Turn a vague clinical prediction request into a written, confirmed pipeline spec — cohort, unit of prediction, index time, observation window, prediction gap, label definition, split strategy, and metric. Use before writing any PyHealth modeling code, whenever the request leaves any of those undecided.
---

# Scope a modeling request

Your job in this stage is to end up with a filled-in spec, not to write code.
Read [glossary.md](../../shared/glossary.md) if any term below is unfamiliar to you or to the
user.

---

## How to ask

**Ask once, in one batch.** Do not interrogate one question at a time. Work out
which fields you can reasonably infer, state those inferences, and ask only about
the ones where a wrong guess would send the whole pipeline in the wrong direction.

**Every question carries a recommended default.** The user may well be new to
this. "I don't know" must be a usable answer — so pair each question with a
default and a one-line reason, and let them say "use the defaults."

**Cap it at about six questions.** More than that and you are designing the study
for them rather than with them. Anything left over you can settle later, at the
step where it actually matters.

Good:

> A few things I need to pin down before writing anything — say "defaults" and
> I'll use my recommendations.
>
> 1. **Who counts as a patient here?** Default: all adults with ≥2 admissions
>    (we need two so we can predict forward without leakage).
> 2. **When do you want the prediction made?** Default: at discharge from a
>    visit, predicting the next one.
> 3. **What's the outcome exactly?** Default: died during the next admission.
> 4. **Which data do you want to feed it?** Default: diagnosis, procedure, and
>    drug codes — the labs add a lot of engineering for a modest gain.
> 5. **What have you got to run this on?** GPU, CPU only, and roughly how long
>    are you willing to wait?
> 6. **What does "done" look like?** A working baseline, a tuned model, or a
>    reproducible benchmark with seeds and confidence intervals?

Bad: "What is your unit of prediction?" with no default, no explanation, and five
more like it queued behind.

---

## The fields

Fill all of these before writing the spec file. Mark each **[asked]**, **[inferred]**, or
**[deferred]** so the user can see what you assumed.

| Field | What it means | Default when unspecified |
|---|---|---|
| **Clinical question** | What decision would this model support? | — must be stated |
| **Prediction task** | mortality, readmission, length-of-stay, drug recommendation, phenotyping, ... | — must be stated |
| **Dataset** | which data, and where it lives on disk | MIMIC-IV if the user has it; otherwise the bring-your-own-data path |
| **Cohort** | inclusion + exclusion criteria | adults, ≥2 visits |
| **Unit of prediction** | patient / visit / window / image / note / signal | visit |
| **Temporal structure** | same-visit, next-visit, or patient-cumulative | **next-visit** |
| **Index time** | the moment of prediction | end of the current visit |
| **Observation window** | how far back inputs may reach | the current visit only (or all prior visits for cumulative tasks) |
| **Prediction gap + horizon** | how far ahead, and over what span | next visit, no explicit gap |
| **Label type** | binary / multiclass / multilabel / regression | binary |
| **Positive / negative / excluded** | the exact criteria | must be explicit — no default |
| **Input modalities** | codes, labs, vitals, notes, images, waveforms | diagnosis + procedure + drug codes |
| **Code systems + mapping** | ICD-9/10, NDC, ATC, LOINC; and whether to group them | keep raw for the baseline; try ICD→CCS when optimizing |
| **Split strategy** | how train/val/test are separated | **by patient**, 80/10/10 |
| **Metric** | how success is measured | ROC-AUC + PR-AUC (binary); see [evaluation.md](../train-and-evaluate/references/metrics-and-reporting.md) |
| **Compute** | GPU or CPU, and time tolerance | ask — this sets the whole scope |
| **Definition of done** | baseline / tuned / benchmarked | working baseline |

---

## Temporal structure — the one decision that matters most

Three shapes, and picking wrong is the most common way a clinical model becomes
silently invalid:

- **Visit-level, same-visit** — features and label from the same admission.
  Only safe when the label is structurally independent of the features, e.g.
  length of stay computed from admit/discharge timestamps. Otherwise leaks.
- **Visit-level, next-visit** — features from visit `i`, label from visit `i+1`.
  Safe by construction. **Use this whenever you are unsure.**
- **Patient-level, cumulative** — all prior visits feed the current prediction.
  Needed for drug recommendation and anything where history is the signal.
  Requires `nested_sequence` processors.

State which one you chose and why, in one sentence, in the spec.

---

## The spec file

Write this to `pipeline_spec.md` in the user's working directory, filled in, and
**stop for confirmation**. This is the one hard gate in the skill.

```markdown
# Pipeline spec — <short name>

## Clinical question
<one paragraph: what decision this supports, and for whom>

## Data
- Dataset: <class or "custom, via config at path>
- Tables / files used: <list>
- Access path: <root dir>
- Cache dir: <path>

## Cohort
- Include: <criteria>
- Exclude: <criteria>
- Expected size: <fill in once the dataset loads; "unknown" until then>

## Prediction setup
- Unit of prediction: <patient | visit | window | record>
- Temporal structure: <same-visit | next-visit | cumulative>  — because <reason>
- Index time: <when>
- Observation window: <how far back>
- Gap / horizon: <how far ahead, over what span>

## Label
- Type: <binary | multiclass | multilabel | regression>
- Positive: <criteria>
- Negative: <criteria>
- Excluded / censored: <criteria>
- Expected positive rate: <fill in once the task generates samples>

## Inputs
| Feature | Source | Code system | Processor |
|---|---|---|---|
| conditions | diagnoses_icd | ICD-9/10-CM | sequence |
| ... | | | |

## Evaluation
- Split: <by patient, 80/10/10, seed 42>
- Primary metric: <e.g. roc_auc>  Secondary: <e.g. pr_auc>
- Seeds for the final report: <42, 43, 44>

## Constraints
- Compute: <GPU/CPU, time budget>
- Definition of done: <baseline | tuned | benchmarked>

## Assumptions I made
- <every [inferred] field, listed plainly so the user can correct it>
```

---

## Red flags to raise during the interview

Say these out loud rather than silently working around them:

- **Same-visit prediction with treatment features.** "Predicting mortality for
  this admission from this admission's drugs" mostly learns that dying patients
  get comfort-care medications. Propose next-visit instead.
- **Discharge-coded diagnoses as inputs for that same visit.** In MIMIC,
  `diagnoses_icd` is timestamped at discharge. It cannot be an admission-time
  feature.
- **A cohort so narrow it will yield a few hundred samples.** Estimate it in
  as soon as the dataset loads, and say the number before anyone trains anything.
- **A very rare outcome (<1%) with accuracy as the metric.** Redirect to PR-AUC
  and say why.
- **Real PHI in a shared or logged workspace.** Stop and flag it.
