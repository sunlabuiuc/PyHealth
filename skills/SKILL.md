---
name: pyhealth
description: Work with PyHealth, the clinical predictive modeling library — install and set it up from scratch, load EHR/imaging/signal datasets or onboard raw patient files, define prediction tasks, choose processors and models, train and evaluate, calibrate probabilities, interpret a trained model, map medical codes (ICD/NDC/ATC/CCS), and contribute new datasets, tasks, models, processors, metrics, calibrators, or interpreters upstream. Use for any request involving pyhealth, MIMIC-III/IV, eICU, OMOP, EHRShot, clinical prediction, or this repository's package code — including "help me get started with PyHealth" and installation or environment problems.
---

# PyHealth

This skill is a **router**. It does not contain the answer; it works out which guides do, and
loads those.

PyHealth is large — 24 datasets, 46 tasks, 48 models, 24 registered processors, 11 calibrators
and set predictors, 12 interpreters, 10 code vocabularies. A request touches a few of those, not
all of them. Your job is to name which few, then read only those guides.

---

## 0. Preflight — is PyHealth actually installed?

Before decomposing anything, check that the thing exists:

```bash
python -P -c "import pyhealth; print(pyhealth.__version__)"
```

The `-P` is load-bearing. Without it Python puts the current directory on the path, so
running this from a PyHealth checkout imports the *source tree* and reports success even
when nothing is installed and no dependency is present.

If that fails, the request cannot proceed no matter how well you route it —
read [set-up-the-environment](guides/set-up-the-environment/SKILL.md) and do the
setup first, then come back here and continue. Do not ask the user to install it
and report back; install it, verify it, and carry on in the same turn.

If it succeeds, say nothing about it and move on. A working environment does not
need announcing.

## 1. Decompose the request

Before reading any guide or writing any code:

**a. What is the goal?**
- **Use** PyHealth — build, train, or analyze something with the existing library
- **Extend** it — write a new component for their own project
- **Contribute** it upstream — same, but it has to pass this repo's CI gate

These have different definitions of done. Contributing has a hard requirement most people miss:
any PR touching `pyhealth/` must *also* add files under `docs/` and `examples/`.

**b. Which subsystems does it touch?** Walk the chain and mark what is involved:

```
raw data → dataset → task → processors → model → training → evaluation
                                                      ↘ calibration
                                                      ↘ interpretation
                        medical codes cut across all of it
```

**c. What already exists?** PyHealth probably ships it. A built-in task in `pyhealth/tasks/`
beats anything you would write. Check before building.

**d. Say the decomposition back to the user** — a short ordered list of what you are going to
do — then load the guides for those steps and nothing else.

> "This needs three things: getting your CSVs into PyHealth's format, defining the mortality
> task, then training. I'll start with the data — the other two depend on how it lands."

Loading every guide "to be safe" defeats the design. Two or three is normal.

## 2. Guides

| Guide | Use it when | Don't use it when |
|---|---|---|
| [set-up-the-environment](guides/set-up-the-environment/SKILL.md) | `import pyhealth` fails, errors, or the user has just cloned the repo | the import already works |
| [scope-a-modeling-request](guides/scope-a-modeling-request/SKILL.md) | a modeling request leaves cohort, timing, label, or metric undecided | the request is already fully specified, or nothing is being trained |
| [bring-your-own-data](guides/bring-your-own-data/SKILL.md) | the data is raw files — CSV, TSV, Parquet | it's a supported dataset |
| [use-a-dataset](guides/use-a-dataset/SKILL.md) | picking or loading a built-in dataset | the data is custom |
| [define-a-task](guides/define-a-task/SKILL.md) | writing what to predict and from what | a built-in task in `pyhealth/tasks/` already fits |
| [choose-processors](guides/choose-processors/SKILL.md) | filling in `input_schema`/`output_schema`, or a processor errored | schemas already work |
| [choose-a-model](guides/choose-a-model/SKILL.md) | selecting a model, or one rejected your schema | the model is already chosen and compatible |
| [train-and-evaluate](guides/train-and-evaluate/SKILL.md) | splitting, training, scoring | no training involved |
| [optimize-a-pipeline](guides/optimize-a-pipeline/SKILL.md) | the user asks to improve a **working** baseline | nothing trains end to end yet |
| [calibrate-predictions](guides/calibrate-predictions/SKILL.md) | probabilities are overconfident, or coverage guarantees are wanted | only ranking metrics matter |
| [interpret-a-model](guides/interpret-a-model/SKILL.md) | "why did it predict this?", or auditing for spurious signal | no trained model exists yet |
| [map-medical-codes](guides/map-medical-codes/SKILL.md) | codes need grouping, normalizing, or translating between systems | raw codes are fine |
| [add-a-component](guides/add-a-component/SKILL.md) | writing a new dataset/task/model/processor/metric/calibrator/interpreter **for the package** | using existing components |
| [ship-a-contribution](guides/ship-a-contribution/SKILL.md) | preparing or debugging a PR against PyHealth | the change stays local |

Fuller descriptions, prerequisites, and outputs: [table-of-contents.md](table-of-contents.md).

**Shared background**, referenced by many guides:
[glossary.md](shared/glossary.md) (clinical ML terms — read if a term is unfamiliar to you or
the user), [data-model.md](shared/data-model.md) (Event, Patient, sample dicts, caching),
[repo-conventions.md](shared/repo-conventions.md) (the CI gate and repo standards).

## 3. Compose

Multiple guides is the normal case. Read them in dependency order — data before task, task
before model, model before interpretation. Do one guide's work at a time and show the user real
output at each boundary before moving on.

Common combinations:

| Request | Guides, in order |
|---|---|
| "help me get started with PyHealth" | set-up-the-environment → scope → *(then whatever they described)* |
| "predict mortality on MIMIC-IV" | scope → use-a-dataset → define-a-task → choose-a-model → train-and-evaluate |
| "I have patient CSVs, can I use PyHealth?" | bring-your-own-data → *(then scope, once it loads)* |
| "add my GRU variant to PyHealth" | add-a-component → ship-a-contribution |
| "why did the model predict this?" | interpret-a-model |
| "our probabilities look overconfident" | calibrate-predictions |
| "add a dataset and a task, then upstream it" | bring-your-own-data → define-a-task → add-a-component → ship-a-contribution |

## 4. Rules that hold across every guide

**Agree before building.** For modeling work, do not write code until a spec is confirmed
(`scope-a-modeling-request`). For contribution work, do not write code until the file list is
agreed (`add-a-component`). One batched round of questions, each with a recommended default so
"use the defaults" is always a real answer. Never a one-question-at-a-time interrogation.

**Assume the user is new to clinical ML** until you learn otherwise. Define terms as you use
them, explain *why* a rule exists rather than just stating it, and show output — a sample count,
a printed sample dict, a tensor shape, a loss curve. A newcomer cannot tell a working pipeline
from a silently empty one without seeing the data.

**Read the source before claiming an API.** Every class name, kwarg, and processor name in these
guides was verified against `pyhealth/`. When you need something not covered, read the file —
do not guess a signature.

**The safety rules are not negotiable**, and each has a reason worth saying out loud:

- **Split by patient** — the same patient in train and test lets the model memorize the person,
  and the test score stops measuring anything.
- **Predict forward, not sideways** — features from visit `i`, label from visit `i+1`.
  Same-visit setups leak whenever the label shapes the features, which in EHR data is nearly
  always. Discharge-coded diagnoses are the classic trap.
- **Touch the test set once**, after everything is selected on validation.
- **`dev=True` answers "does it run", never "is it good"** — ~1000 patients is far too small for
  a metric to mean anything.
- **Report mean ± std over ≥3 seeds.** One seed is mostly noise.
- **No real PHI** in shared caches, logs, or anything published.

**Stop when something looks wrong.** Zero samples, a single-class label, a flat loss, an
ROC-AUC above ~0.95 on a next-visit clinical outcome — these are findings, not things to
proceed past. The last one almost always means leakage.
