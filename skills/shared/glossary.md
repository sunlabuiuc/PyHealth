# Glossary — the words this skill uses

If you are new to clinical predictive modeling, read this once. Everything else
in the skill assumes these terms.

---

**Cohort** — the set of patients your model is about. "Adult ICU patients with at
least two hospital admissions" is a cohort. Defining it means writing down who is
*included* and who is *excluded*, explicitly. Most disagreements about a
clinical model turn out to be disagreements about the cohort.

**Unit of prediction** — what one row of your dataset represents. In PyHealth,
one row is one *sample dict*. Common units:
- **one patient** — a single prediction per person
- **one visit / admission** — a prediction per hospital stay
- **one time window** — a prediction per 24h block, say
- **one image / note / signal recording**

This choice determines the shape of everything downstream, so it is the first
thing to pin down.

**Index time** — the moment you stand at when you make the prediction. Everything
before it is allowed as input; everything after it is the future. Also called
*prediction time*.

**Observation window** — how far back from the index time you look for input
data. "The last 48 hours of labs" or "all prior visits" are observation windows.

**Prediction gap** (or *lead time*) — the distance between the index time and the
start of the period you are predicting. A gap of zero means you predict what is
happening right now; a gap of 24h means you predict what happens starting a day
from now. Gaps matter clinically: a mortality model with no gap is often just
detecting that a patient is already dying.

**Prediction horizon** — how far forward the outcome window extends from the end
of the gap. "30-day readmission" is a 30-day horizon.

**Label** — the thing you are predicting, as a number the model can be scored
against. Four shapes:
- **binary** — 0 or 1 (died / did not die)
- **multiclass** — exactly one of N categories (length-of-stay bucket)
- **multilabel** — any subset of N categories at once (which drugs to give)
- **regression** — a continuous number (days of stay)

**Leakage** — when information the model could not actually have at index time
sneaks into its inputs. It produces excellent scores and a useless model. Two
kinds bite constantly in EHR work:
- **Temporal leakage** — a feature recorded after the index time. Discharge
  diagnoses used to predict something about that same admission is the classic
  case: the code was assigned at discharge, not at admission.
- **Patient leakage** — the same patient appearing in both train and test. The
  model memorizes the person, not the pattern. Fixed by splitting *by patient*.

**Same-visit vs. next-visit** — a same-visit setup takes features and label from
the same admission, which is leakage-prone. A next-visit setup takes features
from visit *i* and the label from visit *i+1*, which is safe by construction.
When you are unsure, use next-visit.

**Train / validation / test split** — three disjoint slices. You *fit* on train,
*choose* between options on validation, and *report* on test. The test set gets
touched exactly once, at the very end. Peeking at test while iterating turns your
final number into an optimistic fiction.

**Code system** — a controlled vocabulary for clinical concepts: ICD-9/ICD-10 for
diagnoses and procedures, NDC for drug products, ATC for drug classes, LOINC for
labs, CCS for grouped categories. Raw code systems are huge (~15k ICD diagnosis
codes); mapping to a coarser system (ICD → CCS, ~285 groups) is often the single
highest-value preprocessing step.

**Class imbalance** — when one label is much rarer than the other. In-hospital
mortality is typically 2–10% positive. Accuracy is meaningless here (predict "no"
always and score 95%); use ROC-AUC and PR-AUC instead, and prefer PR-AUC when
positives are very rare.

---

## PyHealth-specific vocabulary

**Event** — one timestamped row from one source table, attached to a patient.

**Patient** — all of one person's events, time-ordered. What a task's `__call__`
receives.

**Dataset** (`BaseDataset`) — the loader that turns raw files into `Patient`
objects, driven by a YAML table config. See
[bring-your-own-data.md](../guides/bring-your-own-data/SKILL.md).

**Task** (`BaseTask`) — the function that turns one `Patient` into zero or more
*sample dicts*. This is where cohort, window, and label live. See
[tasks.md](../guides/define-a-task/SKILL.md).

**Sample dict** — one training example: a flat `dict` with `patient_id`, a visit
or record id, one key per input feature, and one key per label.

**`input_schema` / `output_schema`** — declarations on the task class mapping each
sample-dict key to a *processor* name.

**Processor** — the thing that turns a sample-dict value into a tensor, learning
any vocabulary or statistics it needs from the training data. Picking a processor
is picking a representation. See [processors.md](../guides/choose-processors/SKILL.md).

**`SampleDataset`** — the output of `dataset.set_task(task)`: all sample dicts
plus fitted processors. This is what you split, load, and hand to a model.

**`dev=True`** — a small-subset mode (~1000 patients) for checking that code runs.
Never a source of reportable numbers.

**`cache_dir`** — where PyHealth stores the parsed dataset and fitted task, so the
second run is fast. Always set it explicitly.
