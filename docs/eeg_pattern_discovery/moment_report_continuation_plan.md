# EEGBCI Moment Report Continuation Plan

Date: 2026-07-08
Status: ready for implementation
Branch: eegbci-pattern-discovery

## Context

The previous EEGBCI implementation plan has already been implemented.

Do not restart from scratch.

The existing implementation already added:

- `EEGBCIDataset`
- `EEGMotorImageryEEGBCI`
- `EEGBCIPatternDiscovery`
- EEGBCI dataset/task exports
- EEGBCI docs/API pages
- offline unit tests
- skipped-by-default real-data smoke test
- `examples/eeg/eegbci/eegbci_pattern_discovery.py`
- `examples/eeg/eegbci/README.md`

The problem is not missing core implementation. The problem is output quality.

The current generated artifacts in `outputs/eegbci_pattern_discovery/` are weak:

- Markdown summary reports only counts.
- All inspected windows collapse to `mixed_frequency_profile`.
- All inspected windows have `confidence=low`.
- All inspected windows have `quality_flags=low_confidence`.
- The summary does not compare task labels with inferred frequency-pattern states.
- The repeated row text `"This is exploratory signal metadata..."` is not useful.

## New North Star

Answer this question:

> What is the brain doing in each moment, according to its frequency patterns?

More precisely:

> For each 2-second EEG segment, infer the most likely functional brain-state
> hypothesis from its frequency-band profile, then compare that hypothesis to the
> experimental task label.

This moves the output from:

> "Did the code produce a CSV?"

to:

> "At this moment, does the EEG look idle, motor-engaged, slow-wave dominant,
> artifact-like, mixed, or ambiguous, and does that match the task?"

## Source Documents

- `docs/eeg_pattern_discovery/brainstorm.md`
- `docs/eeg_pattern_discovery/design.md`
- `docs/eeg_pattern_discovery/pattern_analysis_redesign.md`
- `docs/eeg_pattern_discovery/implementation_plan.md`
- local CEO plan:
  `/Users/vihaanagrawal/.gstack/projects/sunlabuiuc-PyHealth/ceo-plans/2026-07-08-eegbci-moment-report.md`

## Scope

Implement only the moment-report upgrade.

Primary files:

- `examples/eeg/eegbci/eegbci_pattern_discovery.py`
- `tests/core/test_eegbci.py`
- `examples/eeg/eegbci/README.md`
- this continuation plan, as progress is made

Avoid touching:

- `pyhealth/datasets/eegbci.py`
- `pyhealth/tasks/eegbci.py`
- dataset/task exports
- API RST docs

Only touch `pyhealth/tasks/eegbci.py` if implementation proves a field must become
part of the reusable task API. Current decision: it should not.

## Explicit Non-Goals

Do not implement:

- the full static brain-state atlas
- HTML report
- pretrained embedding comparison
- clustering or motif discovery
- subject-shift reliability suite
- clinical or cognitive claims
- new package dependencies
- new PyHealth model-training APIs
- a rewrite of the existing EEGBCI dataset/task work

## CEO Review Decisions

Mode: Selective Expansion.

Baseline approach: analysis-grade moment report now, with the future atlas as the
north star.

Accepted additions:

1. Rest-normalized evidence.
2. Parseable quality flags.
3. Representative window cards.
4. Analysis versioning.

Deferred:

- HTML report. It belongs to the later atlas phase after CSV/Markdown prove useful.

Boundary decision:

- Keep moment-report fields example-only.
- Do not add `state_hypothesis`, `evidence_score`, rest-normalized deltas,
  `task_state_relation`, or related fields to `EEGBCIPatternDiscovery` samples in
  this PR.

Reason:

- These fields depend on cross-window context such as rest baselines and task-label
  comparisons. They belong in the artifact generator, not the reusable per-sample
  PyHealth task.

## Product Thesis

The useful artifact is not "a CSV with bandpower columns."

The useful artifact is a moment-by-moment EEG state ledger.

Each 2-second segment should tell a researcher:

- what the subject was instructed to do
- what the EEG frequency profile looked like
- which functional state hypothesis best fits that profile
- how strong or weak the evidence is
- whether the frequency pattern supports, adds detail to, disagrees with, or is
  ambiguous relative to the experimental task label

## Required CSV Fields

Keep existing useful fields:

- `patient_id`
- `record_id`
- `subject_id`
- `run`
- `run_type`
- `trial_id`
- `event_code`
- `task_label`
- `label_family`
- `label`
- `eegbci_label`
- `model_label`
- `start_time`
- `end_time`
- bandpower absolute and relative fields
- `dominant_band`
- `alpha_beta_ratio`
- `theta_beta_ratio`
- `brain_state_hypothesis`
- `confidence`
- `quality_flags`
- `interpretation`

Add:

- `analysis_version`
- `state_hypothesis`
- `state_confidence`
- `evidence_score`
- `evidence_summary`
- `rest_reference_scope`
- `rest_delta_relative_delta`
- `rest_theta_relative_delta`
- `rest_alpha_relative_delta`
- `rest_beta_relative_delta`
- `rest_gamma_relative_delta`
- `task_state_relation`
- `task_state_rationale`
- `task_state_confidence`
- `is_low_confidence`
- `is_possible_artifact`
- `is_mixed_or_ambiguous`

Use:

```python
ANALYSIS_VERSION = "eegbci_pattern_moment_report_v1"
```

## Functional State Vocabulary

Use these report-level state names:

| State | Meaning |
| --- | --- |
| `idle_alpha_profile` | Alpha is elevated and alpha/beta is high enough to look idle-like. |
| `sensorimotor_engagement_profile` | Beta or low-gamma evidence is elevated enough to look motor-engaged. |
| `slow_wave_dominant_pattern` | Delta/theta evidence dominates, without implying cognition or diagnosis. |
| `possible_artifact_profile` | Gamma spike, extreme power, or noisy profile suggests inspection. |
| `mixed_ambiguous_profile` | No state wins cleanly or several weak signals conflict. |

Do not use `slow_wave_drowsy_profile`. It sounds too cognitive/clinical.

## Rest Baseline Rules

Rest-normalized evidence is required.

Compute baselines before applying `--max-windows` whenever possible.

Baseline fallback order:

1. Same subject and same run rest windows.
2. Same subject, all requested runs, rest windows.
3. All requested subjects/runs, rest windows.
4. `unavailable`.

If no baseline exists:

- set `rest_reference_scope = "unavailable"`
- set rest-normalized deltas to `NaN` or blank
- state the limitation in Markdown

Do not silently use a missing baseline.

## Task-State Relation Rules

Use this deterministic first-pass table:

| Task family | State hypothesis | Relation |
| --- | --- | --- |
| rest | `idle_alpha_profile` | `supports_label` |
| rest | `mixed_ambiguous_profile` | `ambiguous` |
| rest | `possible_artifact_profile` | `not_applicable` |
| motor execution | `sensorimotor_engagement_profile` | `supports_label` |
| motor imagery | `sensorimotor_engagement_profile` | `adds_detail` |
| any motor task | `idle_alpha_profile` | `disagrees` |
| any task | `slow_wave_dominant_pattern` | `adds_detail` |
| any task | `possible_artifact_profile` | `not_applicable` |
| any task | `mixed_ambiguous_profile` | `ambiguous` |

Add one sentence of rationale in `task_state_rationale`.

## Representative Window Cards

Markdown summary must include deterministic representative windows.

Cards to include when present:

- strongest idle-like window
- strongest motor-engaged window
- strongest slow-wave dominant window
- strongest artifact-like window
- most ambiguous window
- strongest task/state disagreement

Selection rules:

1. Pick highest `evidence_score` for each state class.
2. Tie-break by higher `state_confidence`.
3. Tie-break by earliest `subject_id`, then `run`, then `start_time`.
4. If a state class is absent, omit it and list it as absent.
5. Add one disagreement card using the highest-evidence row where
   `task_state_relation == "disagrees"`.

Each card should include:

- subject
- run
- trial id
- task label
- time range
- state
- evidence score
- dominant band
- relative band values
- rest-normalized deltas
- task-state relation
- confidence
- quality flags
- one-line rationale

## Markdown Summary Contract

The summary must not start with the generic exploratory caveat.

Required sections:

1. Executive result.
2. Run configuration.
3. Window coverage.
4. Moment-state summary.
5. Task label x state matrix.
6. Rest-normalized bandpower summary.
7. Confidence and quality audit.
8. Representative windows.
9. Limitations.
10. Next checks.

The non-clinical warning should live in `Limitations`, for example:

> These labels are signal-pattern summaries from short EEG windows. They are not
> clinical findings and should not be read as evidence of a subject's cognition.

The summary must explicitly say when:

- every window is low confidence
- every window maps to the same state
- no rest baseline is available
- output was capped by `--max-windows`

## Implementation Shape

Keep helper functions pure and testable.

Recommended shape inside `examples/eeg/eegbci/eegbci_pattern_discovery.py`:

```python
ANALYSIS_VERSION = "eegbci_pattern_moment_report_v1"

def build_rest_baselines(rows): ...
def annotate_moment_rows(rows, baselines): ...
def derive_state_hypothesis(row): ...
def derive_quality_columns(row): ...
def task_state_relation(row): ...
def select_representative_windows(rows): ...
def render_summary(rows, config): ...
```

Avoid turning `write_summary()` into a giant function.

Do not create new modules unless this file becomes genuinely hard to read.

## Test Plan

Add focused tests around report helpers using synthetic rows.

Required tests:

- rest baseline fallback:
  - same-run rest
  - same-subject all-run rest
  - global rest
  - unavailable baseline
- parseable quality flags:
  - booleans match `quality_flags`
  - ambiguous state sets `is_mixed_or_ambiguous`
- task-state comparison:
  - deterministic relation table
  - rationale is non-empty
- representative windows:
  - deterministic selection
  - tie-breaks are stable
  - absent state classes are handled
- analysis version:
  - appears in every CSV row
  - appears near the top of Markdown
- interpretation language:
  - row-level interpretation does not contain
    `"This is exploratory signal metadata"`
- empty/edge outputs:
  - no rest windows
  - all low-confidence rows
  - all same-state rows
  - `--max-windows=0`

Keep existing tests for dataset/task behavior.

## Verification Commands

Use the project venv. Plain `python` may not exist in this workspace.

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py -v

.venv/bin/python examples/eeg/eegbci/eegbci_pattern_discovery.py \
  --subjects 1 \
  --runs 3 \
  --max-windows 20 \
  --download
```

Then validate:

- CSV contains all required moment-report columns.
- Markdown contains all required sections.
- Markdown does not start with the generic exploratory caveat.
- Markdown includes representative windows.
- Markdown explains all-low-confidence or all-ambiguous outcomes when they happen.

After code changes:

```bash
graphify update .
```

## Progress Log

- 2026-07-08: Created continuation plan after `/office-hours` and
  `/plan-ceo-review`. This plan explicitly continues from the already-implemented
  EEGBCI dataset/task/example work and scopes only the moment-report upgrade.
- 2026-07-08: Refined and challenged the continuation plan through
  `superpowers:brainstorming`. Wrote the approved design to
  `docs/eeg_pattern_discovery/moment_report_refined_design.md`.
- 2026-07-08: Converted the approved refined design into
  `docs/eeg_pattern_discovery/moment_report_implementation_plan.md` using
  `superpowers:writing-plans`. Ran GStack `/plan-eng-review` against that plan
  before code implementation; the review approved the example-owned architecture
  and tightened empty CSV handling with stable `OUTPUT_COLUMNS`.
- 2026-07-08: Added an extensive correctness test matrix to the implementation
  plan, covering rest fallback, state scoring, task-state relation, quality
  booleans, representative windows, Markdown rendering, empty CSV behavior,
  `--max-windows=0`, uncapped baselines, and end-to-end artifact checks.
- 2026-07-08: Added a post-implementation review gate to the implementation
  plan. The recommended workflow is same-chat orchestration, independent
  sub-agent or fresh review context for GStack `/review`, Markdown review output
  at `docs/eeg_pattern_discovery/moment_report_review.md`, then main-chat fix
  implementation and re-verification.
- 2026-07-08: Tightened the post-implementation review gate: final GStack
  `/review` must run in an independent sub-agent or separate session, not inline
  in the same reasoning thread that implemented the feature.
- 2026-07-08: Added an autonomous execution contract to the implementation plan.
  It now explicitly requires Tasks 1-10, focused and full verification, artifact
  checks, `graphify update .`, independent GStack `/review`, review Markdown,
  accepted-fix implementation, and post-fix re-verification before completion.
- 2026-07-08: Started moment-report implementation. Task 1 is complete:
  added the synthetic moment-row fixture test, added `ANALYSIS_VERSION`,
  `REPORT_BANDS`, and `STATE_CONFIDENCE_RANK`, confirmed the focused test failed
  before implementation and passed after implementation.
- 2026-07-08: Task 2 is complete: added synthetic rest-baseline tests, confirmed
  the missing-helper failure, implemented `build_rest_baselines()` and
  `_mean_band_values()`, and verified rest-only averaging plus no-rest fallback.
- 2026-07-08: Task 3 is complete for the planned partial checkpoint: added rest
  fallback and state-profile tests, confirmed missing-helper failures,
  implemented `_baseline_for_row()`, `_clip01()`, and `derive_state_hypothesis()`,
  and verified the state-profile focused test. The fallback-scope test remains
  expected to pass in Task 5 when `annotate_moment_rows()` is added.
- 2026-07-08: Task 4 is complete: added deterministic task/state relation tests
  and parseable quality-boolean tests, confirmed missing-helper failures,
  implemented `derive_task_state_relation()` and `derive_quality_columns()`, and
  verified the focused relation/quality tests.
- 2026-07-08: Task 5 is complete: added annotation/schema tests plus
  band-specific delta, legacy-field preservation, and non-mutation checks;
  confirmed missing schema/annotator failures; implemented `BASE_OUTPUT_COLUMNS`,
  `MOMENT_REPORT_COLUMNS`, `OUTPUT_COLUMNS`, and `annotate_moment_rows()`; and
  verified the annotation/fallback and band-delta focused tests.
- 2026-07-08: Task 6 is complete: added deterministic representative-window
  tests plus ambiguous and disagreement edge checks, confirmed the missing
  selector failure, implemented stable selection helpers, and verified the
  representative-window focused tests.
- 2026-07-08: Task 7 is complete: added Markdown renderer tests covering required
  sections, empty output, all-low-confidence/all-same-state messaging, task/state
  matrix output, representative card details, limitations placement, and removal
  of the old row-level caveat; confirmed missing-renderer failures; implemented
  `render_summary()` and config-aware `write_summary()`; and verified the focused
  renderer tests plus the caveat regression.
- 2026-07-08: Task 8 is complete: added schema, empty CSV, patched-main
  `--max-windows=0`, uncapped-baseline, analysis-version CSV, and invalid parser
  tests; confirmed the old main-flow failures; updated `main()` to annotate all
  rows before capping and to write stable `OUTPUT_COLUMNS`; verified the full
  helper suite and full `tests/core/test_eegbci.py` file.
