# EEGBCI Moment Report Refined Design

Date: 2026-07-08
Status: approved for implementation planning
Scope: EEGBCI pattern-discovery moment-report upgrade

## Purpose

The existing EEGBCI dataset, tasks, tests, docs, and example have already been
implemented. The remaining problem is artifact quality: the generated CSV and
Markdown prove that the pipeline runs, but they do not yet answer the analysis
question.

The upgraded artifact should answer:

> What is the brain doing in each moment, according to its frequency patterns?

More precisely, for each 2-second EEG segment, the report should infer the most
likely frequency-profile state hypothesis, expose the evidence for that
hypothesis, and compare it with the experimental EEGBCI task label.

The design should refine and challenge the continuation plan, not restart the
EEGBCI integration.

## Recommended Approach

Use an example-owned analysis layer with pure helper functions.

Keep `EEGBCIDataset`, `EEGMotorImageryEEGBCI`, and `EEGBCIPatternDiscovery`
stable. Do not add report-only fields to the reusable PyHealth task API unless
implementation proves that a field is intrinsically per-window and reusable
outside the report.

The report flow should be:

```text
EEGBCIDataset
  -> EEGBCIPatternDiscovery
  -> sample_to_row()
  -> build_rest_baselines()
  -> annotate_moment_rows()
  -> write CSV
  -> render_summary()
  -> write Markdown
```

The current public task fields remain useful compatibility data:

- `brain_state_hypothesis`
- `confidence`
- `quality_flags`
- `interpretation`

The upgraded report should primarily use new example-level fields:

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
- parseable quality booleans

## Approaches Considered

### A. Summary-Only Patch

Only rewrite `write_summary()` and leave CSV rows mostly unchanged.

This is small and low-risk, but it does not fix the core defect. Rows can still
collapse into weak `mixed_frequency_profile` metadata with no inspectable
evidence. Reject this as underpowered.

### B. Example-Owned Analysis Helpers

Add pure helpers in `examples/eeg/eegbci/eegbci_pattern_discovery.py` for rest
baselines, state scoring, quality flags, task/state comparison, representative
window selection, and Markdown rendering.

This is the recommended path. It keeps cross-window analysis out of the reusable
task API while making the generated artifact substantially more useful. The main
risk is example-file growth, so helpers must be named, pure, and directly tested.

### C. Promote Moment Fields Into The Task

Add `state_hypothesis`, `evidence_score`, rest deltas, and task/state comparison
to `EEGBCIPatternDiscovery` samples.

Reject this for now. Rest baselines and task/state comparisons depend on
cross-window context. PyHealth tasks currently emit independent samples, so this
would blur the abstraction boundary.

## Architecture

The upgraded moment report is an analysis layer owned by the example. The public
dataset and task API should remain stable.

`--max-windows` should cap the final artifact, but rest baselines should be built
from all available requested rows when feasible. Otherwise a small capped run can
accidentally remove rest evidence and make the analysis look weaker than the
data. If full baseline collection is not feasible, the report must say the
baseline was computed from capped rows.

Keep helper functions pure and testable:

```python
ANALYSIS_VERSION = "eegbci_pattern_moment_report_v1"

def build_rest_baselines(rows): ...
def annotate_moment_rows(rows, baselines): ...
def derive_state_hypothesis(row): ...
def derive_quality_columns(row): ...
def derive_task_state_relation(row): ...
def select_representative_windows(rows): ...
def render_summary(rows, config): ...
```

Do not create a shared module in this phase. If the example later becomes too
large, these helpers can move to an example utility module without changing the
public PyHealth API.

## State Vocabulary

Use frequency-profile names, not cognitive or clinical claims.

| State | Meaning |
| --- | --- |
| `idle_alpha_profile` | Alpha is elevated and alpha/beta is high enough to look idle-like. |
| `sensorimotor_engagement_profile` | Beta or low-gamma evidence is elevated enough to look motor-engaged. |
| `slow_wave_dominant_pattern` | Delta/theta evidence dominates without implying cognition or diagnosis. |
| `possible_artifact_profile` | Gamma spike, extreme power, or noisy profile suggests inspection. |
| `mixed_ambiguous_profile` | No state wins cleanly or several weak signals conflict. |

Do not use `slow_wave_drowsy_profile`; it implies a cognitive state.

## Evidence Scoring

The state scorer should combine:

- relative band shares
- absolute ratios such as alpha/beta and theta/beta
- rest-normalized deltas when available
- quality and artifact checks
- margin between the winning state and alternatives

The output should include:

- `state_hypothesis`
- `state_confidence`
- `evidence_score`
- `evidence_summary`

Confidence should not rise merely because a state wins. It should require a
meaningful margin over alternatives or useful rest-normalized evidence. If all
state scores are weak, choose `mixed_ambiguous_profile` with low confidence and
explain near misses in `evidence_summary`.

## Rest Baselines

Compute rest baselines before applying the final `--max-windows` cap whenever
possible.

Fallback order:

1. Same subject and same run rest windows.
2. Same subject, all requested runs, rest windows.
3. All requested subjects/runs, rest windows.
4. `unavailable`.

Each annotated row must record `rest_reference_scope`. If no baseline exists:

- set `rest_reference_scope = "unavailable"`
- set rest-normalized deltas to blank or NaN
- state the limitation in Markdown

Do not silently substitute missing rest evidence.

## Task-State Relation

Compare the inferred frequency-profile state with the EEGBCI experimental task
label using a deterministic first-pass table.

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

Each row should include a concise `task_state_rationale` and
`task_state_confidence`.

## Markdown Report Contract

The Markdown report should be a compact analysis result, not a run receipt.

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

The executive result must explicitly say when:

- every row is low confidence
- every row maps to the same state
- no rest baseline is available
- output was capped by `--max-windows`
- no windows were produced

Move the non-clinical warning to `Limitations`. Do not repeat it in every row.

## Representative Windows

Select deterministic representative cards when present:

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
4. Omit absent state classes and list them as absent.
5. For the most ambiguous card, pick the `mixed_ambiguous_profile` row with the
   lowest `evidence_score`, then use the same stable tie-breaks.
6. Add one disagreement card using the highest-evidence row where
   `task_state_relation == "disagrees"`.

Cards should show evidence and uncertainty so "strongest" does not imply the
evidence is strong in an absolute sense.

## Edge Cases

Handle these explicitly:

- No rows: write an empty CSV with expected columns and a Markdown report
  explaining no windows were produced.
- `--max-windows=0`: valid command, no final artifact rows, no crash.
- No rest windows: baseline scope is `unavailable`, rest deltas are blank or
  NaN, and the Markdown states the limitation.
- All rows low confidence: state this in the executive result and audit section.
- All rows same state: state that the analysis collapsed and recommend broader
  coverage or threshold review.
- All rows ambiguous: treat as a valid weak result, not a silent failure.
- Missing optional legacy task fields: tolerate them where possible.
- Missing required metadata fields: fail clearly.

## Non-Goals

Do not implement:

- HTML report output
- pretrained embedding comparison
- clustering or motif discovery
- a full static brain-state atlas
- clinical or cognitive claims
- new package dependencies
- changes to dataset/task exports or API RST pages
- a rewrite of the existing EEGBCI dataset/task work

## Test Strategy

Use synthetic rows for report-helper tests. Do not download EEGBCI data in normal
tests.

Required test areas:

- rest baseline fallback: same-run, same-subject, global, unavailable
- state scoring: alpha-like, beta/motor-like, slow-wave, artifact-like,
  ambiguous
- confidence: weak winner remains low confidence; strong margin can become
  medium
- task-state relation: deterministic relation and non-empty rationale
- quality booleans: `is_low_confidence`, `is_possible_artifact`, and
  `is_mixed_or_ambiguous`
- representative windows: deterministic selection and stable tie-breaks
- report rendering: required sections and limitations warning
- CSV schema: `analysis_version` and moment-report fields in every row
- empty and capped outputs: clear Markdown and no crash

Threshold tests should use obvious synthetic examples rather than overfitting to
exact real EEGBCI values.

## Documentation Updates

Update:

- `examples/eeg/eegbci/README.md`
- `docs/eeg_pattern_discovery/moment_report_continuation_plan.md`

The README should describe the upgraded CSV and Markdown report fields. The
continuation plan should record implementation progress as work proceeds.

## Approval

This refined design was reviewed section by section during brainstorming and
approved for implementation planning on 2026-07-08.
