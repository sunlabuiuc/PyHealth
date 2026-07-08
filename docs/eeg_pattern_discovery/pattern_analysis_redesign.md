# EEGBCI Pattern Discovery Output Redesign

Date: 2026-07-08
Status: draft office-hours design note
Branch: eegbci-pattern-discovery
Mode: Builder / open source research

## Problem Statement

The EEGBCI pattern-discovery implementation exists, but the generated artifact does
not yet answer the research question it was designed to answer.

The stronger question is:

> What is the brain doing in each moment, according to its frequency patterns?

More precisely:

> For each 2-second EEG segment, infer the most likely functional brain-state
> hypothesis from its frequency-band profile, then compare that hypothesis to the
> experimental task label.

That moves the project from "does the label match the data?" to:

> At this moment, does the EEG look relaxed, engaged, drowsy, motor-active,
> noisy, mixed, or ambiguous?

The current output at `outputs/eegbci_pattern_discovery/` is mechanically valid and
analytically weak:

- `eegbci_pattern_summary.md` reports only counts.
- All 20 inspected windows are `mixed_frequency_profile`.
- All 20 inspected windows have `confidence=low`.
- All 20 inspected windows have `quality_flags=low_confidence`.
- The Markdown summary does not compare task labels with frequency profiles.
- The interpretation sentence repeats the same caveat per row, which makes the
  artifact feel defensive instead of informative.

This is not a copy problem only. The current artifact does not expose enough
analysis for a researcher to tell whether the pattern-discovery layer found
anything, failed to find anything, or needs better thresholds.

## Evidence From Current Outputs

Source files inspected:

- `outputs/eegbci_pattern_discovery/eegbci_pattern_summary.md`
- `outputs/eegbci_pattern_discovery/eegbci_pattern_windows.csv`
- `examples/eeg/eegbci/eegbci_pattern_discovery.py`
- `pyhealth/tasks/eegbci.py`
- `docs/eeg_pattern_discovery/brainstorm.md`
- `docs/eeg_pattern_discovery/design.md`
- `docs/eeg_pattern_discovery/implementation_plan.md`

Observed current sample:

| Metric | Value |
| --- | --- |
| Windows | 20 |
| Subjects / runs | S001 / R03 only in current inspected output |
| Task labels | rest: 10, execute_right_fist: 6, execute_left_fist: 4 |
| Hypotheses | mixed_frequency_profile: 20 |
| Confidence | low: 20 |
| Quality flags | low_confidence: 20 |

Current task-label medians:

| Task label | Delta rel. | Theta rel. | Alpha rel. | Beta rel. | Gamma rel. | Alpha/beta | Theta/beta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| execute_left_fist | 0.585 | 0.159 | 0.103 | 0.116 | 0.036 | 0.857 | 1.485 |
| execute_right_fist | 0.629 | 0.144 | 0.078 | 0.122 | 0.030 | 0.638 | 1.195 |
| rest | 0.559 | 0.157 | 0.083 | 0.123 | 0.027 | 0.692 | 0.929 |

The data does contain variation. The output just fails to turn it into a useful
analysis. For example, `execute_left_fist` has higher median theta/beta than rest,
and the top alpha/beta window is a rest window. Those are the kinds of signals the
summary should surface.

## Premises

1. The output artifact should answer the analysis question, not merely prove the
   pipeline runs.
2. A weak result is acceptable if the artifact explains why it is weak using data.
3. The non-clinical warning belongs in one clear methods/limitations section, not
   repeated inside every window interpretation.
4. `mixed_frequency_profile` should be treated as an analysis outcome that needs
   explanation, not as useful interpretation by itself.
5. The first pass can remain rules-only. It does not need pretrained embeddings to
   produce a much stronger report.
6. The artifact should be organized around moment-level state hypotheses, not
   around defensive caveats.

## Product Thesis

The useful artifact is not "a CSV with bandpower columns." The useful artifact is a
moment-by-moment brain-state ledger.

Each 2-second segment should tell a researcher:

- What the subject was instructed to do.
- What the EEG frequency profile looked like.
- Which functional state hypothesis best fits that profile.
- How strong or weak the evidence is.
- Whether the frequency pattern agrees with the experimental label, adds detail
  the label does not contain, or looks ambiguous/noisy.

This should feel like a field guide for EEG moments:

| Experimental label | Frequency-pattern question |
| --- | --- |
| rest | Does this window look idle/alpha-heavy, drowsy/slow-wave, noisy, or mixed? |
| motor execution | Does this window show motor-active beta/sensorimotor evidence, or does it still look idle/mixed? |
| motor imagery | Does this window show engagement without movement, or does it collapse into ambiguous/rest-like activity? |
| any label | Is this window clean enough to interpret, or should it be treated as noise/low confidence? |

The current artifact cannot do this because it collapses all windows to the same
state and gives no ranked evidence. A better artifact can still say "ambiguous,"
but it must say why.

## What Strong Output Should Look Like

The Markdown summary should read like a compact research result:

1. Executive result.
   - What was processed.
   - Whether the run produced separable moment-level state hypotheses.
   - Whether confidence was useful or collapsed.
   - The strongest observed signal in one or two sentences.

2. Dataset/run coverage.
   - Subjects, runs, run families, task labels, windows per label.
   - Window size, sample rate, channel mode, preprocessing choices.
   - A warning if the output is capped by `--max-windows`.

3. Moment-state ledger.
   - One row per representative or notable window.
   - Experimental label, inferred state, evidence score, confidence, and note.
   - This is the human-readable answer to "what is the brain doing right now?"

4. Label vs. hypothesis matrix.
   - Crosstab of `task_label` by `brain_state_hypothesis`.
   - Percentages by task label.
   - Explicit statement when all labels collapse to one hypothesis.

5. Bandpower profile by task label.
   - Median relative delta/theta/alpha/beta/gamma by task label.
   - Median alpha/beta and theta/beta ratios.
   - Simple deltas from rest, for example execution beta minus rest beta.

6. Confidence and quality audit.
   - Confidence distribution.
   - Quality flag distribution.
   - Explanation of why low confidence fired.
   - A threshold diagnostic: which rule each window almost matched, if any.

7. Notable windows.
   - Top alpha/beta windows.
   - Top theta/beta windows.
   - Top beta-relative windows.
   - Highest gamma-relative windows as possible artifact candidates.
   - Include trial id, task label, time range, ratios, and a short reason.

8. Interpretation.
   - Replace repeated generic row text with concise, evidence-specific language.
   - Example: `Delta-heavy mixed profile; no rule-specific hypothesis met. Keep as
     low-confidence baseline evidence, not a brain-state claim.`
   - For the summary, state what the artifact can and cannot conclude.

9. Next analysis recommendations.
   - Whether to run more subjects/runs.
   - Whether thresholds are too strict for EEGBCI after normalization.
   - Whether a rest-normalized or subject-normalized report should be generated.

## Functional State Vocabulary

The current vocabulary has the right caution but too little information. Use a
small set of functional hypotheses that describe what the frequency profile looks
like, not what the person definitely experienced:

| State hypothesis | Frequency evidence | Confidence rule | Good wording |
| --- | --- | --- | --- |
| `idle_alpha_profile` | Alpha is elevated and alpha/beta is high | Medium when alpha clearly dominates; low when only mildly elevated | `Idle-like alpha profile` |
| `sensorimotor_engagement_profile` | Beta or low-gamma is elevated without artifact flags | Medium when beta is meaningfully above rest/task baseline | `Motor-engaged frequency profile` |
| `slow_wave_drowsy_profile` | Theta or delta/theta dominates and theta/beta is high | Medium only when slow-wave power is not universal across all labels | `Slow-wave/drowsy-like profile` |
| `possible_artifact_profile` | Gamma spike, extreme power, or noisy band mix | Low until inspected | `Possible artifact or muscle activity` |
| `mixed_ambiguous_profile` | No rule wins, or several weak rules conflict | Low | `Mixed or ambiguous frequency profile` |

This vocabulary is intentionally not clinical. It is still much better than
`mixed_frequency_profile` for every row because it gives the reader a mental model
for what the algorithm is trying to see.

## Evidence Scoring Contract

Each row should expose why the state was chosen. At minimum:

- `state_hypothesis`: one of the functional state names above.
- `state_confidence`: `low`, `medium`, or `high`.
- `evidence_score`: numeric 0.0 to 1.0, even if simple at first.
- `evidence_summary`: compact text, for example
  `delta_rel=0.66; alpha_beta=2.35; beta_rel=0.08`.
- `agreement_with_task`: `supports_label`, `adds_detail`, `disagrees`,
  `ambiguous`, or `not_applicable`.
- `task_comparison_note`: one sentence comparing the inferred state with the
  experimental label.

The first implementation can compute `evidence_score` from heuristic margins:

- Alpha score: alpha relative share plus alpha/beta margin.
- Motor score: beta relative share plus beta-above-rest margin when rest baseline
  is available.
- Slow-wave score: theta/beta and delta/theta dominance.
- Artifact score: gamma relative share and extreme-power flags.
- Mixed score: inverse of the winning margin.

Do not overfit this yet. The point is to make uncertainty inspectable.

## Interpretation Language Contract

Remove this repeated sentence from per-window output:

> This is exploratory signal metadata, not evidence of cognition or a clinical diagnosis.

Keep the safety boundary, but move it into the summary methods section:

> These labels are signal-pattern summaries from short EEG windows. They are not
> clinical findings and should not be read as evidence of a subject's cognition.

Per-window interpretation should be short and data-specific:

| Case | Better interpretation |
| --- | --- |
| No rule match | `Mixed frequency profile; no band-specific rule met. Low confidence.` |
| Alpha-ish rest | `Alpha/beta is elevated versus beta, consistent with an idle-like profile.` |
| Beta-heavy movement | `Beta-relative power is elevated, consistent with active sensorimotor processing.` |
| Gamma-heavy | `Gamma-relative power is elevated; inspect for muscle or movement artifact.` |
| Delta-heavy | `Delta dominates this short window. Treat as low-specificity unless this repeats across runs.` |

## Approaches Considered

### Approach A: Summary-Only Repair

Summary: Keep the current task schema and interpretation rules. Rewrite only the
example summary generator so it computes stronger aggregate tables and removes the
repeated caveat from Markdown/CSV interpretation text.

Effort: S

Risk: Low

Pros:

- Fastest path to useful outputs.
- Smallest source diff.
- Does not change task behavior or public task schema.

Cons:

- Per-row `confidence` still collapses to low on the inspected run.
- Does not explain why thresholds fail unless extra diagnostics are derived in the
  example.
- Leaves weak interpretation rules in `pyhealth/tasks/eegbci.py`.

Reuses:

- Existing `sample_to_row()`.
- Existing CSV fields.
- Pandas aggregation in `examples/eeg/eegbci/eegbci_pattern_discovery.py`.

### Approach B: Analysis-Grade Example Contract

Summary: Upgrade the example into a real artifact generator. Add moment-state
ledger rows, summary tables, near-miss diagnostics, task-vs-rest comparisons,
notable windows, and cleaner interpretation language while keeping the core task
rules conservative.

Effort: M

Risk: Low to medium

Pros:

- Directly fixes the weak output the user sees.
- Preserves the rules-only first implementation boundary.
- Makes low-confidence collapse informative instead of embarrassing.
- Gives researchers enough evidence to decide the next experiment.

Cons:

- More code in the example script.
- Requires focused tests for report sections and edge cases.
- Still depends on simple heuristic thresholds.

Reuses:

- Existing bandpower columns.
- Current task labels and quality flags.
- The original design's "answer whether hypotheses line up, sharpen, or disagree"
  contract.

### Approach C: Rule Engine Redesign

Summary: Redesign `interpret_band_profile()` so it uses richer confidence scoring,
rest-normalized deltas, and graded hypotheses instead of hard thresholds.

Effort: L

Risk: Medium

Pros:

- Fixes the root cause of all windows becoming low-confidence mixed profiles.
- Produces better per-window interpretation.
- Can make confidence meaningful across subjects and runs.

Cons:

- More likely to overclaim from simple bandpower features.
- Needs real-data validation across multiple subjects/runs.
- Changes behavior in the task API, not just the example artifact.

Reuses:

- Existing bandpower computation.
- Existing tests around interpretation helper, with expanded cases.

### Approach D: Moment Atlas

Summary: Generate a richer static "brain-state atlas" artifact in Markdown plus
CSV. Each state gets representative windows, task-label enrichment, evidence
distributions, and next-inspection recommendations.

Effort: L

Risk: Medium

Pros:

- Best match for the bigger question: what the brain appears to be doing moment by
  moment.
- Produces a compelling research artifact, not just a report.
- Sets up the later embedding/cluster atlas without requiring neural models now.

Cons:

- Bigger than a repair pass.
- Needs careful wording to avoid overclaiming.
- More tests and fixture data needed to keep the report deterministic.

Reuses:

- The "Brain-State Atlas Explorer" and "Label Disagreement Mining" ideas from
  `docs/eeg_pattern_discovery/brainstorm.md`.
- Existing CSV fields, plus evidence scoring and task comparison notes.

## Recommendation

Choose Approach B now, with the north-star question from Approach D.

The current artifact fails at the analysis/reporting layer first. Fixing that layer
turns even an all-low-confidence run into a useful result: "the current thresholds
do not produce separable hypotheses on this run, but these bandpower differences
are visible and these windows are worth inspecting." That is a real research
artifact.

But do not keep Approach B small in spirit. The report should be designed as the
first version of a moment atlas:

- "What does this 2-second segment look like?"
- "How confident are we?"
- "Does that fit the task label?"
- "Which windows deserve human inspection?"

Then Approach C can be justified with evidence instead of taste.

## Success Criteria

The output redesign is successful when:

- The summary no longer starts with the generic exploratory caveat.
- The summary has a one-paragraph result that names the strongest finding and the
  biggest limitation.
- The summary directly answers: "What is the brain doing in each moment, according
  to its frequency patterns?"
- The CSV has moment-state fields: `state_hypothesis`, `evidence_score`,
  `evidence_summary`, `agreement_with_task`, and `task_comparison_note`, or an
  explicitly documented equivalent.
- The summary includes a moment-state ledger for representative or notable
  windows.
- The summary includes label-vs-hypothesis percentages.
- The summary includes bandpower medians by task label.
- The summary includes confidence and quality-flag audits.
- The summary includes notable windows ranked by alpha/beta, theta/beta, beta, and
  gamma.
- The summary explicitly explains an all-low-confidence outcome when it happens.
- Per-window interpretation text is short, evidence-specific, and non-clinical
  without repeating legalistic boilerplate.
- Tests cover the report generator on a tiny synthetic row set that includes mixed,
  medium-confidence, and artifact-like windows.

## Open Questions

- Should the CSV keep `interpretation`, or should row-level explanations move to a
  separate `pattern_note` field while `interpretation` remains backward-compatible?
- Should the example default `--max-windows` remain uncapped in docs, or should the
  README encourage larger multi-run output for meaningful summaries?
- Should the next iteration add `analysis_version` to the CSV/summary so future
  rule changes are traceable?
- Should threshold diagnostics live in the example only, or become fields returned
  by `EEGBCIPatternDiscovery`?

## Next Steps

1. Implement Approach B in `examples/eeg/eegbci/eegbci_pattern_discovery.py`.
2. Update `tests/core/test_eegbci.py` with focused tests for summary sections and
   interpretation text.
3. Regenerate `outputs/eegbci_pattern_discovery/`.
4. Update `docs/eeg_pattern_discovery/implementation_plan.md` with the output
   quality correction and verification evidence.
5. Run the existing EEGBCI unit tests plus the example command.

