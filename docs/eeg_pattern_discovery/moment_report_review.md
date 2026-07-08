# EEGBCI Moment Report Implementation Review

Date: 2026-07-08
Branch: eegbci-pattern-discovery
Review path: GStack /review

## Scope Reviewed

Reviewed the committed branch at `a077111` against `origin/master`, with primary focus on the moment-report pass:

- `examples/eeg/eegbci/eegbci_pattern_discovery.py`
- `tests/core/test_eegbci.py`
- `examples/eeg/eegbci/README.md`
- `docs/eeg_pattern_discovery/moment_report_implementation_plan.md`
- `docs/eeg_pattern_discovery/moment_report_continuation_plan.md`

Also checked the existing generated artifacts under `outputs/eegbci_pattern_discovery/` because Task 10 requires CSV and Markdown contract validation.

Public API boundary check: the report-only fields stay out of `pyhealth/tasks/eegbci.py` and `pyhealth/datasets/eegbci.py`. Only the legacy `brain_state_hypothesis` task field appears in reusable task code.

## Verification Commands

Commands run:

```bash
graphify query "Review EEGBCI moment report implementation files, task contracts, artifact generation, and tests" --budget 2000
git diff --stat origin/master...HEAD
git diff --check origin/master...HEAD
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers -v
.venv/bin/python -m pytest tests/core/test_eegbci.py -v
.venv/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
csv = Path("outputs/eegbci_pattern_discovery/eegbci_pattern_windows.csv")
summary = Path("outputs/eegbci_pattern_discovery/eegbci_pattern_summary.md")
required = {
    "analysis_version", "state_hypothesis", "state_confidence", "evidence_score",
    "evidence_summary", "rest_reference_scope", "rest_alpha_relative_delta",
    "task_state_relation", "task_state_rationale", "task_state_confidence",
    "is_low_confidence", "is_possible_artifact", "is_mixed_or_ambiguous",
}
df = pd.read_csv(csv)
print("rows", len(df))
print("missing", sorted(required - set(df.columns)))
print("analysis_version_all", bool((df["analysis_version"] == "eegbci_pattern_moment_report_v1").all()))
text = summary.read_text(encoding="utf-8")
headings = [
    "Executive Result", "Run Configuration", "Window Coverage", "Moment-State Summary",
    "Task Label x State Matrix", "Rest-Normalized Bandpower Summary",
    "Confidence and Quality Audit", "Representative Windows", "Limitations", "Next Checks",
]
print("missing_headings", [h for h in headings if h not in text])
PY
rg "Executive Result|Run Configuration|Window Coverage|Moment-State Summary|Task Label x State Matrix|Rest-Normalized Bandpower Summary|Confidence and Quality Audit|Representative Windows|Limitations|Next Checks" outputs/eegbci_pattern_discovery/eegbci_pattern_summary.md
```

Results:

- Helper suite: 28 passed, 12 subtests passed.
- Full EEGBCI test file: 52 passed, 1 skipped, 12 subtests passed. The skipped test is the opt-in real-data smoke test.
- `git diff --check`: clean.
- Existing artifact schema check: 20 rows, no missing required moment-report columns, all rows use `eegbci_pattern_moment_report_v1`.
- Existing Markdown artifact includes all required headings and representative windows.

Not run:

- I did not rerun the real-data example command because this reviewer was instructed that the only write target is this review document. The existing generated artifacts were inspected read-only.
- I did not run `graphify update .` because it writes graph files and this reviewer is read-only except for this file.

## Findings

1. [P1] (confidence: 9/10) `examples/eeg/eegbci/eegbci_pattern_discovery.py:237`, `examples/eeg/eegbci/eegbci_pattern_discovery.py:444`, `outputs/eegbci_pattern_discovery/eegbci_pattern_summary.md:45` - The report conflates legacy low-confidence flags with the new moment-report state confidence. In the generated artifact, `state_confidence` is `medium` for 16 of 20 rows, but `is_low_confidence` is `True` for all 20 rows because every row inherits legacy `quality_flags=low_confidence`. The Markdown then says "Every window is low confidence" and reports `Low-confidence rows: 20`, contradicting `State confidence: {'low': 4, 'medium': 16}`. This weakens the main artifact the pass is supposed to improve.

2. [P2] (confidence: 8/10) `docs/eeg_pattern_discovery/moment_report_implementation_plan.md:1406`, `tests/core/test_eegbci.py:556`, `tests/core/test_eegbci.py:658`, `tests/core/test_eegbci.py:1325` - The extensive correctness matrix is not fully satisfied as written. Missing named coverage includes `test_state_confidence_requires_margin`, `test_quality_booleans_do_not_depend_on_string_parsing_only`, and `test_parse_int_list_accepts_ranges_and_singletons`. Existing tests cover broad state detection, string-based quality flags, and invalid parser input, but they do not prove these specific required edge cases.

3. [P2] (confidence: 9/10) `docs/eeg_pattern_discovery/moment_report_implementation_plan.md:1318`, `docs/eeg_pattern_discovery/moment_report_implementation_plan.md:1396`, `docs/eeg_pattern_discovery/moment_report_continuation_plan.md:471` - Task 10 remains unchecked and the continuation progress log stops at Task 9. The implementation contract required Task 10 verification, artifact inspection, and graph update tracking before completion. The artifacts exist and several checks pass, but the plan state does not show that the implementation session completed those required steps.

No SQL, shell-injection, LLM trust-boundary, race-condition, CI/CD, or public API boundary issues found in the moment-report pass.

## Required Fixes

1. Separate legacy quality flags from moment-report confidence in the CSV/Markdown contract. Either make `is_low_confidence` mean `state_confidence == "low"` and add a separate legacy flag column/count, or keep the boolean as a legacy-quality indicator and change the Markdown wording so it does not claim every moment-report state is low confidence when most `state_confidence` values are medium. Add a regression test using rows with `quality_flags="low_confidence"` and `state_confidence="medium"`.

2. Add the missing correctness-matrix tests or update the matrix with explicit rationale for merged coverage. Minimum expected tests: margin lowers `state_confidence`, ambiguous state sets `is_mixed_or_ambiguous` without relying on flag text, and `parse_int_list("1,3-5") == [1, 3, 4, 5]`.

3. Complete Task 10 bookkeeping in the plan documents after the main implementation chat reruns the allowed final verification commands and `graphify update .` after any code changes. Mark the Task 10 checkboxes accurately and append a continuation-plan progress entry.

## Fix Implementation Log

- Accepted and fixed Finding 1. `derive_quality_columns()` now makes
  `is_low_confidence` depend on moment-report `state_confidence == "low"` rather
  than legacy `quality_flags=low_confidence`. This keeps the CSV boolean and
  Markdown low-confidence count consistent with the state-confidence audit.
  Added regression coverage for a medium-confidence state with legacy
  `quality_flags="low_confidence"`.
- Accepted and fixed Finding 2. Added matrix coverage for
  `test_state_confidence_requires_margin`,
  `test_quality_booleans_do_not_depend_on_string_parsing_only`, and
  `test_parse_int_list_accepts_ranges_and_singletons`.
- Accepted and fixed Finding 3. Marked Task 10 checkboxes complete in the
  implementation plan and appended final Task 10/post-review progress to the
  continuation plan.

## Post-Fix Verification

Commands run after fixes:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers -k "state_confidence_requires_margin or quality_booleans_do_not or parse_int_list_accepts" -v
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers -v
.venv/bin/python -m pytest tests/core/test_eegbci.py -v
.venv/bin/python examples/eeg/eegbci/eegbci_pattern_discovery.py \
  --subjects 1 \
  --runs 3 \
  --max-windows 20 \
  --download
.venv/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
df = pd.read_csv("outputs/eegbci_pattern_discovery/eegbci_pattern_windows.csv")
required = {
    "analysis_version", "state_hypothesis", "state_confidence", "evidence_score",
    "evidence_summary", "rest_reference_scope", "rest_alpha_relative_delta",
    "task_state_relation", "task_state_rationale", "task_state_confidence",
    "is_low_confidence", "is_possible_artifact", "is_mixed_or_ambiguous",
}
missing = sorted(required - set(df.columns))
summary = Path("outputs/eegbci_pattern_discovery/eegbci_pattern_summary.md").read_text(encoding="utf-8")
assert len(df) == 20
assert not missing
assert (df["analysis_version"] == "eegbci_pattern_moment_report_v1").all()
assert int(df["is_low_confidence"].sum()) == int((df["state_confidence"] == "low").sum())
assert not summary.splitlines()[2].startswith("Brain-state hypotheses are exploratory signal metadata")
assert summary.count("### ") >= 1
PY
rg "Executive Result|Run Configuration|Window Coverage|Moment-State Summary|Task Label x State Matrix|Rest-Normalized Bandpower Summary|Confidence and Quality Audit|Representative Windows|Limitations|Next Checks" outputs/eegbci_pattern_discovery/eegbci_pattern_summary.md
graphify update .
```

Results:

- Focused review-fix tests: 4 passed.
- Full helper suite: 32 passed, 12 subtests passed.
- Full EEGBCI test file: 56 passed, 1 skipped, 12 subtests passed. The skipped
  test is the opt-in real-data smoke test.
- Real-data example regenerated
  `outputs/eegbci_pattern_discovery/eegbci_pattern_windows.csv` and
  `outputs/eegbci_pattern_discovery/eegbci_pattern_summary.md`.
- Refreshed CSV: 20 rows, no missing required moment-report columns,
  `analysis_version` is correct for every row, `state_confidence` counts are
  `{'medium': 16, 'low': 4}`, and `is_low_confidence` count is 4.
- Refreshed Markdown: all required headings are present, the old generic caveat
  is not the opening body text, and at least one representative window card is
  present.
- `graphify update .` completed after code changes.

## Final Verdict

Approved after fixes. The confidence contradiction is resolved, the missing
correctness-matrix tests were added, Task 10 bookkeeping is complete, artifacts
were regenerated and validated, full EEGBCI tests pass, and graphify was updated.
