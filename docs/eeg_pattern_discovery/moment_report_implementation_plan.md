# EEGBCI Moment Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the EEGBCI example artifact from a run receipt into a moment-by-moment frequency-pattern report with rest-normalized evidence, state hypotheses, task-state comparison, representative windows, and explicit limitations.

**Architecture:** Keep `EEGBCIDataset`, `EEGMotorImageryEEGBCI`, and `EEGBCIPatternDiscovery` stable. Add an example-owned analysis layer in `examples/eeg/eegbci/eegbci_pattern_discovery.py` that converts per-window task samples into rows, computes rest baselines across the requested rows, annotates rows, writes CSV, and renders a Markdown analysis report.

**Tech Stack:** Python, pandas, standard library `collections.Counter`, existing PyHealth EEGBCI dataset/task APIs, `unittest` tests in `tests/core/test_eegbci.py`.

## Global Constraints

- Do not add new package dependencies.
- Do not change dataset/task exports or API RST pages for this moment-report pass.
- Keep report-only fields example-level unless implementation proves they are intrinsically per-window and reusable outside the report.
- Do not use clinical or cognitive claims.
- Use `ANALYSIS_VERSION = "eegbci_pattern_moment_report_v1"`.
- Normal tests must use synthetic rows and must not download EEGBCI data.
- `--max-windows` caps the final artifact rows, not the baseline source rows.
- After code changes, run `graphify update .`.

---

## File Structure

- Modify `examples/eeg/eegbci/eegbci_pattern_discovery.py`: add pure report helper functions, update `main()` data flow, write enriched CSV, render Markdown report.
- Modify `tests/core/test_eegbci.py`: add tests for report helpers using synthetic rows.
- Modify `examples/eeg/eegbci/README.md`: document the upgraded CSV and Markdown report.
- Modify `docs/eeg_pattern_discovery/moment_report_continuation_plan.md`: keep the progress log current.

Do not modify:

- `pyhealth/datasets/eegbci.py`
- `pyhealth/tasks/eegbci.py`
- `pyhealth/datasets/__init__.py`
- `pyhealth/tasks/__init__.py`
- `docs/api/datasets/pyhealth.datasets.EEGBCIDataset.rst`
- `docs/api/tasks/pyhealth.tasks.eegbci.rst`
- `docs/api/datasets.rst`
- `docs/api/tasks.rst`

## Data Flow

```text
EEGBCIDataset
  -> dataset.set_task(EEGBCIPatternDiscovery(compute_stft=False))
  -> collect all requested sample_to_row(sample) rows
  -> build_rest_baselines(all_rows)
  -> annotate_moment_rows(all_rows, baselines)
  -> apply --max-windows cap to annotated rows
  -> write eegbci_pattern_windows.csv
  -> render_summary(capped_rows, config)
  -> write eegbci_pattern_summary.md
```

## Autonomous Execution Contract

This plan is intended to be executed end to end by an implementation agent.
It is not a shell script, but it is a complete execution contract.

Required execution flow:

```text
1. Execute Tasks 1-10 in order.
2. For each task:
   - write the failing tests first
   - run the named focused test command and confirm failure
   - implement the minimal code/doc change
   - run the named focused test command and confirm pass
   - update `docs/eeg_pattern_discovery/moment_report_continuation_plan.md`
     when progress meaningfully changes
3. After Task 10:
   - run `.venv/bin/python -m pytest tests/core/test_eegbci.py -v`
   - run the real-data example command when network/data access is available
   - validate generated CSV and Markdown artifacts
   - run `graphify update .`
4. Dispatch an independent sub-agent or separate session to run GStack `/review`.
5. Require the independent reviewer to write
   `docs/eeg_pattern_discovery/moment_report_review.md`.
6. Main implementation chat reads the review document, applies accepted fixes,
   and records deferred or rejected findings with rationale.
7. Rerun focused tests for every fix, then full EEGBCI tests and artifact checks.
8. Update `moment_report_review.md` with fix log, post-fix verification, and
   final verdict.
9. Report completion only after code, docs, tests, artifacts, graph update, and
   review fixes are complete.
```

Use `superpowers:subagent-driven-development` when available for Task 1-10
implementation. Use one independent sub-agent per task or small task group when
the task can be reviewed independently. Keep the main chat responsible for
reviewing sub-agent output, applying final patches, and running verification.

Do not stop for user input unless:

- a required dependency or data source is unavailable
- tests fail in a way that contradicts the approved design
- the independent review finds a scope change that would modify the public
  PyHealth dataset/task APIs
- a destructive or externally visible action would be required

## Task 1: Report Constants And Synthetic Test Fixture

**Files:**
- Modify: `examples/eeg/eegbci/eegbci_pattern_discovery.py`
- Modify: `tests/core/test_eegbci.py`

**Interfaces:**
- Produces: `ANALYSIS_VERSION: str`
- Produces: `REPORT_BANDS: tuple[str, ...]`
- Produces: test helper `_moment_row(**overrides) -> dict`

- [x] **Step 1: Write failing import and fixture test**

Add this test class near the end of `tests/core/test_eegbci.py`, before `TestEEGBCIRealDataSmoke`:

```python
class TestEEGBCIMomentReportHelpers(unittest.TestCase):
    def _moment_row(self, **overrides):
        row = {
            "patient_id": "S001",
            "record_id": "R03",
            "subject_id": 1,
            "run": 3,
            "run_type": "motor_execution_left_right",
            "trial_id": "S001_R03_T0_0",
            "event_code": "T0",
            "task_label": "rest",
            "label_family": "rest",
            "label": 0,
            "eegbci_label": 0,
            "model_label": 0,
            "start_time": 0.0,
            "end_time": 2.0,
            "dominant_band": "alpha",
            "delta_relative": 0.05,
            "theta_relative": 0.10,
            "alpha_relative": 0.55,
            "beta_relative": 0.20,
            "gamma_relative": 0.10,
            "alpha_beta_ratio": 2.75,
            "theta_beta_ratio": 0.50,
            "brain_state_hypothesis": "relaxed_or_idle",
            "confidence": "medium",
            "quality_flags": "",
            "interpretation": "Alpha-dominant profile.",
        }
        row.update(overrides)
        return row

    def test_analysis_version_constant(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import ANALYSIS_VERSION

        self.assertEqual(ANALYSIS_VERSION, "eegbci_pattern_moment_report_v1")
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers::test_analysis_version_constant -v
```

Expected: fail with `ImportError` or `AttributeError` for missing `ANALYSIS_VERSION`.

- [x] **Step 3: Add constants**

Add below the PyHealth imports in `examples/eeg/eegbci/eegbci_pattern_discovery.py`:

```python
ANALYSIS_VERSION = "eegbci_pattern_moment_report_v1"
REPORT_BANDS = ("delta", "theta", "alpha", "beta", "gamma")
STATE_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
```

- [x] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers::test_analysis_version_constant -v
```

Expected: pass.

## Task 2: Rest Baseline Builder

**Files:**
- Modify: `examples/eeg/eegbci/eegbci_pattern_discovery.py`
- Modify: `tests/core/test_eegbci.py`

**Interfaces:**
- Consumes: row dicts with `subject_id`, `run`, `task_label`, and `{band}_relative`
- Produces: `build_rest_baselines(rows: list[dict]) -> dict`

The returned dict must include:

```python
{
    "same_subject_run": {(1, 3): {"delta_relative": 0.05, ...}},
    "same_subject_all_runs": {1: {"delta_relative": 0.06, ...}},
    "global_rest": {"delta_relative": 0.07, ...},
}
```

- [x] **Step 1: Write failing baseline tests**

Add these tests to `TestEEGBCIMomentReportHelpers`:

```python
    def test_build_rest_baselines_uses_rest_rows_only(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import build_rest_baselines

        rows = [
            self._moment_row(task_label="rest", subject_id=1, run=3, alpha_relative=0.50),
            self._moment_row(task_label="execute_left_fist", subject_id=1, run=3, alpha_relative=0.90),
            self._moment_row(task_label="rest", subject_id=1, run=4, alpha_relative=0.70),
        ]

        baselines = build_rest_baselines(rows)

        self.assertAlmostEqual(
            baselines["same_subject_run"][(1, 3)]["alpha_relative"], 0.50
        )
        self.assertAlmostEqual(
            baselines["same_subject_all_runs"][1]["alpha_relative"], 0.60
        )
        self.assertAlmostEqual(baselines["global_rest"]["alpha_relative"], 0.60)

    def test_build_rest_baselines_handles_no_rest_rows(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import build_rest_baselines

        rows = [self._moment_row(task_label="execute_left_fist", label_family="motor_execution")]

        baselines = build_rest_baselines(rows)

        self.assertEqual(baselines["same_subject_run"], {})
        self.assertEqual(baselines["same_subject_all_runs"], {})
        self.assertIsNone(baselines["global_rest"])
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest \
  tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers::test_build_rest_baselines_uses_rest_rows_only \
  tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers::test_build_rest_baselines_handles_no_rest_rows \
  -v
```

Expected: fail because `build_rest_baselines` is missing.

- [x] **Step 3: Implement baseline helpers**

Add below `sample_to_row()`:

```python
def _mean_band_values(rows: list[dict]) -> dict:
    means = {}
    for band in REPORT_BANDS:
        key = f"{band}_relative"
        values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
        if values:
            means[key] = sum(values) / len(values)
    return means


def build_rest_baselines(rows: list[dict]) -> dict:
    rest_rows = [row for row in rows if row.get("task_label") == "rest"]
    same_subject_run = {}
    same_subject_all_runs = {}

    subject_run_keys = sorted({(row["subject_id"], row["run"]) for row in rest_rows})
    for key in subject_run_keys:
        subject_id, run = key
        grouped = [
            row for row in rest_rows
            if row["subject_id"] == subject_id and row["run"] == run
        ]
        same_subject_run[key] = _mean_band_values(grouped)

    subject_keys = sorted({row["subject_id"] for row in rest_rows})
    for subject_id in subject_keys:
        grouped = [row for row in rest_rows if row["subject_id"] == subject_id]
        same_subject_all_runs[subject_id] = _mean_band_values(grouped)

    return {
        "same_subject_run": same_subject_run,
        "same_subject_all_runs": same_subject_all_runs,
        "global_rest": _mean_band_values(rest_rows) if rest_rows else None,
    }
```

- [x] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers -k "baseline or analysis_version" -v
```

Expected: pass.

## Task 3: Rest Fallback And State Scoring

**Files:**
- Modify: `examples/eeg/eegbci/eegbci_pattern_discovery.py`
- Modify: `tests/core/test_eegbci.py`

**Interfaces:**
- Consumes: `build_rest_baselines()` output
- Produces: `_baseline_for_row(row: dict, baselines: dict) -> tuple[str, dict | None]`
- Produces: `derive_state_hypothesis(row: dict) -> dict`

`derive_state_hypothesis()` returns:

```python
{
    "state_hypothesis": "idle_alpha_profile",
    "state_confidence": "medium",
    "evidence_score": 0.72,
    "evidence_summary": "alpha=0.55; beta=0.20; gamma=0.10; alpha_beta=2.75; margin=0.21",
}
```

- [x] **Step 1: Write failing rest fallback and state tests**

Add these tests:

```python
    def test_annotate_rest_fallback_scopes(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            annotate_moment_rows,
            build_rest_baselines,
        )

        rows = [
            self._moment_row(task_label="rest", subject_id=1, run=3, alpha_relative=0.50),
            self._moment_row(task_label="rest", subject_id=1, run=4, alpha_relative=0.70),
            self._moment_row(task_label="execute_left_fist", label_family="motor_execution", subject_id=1, run=3, alpha_relative=0.80),
            self._moment_row(task_label="execute_left_fist", label_family="motor_execution", subject_id=1, run=5, alpha_relative=0.80),
            self._moment_row(task_label="execute_left_fist", label_family="motor_execution", subject_id=2, run=8, alpha_relative=0.80),
        ]

        annotated = annotate_moment_rows(rows, build_rest_baselines(rows))

        self.assertEqual(annotated[2]["rest_reference_scope"], "same_subject_run")
        self.assertEqual(annotated[3]["rest_reference_scope"], "same_subject_all_runs")
        self.assertEqual(annotated[4]["rest_reference_scope"], "global_rest")

    def test_derive_state_hypothesis_detects_profiles(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import derive_state_hypothesis

        cases = [
            (self._moment_row(alpha_relative=0.60, beta_relative=0.12, gamma_relative=0.05, alpha_beta_ratio=5.0), "idle_alpha_profile"),
            (self._moment_row(alpha_relative=0.12, beta_relative=0.48, gamma_relative=0.16, alpha_beta_ratio=0.25), "sensorimotor_engagement_profile"),
            (self._moment_row(delta_relative=0.42, theta_relative=0.36, alpha_relative=0.08, beta_relative=0.08), "slow_wave_dominant_pattern"),
            (self._moment_row(gamma_relative=0.48, alpha_relative=0.10, beta_relative=0.12), "possible_artifact_profile"),
            (self._moment_row(delta_relative=0.18, theta_relative=0.20, alpha_relative=0.22, beta_relative=0.21, gamma_relative=0.19, alpha_beta_ratio=1.05), "mixed_ambiguous_profile"),
        ]

        for row, expected in cases:
            with self.subTest(expected=expected):
                result = derive_state_hypothesis(row)
                self.assertEqual(result["state_hypothesis"], expected)
                self.assertIn(result["state_confidence"], {"low", "medium", "high"})
                self.assertGreaterEqual(result["evidence_score"], 0.0)
                self.assertLessEqual(result["evidence_score"], 1.0)
                self.assertIn("alpha=", result["evidence_summary"])
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers -k "fallback or profiles" -v
```

Expected: fail because `annotate_moment_rows` and `derive_state_hypothesis` are missing.

- [x] **Step 3: Implement fallback and scoring**

Add below `build_rest_baselines()`:

```python
def _baseline_for_row(row: dict, baselines: dict) -> tuple[str, dict | None]:
    subject_run_key = (row["subject_id"], row["run"])
    if subject_run_key in baselines["same_subject_run"]:
        return "same_subject_run", baselines["same_subject_run"][subject_run_key]
    if row["subject_id"] in baselines["same_subject_all_runs"]:
        return "same_subject_all_runs", baselines["same_subject_all_runs"][row["subject_id"]]
    if baselines["global_rest"]:
        return "global_rest", baselines["global_rest"]
    return "unavailable", None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def derive_state_hypothesis(row: dict) -> dict:
    delta = float(row.get("delta_relative", 0.0) or 0.0)
    theta = float(row.get("theta_relative", 0.0) or 0.0)
    alpha = float(row.get("alpha_relative", 0.0) or 0.0)
    beta = float(row.get("beta_relative", 0.0) or 0.0)
    gamma = float(row.get("gamma_relative", 0.0) or 0.0)
    alpha_beta = float(row.get("alpha_beta_ratio", 0.0) or 0.0)
    theta_beta = float(row.get("theta_beta_ratio", 0.0) or 0.0)

    scores = {
        "idle_alpha_profile": _clip01((alpha - 0.25) + min(alpha_beta / 8.0, 0.40)),
        "sensorimotor_engagement_profile": _clip01((beta - 0.20) + max(gamma - 0.12, 0.0) + max(0.0, 1.5 - alpha_beta) / 6.0),
        "slow_wave_dominant_pattern": _clip01((delta + theta) - 0.45 + min(theta_beta / 8.0, 0.20)),
        "possible_artifact_profile": _clip01((gamma - 0.22) * 2.0 + max(delta - 0.50, 0.0)),
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, winning_score = ordered[0]
    runner_up = ordered[1][1]
    margin = winning_score - runner_up

    if winning_score < 0.20 or margin < 0.08:
        state = "mixed_ambiguous_profile"
        evidence_score = round(max(winning_score, 0.10), 3)
        confidence = "low"
    else:
        state = winner
        evidence_score = round(winning_score, 3)
        if winning_score >= 0.65 and margin >= 0.20:
            confidence = "high"
        elif winning_score >= 0.35 and margin >= 0.12:
            confidence = "medium"
        else:
            confidence = "low"

    return {
        "state_hypothesis": state,
        "state_confidence": confidence,
        "evidence_score": evidence_score,
        "evidence_summary": (
            f"delta={delta:.3f}; theta={theta:.3f}; alpha={alpha:.3f}; "
            f"beta={beta:.3f}; gamma={gamma:.3f}; alpha_beta={alpha_beta:.3f}; "
            f"margin={margin:.3f}"
        ),
    }
```

- [x] **Step 4: Run tests to verify current expected partial failure**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers -k "profiles" -v
```

Expected: pass. The fallback test still fails until `annotate_moment_rows()` exists in Task 5.

## Task 4: Task-State Relation And Quality Booleans

**Files:**
- Modify: `examples/eeg/eegbci/eegbci_pattern_discovery.py`
- Modify: `tests/core/test_eegbci.py`

**Interfaces:**
- Produces: `derive_task_state_relation(row: dict) -> dict`
- Produces: `derive_quality_columns(row: dict) -> dict`

- [x] **Step 1: Write failing tests**

Add:

```python
    def test_task_state_relation_table_is_deterministic(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import derive_task_state_relation

        cases = [
            ("rest", "rest", "idle_alpha_profile", "supports_label"),
            ("rest", "rest", "mixed_ambiguous_profile", "ambiguous"),
            ("rest", "rest", "possible_artifact_profile", "not_applicable"),
            ("execute_left_fist", "motor_execution", "sensorimotor_engagement_profile", "supports_label"),
            ("imagine_left_fist", "motor_imagery", "sensorimotor_engagement_profile", "adds_detail"),
            ("execute_left_fist", "motor_execution", "idle_alpha_profile", "disagrees"),
            ("imagine_left_fist", "motor_imagery", "slow_wave_dominant_pattern", "adds_detail"),
        ]

        for task_label, label_family, state, expected in cases:
            with self.subTest(state=state, label_family=label_family):
                result = derive_task_state_relation(
                    self._moment_row(
                        task_label=task_label,
                        label_family=label_family,
                        state_hypothesis=state,
                    )
                )
                self.assertEqual(result["task_state_relation"], expected)
                self.assertIn(result["task_state_confidence"], {"low", "medium", "high"})
                self.assertGreater(len(result["task_state_rationale"]), 20)

    def test_quality_booleans_are_parseable(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import derive_quality_columns

        flags = derive_quality_columns(
            self._moment_row(
                state_hypothesis="possible_artifact_profile",
                state_confidence="low",
                quality_flags="low_confidence; high_gamma",
            )
        )

        self.assertTrue(flags["is_low_confidence"])
        self.assertTrue(flags["is_possible_artifact"])
        self.assertFalse(flags["is_mixed_or_ambiguous"])
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers -k "relation_table or quality_booleans" -v
```

Expected: fail because helper functions are missing.

- [x] **Step 3: Implement relation and quality helpers**

Add:

```python
def derive_task_state_relation(row: dict) -> dict:
    label_family = row.get("label_family", "")
    task_label = row.get("task_label", "")
    state = row.get("state_hypothesis", "")

    if state == "possible_artifact_profile":
        relation = "not_applicable"
        confidence = "medium"
        rationale = "Artifact-like frequency evidence is flagged for inspection instead of task-label comparison."
    elif state == "mixed_ambiguous_profile":
        relation = "ambiguous"
        confidence = "low"
        rationale = "No frequency-profile state won clearly enough to compare strongly with the task label."
    elif task_label == "rest" and state == "idle_alpha_profile":
        relation = "supports_label"
        confidence = "medium"
        rationale = "The idle-like alpha profile is consistent with a rest-labeled EEGBCI window."
    elif label_family == "motor_execution" and state == "sensorimotor_engagement_profile":
        relation = "supports_label"
        confidence = "medium"
        rationale = "The motor-engaged frequency profile is consistent with an execution-labeled window."
    elif label_family == "motor_imagery" and state == "sensorimotor_engagement_profile":
        relation = "adds_detail"
        confidence = "medium"
        rationale = "The motor-engaged frequency profile adds signal detail to an imagery-labeled window."
    elif label_family in {"motor_execution", "motor_imagery"} and state == "idle_alpha_profile":
        relation = "disagrees"
        confidence = "medium"
        rationale = "The idle-like alpha profile does not align with a motor-labeled EEGBCI window."
    elif state == "slow_wave_dominant_pattern":
        relation = "adds_detail"
        confidence = "low"
        rationale = "The slow-wave dominant pattern adds frequency detail but is not a direct task match."
    else:
        relation = "ambiguous"
        confidence = "low"
        rationale = "The task label and frequency-profile state do not have a stronger deterministic mapping."

    return {
        "task_state_relation": relation,
        "task_state_rationale": rationale,
        "task_state_confidence": confidence,
    }


def derive_quality_columns(row: dict) -> dict:
    flags = str(row.get("quality_flags", ""))
    state = row.get("state_hypothesis", "")
    confidence = row.get("state_confidence", row.get("confidence", ""))
    return {
        "is_low_confidence": confidence == "low" or "low_confidence" in flags,
        "is_possible_artifact": state == "possible_artifact_profile" or "artifact" in flags or "high_gamma" in flags,
        "is_mixed_or_ambiguous": state == "mixed_ambiguous_profile" or "ambiguous" in flags,
    }
```

- [x] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers -k "relation_table or quality_booleans" -v
```

Expected: pass.

## Task 5: Row Annotation And CSV Schema

**Files:**
- Modify: `examples/eeg/eegbci/eegbci_pattern_discovery.py`
- Modify: `tests/core/test_eegbci.py`

**Interfaces:**
- Consumes: `derive_state_hypothesis()`, `derive_task_state_relation()`, `derive_quality_columns()`, `_baseline_for_row()`
- Produces: `annotate_moment_rows(rows: list[dict], baselines: dict) -> list[dict]`
- Produces: `BASE_OUTPUT_COLUMNS: tuple[str, ...]`
- Produces: `MOMENT_REPORT_COLUMNS: tuple[str, ...]`
- Produces: `OUTPUT_COLUMNS: tuple[str, ...]`

- [x] **Step 1: Write failing annotation tests**

Add:

```python
    def test_annotate_moment_rows_adds_required_fields(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            ANALYSIS_VERSION,
            annotate_moment_rows,
            build_rest_baselines,
        )

        rows = [
            self._moment_row(task_label="rest", alpha_relative=0.50, beta_relative=0.20),
            self._moment_row(task_label="execute_left_fist", label_family="motor_execution", alpha_relative=0.20, beta_relative=0.45),
        ]

        annotated = annotate_moment_rows(rows, build_rest_baselines(rows))

        row = annotated[1]
        self.assertEqual(row["analysis_version"], ANALYSIS_VERSION)
        self.assertIn(row["state_hypothesis"], {
            "idle_alpha_profile",
            "sensorimotor_engagement_profile",
            "slow_wave_dominant_pattern",
            "possible_artifact_profile",
            "mixed_ambiguous_profile",
        })
        self.assertIn("rest_alpha_relative_delta", row)
        self.assertAlmostEqual(row["rest_alpha_relative_delta"], -0.30)
        self.assertIn("task_state_relation", row)
        self.assertIn("task_state_rationale", row)
        self.assertIn("is_low_confidence", row)

    def test_annotate_moment_rows_marks_unavailable_rest(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            annotate_moment_rows,
            build_rest_baselines,
        )

        rows = [self._moment_row(task_label="execute_left_fist", label_family="motor_execution")]

        annotated = annotate_moment_rows(rows, build_rest_baselines(rows))

        self.assertEqual(annotated[0]["rest_reference_scope"], "unavailable")
        self.assertEqual(annotated[0]["rest_alpha_relative_delta"], "")
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers -k "annotate_moment_rows or fallback_scopes" -v
```

Expected: fail until `annotate_moment_rows()` exists.

- [x] **Step 3: Implement annotation and schema**

Add below `derive_quality_columns()`:

```python
BASE_OUTPUT_COLUMNS = (
    "patient_id",
    "record_id",
    "subject_id",
    "run",
    "run_type",
    "trial_id",
    "event_code",
    "task_label",
    "label_family",
    "label",
    "eegbci_label",
    "model_label",
    "start_time",
    "end_time",
    "dominant_band",
    "alpha_beta_ratio",
    "theta_beta_ratio",
    "brain_state_hypothesis",
    "confidence",
    "quality_flags",
    "interpretation",
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "delta_relative",
    "theta_relative",
    "alpha_relative",
    "beta_relative",
    "gamma_relative",
)

MOMENT_REPORT_COLUMNS = (
    "analysis_version",
    "state_hypothesis",
    "state_confidence",
    "evidence_score",
    "evidence_summary",
    "rest_reference_scope",
    "rest_delta_relative_delta",
    "rest_theta_relative_delta",
    "rest_alpha_relative_delta",
    "rest_beta_relative_delta",
    "rest_gamma_relative_delta",
    "task_state_relation",
    "task_state_rationale",
    "task_state_confidence",
    "is_low_confidence",
    "is_possible_artifact",
    "is_mixed_or_ambiguous",
)

OUTPUT_COLUMNS = BASE_OUTPUT_COLUMNS + MOMENT_REPORT_COLUMNS


def annotate_moment_rows(rows: list[dict], baselines: dict) -> list[dict]:
    annotated = []
    for row in rows:
        next_row = dict(row)
        scope, baseline = _baseline_for_row(next_row, baselines)
        next_row["analysis_version"] = ANALYSIS_VERSION
        next_row["rest_reference_scope"] = scope

        for band in REPORT_BANDS:
            source_key = f"{band}_relative"
            delta_key = f"rest_{band}_relative_delta"
            if baseline and source_key in baseline and next_row.get(source_key) not in ("", None):
                next_row[delta_key] = round(float(next_row[source_key]) - float(baseline[source_key]), 6)
            else:
                next_row[delta_key] = ""

        next_row.update(derive_state_hypothesis(next_row))
        next_row.update(derive_task_state_relation(next_row))
        next_row.update(derive_quality_columns(next_row))
        annotated.append(next_row)
    return annotated
```

- [x] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers -k "annotate_moment_rows or fallback_scopes" -v
```

Expected: pass.

## Task 6: Representative Window Selection

**Files:**
- Modify: `examples/eeg/eegbci/eegbci_pattern_discovery.py`
- Modify: `tests/core/test_eegbci.py`

**Interfaces:**
- Produces: `select_representative_windows(rows: list[dict]) -> dict`

Return shape:

```python
{
    "cards": {"strongest_idle_like": row, "most_ambiguous": row, ...},
    "absent": ["strongest_artifact_like", ...],
}
```

- [x] **Step 1: Write failing representative selection tests**

Add:

```python
    def test_select_representative_windows_is_deterministic(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import select_representative_windows

        rows = [
            self._moment_row(subject_id=2, run=4, start_time=6.0, state_hypothesis="idle_alpha_profile", state_confidence="medium", evidence_score=0.80),
            self._moment_row(subject_id=1, run=3, start_time=4.0, state_hypothesis="idle_alpha_profile", state_confidence="medium", evidence_score=0.80),
            self._moment_row(subject_id=1, run=3, start_time=8.0, state_hypothesis="sensorimotor_engagement_profile", state_confidence="high", evidence_score=0.90),
            self._moment_row(subject_id=1, run=3, start_time=10.0, state_hypothesis="mixed_ambiguous_profile", state_confidence="low", evidence_score=0.12),
            self._moment_row(subject_id=1, run=3, start_time=12.0, state_hypothesis="idle_alpha_profile", task_state_relation="disagrees", state_confidence="medium", evidence_score=0.70),
        ]

        selected = select_representative_windows(rows)

        self.assertEqual(selected["cards"]["strongest_idle_like"]["subject_id"], 1)
        self.assertEqual(selected["cards"]["strongest_motor_engaged"]["state_hypothesis"], "sensorimotor_engagement_profile")
        self.assertEqual(selected["cards"]["most_ambiguous"]["start_time"], 10.0)
        self.assertEqual(selected["cards"]["strongest_task_state_disagreement"]["task_state_relation"], "disagrees")
        self.assertIn("strongest_artifact_like", selected["absent"])
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers::test_select_representative_windows_is_deterministic -v
```

Expected: fail because `select_representative_windows` is missing.

- [x] **Step 3: Implement representative selection**

Add:

```python
def _stable_row_key(row: dict) -> tuple:
    return (
        row.get("subject_id", 0),
        row.get("run", 0),
        float(row.get("start_time", 0.0) or 0.0),
    )


def _strongest_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("evidence_score", 0.0) or 0.0),
            -STATE_CONFIDENCE_RANK.get(row.get("state_confidence", "low"), 0),
            *_stable_row_key(row),
        ),
    )[0]


def select_representative_windows(rows: list[dict]) -> dict:
    definitions = {
        "strongest_idle_like": "idle_alpha_profile",
        "strongest_motor_engaged": "sensorimotor_engagement_profile",
        "strongest_slow_wave": "slow_wave_dominant_pattern",
        "strongest_artifact_like": "possible_artifact_profile",
    }
    cards = {}
    absent = []

    for card_name, state in definitions.items():
        candidate = _strongest_row([row for row in rows if row.get("state_hypothesis") == state])
        if candidate is None:
            absent.append(card_name)
        else:
            cards[card_name] = candidate

    ambiguous = [row for row in rows if row.get("state_hypothesis") == "mixed_ambiguous_profile"]
    if ambiguous:
        cards["most_ambiguous"] = sorted(
            ambiguous,
            key=lambda row: (
                float(row.get("evidence_score", 0.0) or 0.0),
                -STATE_CONFIDENCE_RANK.get(row.get("state_confidence", "low"), 0),
                *_stable_row_key(row),
            ),
        )[0]
    else:
        absent.append("most_ambiguous")

    disagreement = _strongest_row(
        [row for row in rows if row.get("task_state_relation") == "disagrees"]
    )
    if disagreement is None:
        absent.append("strongest_task_state_disagreement")
    else:
        cards["strongest_task_state_disagreement"] = disagreement

    return {"cards": cards, "absent": absent}
```

- [x] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers::test_select_representative_windows_is_deterministic -v
```

Expected: pass.

## Task 7: Markdown Summary Renderer

**Files:**
- Modify: `examples/eeg/eegbci/eegbci_pattern_discovery.py`
- Modify: `tests/core/test_eegbci.py`

**Interfaces:**
- Produces: `render_summary(rows: list[dict], config: dict) -> str`
- Updates: `write_summary(rows: list[dict], path: Path, config: dict) -> None`

- [ ] **Step 1: Write failing renderer tests**

Add:

```python
    def test_render_summary_contains_required_sections_and_limitations(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            ANALYSIS_VERSION,
            annotate_moment_rows,
            build_rest_baselines,
            render_summary,
        )

        rows = [
            self._moment_row(task_label="execute_left_fist", label_family="motor_execution")
        ]
        annotated = annotate_moment_rows(rows, build_rest_baselines(rows))
        summary = render_summary(
            annotated,
            {
                "subjects": [1],
                "runs": [3],
                "max_windows": 1,
                "baseline_row_count": 1,
                "output_was_capped": True,
            },
        )

        self.assertIn(ANALYSIS_VERSION, summary.splitlines()[2])
        for heading in [
            "## Executive Result",
            "## Run Configuration",
            "## Window Coverage",
            "## Moment-State Summary",
            "## Task Label x State Matrix",
            "## Rest-Normalized Bandpower Summary",
            "## Confidence and Quality Audit",
            "## Representative Windows",
            "## Limitations",
            "## Next Checks",
        ]:
            self.assertIn(heading, summary)
        self.assertIn("No rest baseline was available", summary)
        self.assertIn("Output was capped by `--max-windows`", summary)
        self.assertNotIn("Brain-state hypotheses are exploratory signal metadata", summary.splitlines()[2])

    def test_render_summary_handles_empty_rows(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import render_summary

        summary = render_summary(
            [],
            {
                "subjects": [1],
                "runs": [3],
                "max_windows": 0,
                "baseline_row_count": 0,
                "output_was_capped": True,
            },
        )

        self.assertIn("No windows were produced", summary)
        self.assertIn("## Limitations", summary)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers -k "render_summary" -v
```

Expected: fail because `render_summary` is missing.

- [ ] **Step 3: Implement renderer**

Replace existing `write_summary()` with:

```python
def _format_count_lines(counter: Counter) -> list[str]:
    if not counter:
        return ["- None"]
    return [f"- {label}: {count}" for label, count in counter.most_common()]


def _format_card(row: dict) -> list[str]:
    bands = ", ".join(
        f"{band}={float(row.get(f'{band}_relative', 0.0) or 0.0):.3f}"
        for band in REPORT_BANDS
    )
    deltas = ", ".join(
        f"{band}={row.get(f'rest_{band}_relative_delta', '')}"
        for band in REPORT_BANDS
    )
    return [
        f"- Subject {row.get('subject_id')} run {row.get('run')} trial {row.get('trial_id')}",
        f"  - Task: {row.get('task_label')} from {row.get('start_time')}s to {row.get('end_time')}s",
        f"  - State: {row.get('state_hypothesis')} ({row.get('state_confidence')}, evidence {row.get('evidence_score')})",
        f"  - Dominant band: {row.get('dominant_band')}; relative bands: {bands}",
        f"  - Rest deltas: {deltas}; scope: {row.get('rest_reference_scope')}",
        f"  - Task relation: {row.get('task_state_relation')} ({row.get('task_state_confidence')})",
        f"  - Flags: low_confidence={row.get('is_low_confidence')}, possible_artifact={row.get('is_possible_artifact')}, mixed_or_ambiguous={row.get('is_mixed_or_ambiguous')}",
        f"  - Rationale: {row.get('task_state_rationale')}",
    ]


def render_summary(rows: list[dict], config: dict) -> str:
    state_counts = Counter(row.get("state_hypothesis", "missing") for row in rows)
    task_counts = Counter(row.get("task_label", "missing") for row in rows)
    confidence_counts = Counter(row.get("state_confidence", "missing") for row in rows)
    relation_counts = Counter(row.get("task_state_relation", "missing") for row in rows)
    unavailable_rest = sum(row.get("rest_reference_scope") == "unavailable" for row in rows)
    low_confidence = sum(bool(row.get("is_low_confidence")) for row in rows)
    artifacts = sum(bool(row.get("is_possible_artifact")) for row in rows)
    ambiguous = sum(bool(row.get("is_mixed_or_ambiguous")) for row in rows)
    representatives = select_representative_windows(rows)

    executive = []
    if not rows:
        executive.append("No windows were produced for the requested configuration.")
    else:
        top_state, top_state_count = state_counts.most_common(1)[0]
        executive.append(
            f"Processed {len(rows)} windows. Most common state: `{top_state}` ({top_state_count}/{len(rows)})."
        )
        if low_confidence == len(rows):
            executive.append("Every window is low confidence.")
        if len(state_counts) == 1:
            executive.append("Every window maps to the same state; broaden coverage or review thresholds.")
        if unavailable_rest == len(rows):
            executive.append("No rest baseline was available for the emitted rows.")
    if config.get("output_was_capped"):
        executive.append("Output was capped by `--max-windows`.")

    lines = [
        "# EEGBCI Pattern Discovery Moment Report",
        "",
        f"Analysis version: `{ANALYSIS_VERSION}`",
        "",
        "## Executive Result",
        "",
        *[f"- {item}" for item in executive],
        "",
        "## Run Configuration",
        "",
        f"- Subjects: {config.get('subjects')}",
        f"- Runs: {config.get('runs')}",
        f"- Max windows: {config.get('max_windows')}",
        f"- Baseline source rows: {config.get('baseline_row_count')}",
        "",
        "## Window Coverage",
        "",
        f"- Output windows: {len(rows)}",
        f"- Task labels: {dict(task_counts)}",
        "",
        "## Moment-State Summary",
        "",
        *_format_count_lines(state_counts),
        "",
        "## Task Label x State Matrix",
        "",
    ]

    matrix = Counter(
        (row.get("task_label", "missing"), row.get("state_hypothesis", "missing"))
        for row in rows
    )
    if matrix:
        for (task_label, state), count in sorted(matrix.items()):
            lines.append(f"- {task_label} x {state}: {count}")
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Rest-Normalized Bandpower Summary",
        "",
        f"- Rows with unavailable rest baseline: {unavailable_rest}",
    ])
    for band in REPORT_BANDS:
        key = f"rest_{band}_relative_delta"
        values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
        if values:
            lines.append(f"- {band}: mean delta {sum(values) / len(values):.3f}")
        else:
            lines.append(f"- {band}: unavailable")

    lines.extend([
        "",
        "## Confidence and Quality Audit",
        "",
        f"- State confidence: {dict(confidence_counts)}",
        f"- Task-state relations: {dict(relation_counts)}",
        f"- Low-confidence rows: {low_confidence}",
        f"- Possible artifact rows: {artifacts}",
        f"- Mixed or ambiguous rows: {ambiguous}",
        "",
        "## Representative Windows",
        "",
    ])
    if representatives["cards"]:
        for card_name, row in representatives["cards"].items():
            lines.append(f"### {card_name.replace('_', ' ').title()}")
            lines.extend(_format_card(row))
            lines.append("")
    else:
        lines.append("- None")
    if representatives["absent"]:
        lines.append(f"- Absent representative classes: {', '.join(representatives['absent'])}")

    lines.extend([
        "",
        "## Limitations",
        "",
        "- These labels are signal-pattern summaries from short EEG windows. They are not clinical findings and should not be read as evidence of a subject's cognition.",
    ])
    if unavailable_rest:
        lines.append("- No rest baseline was available for at least one emitted row.")
    if config.get("output_was_capped"):
        lines.append("- The output was capped, so the artifact may not represent all requested windows.")

    lines.extend([
        "",
        "## Next Checks",
        "",
        "- Run with broader subjects/runs to verify that state diversity improves.",
        "- Inspect possible artifact rows before drawing conclusions from state counts.",
        "- Compare rest-normalized deltas against the raw relative band shares.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def write_summary(rows: list[dict], path: Path, config: dict) -> None:
    path.write_text(render_summary(rows, config), encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers -k "render_summary" -v
```

Expected: pass.

## Task 8: Main Flow, Empty CSV, And Max-Windows Semantics

**Files:**
- Modify: `examples/eeg/eegbci/eegbci_pattern_discovery.py`
- Modify: `tests/core/test_eegbci.py`

**Interfaces:**
- Updates: `main()` so all requested rows are collected before truncation.
- Produces: empty CSV with stable columns when `--max-windows=0`.

- [ ] **Step 1: Write failing schema test**

Add:

```python
    def test_moment_report_columns_are_declared(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            MOMENT_REPORT_COLUMNS,
            OUTPUT_COLUMNS,
        )

        for column in [
            "patient_id",
            "task_label",
            "alpha_relative",
            "analysis_version",
            "state_hypothesis",
            "state_confidence",
            "evidence_score",
            "evidence_summary",
            "rest_reference_scope",
            "rest_alpha_relative_delta",
            "task_state_relation",
            "task_state_rationale",
            "task_state_confidence",
            "is_low_confidence",
            "is_possible_artifact",
            "is_mixed_or_ambiguous",
        ]:
            self.assertIn(column, OUTPUT_COLUMNS)
        self.assertIn("analysis_version", MOMENT_REPORT_COLUMNS)
```

- [ ] **Step 2: Run focused helper tests**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers -v
```

Expected: pass after previous tasks and the schema declaration.

- [ ] **Step 3: Update main flow**

Replace the row collection and output block in `main()` with:

```python
    all_rows = [sample_to_row(sample) for sample in sample_dataset]
    baselines = build_rest_baselines(all_rows)
    annotated_rows = annotate_moment_rows(all_rows, baselines)
    output_rows = (
        annotated_rows[: args.max_windows]
        if args.max_windows is not None
        else annotated_rows
    )
    output_was_capped = (
        args.max_windows is not None and len(annotated_rows) > len(output_rows)
    )

    csv_path = output_dir / "eegbci_pattern_windows.csv"
    summary_path = output_dir / "eegbci_pattern_summary.md"
    pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS).to_csv(csv_path, index=False)
    write_summary(
        output_rows,
        summary_path,
        {
            "subjects": parse_int_list(args.subjects),
            "runs": parse_int_list(args.runs),
            "max_windows": args.max_windows,
            "baseline_row_count": len(all_rows),
            "output_was_capped": output_was_capped,
        },
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
```

This deliberately uses `OUTPUT_COLUMNS` even when `output_rows` is empty, so
`--max-windows=0` and no-window runs still produce a parseable CSV contract.

- [ ] **Step 4: Run full EEGBCI unit test file**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py -v
```

Expected: pass, with the real-data smoke test skipped unless `PYHEALTH_RUN_REAL_EEGBCI=1`.

## Task 9: README And Progress Documentation

**Files:**
- Modify: `examples/eeg/eegbci/README.md`
- Modify: `docs/eeg_pattern_discovery/moment_report_continuation_plan.md`

**Interfaces:**
- Produces: README section describing upgraded moment report fields and limitations.
- Produces: continuation plan progress entry for implementation.

- [ ] **Step 1: Update README output description**

Replace the CSV paragraph in `examples/eeg/eegbci/README.md` with:

```markdown
The CSV has one row per emitted 2-second window. Key columns include subject/run
metadata, `event_code`, decoded `task_label`, raw EEGBCI numeric label
(`eegbci_label` / `label`), PyHealth model-local label (`model_label`),
absolute window timing, band powers, relative band powers, `dominant_band`,
frequency ratios, legacy `brain_state_hypothesis`, `confidence`,
`quality_flags`, and `interpretation`.

The moment-report columns add analysis-grade fields:

- `analysis_version`
- `state_hypothesis`, `state_confidence`, and `evidence_score`
- `evidence_summary`
- `rest_reference_scope` and rest-normalized relative band deltas
- `task_state_relation`, `task_state_rationale`, and `task_state_confidence`
- `is_low_confidence`, `is_possible_artifact`, and `is_mixed_or_ambiguous`

The Markdown report summarizes state counts, task-label/state agreement,
rest-normalized bandpower deltas, confidence and quality flags, representative
windows, limitations, and next checks. These labels are signal-pattern
summaries from short EEG windows, not clinical findings or evidence of a
subject's cognition.
```

- [ ] **Step 2: Update continuation plan progress**

Append to `docs/eeg_pattern_discovery/moment_report_continuation_plan.md`:

```markdown
- 2026-07-08: Converted the refined design into
  `docs/eeg_pattern_discovery/moment_report_implementation_plan.md` and ran
  GStack `/plan-eng-review` against the plan before code implementation.
```

- [ ] **Step 3: Verify docs mention the plan**

Run:

```bash
rg "moment_report_implementation_plan|GStack `/plan-eng-review`" docs/eeg_pattern_discovery examples/eeg/eegbci/README.md
```

Expected: matches in the continuation plan and README content.

## Task 10: Manual Artifact Verification

**Files:**
- No planned code changes unless verification exposes a bug.

**Interfaces:**
- Verifies: local synthetic/unit coverage and real-data example behavior.

- [ ] **Step 1: Run unit tests**

Run:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py -v
```

Expected: pass, with real-data smoke skipped unless explicitly enabled.

- [ ] **Step 2: Run the example on a tiny real-data request**

Run:

```bash
.venv/bin/python examples/eeg/eegbci/eegbci_pattern_discovery.py \
  --subjects 1 \
  --runs 3 \
  --max-windows 20 \
  --download
```

Expected:

- `outputs/eegbci_pattern_discovery/eegbci_pattern_windows.csv` is written.
- `outputs/eegbci_pattern_discovery/eegbci_pattern_summary.md` is written.
- CSV includes all `MOMENT_REPORT_COLUMNS`.
- Markdown includes all required sections.
- Markdown does not start with the old generic exploratory caveat.

- [ ] **Step 3: Inspect artifact schema**

Run:

```bash
.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/eegbci_pattern_discovery/eegbci_pattern_windows.csv")
required = {
    "analysis_version",
    "state_hypothesis",
    "state_confidence",
    "evidence_score",
    "evidence_summary",
    "rest_reference_scope",
    "rest_alpha_relative_delta",
    "task_state_relation",
    "task_state_rationale",
    "task_state_confidence",
    "is_low_confidence",
    "is_possible_artifact",
    "is_mixed_or_ambiguous",
}
missing = sorted(required - set(df.columns))
print("rows", len(df))
print("missing", missing)
assert not missing
assert (df["analysis_version"] == "eegbci_pattern_moment_report_v1").all()
PY
```

Expected:

```text
rows 20
missing []
```

- [ ] **Step 4: Inspect Markdown contract**

Run:

```bash
rg "Executive Result|Run Configuration|Window Coverage|Moment-State Summary|Task Label x State Matrix|Rest-Normalized Bandpower Summary|Confidence and Quality Audit|Representative Windows|Limitations|Next Checks" outputs/eegbci_pattern_discovery/eegbci_pattern_summary.md
```

Expected: all required headings match.

- [ ] **Step 5: Update Graphify**

Run:

```bash
graphify update .
```

Expected: graph update completes. Dirty `graphify-out/` files are expected.

## Extensive Correctness Test Matrix

Add these tests to `TestEEGBCIMomentReportHelpers` while implementing Tasks 1-8.
The goal is to prove the report helpers are correct under realistic edge cases,
not just that the happy path produces rows.

| Test name | Fixture | Assertions |
| --- | --- | --- |
| `test_analysis_version_constant` | Import `ANALYSIS_VERSION`. | Exact value is `eegbci_pattern_moment_report_v1`. |
| `test_moment_report_columns_are_declared` | Import `OUTPUT_COLUMNS` and `MOMENT_REPORT_COLUMNS`. | Base row columns and report columns are present; `analysis_version` is in `MOMENT_REPORT_COLUMNS`. |
| `test_build_rest_baselines_uses_rest_rows_only` | Rest and non-rest rows for one subject across two runs. | Non-rest rows do not affect rest averages; same-run, same-subject, and global means are correct. |
| `test_build_rest_baselines_handles_no_rest_rows` | Only motor rows. | Same-run and same-subject baseline maps are empty; global baseline is `None`. |
| `test_annotate_rest_fallback_scopes` | One same-run rest candidate, one same-subject fallback, and one global fallback. | Rows receive `same_subject_run`, `same_subject_all_runs`, and `global_rest` in that order. |
| `test_annotate_moment_rows_marks_unavailable_rest` | Motor-only rows. | `rest_reference_scope == "unavailable"` and all rest delta columns are blank strings. |
| `test_rest_delta_values_are_band_specific` | Rest row with different values for each band plus one motor row. | `rest_delta_relative_delta`, `rest_theta_relative_delta`, `rest_alpha_relative_delta`, `rest_beta_relative_delta`, and `rest_gamma_relative_delta` equal motor minus rest for the matching band. |
| `test_derive_state_hypothesis_detects_idle_alpha_profile` | High alpha, low beta/gamma, high alpha/beta. | State is `idle_alpha_profile`; confidence is not outside `low`, `medium`, `high`; score is between 0 and 1. |
| `test_derive_state_hypothesis_detects_sensorimotor_engagement_profile` | High beta or low-gamma, low alpha/beta. | State is `sensorimotor_engagement_profile`; evidence summary includes beta and alpha/beta values. |
| `test_derive_state_hypothesis_detects_slow_wave_dominant_pattern` | Delta plus theta dominates. | State is `slow_wave_dominant_pattern`; no cognitive or clinical wording appears in evidence summary. |
| `test_derive_state_hypothesis_detects_possible_artifact_profile` | Gamma spike or extreme high-frequency share. | State is `possible_artifact_profile`; `derive_quality_columns()` marks `is_possible_artifact`. |
| `test_derive_state_hypothesis_marks_weak_margin_ambiguous` | Balanced band shares with no clear winner. | State is `mixed_ambiguous_profile`; confidence is `low`. |
| `test_state_confidence_requires_margin` | Two rows with same winning state, one clear margin and one narrow margin. | Narrow-margin row has lower confidence than clear-margin row. |
| `test_task_state_relation_table_is_deterministic` | Rows covering rest, motor execution, motor imagery, slow-wave, artifact, and ambiguous states. | Relation values match the approved decision table; every rationale is non-empty. |
| `test_task_state_relation_idle_motor_disagrees` | Motor task row with `idle_alpha_profile`. | Relation is `disagrees`; confidence is parseable; rationale mentions motor-labeled mismatch without clinical claims. |
| `test_task_state_relation_artifact_not_applicable` | Any task row with `possible_artifact_profile`. | Relation is `not_applicable`; rationale says inspection rather than task comparison. |
| `test_quality_booleans_are_parseable` | Low-confidence artifact row with text flags. | `is_low_confidence` and `is_possible_artifact` are true; ambiguous flag follows state/flag input. |
| `test_quality_booleans_do_not_depend_on_string_parsing_only` | Row with `state_hypothesis="mixed_ambiguous_profile"` and empty `quality_flags`. | `is_mixed_or_ambiguous` is true because the state is ambiguous. |
| `test_annotate_moment_rows_adds_required_fields` | One rest row and one motor row. | Every `MOMENT_REPORT_COLUMNS` entry exists in every annotated row. |
| `test_annotate_moment_rows_preserves_legacy_fields` | Row with existing `brain_state_hypothesis`, `confidence`, `quality_flags`, and `interpretation`. | Legacy fields remain unchanged after annotation. |
| `test_annotate_moment_rows_does_not_mutate_input_rows` | Keep a copy of input rows before annotation. | Original rows do not gain report fields after `annotate_moment_rows()`. |
| `test_select_representative_windows_is_deterministic` | Multiple candidate rows with tied evidence/confidence and different subject/run/start time. | Stable tie-break chooses earliest subject, run, and start time. |
| `test_select_representative_windows_lists_absent_classes` | Rows missing artifact and slow-wave classes. | Missing representative card names appear in `absent`. |
| `test_select_representative_windows_picks_lowest_evidence_ambiguous` | Multiple ambiguous rows with different evidence scores. | `most_ambiguous` chooses the lowest evidence score, then stable tie-breaks. |
| `test_select_representative_windows_picks_strongest_disagreement` | Multiple disagreement rows. | Highest evidence disagreement is selected; ties use confidence and stable ordering. |
| `test_render_summary_contains_required_sections_and_limitations` | Annotated row with unavailable rest and capped config. | All required headings appear; missing rest and cap limitations appear. |
| `test_render_summary_handles_empty_rows` | Empty row list with `max_windows=0`. | Summary says no windows were produced and still includes Limitations and Next Checks. |
| `test_render_summary_reports_all_low_confidence` | Rows where every `is_low_confidence` is true. | Executive Result states every window is low confidence. |
| `test_render_summary_reports_all_same_state` | Rows all mapped to one state. | Executive Result states every window maps to the same state. |
| `test_render_summary_reports_task_state_matrix` | Rows spanning at least two task labels and two states. | Matrix contains each task/state count deterministically. |
| `test_render_summary_includes_representative_window_details` | Annotated rows with at least one representative card. | Card includes subject, run, trial id, time range, state, evidence, dominant band, rest deltas, relation, confidence, flags, and rationale. |
| `test_render_summary_moves_nonclinical_warning_to_limitations` | Any non-empty summary. | Warning appears under `## Limitations` and not as the opening body text. |
| `test_summary_text_does_not_repeat_old_row_level_caveat` | Annotated rows with legacy interpretation text. | `"This is exploratory signal metadata"` is absent from generated summary and new row-level interpretation. |
| `test_empty_dataframe_uses_output_columns` | Build `pd.DataFrame([], columns=OUTPUT_COLUMNS)`. | Empty CSV header contains base and report columns. |
| `test_main_max_windows_zero_writes_empty_artifacts` | Patch dataset/task iteration to produce rows, invoke `main()` with `--max-windows 0` and temp output dir. | CSV has zero rows with `OUTPUT_COLUMNS`; Markdown says no windows were produced and output was capped. |
| `test_main_baseline_uses_uncapped_rows` | Patch task iteration so rest row appears after the first emitted row; run with `--max-windows 1`. | Emitted row can still receive non-`unavailable` rest baseline from rows beyond the cap. |
| `test_main_writes_analysis_version_to_every_csv_row` | Patch task iteration to produce two rows. | CSV `analysis_version` column exists and every value equals `ANALYSIS_VERSION`. |
| `test_parse_int_list_accepts_ranges_and_singletons` | Existing parser with `"1,3-5"`. | Result is `[1, 3, 4, 5]`. |
| `test_parse_int_list_rejects_invalid_input_loudly` | Existing parser with `"a"` or malformed range. | Raises `ValueError`; no silent fallback. |

Run the complete helper suite with:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIMomentReportHelpers -v
```

Run the full EEGBCI test file with:

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py -v
```

Manual real-data artifact checks still belong in Task 10 because they validate
the end-to-end generated CSV and Markdown, not just pure helper behavior.

## Post-Implementation Review Gate

After implementation and verification finish, run a fresh review before calling
the work complete.

Recommended workflow:

```text
main implementation chat
  -> complete Tasks 1-10
  -> run unit tests, artifact checks, and graphify update
  -> dispatch independent sub-agent or separate session for GStack /review
  -> independent reviewer writes review Markdown
  -> main implementation chat applies fixes
  -> rerun focused tests and full EEGBCI tests
```

Use the strongest available review path:

1. Prefer GStack `/review` for the final pre-landing diff review because it is
   designed to inspect the branch against the base branch and catch regressions,
   trust-boundary mistakes, structural issues, and missing tests.
2. If `/review` is unavailable or the review is not PR/diff-shaped, use the
   `bug-review` skill against the changed files and generated artifacts.
3. If both are available, run `/review` first, then use `bug-review` only for
   focused follow-up on risky areas the review identifies.

The review must create:

- `docs/eeg_pattern_discovery/moment_report_review.md`

Required review document sections:

```markdown
# EEGBCI Moment Report Implementation Review

Date: 2026-07-08
Branch: eegbci-pattern-discovery
Review path: GStack /review

## Scope Reviewed

## Verification Commands

## Findings

## Required Fixes

## Fix Implementation Log

## Post-Fix Verification

## Final Verdict
```

Review independence rule:

- The same chat can orchestrate the whole workflow because it has the project
  context and plan history.
- The actual GStack `/review` must run in an independent sub-agent or separate
  session. Do not run the final review inline in the same reasoning thread that
  implemented the feature.
- The independent reviewer must inspect the diff cold, write
  `docs/eeg_pattern_discovery/moment_report_review.md`, and stop after the
  review document is complete.
- Fixes should be implemented by the main implementation chat after the review
  is written, so there is one accountable thread for code changes and
  verification.

Do not mark the implementation complete until:

- review findings are written to Markdown
- each accepted finding is fixed or explicitly documented as deferred
- focused tests for each fix pass
- `.venv/bin/python -m pytest tests/core/test_eegbci.py -v` passes
- generated CSV and Markdown artifacts still satisfy the contract
- `graphify update .` has been run after code changes

## Self-Review

### Spec Coverage

- Rest-normalized evidence: Task 2, Task 3, Task 5, Task 7.
- Parseable quality flags: Task 4, Task 5, Task 7.
- Representative windows: Task 6, Task 7.
- Analysis versioning: Task 1, Task 5, Task 7, Task 8.
- Task-state comparison: Task 4, Task 5, Task 7.
- Stable dataset/task API boundary: Global Constraints and File Structure.
- Empty/capped output behavior: Task 7, Task 8, Task 10.
- README and continuation docs: Task 9.

### Placeholder Scan

No implementation steps use TBD, TODO, or open-ended "handle edge cases" instructions. Each test and implementation step includes concrete code or exact commands.

### Type Consistency

The plan consistently uses:

- `build_rest_baselines(rows) -> dict`
- `derive_state_hypothesis(row) -> dict`
- `derive_task_state_relation(row) -> dict`
- `derive_quality_columns(row) -> dict`
- `annotate_moment_rows(rows, baselines) -> list[dict]`
- `select_representative_windows(rows) -> dict`
- `render_summary(rows, config) -> str`
- `write_summary(rows, path, config) -> None`

## GSTACK ENGINEERING REVIEW

Status: DONE

### Review Scope

Reviewed this implementation plan against:

- `docs/eeg_pattern_discovery/moment_report_refined_design.md`
- `docs/eeg_pattern_discovery/moment_report_continuation_plan.md`
- `/Users/vihaanagrawal/.gstack/projects/sunlabuiuc-PyHealth/ceo-plans/2026-07-08-eegbci-moment-report.md`
- current `examples/eeg/eegbci/eegbci_pattern_discovery.py`
- current `tests/core/test_eegbci.py`

### Architecture Verdict

The architecture boundary is right.

```text
Reusable PyHealth APIs                  Example-owned artifact layer
----------------------                  ----------------------------
EEGBCIDataset
  -> EEGMotorImageryEEGBCI samples
  -> EEGBCIPatternDiscovery bandpower
                                        sample_to_row()
                                        build_rest_baselines()
                                        annotate_moment_rows()
                                        select_representative_windows()
                                        render_summary()
                                        CSV + Markdown
```

The plan avoids putting cross-window concepts into `EEGBCIPatternDiscovery`.
That matters because rest baselines, capped output status, and task-state
comparison are artifact-level context, not independent per-sample labels.

### Data Flow Review

The critical data-flow fix is present:

```text
collect all requested rows
  -> compute rest baselines
  -> annotate all rows
  -> apply --max-windows
  -> write outputs
```

This prevents `--max-windows` from accidentally deleting the rest evidence
needed to interpret non-rest rows.

### Findings And Required Adjustments

| Severity | Finding | Plan Adjustment |
| --- | --- | --- |
| High | Empty CSV handling cannot depend on `sample_dataset[0]`; `--max-windows=0` and no-window requests are valid edge cases. | Added `BASE_OUTPUT_COLUMNS`, `MOMENT_REPORT_COLUMNS`, and `OUTPUT_COLUMNS`; Task 8 now writes `pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)`. |
| Medium | The example may collect more rows than it emits. For the default subjects/runs this is acceptable, but users should see the baseline source count. | `render_summary()` config includes `baseline_row_count`; Markdown Run Configuration reports it. |
| Medium | State scoring thresholds are heuristic and could become brittle if tests overfit exact values. | Tests use obvious synthetic profiles and assert categories plus score bounds, not exact score formulas. |
| Low | The implementation plan adds many helpers in one example file. That is acceptable for this phase, but file growth needs containment. | Helpers are pure, named, and directly tested. No shared module until the example becomes genuinely hard to maintain. |

### Edge Case Coverage

```text
No rows
  -> render_summary([]) says no windows were produced
  -> CSV writes stable OUTPUT_COLUMNS

No rest rows
  -> rest_reference_scope = "unavailable"
  -> rest deltas = ""
  -> Markdown limitation states missing rest baseline

All low confidence
  -> Executive Result says every window is low confidence
  -> Confidence and Quality Audit counts low-confidence rows

All same state
  -> Executive Result says every window maps to the same state
  -> Next Checks recommends broader coverage or threshold review

Capped output
  -> baselines computed before cap
  -> Executive Result says output was capped
```

### Test Coverage Review

The plan has the right test shape:

- Pure helper tests are synthetic and offline.
- Dataset/task tests remain intact.
- Rest fallback covers same subject/run, same subject/all runs, global rest, and unavailable.
- Task-state relation table is deterministic.
- Representative selection tests tie-break on evidence, confidence, subject, run, and start time.
- Markdown tests assert required sections and important limitations.
- Manual verification checks the real generated CSV/Markdown artifact.

One implementation note: when writing the tests, keep the helper class before
`TestEEGBCIRealDataSmoke` so normal test runs do not inherit the real-data skip.

### Performance And Blast Radius

Blast radius is low. The plan touches one example script, one existing test file,
one README, and this planning doc. No model APIs, dataset APIs, task exports, or
Sphinx API pages change.

The only meaningful performance tradeoff is collecting all requested rows before
applying `--max-windows`. That is the correct tradeoff for this artifact because
baseline quality is the point of the report. If future users request large
subject/run sets, add a separate `--baseline-max-windows` or streaming baseline
path later. Do not complicate this PR now.

### Recommendation

Proceed with implementation exactly in this plan order. The first implementation
checkpoint should be after Task 5, because that is where the CSV contract and
core data flow become real. The second checkpoint should be after Task 8, before
real-data manual verification.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| Eng Review | `/plan-eng-review` | Architecture, data flow, edge cases, tests | 1 | DONE | Added stable output schema requirement; confirmed example-owned analysis boundary and baseline-before-cap flow. |

**VERDICT:** APPROVED FOR IMPLEMENTATION after the `OUTPUT_COLUMNS` adjustment above.
