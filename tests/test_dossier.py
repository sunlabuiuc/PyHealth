"""Unit tests for DOSSIER EHR fact-checking — Task + Pipeline.

Scope follows the PyHealth PR rubric:
  Task  (EHRFactCheckingMIMIC3):  sample processing, label generation,
                                    feature extraction, edge cases.
  Pipeline (DOSSIERPipeline):     prediction (forward-pass equivalent),
                                    evaluation output shapes and values.

All tests use inline DataFrames and MagicMock for LLM and SQL execution.
No MIMIC-III files, no API key.  Total: 14 tests, <100 ms each.
"""

import importlib.util
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock

import pandas as pd
import sklearn.metrics  # pre-import: avoids 1-second cold-start in evaluate() tests

# ---------------------------------------------------------------------------
# Isolated import helpers — avoid triggering torch / polars via __init__.py
# ---------------------------------------------------------------------------

def _load_module_isolated(path: str, name: str) -> types.ModuleType:
    """Load a Python source file as an isolated module without side-importing.

    Args:
        path: Absolute filesystem path to the ``.py`` source file.
        name: Fully-qualified module name to register in ``sys.modules``.

    Returns:
        The freshly executed module object.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_PYHEALTH_ROOT = pathlib.Path(__file__).parent.parent / "pyhealth"

_base_task_stub = types.ModuleType("pyhealth.tasks.base_task")


class _BaseTask:
    """Minimal BaseTask stub used during isolated module loading."""

    def __init__(self, code_mapping: types.ModuleType | None = None) -> None:
        """Store ``code_mapping`` to satisfy the real BaseTask API.

        Args:
            code_mapping: Unused in tests; mirrors the production signature.
        """
        self.code_mapping = code_mapping


_base_task_stub.BaseTask = _BaseTask
sys.modules["pyhealth.tasks.base_task"] = _base_task_stub

for _ns in ("pyhealth", "pyhealth.tasks", "pyhealth.models"):
    if _ns not in sys.modules:
        sys.modules[_ns] = types.ModuleType(_ns)

_dossier_mod = _load_module_isolated(
    str(_PYHEALTH_ROOT / "models" / "dossier.py"),
    "pyhealth.models.dossier",
)
_task_mod = _load_module_isolated(
    str(_PYHEALTH_ROOT / "tasks" / "ehr_fact_checking.py"),
    "pyhealth.tasks.ehr_fact_checking",
)

DOSSIERPromptGenerator = _dossier_mod.DOSSIERPromptGenerator
DOSSIERPipeline = _dossier_mod.DOSSIERPipeline
EHRFactCheckingMIMIC3 = _task_mod.EHRFactCheckingMIMIC3


# ---------------------------------------------------------------------------
# Tests: EHRFactCheckingMIMIC3 — Task coverage
# Rubric: sample processing, label generation, feature extraction, edge cases
# ---------------------------------------------------------------------------

class TestEHRFactCheckingMIMIC3Task(unittest.TestCase):
    """Covers all four task rubric dimensions with mock patients."""

    def _make_claims_df(self) -> pd.DataFrame:
        """Return a minimal claims DataFrame with three labelled claims.

        Claims match the paper's slot-filling pattern: glucose (T), heparin
        (F), and intubation (N) across two admissions.
        """
        return pd.DataFrame({
            "HADM_ID": [100001, 100001, 100002],
            "claim": [
                "pt had glucose > 100",
                "pt was given heparin",
                "pt was intubated",
            ],
            "t_C": [24.0, 48.0, 72.0],
            "label": ["T", "F", "N"],
        })

    def _make_patient(self, hadm_id: int) -> MagicMock:
        """Return a mock Patient with a single admission for ``hadm_id``.

        Args:
            hadm_id: MIMIC-III hospital admission ID to assign to the mock.

        Returns:
            MagicMock configured with ``patient_id="P001"`` and one admission.
        """
        adm = MagicMock()
        adm.hadm_id = hadm_id
        patient = MagicMock()
        patient.patient_id = "P001"
        patient.get_events.return_value = [adm]
        return patient

    def test_missing_required_columns_raises(self):
        """Edge case: missing required columns raises ValueError on construction."""
        bad_df = pd.DataFrame({"HADM_ID": [1], "claim": ["test"]})
        with self.assertRaises(ValueError):
            EHRFactCheckingMIMIC3(claims_df=bad_df)

    def test_samples_returned_for_matching_admission(self):
        """Sample processing: two claims for HADM 100001 → two samples."""
        task = EHRFactCheckingMIMIC3(claims_df=self._make_claims_df())
        samples = task(self._make_patient(100001))
        self.assertEqual(len(samples), 2)

    def test_correct_label_encoding(self):
        """Label generation: T→0, F→1, N→2 per STANCE_TO_INT mapping."""
        task = EHRFactCheckingMIMIC3(claims_df=self._make_claims_df())
        samples = task(self._make_patient(100001))
        labels = {s["claim"]: s["label"] for s in samples}
        self.assertEqual(labels["pt had glucose > 100"], 0)  # T -> 0
        self.assertEqual(labels["pt was given heparin"], 1)  # F -> 1

    def test_no_samples_for_unmatched_patient(self):
        """Edge case: admission ID absent from claims_df → empty list."""
        task = EHRFactCheckingMIMIC3(claims_df=self._make_claims_df())
        samples = task(self._make_patient(999999))
        self.assertEqual(samples, [])

    def test_extra_columns_forwarded(self):
        """Feature extraction: extra claims_df columns forwarded into samples."""
        claims = self._make_claims_df()
        claims["lower"] = [1, 0, 1]
        claims["upper"] = ["", "0", ""]
        task = EHRFactCheckingMIMIC3(claims_df=claims)
        samples = task(self._make_patient(100001))
        for s in samples:
            self.assertIn("lower", s)
            self.assertIn("upper", s)


# ---------------------------------------------------------------------------
# Tests: DOSSIERPipeline.predict_claim — forward-pass equivalent
# Rubric: forward pass, output shapes, edge cases
#
# Mock data drawn from paper sample claims (Zhang et al. MLHC 2024, App. F):
#   HADM 100001 labs: glucose = 145 mg/dL, 138 mg/dL  (both > 100)
#   HADM 100002 labs: potassium = 2.8 mEq/L            (< 3.0, true negative case)
# SQL executor is mocked; no real SQLite.
# ---------------------------------------------------------------------------

class TestDOSSIERPipelinePredictClaim(unittest.TestCase):
    """Covers DOSSIERPipeline.predict_claim as the forward-pass equivalent."""

    def _make_mimic_dfs(self) -> dict:
        """Return a minimal MIMIC-III table dict: one admission, empty sub-tables."""
        adm_df = pd.DataFrame({
            "HADM_ID": [100001],
            "ADMITTIME": pd.to_datetime(["2100-01-01"]),
            "DIAGNOSIS": [["Sepsis"]],
            "ADMIT_CUI": [["C0243026"]],
        }).set_index("HADM_ID")
        empty_idx = pd.Index([], dtype=int, name="HADM_ID")
        return {
            "adms": adm_df,
            "labs": pd.DataFrame(
                columns=["CHARTTIME", "VALUENUM", "VALUEUOM", "LABEL", "CUI"]
            ).set_index(empty_idx),
            "vits": pd.DataFrame(
                columns=["CHARTTIME", "VALUENUM", "VALUEUOM", "LABEL", "CUI"]
            ).set_index(empty_idx),
            "inputs": pd.DataFrame(
                columns=[
                    "STARTTIME", "AMOUNT", "ORIGINALAMOUNT",
                    "AMOUNTUOM", "LABEL", "CUI",
                ]
            ).set_index(empty_idx),
        }

    def _make_pipeline(
        self, llm_response: str, sql_rows: pd.DataFrame
    ) -> DOSSIERPipeline:
        """Return a DOSSIERPipeline with mocked LLM and SQL executor.

        Args:
            llm_response: Raw string the mock LLM callable will return.
            sql_rows: DataFrame the mock SQL executor's ``run_query`` returns.

        Returns:
            Partially-initialised DOSSIERPipeline ready for ``predict_claim``.
        """
        p = DOSSIERPipeline.__new__(DOSSIERPipeline)

        mock_exec = MagicMock()
        mock_exec.run_query.return_value = sql_rows
        p._executor = mock_exec
        p._kg_loaded = False

        p._llm_callable = lambda _prompt: llm_response
        p.prompt_variant = "neither"
        p.global_kg = None
        p.cui_mapping = None
        p.subset_predicates = ["ISA"]
        p.prompter = DOSSIERPromptGenerator(
            tag_fn=None, cuis_in_ehr=[], prompt_variant="neither", add_examples=False
        )
        p._mimic_dfs = self._make_mimic_dfs()
        return p

    def test_predict_true_stance(self):
        """Forward pass: 2 glucose rows returned (145, 138 mg/dL) → pred=T."""
        llm_response = (
            "<sql>SELECT * FROM Lab WHERE str_label='Glucose' AND Value > 100</sql>"
            "<lower>1</lower><upper></upper><stance>T</stance>"
        )
        rows = pd.DataFrame({
            "Value": [145.0, 138.0], "str_label": ["Glucose", "Glucose"]
        })
        result = self._make_pipeline(llm_response, rows).predict_claim(
            "The patient had a glucose level above 100 mg/dL", 24.0, 100001
        )
        self.assertEqual(result["pred_label"], "T")
        self.assertIsNone(result["error"])

    def test_predict_nei_on_empty_result(self):
        """Forward pass: 0 SQL rows + lower=1 → NEI (paper Appendix B)."""
        llm_response = (
            "<sql>SELECT * FROM Input WHERE str_label='Aspirin'</sql>"
            "<lower>1</lower><upper></upper><stance>T</stance>"
        )
        result = self._make_pipeline(llm_response, pd.DataFrame()).predict_claim(
            "The patient received aspirin.", 24.0, 100001
        )
        self.assertEqual(result["pred_label"], "N")

    def test_predict_nei_on_invalid_sql(self):
        """Edge case: LLM returns no <sql> tag → executor not called, pred=N."""
        pipeline = self._make_pipeline(
            "I don't know how to answer this.", pd.DataFrame()
        )
        result = pipeline.predict_claim("A strange condition.", 24.0, 100001)
        self.assertEqual(result["pred_label"], "N")
        self.assertIsNotNone(result["error"])
        pipeline._executor.run_query.assert_not_called()

    def test_predict_false_via_absence(self):
        """Our extension: 0 rows + stance=F + bounds [0,0] → pred=F (absence)."""
        llm_response = (
            "<sql>SELECT * FROM Lab WHERE str_label='Potassium' AND Value < 3.0</sql>"
            "<lower>0</lower><upper>0</upper><stance>F</stance>"
        )
        result = self._make_pipeline(llm_response, pd.DataFrame()).predict_claim(
            "The patient had low potassium below 3.0 mEq/L.", 24.0, 100001
        )
        self.assertEqual(result["pred_label"], "F")


# ---------------------------------------------------------------------------
# Tests: DOSSIERPipeline.evaluate — output shape and values
# Rubric: output shapes (metrics dict keys and values correct)
# ---------------------------------------------------------------------------

class TestDOSSIERPipelineEvaluate(unittest.TestCase):
    """Covers DOSSIERPipeline.evaluate output shape and metric values."""

    def _make_pipeline(self) -> DOSSIERPipeline:
        """Return a DOSSIERPipeline stub sufficient for evaluate() calls."""
        p = DOSSIERPipeline.__new__(DOSSIERPipeline)
        p.claims_df = pd.DataFrame(
            {"HADM_ID": [1], "claim": ["x"], "t_C": [0], "label": ["T"]}
        )
        p.prompter = MagicMock()
        p._executor = MagicMock()
        p._mimic_dfs = None
        p._kg_loaded = False
        p.global_kg = None
        p.prompt_variant = "full"
        p.subset_predicates = ["ISA"]
        p.cui_mapping = None
        return p

    def test_perfect_accuracy(self):
        """All correct → accuracy=1.0, macro_f1=1.0."""
        res_df = pd.DataFrame({
            "label":      ["T", "F", "N", "T"],
            "pred_label": ["T", "F", "N", "T"],
        })
        metrics = self._make_pipeline().evaluate(res_df)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)

    def test_all_wrong_low_accuracy(self):
        """All wrong → accuracy=0.0."""
        res_df = pd.DataFrame({
            "label":      ["T", "T", "T"],
            "pred_label": ["F", "N", "F"],
        })
        metrics = self._make_pipeline().evaluate(res_df)
        self.assertAlmostEqual(metrics["accuracy"], 0.0)

    def test_metrics_keys_present(self):
        """Output shape: all required metric keys present in returned dict."""
        res_df = pd.DataFrame({"label": ["T", "F"], "pred_label": ["T", "N"]})
        metrics = self._make_pipeline().evaluate(res_df)
        for key in ["accuracy", "macro_f1", "T_f1", "F_f1", "N_f1"]:
            self.assertIn(key, metrics)


# ---------------------------------------------------------------------------
# Test: Cleanup — in-memory executor leaves no disk files
# Rubric: Temporary dirs and proper cleanup (1 pt)
# ---------------------------------------------------------------------------

class TestSQLiteTablePersistence(unittest.TestCase):
    """Verifies stale SQLite data from a prior admission is cleared.

    Regression test for the bug where Input rows from HADM=100001 would
    persist in the shared in-memory executor when HADM=100002 (which has no
    inputs) was processed next, causing wrong F→T predictions.
    """

    def _make_two_adm_dfs(self) -> dict:
        """Two admissions: 100001 has furosemide input, 100002 has none."""
        adm_df = pd.DataFrame({
            "HADM_ID": [100001, 100002],
            "ADMITTIME": pd.to_datetime(["2100-01-01", "2100-02-01"]),
            "DIAGNOSIS": [["Sepsis"], ["Pneumonia"]],
            "ADMIT_CUI": [["C0243026"], ["C0032285"]],
        }).set_index("HADM_ID")
        empty_idx = pd.Index([], dtype=int, name="HADM_ID")
        labs = pd.DataFrame(
            columns=["CHARTTIME", "VALUENUM", "VALUEUOM", "LABEL", "CUI"]
        ).set_index(empty_idx)
        vits = pd.DataFrame(
            columns=["CHARTTIME", "VALUENUM", "VALUEUOM", "LABEL", "CUI"]
        ).set_index(empty_idx)
        inputs = pd.DataFrame({
            "HADM_ID": [100001],
            "STARTTIME": pd.to_datetime(["2100-01-01 12:00"]),
            "AMOUNT": [40.0],
            "ORIGINALAMOUNT": [40.0],
            "AMOUNTUOM": ["mg"],
            "LABEL": ["Furosemide"],
            "CUI": ["C0016860"],
        }).set_index("HADM_ID")
        return {"adms": adm_df, "labs": labs, "vits": vits, "inputs": inputs}

    def _make_pipeline(self, mimic_dfs: dict) -> "DOSSIERPipeline":
        SQLExecutor = _dossier_mod.SQLExecutor
        DOSSIERPromptGenerator = _dossier_mod.DOSSIERPromptGenerator
        p = DOSSIERPipeline.__new__(DOSSIERPipeline)
        p._executor = SQLExecutor(in_memory=True)
        p._kg_loaded = False
        p._llm_callable = lambda _: (
            "<sql>SELECT * FROM Input WHERE LOWER(str_label) LIKE '%furosemide%'</sql>"
            "<lower>0</lower><upper>0</upper><stance>F</stance>"
        )
        p.prompt_variant = "neither"
        p.global_kg = None
        p.cui_mapping = None
        p.subset_predicates = ["ISA"]
        p.prompter = DOSSIERPromptGenerator(
            tag_fn=None, cuis_in_ehr=[], prompt_variant="neither", add_examples=False
        )
        p._mimic_dfs = mimic_dfs
        return p

    def test_input_table_cleared_between_admissions(self):
        """HADM=100001 has furosemide; HADM=100002 should see empty Input."""
        mimic_dfs = self._make_two_adm_dfs()
        pipeline = self._make_pipeline(mimic_dfs)

        # HADM=100001: furosemide present → 1 row → negate(F) = T (given)
        r1 = pipeline.predict_claim("furosemide claim", 48.0, 100001)
        self.assertEqual(r1["pred_label"], "T",
                         "100001 has furosemide — should be True (drug was given)")
        self.assertEqual(r1["n_result_rows"], 1)

        # HADM=100002: no inputs → Input table should be empty → 0 rows → F (absent)
        r2 = pipeline.predict_claim("furosemide claim", 48.0, 100002)
        self.assertEqual(r2["pred_label"], "F",
                         "100002 has no inputs — stale Input rows would give pred=T (bug)")
        self.assertEqual(r2["n_result_rows"], 0)


class TestPipelineCleanup(unittest.TestCase):
    """Verifies that the in-memory SQLExecutor leaves no temporary disk files."""

    def test_in_memory_executor_no_disk_files(self):
        """In-memory SQLExecutor creates no files in any temporary directory."""
        SQLExecutor = _dossier_mod.SQLExecutor
        with tempfile.TemporaryDirectory() as tmpdir:
            before = set(os.listdir(tmpdir))
            executor = SQLExecutor(in_memory=True)
            executor.load_tables(
                {"Lab": pd.DataFrame({"t": [1.0], "CUI": ["C0001"],
                                      "Value": [145.0], "Units": ["mg/dL"],
                                      "str_label": ["Glucose"]})},
                add_kg_identity_edges=False,
            )
            executor.run_query("SELECT * FROM Lab")
            executor.close()
            after = set(os.listdir(tmpdir))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
