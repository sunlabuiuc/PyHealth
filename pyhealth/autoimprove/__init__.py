"""Automatic, guided improvement of PyHealth pipelines.

An :class:`AutoImprover` searches over model, argument and training choices,
guided by an LLM (:class:`LLMProposer`) or at random (:class:`RandomProposer`),
scoring every proposal on validation patients and keeping the test split for
one final comparison. Among the configs that beat the starting pipeline, it
selects the most interpretable one within noise of the best, and its report
(:meth:`AutoImprover.report`) explains the choice and how the search got there.
"""

from .analysis import AnalysisSpec, sample_columns
from .codegen import CODE_CONTRACT, task_context
from .evaluate import (
    check_models,
    describe_error,
    fit_predict,
    fold_parts,
    out_of_fold,
    patient_folds,
    predict,
    run_config,
    run_config_cv,
    score,
    score_predictions,
    train_pipeline,
)
from .guardrails import EvaluationSpec, clearly_worse, metrics_menu
from .evidence import (
    EvidenceProposer,
    add_literature,
    evidence_lines,
    literature_section,
)
from .features import EngineeredDataset, FeatureSpace, FeatureSpec
from .improver import (
    DEFAULT_METRIC,
    TURN_KINDS,
    AutoImprover,
    ImproveResult,
    Trial,
    paired_gain,
    select_simplest,
    select_trial,
)
from .knowledge import Document, Evidence, KnowledgeBase, parse_effect, pubmed_query
from .llm import LLM, ClaudeLLM, GeminiLLM, OpenAILLM
from .proposers import (
    STRATEGIES,
    LLMProposer,
    Proposer,
    ProposerExhausted,
    QueueProposer,
    RandomProposer,
)
from .report import Report, build_report, load_run, unverified_numbers
from .sandbox import SandboxError, run_sandboxed, static_check
from .space import (
    CODE_MODEL,
    DEFAULT_MODELS,
    LADDER,
    TRAIN_PARAMS,
    ModelSpec,
    Param,
    PipelineConfig,
    SearchSpace,
    config_diff,
    with_xgboost,
)

__all__ = [
    "CODE_CONTRACT",
    "CODE_MODEL",
    "DEFAULT_METRIC",
    "DEFAULT_MODELS",
    "LADDER",
    "LLM",
    "STRATEGIES",
    "TRAIN_PARAMS",
    "TURN_KINDS",
    "AnalysisSpec",
    "AutoImprover",
    "EvaluationSpec",
    "SandboxError",
    "clearly_worse",
    "fit_predict",
    "metrics_menu",
    "out_of_fold",
    "run_sandboxed",
    "sample_columns",
    "score_predictions",
    "static_check",
    "task_context",
    "ClaudeLLM",
    "Document",
    "EngineeredDataset",
    "Evidence",
    "EvidenceProposer",
    "FeatureSpace",
    "FeatureSpec",
    "GeminiLLM",
    "ImproveResult",
    "KnowledgeBase",
    "LLMProposer",
    "ModelSpec",
    "OpenAILLM",
    "Param",
    "PipelineConfig",
    "Proposer",
    "ProposerExhausted",
    "QueueProposer",
    "RandomProposer",
    "Report",
    "SearchSpace",
    "Trial",
    "add_literature",
    "build_report",
    "check_models",
    "config_diff",
    "describe_error",
    "evidence_lines",
    "fold_parts",
    "literature_section",
    "load_run",
    "paired_gain",
    "parse_effect",
    "patient_folds",
    "predict",
    "pubmed_query",
    "run_config",
    "run_config_cv",
    "score",
    "select_simplest",
    "select_trial",
    "train_pipeline",
    "unverified_numbers",
    "with_xgboost",
]
