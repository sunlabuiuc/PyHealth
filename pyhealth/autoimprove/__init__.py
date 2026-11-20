"""Automatic, guided improvement of PyHealth pipelines.

An :class:`AutoImprover` searches over model, argument and training choices,
guided by an LLM (:class:`LLMProposer`) or at random (:class:`RandomProposer`),
scoring every proposal on validation patients and keeping the test split for
one final comparison. Among the configs that beat the starting pipeline, it
selects the most interpretable one within noise of the best, and its report
(:meth:`AutoImprover.report`) explains the choice and how the search got there.
"""

from .evaluation.guardrails import EvaluationSpec, clearly_worse, metrics_menu
from .evaluation.records import ImproveResult, Trial
from .evaluation.report import Report, build_report, load_run, unverified_numbers
from .evaluation.selection import paired_gain, select_simplest, select_trial
from .evaluation.training import (
    check_models,
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
from .config import RunConfig
from .improver import DEFAULT_METRIC, AutoImprover
from .search.analysis import AnalysisSpec, sample_columns
from .search.features import EngineeredDataset, FeatureSpace, FeatureSpec
from .search.literature import (
    Document,
    Evidence,
    EvidenceProposer,
    KnowledgeBase,
    add_literature,
    evidence_lines,
    literature_section,
    parse_effect,
    pubmed_query,
)
from .search.prompts import CODE_CONTRACT, STRATEGIES, task_context
from .search.proposers import (
    LLMProposer,
    Proposer,
    ProposerExhausted,
    QueueProposer,
    RandomProposer,
)
from .search.space import (
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
from .search.turns import TURN_KINDS
from .utils.data import describe_error
from .utils.llm import LLM, ClaudeLLM, GeminiLLM, OpenAILLM
from .utils.sandbox import SandboxError, run_sandboxed, static_check

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
    "RunConfig",
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
