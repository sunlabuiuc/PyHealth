"""Automatic, guided improvement of PyHealth pipelines.

An :class:`AutoImprover` searches over model, argument and training choices,
guided by an LLM (:class:`LLMProposer`) or at random (:class:`RandomProposer`),
scoring every proposal on validation patients and keeping the test split for
one final comparison. Among the configs that beat the starting pipeline, it
selects the most interpretable one within noise of the best, and its report
(:meth:`AutoImprover.report`) explains the choice and how the search got there.
"""

from .evaluate import (
    check_models,
    describe_error,
    fold_parts,
    patient_folds,
    predict,
    run_config,
    run_config_cv,
    score,
    train_pipeline,
)
from .features import EngineeredDataset, FeatureSpace, FeatureSpec
from .improver import (
    DEFAULT_METRIC,
    AutoImprover,
    ImproveResult,
    Trial,
    paired_gain,
    select_simplest,
    select_trial,
)
from .knowledge import Document, Evidence, KnowledgeBase, parse_effect, pubmed_query
from .llm import LLM, ClaudeLLM
from .proposers import (
    STRATEGIES,
    LLMProposer,
    Proposer,
    ProposerExhausted,
    QueueProposer,
    RandomProposer,
)
from .report import Report, build_report, load_run, unverified_numbers
from .space import (
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
    "DEFAULT_METRIC",
    "DEFAULT_MODELS",
    "LADDER",
    "LLM",
    "STRATEGIES",
    "TRAIN_PARAMS",
    "AutoImprover",
    "ClaudeLLM",
    "Document",
    "EngineeredDataset",
    "Evidence",
    "FeatureSpace",
    "FeatureSpec",
    "ImproveResult",
    "KnowledgeBase",
    "LLMProposer",
    "ModelSpec",
    "Param",
    "PipelineConfig",
    "Proposer",
    "ProposerExhausted",
    "QueueProposer",
    "RandomProposer",
    "Report",
    "SearchSpace",
    "Trial",
    "build_report",
    "check_models",
    "config_diff",
    "describe_error",
    "fold_parts",
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
