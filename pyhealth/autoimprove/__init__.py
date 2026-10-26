"""Automatic, guided improvement of PyHealth pipelines.

An :class:`AutoImprover` searches over model, argument and training choices,
guided by an LLM (:class:`LLMProposer`) or at random (:class:`RandomProposer`),
scoring every proposal on validation patients and keeping the test split for
one final comparison.
"""

from .evaluate import (
    check_models,
    describe_error,
    patient_folds,
    predict,
    run_config,
    run_config_cv,
    score,
    train_pipeline,
)
from .improver import (
    DEFAULT_METRIC,
    AutoImprover,
    ImproveResult,
    Trial,
    paired_gain,
    select_trial,
)
from .llm import LLM, ClaudeLLM
from .proposers import STRATEGIES, LLMProposer, Proposer, RandomProposer
from .space import (
    DEFAULT_MODELS,
    TRAIN_PARAMS,
    ModelSpec,
    Param,
    PipelineConfig,
    SearchSpace,
    with_xgboost,
)

__all__ = [
    "DEFAULT_METRIC",
    "DEFAULT_MODELS",
    "LLM",
    "STRATEGIES",
    "TRAIN_PARAMS",
    "AutoImprover",
    "ClaudeLLM",
    "ImproveResult",
    "LLMProposer",
    "ModelSpec",
    "Param",
    "PipelineConfig",
    "Proposer",
    "RandomProposer",
    "SearchSpace",
    "Trial",
    "check_models",
    "describe_error",
    "paired_gain",
    "patient_folds",
    "predict",
    "run_config",
    "run_config_cv",
    "score",
    "select_trial",
    "train_pipeline",
    "with_xgboost",
]
