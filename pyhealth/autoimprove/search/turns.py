"""The kinds of turn: what one agent does when it is its turn to propose.

Each function takes the :class:`~pyhealth.autoimprove.AutoImprover` running the
search as its first argument, asks its proposer for one attempt, checks it,
and hands it to the improver to score (or, for evaluation and analysis turns,
runs it).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from ..evaluation import guardrails
from ..evaluation.guardrails import decide, metrics_menu, run_evaluations
from ..evaluation.records import Trial
from ..evaluation.selection import _one_se_sets, paired_gain
from ..utils.data import describe_error
from ..utils.sandbox import SandboxError
from .analysis import format_results, run_analysis_job
from .features import FeatureSpec
from .prompts import task_context
from .proposers import ProposalContext, ProposerExhausted
from .space import _derived, _key


logger = logging.getLogger(__name__)

MAX_PROPOSER_FAILURES = 3
"""Proposer errors in a row (API errors, refusals) before a run stops."""


TURN_KINDS = ("model", "features", "code", "evaluation", "analysis")
"""The kinds of turn a ``cycle_plan`` can list."""


GENERATED_KINDS = ("code", "evaluation", "analysis")
"""Turn kinds whose proposals are code that runs in its own process."""


def model_turn(state, cycle: int, agent: str, strategy: str, kind: str = "model") -> None:
    """A model turn, or a code turn (``kind="code"``): propose, check, score."""
    feedback = ""
    accepted = state._accepted()
    propose = (state.proposer.propose_code if kind == "code"
               else state.proposer.propose)
    for _ in range(2):  # one retry when the proposal is invalid or repeated
        context = build_context(state, agent, strategy, feedback, kind=kind)
        try:
            proposal = propose(context)
        except ValueError as e:
            feedback = f"your last proposal was rejected: {e}"
            continue
        except Exception as e:  # noqa: BLE001 - an API error or refusal
            skip_turn(state, agent, cycle, e)
            return
        state._proposer_failures = 0
        try:
            config, notes = state.space.validate(proposal)
        except ValueError as e:
            feedback = f"your last proposal was rejected: {e}"
            continue
        if _key(config, accepted) in context.tried:
            feedback = "your last proposal repeated an earlier trial"
            if notes:
                feedback += (
                    " once it was changed to fit the search space ("
                    + "; ".join(notes)
                    + "). Stay inside the ranges listed above"
                )
            continue
        state._evaluate(config, cycle, agent, strategy, notes, features=accepted)
        return
    logger.warning(
        "%s made no usable proposal in cycle %d: %s", agent, cycle, feedback
    )


def skip_turn(state, agent: str, cycle: int, error: Exception) -> None:
    """Skips a turn whose proposer failed, e.g. an API error or a refusal.

    Raises:
        RuntimeError: After :data:`MAX_PROPOSER_FAILURES` failures in a
            row (e.g. missing credentials), so a broken backend does not
            spend the whole budget.
    """
    state._proposer_failures += 1
    text = describe_error(error)
    if state._proposer_failures >= MAX_PROPOSER_FAILURES:
        raise RuntimeError(
            f"the proposer failed {state._proposer_failures} turns in a row; "
            f"last error: {text}"
        ) from error
    logger.warning("%s's proposer failed in cycle %d, turn skipped: %s",
                   agent, cycle, text)


def feature_turn(state, cycle: int, agent: str, strategy: str) -> None:
    """One feature proposal: validate, test for leaks, score, accept or not."""
    base = state._incumbent(cycle)
    if base is None:
        return
    accepted = state._accepted()
    tried = {name for t in state.trials for name in t.added}
    have = {s["name"] for s in accepted}
    feedback = ""
    for _ in range(2):  # one retry when the proposal is invalid
        context = build_context(state, agent, strategy, feedback, kind="features")
        try:
            proposal = state.proposer.propose_features(context)
        except ProposerExhausted:
            state._exhausted = True
            return
        except ValueError as e:
            feedback = f"your last proposal was rejected: {e}"
            continue
        except Exception as e:  # noqa: BLE001 - an API error or refusal
            skip_turn(state, agent, cycle, e)
            return
        state._proposer_failures = 0
        try:
            specs = state.features.validate_many(list(proposal.get("features") or []))
        except ValueError as e:
            feedback = f"your last proposal was rejected: {e}"
            continue
        new = [s for s in specs if s.name not in have and s.name not in tried]
        if not new:
            feedback = ("every feature in your last proposal was already tried ("
                        + ", ".join(s.name for s in specs) + "); propose new ones")
            continue
        leaks = {k: v for k, v in state.features.future_blind(
            new, state.train_dataset).items() if v}
        if leaks:
            feedback = "the future-blind test rejected " + "; ".join(
                f"{k}: {v}" for k, v in leaks.items())
            continue
        hypothesis = {k: proposal[k] for k in ("expected", "kind") if proposal.get(k)}
        config = _derived(base.config, str(proposal.get("rationale", "")),
                          hypothesis)
        trial = state._evaluate(
            config, cycle, agent, strategy, features=accepted + [
                s.to_dict() for s in new], added=[s.name for s in new],
            parent=base, save=False,
        )
        judge(state, trial, base, new)
        state._save_trial(trial)
        return
    logger.warning(
        "%s made no usable feature proposal in cycle %d: %s", agent, cycle, feedback
    )


def judge(state, trial: Trial, base: Trial, new: Sequence[FeatureSpec]) -> None:
    """Accepts a feature group if its paired gain over ``base`` clears the bound."""
    if trial.error is not None or trial.stopped_early:
        trial.accepted = False
        return
    labels = state._labels()
    for spec in new:
        column = state._raw(spec, "train")
        filled = np.where(np.isnan(column), np.nanmedian(column)
                          if np.isfinite(column).any() else 0.0, column)
        if len(np.unique(labels)) > 1 and np.ptp(filled) > 0:
            auc = roc_auc_score(labels, filled)
            if max(auc, 1 - auc) > state.suspicion_auc:
                trial.suspicious.append(spec.name)
                trial.notes.append(
                    f"{spec.name} alone has ROC-AUC {max(auc, 1 - auc):.3f}: "
                    "possible leakage; held for review"
                )
    gain, se = paired_gain(trial.fold_scores, base.fold_scores, state.maximize)
    bound = gain - (state.promote_se or 1.0) * se
    trial.accepted = bound > 0 and not trial.suspicious
    trial.notes.append(
        f"vs #{base.trial_id} without the new features: {gain:+.4f} ± {se:.4f}; "
        + ("accepted" if trial.accepted else "not accepted")
    )


def evaluation_turn(state, cycle: int, agent: str) -> None:
    """One evaluation proposal: check it, score every trial, adopt or not."""
    feedback = ""
    for _ in range(2):  # one retry when the proposal is invalid
        context = build_context(state, agent, "explore", feedback, kind="evaluation")
        try:
            spec = state.proposer.propose_evaluation(context)
        except ValueError as e:
            feedback = f"your last proposal was rejected: {e}"
            continue
        except Exception as e:  # noqa: BLE001 - an API error or refusal
            skip_turn(state, agent, cycle, e)
            return
        state._proposer_failures = 0
        unknown = [c for c in spec.needs if c not in state._columns()]
        if unknown:
            feedback = (f"your last proposal needs unknown columns {unknown}; "
                        f"use only {sorted(state._columns())}")
            continue
        if any(e["spec"].name == spec.name for e in state.evaluations):
            feedback = f"an evaluation named {spec.name!r} exists; propose another"
            continue
        record = {"spec": spec, "status": "rejected", "reason": "",
                  "scores": {}, "cycle": cycle, "agent": agent}
        trials = state._scorable()
        base_id = state._baseline_id()
        if base_id is None or base_id not in {t.trial_id for t in trials}:
            record["reason"] = "no baseline with stored out-of-fold predictions"
        else:
            try:
                result = run_evaluations(
                    spec, *guardrails.evaluation_inputs(state, spec, trials), checks=True,
                    timeout=state.eval_timeout,
                    log_path=state.store.log_path(f"evaluation_{spec.name}"))
            except SandboxError as e:
                record["reason"] = e.brief
            else:
                ids = [t.trial_id for t in trials]
                record["scores"] = dict(zip(ids, result["values"]))
                fails = state._guardrail_fails()
                candidates, _, _ = _one_se_sets(
                    [t for t in state.trials if t.accepted is not False
                     and not fails(t)],
                    state.maximize, state.promote_se or 1.0)
                record["status"], record["reason"] = decide(
                    spec, result, ids, base_id,
                    [t.trial_id for t in candidates if t.trial_id != base_id],
                    sum(e["status"] == "active" for e in state.evaluations),
                    state.max_guardrails, state.guardrail_se)
        state.evaluations.append(record)
        guardrails.save_evaluations(state)
        logger.info("evaluation %s: %s (%s)", spec.name, record["status"],
                    record["reason"])
        return
    logger.warning("%s made no usable evaluation in cycle %d: %s", agent, cycle,
                   feedback)


def analysis_turn(state, cycle: int, agent: str) -> None:
    """One analysis: run it on the training split, keep its findings."""
    feedback = ""
    for _ in range(2):  # one retry when the proposal is invalid or fails
        context = build_context(state, agent, "explore", feedback, kind="analysis")
        try:
            spec = state.proposer.propose_analysis(context)
        except ValueError as e:
            feedback = f"your last proposal was rejected: {e}"
            continue
        except Exception as e:  # noqa: BLE001 - an API error or refusal
            skip_turn(state, agent, cycle, e)
            return
        state._proposer_failures = 0
        number = len(state.analyses)
        if state.run_dir:
            state.store.write_text(os.path.join("analysis", f"{number}_{spec.name}.py"),
                        spec.code)
        try:
            results = run_analysis_job(
                spec, state._analysis_columns(), timeout=state.eval_timeout,
                log_path=state.store.log_path(f"analysis_{number}_{spec.name}"))
        except SandboxError as e:
            feedback = f"your last analysis failed: {e.brief}"
            continue
        text = format_results(results)
        try:
            found = state.proposer.summarize_analysis(spec, text, context)
        except Exception as e:  # noqa: BLE001 - keep a plain summary instead
            logger.warning("findings not summarised: %s", describe_error(e))
            found = []
        if not found and text:
            found = [f"{spec.question or spec.name}: "
                     + " ".join(text.split())[:280]]
        state.findings = (state.findings + list(found))[-state.max_findings:]
        record = {**spec.to_dict(), "cycle": cycle, "agent": agent,
                  "results": results, "findings": list(found)}
        state.analyses.append(record)
        if state.run_dir:
            state.store.write_text(os.path.join("analysis", f"{number}_{spec.name}.json"),
                        json.dumps(record, indent=1, default=str))
            state.store.write_text("findings.json", json.dumps(state.findings, indent=1))
        return
    logger.warning("%s made no usable analysis in cycle %d: %s", agent, cycle,
                   feedback)


def build_context(state, agent, strategy, feedback, kind="model") -> ProposalContext:
    result = state.result()
    done = result.leaderboard(k=len(state.trials))
    card = ""
    if state.task_card is not None:
        card = "\n".join(f"- {name}: {item['text']}"
                          for name, item in state.task_card.design())
    return ProposalContext(
        agent=agent,
        strategy=strategy,
        space=state.space,
        task=state._task,
        metric=state.metric,
        maximize=state.maximize,
        own_trials=[t for t in state.trials if t.agent == agent][-8:],
        leaderboard=done[:8],
        worst=list(reversed(done[8:]))[:3] if len(done) > 8 else [],
        tried={_key(t.config, t.features) for t in state.trials},
        lessons=state.lessons,
        feedback=feedback,
        selected=result.best,
        selection_rule=state.selection_rule() if state.promote_se is not None else "",
        kind=kind,
        features=state.features,
        accepted=state._accepted(),
        feature_trials=[t for t in state.trials if t.added][-12:],
        card=card,
        tried_features=sorted({n for t in state.trials for n in t.added}),
        findings=list(state.findings),
        evaluations=evaluation_lines(state),
        **(generated_context(state, kind, done) if kind in GENERATED_KINDS else {}),
    )


def generated_context(state, kind: str, done: Sequence[Trial]) -> dict[str, Any]:
    """Context only code, evaluation and analysis prompts need."""
    if state._task_source is None:
        state._task_source = task_context(state.task, state.train_dataset)
    out: dict[str, Any] = {"task_source": state._task_source,
                           "columns": sorted(state._analysis_columns())}
    if kind == "evaluation":
        out["metrics_menu"] = metrics_menu(state.mode)
        out["columns"] = sorted(c for c in state._columns() if c != "patient")
    if kind == "code":
        best_code = [t for t in done if t.config.code][:2]
        out["code_examples"] = [
            (f"#{t.trial_id} [{state.metric}={t.score:.4f}]", t.config.code)
            for t in best_code
        ]
    return out


def evaluation_lines(state) -> list[str]:
    lines = []
    fails = state._guardrail_fails()
    for e in state.evaluations:
        spec = e["spec"]
        metrics = ", ".join(f"{m} ({d} is better)" for m, d in spec.metrics.items())
        line = f"{spec.name} [{e['status']}]: {metrics}"
        if e["status"] == "active":
            failing = [t.trial_id for t in state.trials if t.error is None and any(
                k.startswith(spec.name + ".") for k in fails(t))]
            line += ("; fails: " + ", ".join(f"#{i}" for i in failing)
                     if failing else "; no trial fails it")
        elif e.get("reason"):
            line += f"; {e['reason']}"
        lines.append(line)
    return lines


def take_turn(state, cycle: int, kind: str, agent: str, strategy: str) -> None:
    """One agent's turn in a round of this kind."""
    if kind == "features":
        feature_turn(state, cycle, agent, strategy)
    elif kind == "evaluation":
        evaluation_turn(state, cycle, agent)
    elif kind == "analysis":
        analysis_turn(state, cycle, agent)
    else:  # "model" or "code"
        model_turn(state, cycle, agent, strategy, kind=kind)


ADDS_NO_TRIALS = ("evaluation", "analysis")
"""Kinds of round that add no trials, so they don't count towards patience."""
