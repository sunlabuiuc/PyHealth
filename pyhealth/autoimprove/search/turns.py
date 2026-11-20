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
from ..utils.errors import describe_error
from ..utils.sandbox import SandboxError
from .analysis import format_results, run_analysis_job
from .codegen import task_context
from .context import ProposalContext
from .features import FeatureSpec
from .proposers import ProposerExhausted
from .space import _derived, _key


logger = logging.getLogger(__name__)

MAX_PROPOSER_FAILURES = 3
"""Proposer errors in a row (API errors, refusals) before a run stops."""


TURN_KINDS = ("model", "features", "code", "evaluation", "analysis")
"""The kinds of turn a ``cycle_plan`` can list."""


GENERATED_KINDS = ("code", "evaluation", "analysis")
"""Turn kinds whose proposals are code that runs in its own process."""


def model_turn(imp, cycle: int, agent: str, strategy: str, kind: str = "model") -> None:
    """A model turn, or a code turn (``kind="code"``): propose, check, score."""
    feedback = ""
    accepted = imp._accepted()
    propose = (imp.proposer.propose_code if kind == "code"
               else imp.proposer.propose)
    for _ in range(2):  # one retry when the proposal is invalid or repeated
        context = build_context(imp, agent, strategy, feedback, kind=kind)
        try:
            proposal = propose(context)
        except ValueError as e:
            feedback = f"your last proposal was rejected: {e}"
            continue
        except Exception as e:  # noqa: BLE001 - an API error or refusal
            skip_turn(imp, agent, cycle, e)
            return
        imp._proposer_failures = 0
        try:
            config, notes = imp.space.validate(proposal)
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
        imp._evaluate(config, cycle, agent, strategy, notes, features=accepted)
        return
    logger.warning(
        "%s made no usable proposal in cycle %d: %s", agent, cycle, feedback
    )


def skip_turn(imp, agent: str, cycle: int, error: Exception) -> None:
    """Skips a turn whose proposer failed, e.g. an API error or a refusal.

    Raises:
        RuntimeError: After :data:`MAX_PROPOSER_FAILURES` failures in a
            row (e.g. missing credentials), so a broken backend does not
            spend the whole budget.
    """
    imp._proposer_failures += 1
    text = describe_error(error)
    if imp._proposer_failures >= MAX_PROPOSER_FAILURES:
        raise RuntimeError(
            f"the proposer failed {imp._proposer_failures} turns in a row; "
            f"last error: {text}"
        ) from error
    logger.warning("%s's proposer failed in cycle %d, turn skipped: %s",
                   agent, cycle, text)


def feature_turn(imp, cycle: int, agent: str, strategy: str) -> None:
    """One feature proposal: validate, test for leaks, score, accept or not."""
    base = imp._incumbent(cycle)
    if base is None:
        return
    accepted = imp._accepted()
    tried = {name for t in imp.trials for name in t.added}
    have = {s["name"] for s in accepted}
    feedback = ""
    for _ in range(2):  # one retry when the proposal is invalid
        context = build_context(imp, agent, strategy, feedback, kind="features")
        try:
            proposal = imp.proposer.propose_features(context)
        except ProposerExhausted:
            imp._exhausted = True
            return
        except ValueError as e:
            feedback = f"your last proposal was rejected: {e}"
            continue
        except Exception as e:  # noqa: BLE001 - an API error or refusal
            skip_turn(imp, agent, cycle, e)
            return
        imp._proposer_failures = 0
        try:
            specs = imp.features.validate_many(list(proposal.get("features") or []))
        except ValueError as e:
            feedback = f"your last proposal was rejected: {e}"
            continue
        new = [s for s in specs if s.name not in have and s.name not in tried]
        if not new:
            feedback = ("every feature in your last proposal was already tried ("
                        + ", ".join(s.name for s in specs) + "); propose new ones")
            continue
        leaks = {k: v for k, v in imp.features.future_blind(
            new, imp.train_dataset).items() if v}
        if leaks:
            feedback = "the future-blind test rejected " + "; ".join(
                f"{k}: {v}" for k, v in leaks.items())
            continue
        hypothesis = {k: proposal[k] for k in ("expected", "kind") if proposal.get(k)}
        config = _derived(base.config, str(proposal.get("rationale", "")),
                          hypothesis)
        trial = imp._evaluate(
            config, cycle, agent, strategy, features=accepted + [
                s.to_dict() for s in new], added=[s.name for s in new],
            parent=base, save=False,
        )
        judge(imp, trial, base, new)
        imp._save_trial(trial)
        return
    logger.warning(
        "%s made no usable feature proposal in cycle %d: %s", agent, cycle, feedback
    )


def judge(imp, trial: Trial, base: Trial, new: Sequence[FeatureSpec]) -> None:
    """Accepts a feature group if its paired gain over ``base`` clears the bound."""
    if trial.error is not None or trial.stopped_early:
        trial.accepted = False
        return
    labels = imp._labels()
    for spec in new:
        column = imp._raw(spec, "train")
        filled = np.where(np.isnan(column), np.nanmedian(column)
                          if np.isfinite(column).any() else 0.0, column)
        if len(np.unique(labels)) > 1 and np.ptp(filled) > 0:
            auc = roc_auc_score(labels, filled)
            if max(auc, 1 - auc) > imp.suspicion_auc:
                trial.suspicious.append(spec.name)
                trial.notes.append(
                    f"{spec.name} alone has ROC-AUC {max(auc, 1 - auc):.3f}: "
                    "possible leakage; held for review"
                )
    gain, se = paired_gain(trial.fold_scores, base.fold_scores, imp.maximize)
    bound = gain - (imp.promote_se or 1.0) * se
    trial.accepted = bound > 0 and not trial.suspicious
    trial.notes.append(
        f"vs #{base.trial_id} without the new features: {gain:+.4f} ± {se:.4f}; "
        + ("accepted" if trial.accepted else "not accepted")
    )


def evaluation_turn(imp, cycle: int, agent: str) -> None:
    """One evaluation proposal: check it, score every trial, adopt or not."""
    feedback = ""
    for _ in range(2):  # one retry when the proposal is invalid
        context = build_context(imp, agent, "explore", feedback, kind="evaluation")
        try:
            spec = imp.proposer.propose_evaluation(context)
        except ValueError as e:
            feedback = f"your last proposal was rejected: {e}"
            continue
        except Exception as e:  # noqa: BLE001 - an API error or refusal
            skip_turn(imp, agent, cycle, e)
            return
        imp._proposer_failures = 0
        unknown = [c for c in spec.needs if c not in imp._columns()]
        if unknown:
            feedback = (f"your last proposal needs unknown columns {unknown}; "
                        f"use only {sorted(imp._columns())}")
            continue
        if any(e["spec"].name == spec.name for e in imp.evaluations):
            feedback = f"an evaluation named {spec.name!r} exists; propose another"
            continue
        record = {"spec": spec, "status": "rejected", "reason": "",
                  "scores": {}, "cycle": cycle, "agent": agent}
        trials = imp._scorable()
        base_id = imp._baseline_id()
        if base_id is None or base_id not in {t.trial_id for t in trials}:
            record["reason"] = "no baseline with stored out-of-fold predictions"
        else:
            try:
                result = run_evaluations(
                    spec, *guardrails.evaluation_inputs(imp, spec, trials), checks=True,
                    timeout=imp.eval_timeout,
                    log_path=imp.store.log_path(f"evaluation_{spec.name}"))
            except SandboxError as e:
                record["reason"] = e.brief
            else:
                ids = [t.trial_id for t in trials]
                record["scores"] = dict(zip(ids, result["values"]))
                fails = imp._guardrail_fails()
                candidates, _, _ = _one_se_sets(
                    [t for t in imp.trials if t.accepted is not False
                     and not fails(t)],
                    imp.maximize, imp.promote_se or 1.0)
                record["status"], record["reason"] = decide(
                    spec, result, ids, base_id,
                    [t.trial_id for t in candidates if t.trial_id != base_id],
                    sum(e["status"] == "active" for e in imp.evaluations),
                    imp.max_guardrails, imp.guardrail_se)
        imp.evaluations.append(record)
        guardrails.save_evaluations(imp)
        logger.info("evaluation %s: %s (%s)", spec.name, record["status"],
                    record["reason"])
        return
    logger.warning("%s made no usable evaluation in cycle %d: %s", agent, cycle,
                   feedback)


def analysis_turn(imp, cycle: int, agent: str) -> None:
    """One analysis: run it on the training split, keep its findings."""
    feedback = ""
    for _ in range(2):  # one retry when the proposal is invalid or fails
        context = build_context(imp, agent, "explore", feedback, kind="analysis")
        try:
            spec = imp.proposer.propose_analysis(context)
        except ValueError as e:
            feedback = f"your last proposal was rejected: {e}"
            continue
        except Exception as e:  # noqa: BLE001 - an API error or refusal
            skip_turn(imp, agent, cycle, e)
            return
        imp._proposer_failures = 0
        number = len(imp.analyses)
        if imp.run_dir:
            imp.store.write_text(os.path.join("analysis", f"{number}_{spec.name}.py"),
                        spec.code)
        try:
            results = run_analysis_job(
                spec, imp._analysis_columns(), timeout=imp.eval_timeout,
                log_path=imp.store.log_path(f"analysis_{number}_{spec.name}"))
        except SandboxError as e:
            feedback = f"your last analysis failed: {e.brief}"
            continue
        text = format_results(results)
        try:
            found = imp.proposer.summarize_analysis(spec, text, context)
        except Exception as e:  # noqa: BLE001 - keep a plain summary instead
            logger.warning("findings not summarised: %s", describe_error(e))
            found = []
        if not found and text:
            found = [f"{spec.question or spec.name}: "
                     + " ".join(text.split())[:280]]
        imp.findings = (imp.findings + list(found))[-imp.max_findings:]
        record = {**spec.to_dict(), "cycle": cycle, "agent": agent,
                  "results": results, "findings": list(found)}
        imp.analyses.append(record)
        if imp.run_dir:
            imp.store.write_text(os.path.join("analysis", f"{number}_{spec.name}.json"),
                        json.dumps(record, indent=1, default=str))
            imp.store.write_text("findings.json", json.dumps(imp.findings, indent=1))
        return
    logger.warning("%s made no usable analysis in cycle %d: %s", agent, cycle,
                   feedback)


def build_context(imp, agent, strategy, feedback, kind="model") -> ProposalContext:
    result = imp.result()
    done = result.leaderboard(k=len(imp.trials))
    card = ""
    if imp.task_card is not None:
        card = "\n".join(f"- {name}: {item['text']}"
                          for name, item in imp.task_card.design())
    return ProposalContext(
        agent=agent,
        strategy=strategy,
        space=imp.space,
        task=imp._task,
        metric=imp.metric,
        maximize=imp.maximize,
        own_trials=[t for t in imp.trials if t.agent == agent][-8:],
        leaderboard=done[:8],
        worst=list(reversed(done[8:]))[:3] if len(done) > 8 else [],
        tried={_key(t.config, t.features) for t in imp.trials},
        lessons=imp.lessons,
        feedback=feedback,
        selected=result.best,
        selection_rule=imp.selection_rule() if imp.promote_se is not None else "",
        kind=kind,
        features=imp.features,
        accepted=imp._accepted(),
        feature_trials=[t for t in imp.trials if t.added][-12:],
        card=card,
        tried_features=sorted({n for t in imp.trials for n in t.added}),
        findings=list(imp.findings),
        evaluations=evaluation_lines(imp),
        **(generated_context(imp, kind, done) if kind in GENERATED_KINDS else {}),
    )


def generated_context(imp, kind: str, done: Sequence[Trial]) -> dict[str, Any]:
    """Context only code, evaluation and analysis prompts need."""
    if imp._task_source is None:
        imp._task_source = task_context(imp.task, imp.train_dataset)
    out: dict[str, Any] = {"task_source": imp._task_source,
                           "columns": sorted(imp._analysis_columns())}
    if kind == "evaluation":
        out["metrics_menu"] = metrics_menu(imp.mode)
        out["columns"] = sorted(c for c in imp._columns() if c != "patient")
    if kind == "code":
        best_code = [t for t in done if t.config.code][:2]
        out["code_examples"] = [
            (f"#{t.trial_id} [{imp.metric}={t.score:.4f}]", t.config.code)
            for t in best_code
        ]
    return out


def evaluation_lines(imp) -> list[str]:
    lines = []
    fails = imp._guardrail_fails()
    for e in imp.evaluations:
        spec = e["spec"]
        metrics = ", ".join(f"{m} ({d} is better)" for m, d in spec.metrics.items())
        line = f"{spec.name} [{e['status']}]: {metrics}"
        if e["status"] == "active":
            failing = [t.trial_id for t in imp.trials if t.error is None and any(
                k.startswith(spec.name + ".") for k in fails(t))]
            line += ("; fails: " + ", ".join(f"#{i}" for i in failing)
                     if failing else "; no trial fails it")
        elif e.get("reason"):
            line += f"; {e['reason']}"
        lines.append(line)
    return lines
