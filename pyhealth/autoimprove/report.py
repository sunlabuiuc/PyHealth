"""The run report: the selected model, why it was chosen, and how the search got there.

A report is built from a run's decision log (the trials) and, when they
exist, the change attribution and the test comparison. It needs no model and
no data, so it can be rebuilt at any time from a run directory
(:func:`load_run`). Every statement is labelled with its evidence:

- **measured:** a score or a gain read from the trials;
- **tested:** a controlled comparison, either an attribution run or two
  trials that differ in one setting;
- **hypothesis:** a proposer's rationale or lesson, not checked by a run.

Reports hold definitions, configs and aggregate scores only, never samples.
"""

from __future__ import annotations

import html
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from pyhealth.tasks.task_card import DIRECTIONS, TaskCard

from .space import LADDER, PipelineConfig, SearchSpace, config_diff, with_xgboost

EVIDENCE = {
    "measured": "a score or gain read from the trials",
    "tested": "a controlled comparison: an attribution run, or two trials that "
              "differ in one setting",
    "hypothesis": "a proposer's rationale or lesson, not checked by a run",
}
"""What each evidence label means."""


@dataclass
class Section:
    """One report section: a title and blocks of content.

    A block is ``("p", text)``, ``("list", [text, ...])`` or
    ``("table", [header, ...], [[cell, ...], ...])``.

    Examples:
        >>> Section("Summary", [("p", "Selected: #3.")]).title
        'Summary'
    """

    title: str
    blocks: list[tuple] = field(default_factory=list)


@dataclass
class Report:
    """A rendered run report.

    Args:
        title: The report title.
        sections: The sections, in order.
        data: The same content as structured data, for ``report.json``.

    Examples:
        >>> report = Report("Run", [Section("Summary", [("p", "Selected: #3.")])])
        >>> print(report.to_markdown())
        # Run
        <BLANKLINE>
        ## Summary
        <BLANKLINE>
        Selected: #3.
        <BLANKLINE>
    """

    title: str
    sections: list[Section]
    data: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """The report as Markdown."""
        out = [f"# {self.title}", ""]
        for section in self.sections:
            out += [f"## {section.title}", ""]
            for block in section.blocks:
                if block[0] == "p":
                    out += [block[1], ""]
                elif block[0] == "list":
                    out += [f"- {item}" for item in block[1]] + [""]
                elif block[0] == "table":
                    headers, rows = block[1], block[2]
                    out.append("| " + " | ".join(headers) + " |")
                    out.append("|" + "---|" * len(headers))
                    for row in rows:
                        cells = [str(c).replace("|", "\\|") for c in row]
                        out.append("| " + " | ".join(cells) + " |")
                    out.append("")
        return "\n".join(out)

    def to_html(self) -> str:
        """The report as a self-contained HTML page."""
        body = [f"<h1>{_inline(self.title)}</h1>"]
        for section in self.sections:
            body.append(f"<h2>{_inline(section.title)}</h2>")
            for block in section.blocks:
                if block[0] == "p":
                    body.append(f"<p>{_inline(block[1])}</p>")
                elif block[0] == "list":
                    items = "".join(f"<li>{_inline(i)}</li>" for i in block[1])
                    body.append(f"<ul>{items}</ul>")
                elif block[0] == "table":
                    head = "".join(f"<th>{_inline(h)}</th>" for h in block[1])
                    rows = "".join(
                        "<tr>" + "".join(f"<td>{_inline(str(c))}</td>" for c in row)
                        + "</tr>"
                        for row in block[2]
                    )
                    body.append(
                        f'<div class="table"><table><thead><tr>{head}</tr></thead>'
                        f"<tbody>{rows}</tbody></table></div>"
                    )
        return _PAGE.format(title=html.escape(self.title), body="\n".join(body))

    def to_json(self) -> str:
        """The structured content as JSON."""
        return json.dumps(self.data, indent=1, default=str)

    def save(self, path: str) -> None:
        """Writes ``path`` (HTML) and the ``.md`` and ``.json`` next to it."""
        stem = path.removesuffix(".html")
        os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
        for suffix, text in ((".html", self.to_html()), (".md", self.to_markdown()),
                             (".json", self.to_json())):
            with open(stem + suffix, "w") as f:
                f.write(text)


def load_run(
    run_dir: str,
    space: SearchSpace | None = None,
    metric: str | None = None,
    maximize: bool | None = None,
    title: str = "AutoImprove report",
) -> Report:
    """Builds the report of a run from its directory alone.

    Reads ``trials.jsonl`` and, when present, ``run.json``, ``lessons.json``,
    ``card.json``, ``attribution.json``, ``explain.json``, ``test.json`` and
    ``review.json``. Runs written before ``run.json``
    existed need ``metric`` (and ``maximize`` for a metric where lower is
    better).

    Args:
        run_dir: The improver's ``run_dir``.
        space: The search space, for plain-language settings and the
            ladder. Defaults to the default space plus XGBoost.
        metric: The metric, if ``run.json`` is missing.
        maximize: Whether larger is better, if ``run.json`` is missing.
        title: The report title.

    Examples:
        >>> load_run("runs/mortality").save("runs/mortality/report.html")  # doctest: +SKIP
    """
    from .improver import Trial

    def read(name, default=None):
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            return default
        with open(path) as f:
            return json.load(f)

    with open(os.path.join(run_dir, "trials.jsonl")) as f:
        trials = [Trial.from_dict(json.loads(line)) for line in f if line.strip()]
    settings = read("run.json", {})
    metric = metric or settings.get("metric") or (trials[0].metric_name if trials else "")
    if maximize is None:
        maximize = settings.get("maximize", True)
    cv = any(t.fold_scores for t in trials)
    return build_report(
        trials, metric=metric, maximize=maximize,
        promote_se=settings.get("promote_se", 1.0 if cv else None),
        one_se=settings.get("one_se", True), space=space,
        task=settings.get("task", ""), lessons=read("lessons.json", []),
        test=read("test.json"), attribution=read("attribution.json"),
        dropped_models=settings.get("dropped_models", {}),
        rule=settings.get("rule", ""), title=title,
        card=read("card.json"), explanation=read("explain.json"),
        review=read("review.json", {}),
    )


def build_report(
    trials: Sequence[Any],
    *,
    metric: str,
    maximize: bool = True,
    promote_se: float | None = 1.0,
    one_se: bool = True,
    space: SearchSpace | None = None,
    task: str = "",
    lessons: Sequence[str] = (),
    test: dict[str, Any] | None = None,
    attribution: dict[str, Any] | None = None,
    dropped_models: dict[str, str] | None = None,
    rule: str = "",
    title: str = "AutoImprove report",
    card: Any = None,
    explanation: dict[str, Any] | None = None,
    review: dict[str, Any] | None = None,
) -> Report:
    """Builds a report from a run's trials and, optionally, its test and attribution.

    Args:
        trials: The trials, in the order run (``Trial`` objects).
        metric: The metric the search optimised.
        maximize: Whether larger scores are better.
        promote_se: The selection's standard-error bound (``None`` without CV).
        one_se: Whether the one-SE rule chose among the candidates.
        space: For plain-language settings and the ladder; defaults to the
            default space plus XGBoost.
        task: The task summary the proposer saw.
        lessons: The proposer's lessons.
        test: The result of ``AutoImprover.evaluate_on_test``.
        attribution: The result of ``AutoImprover.attribute``.
        dropped_models: Models removed before the search, with the error.
        rule: The selection rule in words.
        title: The report title.
        card: The task's :class:`~pyhealth.tasks.TaskCard` (or its dict).
        explanation: The result of ``AutoImprover.explain``.
        review: Clinician decisions on flags, from ``AutoImprover.approve``.

    Examples:
        >>> from pyhealth.autoimprove import Trial
        >>> base = Trial(0, -1, "baseline", "baseline", PipelineConfig("MLP"),
        ...              metric_name="pr_auc", score=0.30)
        >>> build_report([base], metric="pr_auc").sections[0].title
        'Summary'
    """
    from .improver import FOLD_TIE, paired_gain, select_simplest, select_trial

    space = space or with_xgboost(SearchSpace())
    if isinstance(card, dict):
        card = TaskCard.from_dict(card)
    review = review or {}
    sign = 1 if maximize else -1
    done = [t for t in trials if t.error is None and t.score is not None]
    if not done:
        return Report(title, [Section("Summary", [("p", "No trial finished.")])],
                      {"trials": [t.to_dict() for t in trials]})
    baseline = next((t for t in done if t.agent == "baseline"), None)
    # gains are shown against the baseline, or the first trial without one
    reference = baseline or done[0]
    cv = bool(reference.fold_scores) and promote_se is not None
    if cv and one_se:
        selected, top = select_simplest(done, maximize, promote_se, space.rung)
    elif cv:
        selected = top = select_trial(done, maximize, promote_se)
    else:
        selected = top = max(done, key=lambda t: sign * t.score)

    def vs(a, b):
        """Gain of a over b: (mean, se, folds better, n folds)."""
        if not (a.fold_scores and len(a.fold_scores) == len(b.fold_scores)):
            return sign * (a.score - b.score), None, None, None
        gain, se = paired_gain(a.fold_scores, b.fold_scores, maximize)
        better = sum(sign * (x - y) > FOLD_TIE
                     for x, y in zip(a.fold_scores, b.fold_scores))
        return gain, se, better, len(a.fold_scores)

    gains = {t.trial_id: vs(t, reference) for t in done}
    candidates = {reference.trial_id} | {
        t.trial_id for t in done
        if cv and t is not reference and gains[t.trial_id][1] is not None
        and (baseline is None
             or gains[t.trial_id][0] - promote_se * gains[t.trial_id][1] > 0)
    }
    near = {
        t.trial_id for t in done
        if t.trial_id in candidates and (t is top or (cv and vs(top, t)[0] <= vs(top, t)[1]))
    }
    ctx = {
        "metric": metric, "sign": sign, "reference": reference, "selected": selected,
        "top": top, "gains": gains, "candidates": candidates, "near": near,
        "space": space, "cv": cv, "promote_se": promote_se, "vs": vs,
        "one_se": one_se, "done": done, "baseline": baseline,
        "start": "start" if baseline else "first trial",
    }
    flags = [{**f, "review": review.get(f["id"])}
             for f in flags_for(card, explanation, test)]
    sections = [
        _summary(ctx, test, flags),
        _task_section(task, dropped_models or {}, card),
        _model_section(ctx, rule, attribution, test),
    ]
    if explanation:
        sections.append(_decides_section(explanation))
    if test:
        sections.append(_test_section(ctx, test))
    sections.append(_trace_section(ctx, trials, lessons))
    sections.append(_review_section(flags))
    sections.append(_checklist_section(ctx, card, explanation, test, attribution))
    sections.append(Section("How to read this report", [
        ("list", [f"**{k}**: {v}." for k, v in EVIDENCE.items()]),
        ("p", "Scores are cross-validation means over patient-grouped folds of "
              "the training patients unless marked as test. ± is one standard "
              "error of the fold-by-fold difference (a paired SE)." if cv else
              "Scores are on the validation split unless marked as test."),
    ]))
    data = {
        "metric": metric, "maximize": maximize, "rule": rule,
        "selected": selected.trial_id, "top": top.trial_id,
        "reference": reference.trial_id, "candidates": sorted(candidates),
        "settings": space.explain(selected.config),
        "rung": space.rung(selected.config),
        "verdicts": {t.trial_id: _verdict(t, ctx) for t in trials},
        "single_setting_effects": _single_setting_effects(ctx, done),
        "lessons": [{"text": x, "evidence": "hypothesis",
                     "unverified_numbers": unverified_numbers(x, trials)}
                    for x in lessons],
        "attribution": attribution, "test": test, "task": task,
        "dropped_models": dropped_models or {},
        "card": card.to_dict() if card is not None else None,
        "explanation": explanation, "flags": flags,
        "trials": [t.to_dict() for t in trials],
    }
    return Report(title, sections, data)


# ----------------------------------------------------------------- sections
def _summary(ctx, test, flags) -> Section:
    sel, ref, metric = ctx["selected"], ctx["reference"], ctx["metric"]
    space = ctx["space"]
    blocks = [("p", (f"**Selected: #{sel.trial_id}, "
                     f"{_model_words(sel.config, space)}.** "
                     f"{_score_line(sel, ref, ctx)} *(measured)*"))]
    if test and "best" in test:
        b = test["best"]
        line = f"On the test patients, {metric} was {_ci(b)}"
        if "baseline" in test:
            line += f" against {_ci(test['baseline'])} for the starting pipeline"
            if "difference" in test:
                line += f"; difference {_ci(test['difference'], signed=True)}"
        blocks.append(("p", line + ". *(measured)*"))
        clinical = b.get("clinical")
        if clinical:
            blocks.append(("p", _clinical_line(clinical) + " *(measured)*"))
    blocks.append(("p", "**Why this model.** " + _why(ctx)))
    open_flags = [f for f in flags if not f.get("review")]
    if flags:
        blocks.append(("p", (f"**Clinical review.** {len(flags)} item(s) flagged for a "
                             f"clinician, {len(open_flags)} still open: see "
                             "*Clinician review*.")))
    return Section("Summary", blocks)


def _task_section(task: str, dropped: dict[str, str], card) -> Section:
    blocks = []
    if card is not None:
        rows = [[name, item["text"], item["rationale"] or "-"]
                for name, item in card.design()]
        if rows:
            blocks += [("p", "**The task's design**, from its task card:"),
                       ("table", ["choice", "what", "why"], rows)]
        if card.features:
            by_group: dict[str, int] = {}
            for info in card.features.values():
                by_group[info.group or "no group"] = by_group.get(
                    info.group or "no group", 0) + 1
            with_direction = sum(1 for i in card.features.values() if i.direction)
            blocks.append(("p", (
                f"**Feature dictionary:** {len(card.features)} features ("
                + ", ".join(f"{g} {n}" for g, n in sorted(by_group.items()))
                + f"); {with_direction} with an expected direction of effect.")))
        if card.notes:
            blocks.append(("p", f"**Notes.** {card.notes}"))
    else:
        blocks.append(("p", ("No task card was given, so the cohort, outcome and "
                             "features have no recorded rationale.")))
    lines = [x.strip().removeprefix("- ") for x in task.splitlines() if x.strip()]
    if lines:
        blocks += [("p", "**Data the search used:**"), ("list", lines)]
    else:
        blocks.append(("p", "No task description was recorded for this run."))
    if dropped:
        blocks.append(("p", ("Models removed before the search because they do "
                             "not run on this data:")))
        blocks.append(("list", [f"{k}: {v}" for k, v in dropped.items()]))
    return Section("Task and data", blocks)


def _model_section(ctx, rule, attribution, test) -> Section:
    sel, top, space, metric = ctx["selected"], ctx["top"], ctx["space"], ctx["metric"]
    blocks = [("p", "The selected model and its settings, in plain words:"),
              ("list", space.explain(sel.config))]
    if rule:
        blocks.append(("p", f"**Selection rule.** {rule}"))
    if ctx["cv"]:
        rows = []
        cands = sorted((t for t in ctx["done"] if t.trial_id in ctx["candidates"]),
                       key=lambda t: -ctx["sign"] * t.score)
        for t in cands:
            gain, se, better, n = ctx["gains"][t.trial_id]
            rows.append([
                f"#{t.trial_id}" + (" (selected)" if t is sel else "")
                + (" (top)" if t is top and t is not sel else ""),
                t.config.model, space.rung(t.config), f"{t.score:.3f}",
                ctx["start"] if t is ctx["reference"] else _pm(gain, se, better, n),
                "yes" if t.trial_id in ctx["near"] else "no",
            ])
        blocks.append(("p", (f"Candidates ({len(cands)} of {len(ctx['gains'])} "
                             "finished trials) *(measured)*:")))
        blocks.append(("table", ["trial", "model", "rung", f"CV {metric}",
                                 f"gain vs {ctx['start']}", "within 1 SE of top"], rows))
        blocks.append(("p", "Ladder rungs: " + "; ".join(
            f"{k} = {v}" for k, v in LADDER.items()) + "."))
    if top is not sel:
        gap, se, better, n = ctx["vs"](top, sel)
        line = (f"**Price of interpretability.** The highest-scoring candidate, "
                f"#{top.trial_id} ({top.config.model}, rung {space.rung(top.config)}), "
                f"scored {_pm(gap, se, better, n)} more in CV")
        if test and "top_minus_best" in test:
            line += (f"; on test the difference was "
                     f"{_ci(test['top_minus_best'], signed=True)}")
        blocks.append(("p", line + ". *(measured)*"))
    if attribution and attribution.get("rows"):
        rows = [[r["setting"], f"{_short(r['from'])} → {_short(r['to'])}",
                 _pm(r["contribution"], r.get("se"), r.get("folds_better"),
                     len(sel.fold_scores) or None),
                 f"{r['score_without']:.3f}"]
                for r in attribution["rows"]]
        blocks.append(("p", (f"**What each change contributed** to #"
                             f"{attribution['trial_id']}, against "
                             f"{attribution['reference']}: each setting was "
                             "reverted alone and the result scored on the same "
                             "folds. *(tested)*")))
        blocks.append(("table", ["setting", "change", f"contribution to {metric}",
                                 "score with it reverted"], rows))
        blocks.append(("p", ("Contributions need not add up to the total gain: "
                             "settings can interact.")))
    elif attribution is not None:
        blocks.append(("p", (f"No setting differs from "
                             f"{attribution.get('reference')}, so there is nothing "
                             "to attribute.")))
    return Section("The model and why", blocks)


def _trace_section(ctx, trials, lessons) -> Section:
    from .improver import _parent_of

    metric, space = ctx["metric"], ctx["space"]
    blocks = []
    cycles: dict[int, list] = {}
    for t in trials:
        cycles.setdefault(t.cycle, []).append(t)
    timeline = []
    for cycle, ts in sorted(cycles.items()):
        name = "start" if cycle < 0 else f"cycle {cycle}"
        parts = [f"#{t.trial_id} {t.config.model} "
                 + ("failed" if t.error else f"{t.score:.3f}") for t in ts]
        timeline.append(f"{name}: " + ", ".join(parts))
    blocks += [("p", "**Timeline.**"), ("list", timeline)]

    rows = []
    for t in trials:
        parent, diff, how = t.parent, t.change, ""
        if parent is None and t.agent != "baseline":
            # logs written before parents were recorded: infer the nearest
            earlier = [x for x in trials if x.trial_id < t.trial_id]
            nearest = _parent_of(t.config, earlier)
            if nearest is not None:
                parent, how = nearest.trial_id, " (inferred)"
                diff = config_diff(nearest.config, t.config)
        change = ", ".join(f"{k} {v}" for k, v in diff.items()) or "-"
        if parent is not None:
            change = f"from #{parent}{how}: {change}"
        hyp = t.config.rationale or "-"
        expected = t.config.hypothesis.get("expected")
        if expected:
            hyp += f" (expected: {expected})"
        if t.error:
            result = "failed"
        else:
            gain, se, better, n = ctx["gains"][t.trial_id]
            result = f"{t.score:.3f}" + (
                "" if t is ctx["reference"] else f"; {_pm(gain, se, better, n)}")
        rows.append([f"#{t.trial_id}", f"{t.agent} ({t.strategy})", t.config.model,
                     change, hyp, result, _verdict(t, ctx)])
    blocks += [("p", ("**Every trial.** The hypothesis is the proposer's, stated "
                      f"before the trial ran *(hypothesis)*; results are CV {metric} "
                      "and the gain over the start *(measured)*; the verdict is the "
                      "selection rule's.")),
               ("table", ["trial", "agent", "model", "change", "hypothesis",
                          "result", "verdict"], rows)]

    effects = _single_setting_effects(ctx, [t for t in trials if t.error is None])
    worked = [e for e in effects if e["class"] == "helped"]
    hurt = [e for e in effects if e["class"] == "hurt"]
    flat = [e for e in effects if e["class"] == "no clear effect"]
    blocks.append(("p", ("**What worked and what didn't.** Pairs of trials that "
                         "differ in exactly one setting give a controlled "
                         "comparison *(tested)*:")))
    if effects:
        for label, group in (("Helped", worked), ("Hurt", hurt),
                             ("No clear effect", flat)):
            if group:
                blocks.append(("list", [f"{label}: {e['text']}" for e in group[:8]]))
    else:
        blocks.append(("p", "No two finished trials differ in exactly one setting."))
    families = {}
    for trial in ctx["done"]:
        best = families.get(trial.config.model)
        if best is None or ctx["sign"] * trial.score > ctx["sign"] * best.score:
            families[trial.config.model] = trial
    fam = []
    for model, t in sorted(families.items(), key=lambda kv: -ctx["sign"] * kv[1].score):
        gain, se, better, n = ctx["gains"][t.trial_id]
        tail = ("" if t is ctx["reference"]
                else f", {_pm(gain, se, better, n)} vs {ctx['start']}")
        fam.append(f"{model} (rung {space.rung(t.config)}): best #{t.trial_id}, "
                   f"{t.score:.3f}{tail}")
    blocks += [("p", "Best trial of each model *(measured)*:"), ("list", fam)]

    if lessons:
        items = []
        for lesson in lessons:
            bad = unverified_numbers(lesson, trials)
            note = f" — numbers not found in the trials: {', '.join(bad)}" if bad else ""
            items.append(f"{lesson} *(hypothesis{note})*")
        blocks += [("p", "**Lessons** written by the proposer:"), ("list", items)]
    return Section("Reasoning trace", blocks)


def _test_section(ctx, test) -> Section:
    metric = test.get("metric", ctx["metric"])
    names = {"best": "selected", "baseline": "starting pipeline",
             "top": "highest-scoring candidate"}
    present = [(k, n) for k, n in names.items() if k in test]
    rows = []
    for key, name in present:
        r = test[key]
        trial = f"#{r['trial_id']}" if "trial_id" in r else ""
        row = [name, trial, r.get("config", {}).get("model", ""), _ci(r)]
        clinical = r.get("clinical")
        if clinical:
            prev = clinical["prevalence"]
            row.append(f"{r['estimate'] / prev:.1f}×" if prev else "-")
        rows.append(row)
    headers = ["config", "trial", "model", f"test {metric}"]
    if any(test[k].get("clinical") for k, _ in present):
        headers.append("lift over prevalence")
    blocks = [("p", ("Each config was retrained on all training patients and "
                     "scored once on the test patients; intervals are "
                     "patient-level bootstrap 95% CIs. *(measured)*")),
              ("table", headers, rows)]
    diffs = []
    if "difference" in test:
        diffs.append(f"selected − starting pipeline: "
                     f"{_ci(test['difference'], signed=True)}")
    if "top_minus_best" in test:
        diffs.append(f"highest-scoring − selected: "
                     f"{_ci(test['top_minus_best'], signed=True)}")
    if diffs:
        blocks += [("p", "Paired differences on the same test patients:"),
                   ("list", diffs)]
    clinical = {k: test[k]["clinical"] for k, _ in present if test[k].get("clinical")}
    if "best" not in clinical:
        return Section("Performance in clinical terms", blocks)
    sel = clinical["best"]
    n_text = f"{sel['n']}" if sel["n"] is not None else "fewer than 11"
    blocks.append(("p", (f"Test set: {n_text} patients, prevalence "
                         f"{sel['prevalence']:.1%}.")))
    cal_rows = []
    for key, name in present:
        if key in clinical:
            c = clinical[key]["calibration"]
            cal_rows.append([name, f"{c.get('oe_ratio', float('nan')):.2f}",
                             f"{c.get('calibration_slope', float('nan')):.2f}",
                             f"{c.get('calibration_intercept', float('nan')):+.2f}",
                             f"{c.get('brier', float('nan')):.3f}"])
    blocks += [("p", ("**Calibration**: observed / expected events (ideal 1), "
                      "slope (ideal 1; below 1 means predictions are too "
                      "extreme) and intercept (ideal 0). *(measured)*")),
               ("table", ["config", "observed / expected", "slope", "intercept",
                          "Brier"], cal_rows)]
    if sel["calibration_bins"]:
        blocks.append(("table", ["predicted risk (bin mean)", "observed rate",
                                 "test samples"],
                       [[f"{b['mean_predicted']:.1%}", f"{b['observed']:.1%}",
                         b["n"]] for b in sel["calibration_bins"]]))
    alert_rows = []
    for label, a in sel["alerts"].items():
        if a.get("suppressed"):
            alert_rows.append([label, "fewer than 11", "-", "-", "-"])
            continue
        base = clinical.get("baseline", {}).get("alerts", {}).get(label, {})
        tail = (f" (start: {base['sensitivity']:.0%})"
                if base and not base.get("suppressed") else "")
        alert_rows.append([label, a["n_flagged"], f"{a['sensitivity']:.0%}{tail}",
                           f"{a['ppv']:.0%}", f"{a['nne']:.1f}"])
    blocks += [("p", ("**Acting on the highest risks**: flag the top share of test "
                      "samples by predicted risk. Sensitivity is the share of all "
                      "events caught, PPV the share of flags that are events, and "
                      "NNE the flags per event caught. *(measured)*")),
               ("table", ["flag", "samples flagged", "sensitivity", "PPV", "NNE"],
                alert_rows)]
    nb_rows = []
    base_nb = {r["threshold"]: r for r in clinical.get("baseline", {}).get(
        "net_benefit", [])}
    for r in sel["net_benefit"]:
        b = base_nb.get(r["threshold"])
        nb_rows.append([f"{r['threshold']:.1%}", f"{r['model']:+.4f}",
                        f"{b['model']:+.4f}" if b else "-", f"{r['treat_all']:+.4f}"])
    blocks += [("p", ("**Net benefit** at risk thresholds where one might act "
                      "(decision curve): true positives per patient minus false "
                      "positives weighted by the threshold odds. Acting on no one "
                      "scores 0. It takes predicted risks at face value, so a model "
                      "whose risks are inflated (observed / expected well below 1) "
                      "acts on too many patients at each threshold; compare after "
                      "recalibration. *(measured)*")),
               ("table", ["threshold", "selected", "starting pipeline",
                          "act on everyone"], nb_rows)]
    sub_rows, gaps = [], []
    for attribute, sub in sel["subgroups"].items():
        for value, r in sub["groups"].items():
            sub_rows.append([
                attribute, value,
                r["n"] if r["n"] is not None else "fewer than 11",
                r["positives"] if r["positives"] is not None else "fewer than 11",
                f"{r['roc_auc']:.3f}" if "roc_auc" in r else "too few events",
                f"{r['pr_auc']:.3f}" if "pr_auc" in r else "-",
                f"{r['oe_ratio']:.2f}" if "oe_ratio" in r else "-",
            ])
        for metric_name, gap in sub["max_gap"].items():
            gaps.append(f"{attribute}: largest {metric_name} gap {gap:.3f}")
    if sub_rows:
        blocks += [("p", "**Subgroups** (counts are patients). *(measured)*"),
                   ("table", ["attribute", "group", "n", "events", "ROC-AUC",
                              "PR-AUC", "observed / expected"], sub_rows)]
        if gaps:
            blocks.append(("list", gaps))
    return Section("Performance in clinical terms", blocks)


def _decides_section(explanation) -> Section:
    blocks = [("p", (f"How the selected model (#{explanation.get('trial_id')}) "
                     f"weighs its inputs, from {explanation['method']} on "
                     f"{explanation['n_samples']} training samples. Contributions "
                     "are on the log-odds scale. *(measured)*"))]
    if explanation["groups"]:
        blocks.append(("table", ["clinical factor group", "share of importance"],
                       [[g["group"], f"{g['share']:.0%}"]
                        for g in explanation["groups"]]))
    rows, shapes = [], []
    for i, f in enumerate(explanation["features"], 1):
        rows.append([i, f["name"], f["meaning"] or "-", f["group"] or "-",
                     f"{f['share']:.1%}", _direction(f["learned"]),
                     _direction(f["expected"]), f["check"]])
        if f["bins"]:
            shapes.append(f"{f['name']}: " + "; ".join(
                f"{b['label']}: {b['effect']:+.2f}" for b in f["bins"]))
    if rows:
        blocks += [("p", ("**Top features**, with the direction the model learned "
                          "and the direction the task card expects:")),
                   ("table", ["#", "feature", "meaning", "group", "importance",
                              "learned", "expected", "check"], rows)]
    if shapes:
        blocks += [("p", ("**Effect shapes**: mean contribution by value bin; every "
                          "bin holds at least 11 patients and its cut points are "
                          "midpoints, not patient values.")),
                   ("list", shapes)]
    stability = explanation.get("stability")
    if stability and stability.get("mean_shared") is not None:
        blocks.append(("p", (f"**Stability:** across {stability['models']} fold "
                             f"models, on average {stability['mean_shared']:.1f} of "
                             f"the top {stability['k']} features are shared (at "
                             f"least {stability['min_shared']}). *(tested)*")))
    if explanation.get("skipped"):
        blocks.append(("p", "Not attributed per column: " + ", ".join(
            f"{k} ({v})" for k, v in explanation["skipped"].items()) + "."))
    return Section("How it decides", blocks)


def _review_section(flags) -> Section:
    if not flags:
        return Section("Clinician review", [("p", "Nothing was flagged.")])
    rows = []
    for f in flags:
        r = f.get("review")
        status = (f"{r['decision']} by {r['reviewer']} on {r['date']}"
                  + (f": {r['note']}" if r.get("note") else "")) if r else "open"
        rows.append([f["id"], f["detail"], status])
    return Section("Clinician review", [
        ("p", ("Items a clinician should decide on before the model is used. "
               "Record decisions with ``AutoImprover.approve``.")),
        ("table", ["flag", "why", "decision"], rows),
    ])


def _checklist_section(ctx, card, explanation, test, attribution) -> Section:
    clinical = (test or {}).get("best", {}).get("clinical")
    subgroups = bool(clinical and clinical.get("subgroups"))
    missing = "not covered"

    def where(ok, text, hint):
        return text if ok else f"{missing}: {hint}"

    rows = [
        ("Data source and setting", where(card is not None, "Task and data",
                                          "give a task card")),
        ("Eligibility of participants", where(
            card is not None and bool(card.population["text"]),
            "Task and data: population and exclusions", "give a task card")),
        ("Outcome: definition and timing", where(
            card is not None and bool(card.outcome["text"]),
            "Task and data: outcome, prediction time, horizon", "give a task card")),
        ("Predictors: definition and timing", where(
            card is not None and bool(card.features),
            "Task and data: feature dictionary", "add features to the task card")),
        ("Sample size and number of events", "Task and data: data the search used"),
        ("Handling of missing data", where(
            card is not None and bool(card.notes), "Task and data: notes",
            "describe it in the task card notes")),
        ("Model type, predictor selection and tuning",
         "The model and why; Reasoning trace"),
        ("Internal validation", "patient-grouped cross-validation (The model and why)"
         if ctx["cv"] else "validation split"),
        ("Class imbalance", "The model and why: settings in plain words"),
        ("Model specification", "The model and why" + (
            ", with the attribution of each setting" if attribution else "")),
        ("Discrimination and calibration, with uncertainty", where(
            bool(clinical), "Performance in clinical terms",
            "run evaluate_on_test")),
        ("Clinical utility", where(bool(clinical), "Performance in clinical terms: "
                                   "alerts and net benefit", "run evaluate_on_test")),
        ("Fairness: performance in subgroups", where(
            subgroups, "Performance in clinical terms: subgroups",
            "pass subgroups to evaluate_on_test")),
        ("Interpretability: what drives predictions", where(
            bool(explanation), "How it decides", "run explain")),
        ("Limitations, ethics, open science, patient and public involvement",
         f"{missing}: for the authors"),
    ]
    return Section("Reporting checklist", [
        ("p", ("Where this report covers the main items of the TRIPOD+AI "
               "reporting guideline for prediction models (Collins et al., BMJ "
               "2024). Item names are paraphrased; check the published checklist "
               "for the full items.")),
        ("table", ["item", "where"], [list(r) for r in rows]),
    ])


def flags_for(card, explanation, test) -> list[dict[str, str]]:
    """Items for clinician review, from the task card, explanation and test.

    Examples:
        >>> [f["id"] for f in flags_for(None, None, None)]
        ['no_task_card']
    """
    out = []
    if card is None:
        out.append({"id": "no_task_card", "kind": "task",
                    "detail": "No task card: the cohort, outcome and features have "
                              "no recorded clinical rationale."})
    for f in (explanation or {}).get("features", []):
        if f["check"] == "opposite":
            out.append({
                "id": f"direction:{f['name']}", "kind": "direction",
                "detail": (f"{f['name']}: learned {_direction(f['learned'])}, "
                           f"expected {_direction(f['expected'])}. Possible causes: "
                           "a wrong expectation, confounding by indication, overlap "
                           "with a related feature (e.g. the lowest and highest "
                           "value of one measure), a coding artifact, a data error "
                           "or leakage."),
            })
        elif card is not None and not f["in_card"]:
            out.append({"id": f"no_rationale:{f['name']}", "kind": "rationale",
                        "detail": f"{f['name']} is a top feature with no entry in "
                                  "the task card's dictionary."})
    clinical = (test or {}).get("best", {}).get("clinical")
    if clinical:
        cal = clinical["calibration"]
        slope, oe = cal.get("calibration_slope"), cal.get("oe_ratio")
        bad = [x for x in (slope, oe) if x is not None and not 0.8 <= x <= 1.25]
        if bad:
            out.append({"id": "calibration", "kind": "calibration",
                        "detail": (f"Calibration slope {slope:.2f}, observed / "
                                   f"expected {oe:.2f}: recalibrate before reading "
                                   "predictions as risks.")})
        for attribute, sub in clinical.get("subgroups", {}).items():
            gap = sub["max_gap"].get("roc_auc")
            if gap is not None and gap >= 0.1:
                out.append({"id": f"subgroup_gap:{attribute}", "kind": "fairness",
                            "detail": f"ROC-AUC differs by {gap:.2f} between "
                                      f"{attribute} groups."})
    return out


def _clinical_line(clinical) -> str:
    parts = []
    alert = clinical["alerts"].get("top 10%")
    if alert and not alert.get("suppressed"):
        parts.append(f"Flagging the top 10% of test samples by risk would catch "
                     f"{alert['sensitivity']:.0%} of events, with {alert['ppv']:.0%} "
                     f"of flags being events")
    cal = clinical["calibration"]
    if "oe_ratio" in cal:
        parts.append(f"observed / expected events {cal['oe_ratio']:.2f}, "
                     f"calibration slope {cal.get('calibration_slope', float('nan')):.2f}")
    return "; ".join(parts) + "." if parts else ""


def _direction(code) -> str:
    if code is None:
        return "-"
    words = {"+": "higher → more risk", "-": "higher → less risk",
             "u": "U-shaped", "": "none", "flat": "flat"}
    return words.get(code, DIRECTIONS.get(code, str(code)))


# ------------------------------------------------------------------ helpers
def _why(ctx) -> str:
    sel, top, ref, space = ctx["selected"], ctx["top"], ctx["reference"], ctx["space"]
    if not ctx["cv"]:
        return "It has the best score; this run has no fold scores to test consistency."
    rung = space.rung(sel.config)
    words = f"rung {rung}: {LADDER.get(rung, '')}"
    n = len(ctx["candidates"])
    if ctx["baseline"] is None:
        lead = ("There is no starting pipeline to compare with, so every trial "
                "was a candidate.")
        if sel is top:
            return f"{lead} It has the best score of the {n} ({words}). *(measured)*"
        gap, se, better, k = ctx["vs"](top, sel)
        return (f"{lead} It is the most interpretable trial ({words}) whose score "
                f"is within one standard error of the best, #{top.trial_id}: "
                f"{_pm(gap, se, better, k)}. *(measured)*")
    if sel is ref and len(ctx["candidates"]) == 1:
        return ("No trial beat the starting pipeline consistently across folds, so "
                "it stays selected. *(measured)*")
    if not ctx["one_se"]:
        return (f"Of the {n} candidates, its gain over the starting pipeline has "
                f"the highest lower bound (mean minus standard error), so it is "
                f"the most consistent improvement ({words}). *(measured)*")
    if sel is top:
        tail = ("" if n <= 2 else
                " No simpler candidate scored within one standard error of it.")
        return (f"It beat the starting pipeline consistently across folds and has "
                f"the highest score of the {n} candidates ({words}).{tail} "
                "*(measured)*")
    gap, se, better, n = ctx["vs"](top, sel)
    return (f"It is the most interpretable candidate ({words}) whose score is within "
            f"one standard error of the highest-scoring candidate, #{top.trial_id} "
            f"({top.config.model}, rung {space.rung(top.config)}). Choosing it costs "
            f"{_pm(gap, se, better, n)} in CV score, which is within noise. "
            "*(measured)*")


def _verdict(t, ctx) -> str:
    if t.error:
        return "failed: " + t.error.split(" at ")[0][:120]
    sel, top = ctx["selected"], ctx["top"]
    if t is sel:
        return "selected"
    if not ctx["cv"]:
        return f"lower score than #{sel.trial_id}"
    if t is ctx["baseline"]:
        return "starting pipeline; trials that beat it consistently replace it"
    gain, se, _, _ = ctx["gains"][t.trial_id]
    space = ctx["space"]
    if t.trial_id in ctx["candidates"]:
        if t is top:
            return (f"highest score, but #{sel.trial_id} is within one SE and "
                    f"simpler (rung {space.rung(sel.config)} vs {space.rung(t.config)})")
        if t.trial_id in ctx["near"]:
            return (f"within one SE of the top, but #{sel.trial_id} is simpler or "
                    "scores higher on the same rung")
        return (("beat the start, but " if ctx["baseline"] else "")
                + f"more than one SE below the top candidate #{top.trial_id}")
    if se is None or gain <= 0:
        return f"no gain over the start ({_signed(gain)})"
    bound = gain - (ctx["promote_se"] or 1.0) * se
    return (f"gain {_signed(gain)} is not consistent across folds (lower bound "
            f"{_signed(bound)} is not above 0)")


def _single_setting_effects(ctx, done) -> list[dict[str, Any]]:
    """Effects of single-setting changes between pairs of trials, clearest first."""
    out = []
    for a, b in combinations(sorted(done, key=lambda t: t.trial_id), 2):
        if a.config.model != b.config.model:
            continue
        diff = config_diff(a.config, b.config)
        if len(diff) != 1:
            continue
        setting, change = next(iter(diff.items()))
        gain, se, better, n = ctx["vs"](b, a)
        if se is None:
            continue
        cls = ("helped" if gain - se > 0 else "hurt" if gain + se < 0
               else "no clear effect")
        out.append({
            "setting": setting, "change": change, "from": a.trial_id,
            "to": b.trial_id, "model": a.config.model, "gain": gain, "se": se,
            "folds_better": better, "class": cls,
            "text": (f"{a.config.model} {setting} {change} (#{a.trial_id} → "
                     f"#{b.trial_id}): {_pm(gain, se, better, n)}"),
        })
    return sorted(out, key=lambda e: -abs(e["gain"]) / max(e["se"], 1e-9))


def unverified_numbers(text: str, trials: Sequence[Any]) -> list[str]:
    """Decimal numbers in ``text`` that match no value in the trials.

    A number matches if some score, fold score, metric, config value, run
    time, or difference between two trials' scores equals it at the
    precision it is written with. Use it to check that an LLM's statement
    quotes the log correctly.

    Examples:
        >>> from pyhealth.autoimprove import PipelineConfig, Trial
        >>> t = [Trial(0, 0, "a", "explore", PipelineConfig("MLP"), score=0.2950),
        ...      Trial(1, 0, "a", "explore", PipelineConfig("MLP"), score=0.2445)]
        >>> unverified_numbers("#0 scored 0.295, #1 0.244 (-0.051); not 0.31", t)
        ['0.31']
    """
    known: list[float] = []
    scores = []
    for t in trials:
        values = [t.score, t.seconds, *t.fold_scores, *t.metrics.values(),
                  *t.config.model_params.values(), *t.config.train_params.values()]
        for v in values:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                known.append(float(v))
        if t.score is not None:
            scores.append(float(t.score))
    known += [a - b for a in scores for b in scores if a != b]
    out = []
    for raw in re.findall(r"(?<![\w.])[-+]?\d*\.\d+", text):
        value = float(raw)
        decimals = len(raw.split(".")[1])
        tol = 0.5 * 10 ** -decimals + 1e-12
        if not any(abs(value - k) <= tol or abs(abs(value) - abs(k)) <= tol
                   for k in known):
            out.append(raw)
    return out


def _model_words(config: PipelineConfig, space: SearchSpace) -> str:
    spec = space.models.get(config.model)
    if spec is None or not spec.description:
        return config.model
    return f"{config.model} ({spec.description})"


def _score_line(t, ref, ctx) -> str:
    metric = ctx["metric"]
    if t is ref:
        return f"CV {metric} {t.score:.3f}."
    gain, se, better, n = ctx["gains"][t.trial_id]
    against = ("the starting pipeline" if ctx["baseline"]
               else f"the first trial, #{ref.trial_id}")
    return (f"CV {metric} {t.score:.3f} against {ref.score:.3f} for {against}: "
            f"{_pm(gain, se, better, n)}.")


def _pm(gain, se, better=None, n=None) -> str:
    text = _signed(gain)
    if se is not None:
        text += f" ± {se:.3f}"
    if better is not None and n:
        text += f" (better in {better}/{n} folds)"
    return text


def _signed(value) -> str:
    return f"{value:+.3f}"


def _ci(r: dict[str, Any], signed: bool = False) -> str:
    fmt = (lambda v: f"{v:+.3f}") if signed else (lambda v: f"{v:.3f}")
    text = fmt(r["estimate"])
    if r.get("lower") is not None and r.get("upper") is not None:
        text += f" [{fmt(r['lower'])}, {fmt(r['upper'])}]"
    return text


def _short(value: Any) -> str:
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def _inline(text: str) -> str:
    """Escapes text and renders **bold** and *italic*."""
    text = html.escape(str(text))
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --ink: #1d2730; --muted: #5b6b78; --rule: #d8e0e6; --bg: #ffffff;
  --head: #f3f6f8; --accent: #0b6e69; }}
@media (prefers-color-scheme: dark) {{ :root {{ --ink: #e2e9ee; --muted: #9fb0bc;
  --rule: #2c3a44; --bg: #121a20; --head: #1a252d; --accent: #4cc3bb; }} }}
body {{ margin: 0 auto; max-width: 1080px; padding: 32px 16px 64px;
  font: 15px/1.55 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  color: var(--ink); background: var(--bg); }}
h1 {{ font-size: 26px; margin: 0 0 8px; }}
h2 {{ font-size: 19px; margin-top: 36px; padding-top: 14px;
  border-top: 1px solid var(--rule); color: var(--accent); }}
p, li {{ max-width: 80ch; }}
em {{ color: var(--muted); }}
.table {{ overflow-x: auto; margin: 12px 0; border: 1px solid var(--rule);
  border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th, td {{ text-align: left; vertical-align: top; padding: 6px 10px;
  border-bottom: 1px solid var(--rule); }}
th {{ background: var(--head); font-weight: 600; white-space: nowrap; }}
tr:last-child td {{ border-bottom: 0; }}
</style></head>
<body>
{body}
</body></html>
"""
