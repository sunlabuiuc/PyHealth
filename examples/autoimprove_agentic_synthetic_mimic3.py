"""Let an LLM write models, evaluations and data analyses with AutoImprover.

Starts from a plain RNN on mortality prediction and runs four kinds of turn,
in a repeating cycle:

- analysis: the LLM asks a question of the training data; the answer comes
  back as aggregate tables with groups under 11 patients withheld, and its
  findings go into every later prompt;
- model: the LLM picks a model and settings from the search space;
- code: the LLM writes its own BaseModel and training loop, which runs in its
  own process and is scored on the same folds as every other attempt;
- evaluation: the LLM writes a new way to score predictions (calibration, a
  subgroup, alert yield...), usually with pyhealth.metrics; if it passes the
  checks it becomes a guardrail, and attempts clearly worse than the starting
  model on it can't be selected.

Generated code runs in a separate process with a time limit, no network and
no credentials, after a static import check. That guards against mistakes,
not against deliberately hostile code: run it on a machine with nothing to
lose.

Needs an LLM (``pip install anthropic`` and set ANTHROPIC_API_KEY for
Claude). Uses PyHealth's public synthetic MIMIC-III (downloaded on first
use). Its labels carry little signal, so the example shows how the loop
works, not how much it gains.

Usage:
    python examples/autoimprove_agentic_synthetic_mimic3.py --run-dir runs/agentic
    python examples/autoimprove_agentic_synthetic_mimic3.py --llm openai

Settings (the plan, rounds, agents, baseline...) come from
``examples/configs/autoimprove/agentic_synthetic_mimic3.yaml``; edit it or
pass ``--config``.
"""

import argparse
from pathlib import Path

from pyhealth.autoimprove import AutoImprover, RunConfig
from pyhealth.datasets import MIMIC3Dataset, PatientSplit
from pyhealth.tasks import MortalityPredictionMIMIC3

SYNTHETIC_ROOT = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III"
CONFIG = (Path(__file__).parent / "configs" / "autoimprove"
          / "agentic_synthetic_mimic3.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG), help="the run's YAML settings")
    parser.add_argument("--llm", choices=["claude", "openai", "gemini"], default=None,
                        help="use this LLM instead of the config's backend")
    parser.add_argument("--run-dir", default=None,
                        help="save trials, code and logs here; rerun to resume")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    task = MortalityPredictionMIMIC3()
    dataset = MIMIC3Dataset(
        root=SYNTHETIC_ROOT,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        cache_dir=args.cache_dir,
    )
    train, val, test = dataset.set_task(
        task, split=PatientSplit(ratios=(0.6, 0.2, 0.2), seed=0))

    config = RunConfig.from_yaml(args.config)
    if args.llm:
        config = config.with_overrides(proposer={"type": "llm", "backend": args.llm})
    if args.run_dir:
        config = config.with_overrides(output={"run_dir": args.run_dir})
    improver = AutoImprover(train, val, config, task=task)
    result = improver.run()
    print(result.summary())

    print("\nfindings from data analysis:")
    for finding in improver.findings:
        print(" -", finding)
    print("\nproposed evaluations:")
    failures = improver.guardrail_failures()
    for e in improver.evaluations:
        print(f" - {e['spec'].name}: {e['status']} ({e['reason']})")
    if failures:
        print("removed by guardrails:", failures)

    test_scores = improver.evaluate_on_test(test, n_boot=200)
    for name in ("baseline", "best"):
        s = test_scores.get(name)
        if s:
            kind = "generated code" if s["config"].get("code") else s["config"]["model"]
            print(f"test {name}: {s['estimate']:.3f} "
                  f"[{s['lower']:.3f}, {s['upper']:.3f}] ({kind})")
    for key, values in test_scores.get("guardrails", {}).items():
        print(f"test guardrail {key}: {values}")
    if args.run_dir:
        improver.report()
        print(f"\nreport: {args.run_dir}/report.html; generated code in "
              f"{args.run_dir}/code/")


if __name__ == "__main__":  # needed: generated code runs in child processes
    main()
