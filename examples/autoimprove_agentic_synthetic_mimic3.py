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
    python examples/autoimprove_agentic_synthetic_mimic3.py --llm openai --rounds 2
"""

import argparse

from pyhealth.autoimprove import (
    AutoImprover,
    ClaudeLLM,
    GeminiLLM,
    LLMProposer,
    OpenAILLM,
    PipelineConfig,
)
from pyhealth.datasets import MIMIC3Dataset, PatientSplit
from pyhealth.tasks import MortalityPredictionMIMIC3

SYNTHETIC_ROOT = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III"
PLAN = ("analysis", "model", "code", "evaluation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["claude", "openai", "gemini"],
                        default="claude")
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=1,
                        help="passes through analysis, model, code, evaluation")
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

    llm = {"claude": ClaudeLLM, "openai": OpenAILLM, "gemini": GeminiLLM}[args.llm]()
    improver = AutoImprover(
        train, val, LLMProposer(llm),
        baseline=PipelineConfig("RNN", {"hidden_dim": 128}, {"epochs": 5}),
        metric="roc_auc",
        task=task,                 # its source code goes into code prompts
        cycle_plan=PLAN,
        cv_folds=3,                # guardrails compare trials fold by fold
        n_agents=args.agents,
        n_cycles=len(PLAN) * args.rounds,
        patience=len(PLAN),
        code_timeout=900,
        run_dir=args.run_dir,
    )
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
