"""Improve a mortality-prediction pipeline automatically with AutoImprover.

Starts from a plain RNN, lets a population of agents propose better
configurations (model, arguments, training settings), scores each one on the
validation patients, and finally compares the best one with the starting
pipeline on the test patients.

Settings come from ``examples/configs/autoimprove/synthetic_mimic3.yaml``
(rounds, agents, baseline...); edit it or pass ``--config``. With
``--llm claude`` the proposals come from Claude (``pip install anthropic``
and set ANTHROPIC_API_KEY). Prompts hold the search space, dataset sizes,
aggregate label counts and validation scores, never patient records. Without
it, random search is used, which needs no API key.

Uses PyHealth's public synthetic MIMIC-III (downloaded on first use; no
credentials; about 2 minutes per pass on a laptop CPU). Its labels carry
little signal, so expect validation AUROCs near 0.5: the example shows how the
loop works, not how much it gains. Point it at your own data and task for that.

Usage:
    python examples/autoimprove_synthetic_mimic3.py
    python examples/autoimprove_synthetic_mimic3.py --llm claude
    python examples/autoimprove_synthetic_mimic3.py --config my_run.yaml
"""

import argparse
from pathlib import Path

from pyhealth.autoimprove import AutoImprover, RunConfig
from pyhealth.datasets import MIMIC3Dataset, PatientSplit
from pyhealth.tasks import MortalityPredictionMIMIC3

SYNTHETIC_ROOT = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III"
CONFIG = Path(__file__).parent / "configs" / "autoimprove" / "synthetic_mimic3.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG), help="the run's YAML settings")
    parser.add_argument("--llm", choices=["claude", "openai", "gemini"], default=None,
                        help="propose with this LLM instead of the config's proposer")
    parser.add_argument("--run-dir", default=None,
                        help="save trials here; rerun with it to resume")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    dataset = MIMIC3Dataset(
        root=SYNTHETIC_ROOT,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        cache_dir=args.cache_dir,
    )
    train, val, test = dataset.set_task(
        MortalityPredictionMIMIC3(),
        split=PatientSplit(ratios=(0.6, 0.2, 0.2), seed=0),
    )

    config = RunConfig.from_yaml(args.config)
    if args.llm:
        config = config.with_overrides(proposer={"type": "llm", "backend": args.llm})
    if args.run_dir:
        config = config.with_overrides(output={"run_dir": args.run_dir})
    improver = AutoImprover(train, val, config)
    result = improver.run()
    print(result.summary())
    print("lessons:", result.lessons)

    test_scores = improver.evaluate_on_test(test, n_boot=200)
    for name in ("baseline", "best"):
        s = test_scores.get(name)
        if s:
            print(f"test {name}: {s['estimate']:.3f} "
                  f"[{s['lower']:.3f}, {s['upper']:.3f}] {s['config']}")
    if "difference" in test_scores:
        d = test_scores["difference"]
        print(f"best - baseline: {d['estimate']:+.3f} "
              f"[{d['lower']:+.3f}, {d['upper']:+.3f}]")


if __name__ == "__main__":
    main()
