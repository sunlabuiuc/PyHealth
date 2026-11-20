"""RunConfig: a run's settings as typed YAML."""

import os
import random
import tempfile
import unittest

import yaml
from pydantic import ValidationError

from pyhealth.autoimprove import (
    AutoImprover,
    PipelineConfig,
    RandomProposer,
    RunConfig,
)
from pyhealth.datasets import create_sample_dataset

YAML = """
search: {metric: roc_auc, proposals_per_round: 2, instructions: Prefer simple models.}
budget: {rounds: 1}
baseline: {model: LogisticRegression, model_params: {embedding_dim: 8}}
space:
  models: [LogisticRegression, MLP]
  train_params:
    epochs: {kind: int, low: 1, high: 2, default: 1}
evaluation: {check: false}
"""


def _samples(n, seed, offset):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        codes = rng.sample([f"c{j}" for j in range(8)], 3)
        out.append({"patient_id": f"p{offset + i}", "visit_id": f"v{offset + i}",
                    "conditions": codes, "label": int("c1" in codes)})
    return out


class TestRunConfig(unittest.TestCase):
    def test_yaml_round_trip(self):
        config = RunConfig.from_yaml_text(YAML)
        self.assertEqual(RunConfig.from_yaml_text(config.to_yaml()), config)
        # defaults are left out of the written YAML
        self.assertNotIn("confirm", yaml.safe_load(config.to_yaml()))

    def test_typos_and_wrong_types_are_errors(self):
        with self.assertRaisesRegex(ValidationError, "roundz"):
            RunConfig.from_yaml_text("budget: {roundz: 3}")
        with self.assertRaisesRegex(ValidationError, "proposals_per_round"):
            RunConfig(search={"proposals_per_round": 0})
        with self.assertRaisesRegex(ValidationError, "unknown models"):
            RunConfig(space={"models": ["NotAModel"]})

    def test_baseline_from_a_pipeline_config(self):
        baseline = PipelineConfig("MLP", {"hidden_dim": 16}, {"epochs": 2})
        config = RunConfig(baseline=baseline)
        self.assertEqual(config.baseline.to_pipeline(), baseline)
        self.assertEqual(
            RunConfig().with_overrides(baseline=baseline).baseline.to_pipeline(),
            baseline)

    def test_overrides_merge_into_sections(self):
        config = RunConfig(budget={"rounds": 4, "patience": 3})
        changed = config.with_overrides(budget={"rounds": 1})
        self.assertEqual((changed.budget.rounds, changed.budget.patience), (1, 3))
        self.assertEqual(config.budget.rounds, 4)  # the original is unchanged
        with self.assertRaises(ValidationError):
            config.with_overrides(budget={"rounds": -1})

    def test_space_section_builds_a_search_space(self):
        space = RunConfig.from_yaml_text(YAML).space.build()
        self.assertEqual(sorted(space.models), ["LogisticRegression", "MLP"])
        self.assertEqual(space.train_params["epochs"].high, 2)
        self.assertIn("lr", space.train_params)  # defaults are kept


class TestRunFromYaml(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema = ({"conditions": "sequence"}, {"label": "binary"})
        cls.train = create_sample_dataset(_samples(60, 0, 0), *schema)
        kw = {"input_processors": cls.train.input_processors,
              "output_processors": cls.train.output_processors}
        cls.val = create_sample_dataset(_samples(30, 1, 1000), *schema, **kw)

    def test_a_yaml_file_configures_and_is_saved_with_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.yaml")
            run_dir = os.path.join(tmp, "run")
            with open(path, "w") as f:
                f.write(YAML + f"output: {{run_dir: {run_dir}}}\n")
            improver = AutoImprover(self.train, self.val, path,
                                    proposer=RandomProposer(seed=0))
            self.assertEqual(improver.metric, "roc_auc")
            self.assertEqual(sorted(improver.space.models),
                             ["LogisticRegression", "MLP"])
            result = improver.run()
            self.assertEqual(result.trials[0].agent, "baseline")
            self.assertEqual(len(result.trials), 3)  # baseline + 2 proposals
            saved = RunConfig.from_yaml(os.path.join(run_dir, "config.yaml"))
            self.assertEqual(saved, RunConfig.from_yaml(path))

    def test_the_proposer_comes_from_the_config_by_default(self):
        config = RunConfig(evaluation={"check": False})
        improver = AutoImprover(self.train, self.val, config)
        self.assertIsInstance(improver.proposer, RandomProposer)

    def test_default_plan(self):
        improver = AutoImprover(self.train, self.val, RunConfig())
        self.assertEqual(improver.cycle_plan, ("model",))
        with self.assertRaisesRegex(ValueError, "search.plan needs kinds"):
            AutoImprover(self.train, self.val, RunConfig(search={"plan": ["modle"]}))


if __name__ == "__main__":
    unittest.main()
