"""Rounds: batched proposals scored together, or one agent after another."""

import random
import unittest
from unittest import mock

from pyhealth.autoimprove import (
    AutoImprover,
    DEFAULT_MODELS,
    Param,
    PipelineConfig,
    Proposer,
    RandomProposer,
    RunConfig,
    SearchSpace,
)
from pyhealth.autoimprove.evaluation import training
from pyhealth.datasets import create_sample_dataset

SPACE = SearchSpace(
    {k: DEFAULT_MODELS[k] for k in ("LogisticRegression", "MLP")},
    {"epochs": Param("int", 1, 1, default=1),
     "lr": Param("float", 1e-3, 1e-1, log=True, default=1e-2),
     "batch_size": Param("choice", choices=(16,), default=16)},
)
BASELINE = PipelineConfig("LogisticRegression", {"embedding_dim": 8})


def _samples(n, seed, offset):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        codes = rng.sample([f"c{j}" for j in range(8)], 3)
        out.append({"patient_id": f"p{offset + i}", "visit_id": f"v{offset + i}",
                    "conditions": codes, "label": int("c1" in codes)})
    return out


class Recording(Proposer):
    """Proposes fixed configs and records what each call could see."""

    def __init__(self, configs):
        self.configs = list(configs)
        self.seen = []  # (agent, trials on the leaderboard, instructions)

    def propose(self, context):
        self.seen.append((context.agent, len(context.leaderboard), context.instructions))
        return self.configs.pop(0)


def _mlp(dim):
    return PipelineConfig("MLP", {"embedding_dim": 8, "hidden_dim": dim})


class TestRounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema = ({"conditions": "sequence"}, {"label": "binary"})
        cls.train = create_sample_dataset(_samples(80, 0, 0), *schema)
        kw = {"input_processors": cls.train.input_processors,
              "output_processors": cls.train.output_processors}
        cls.val = create_sample_dataset(_samples(40, 1, 1000), *schema, **kw)

    def _run(self, proposer, rounds=1, agents=3, **search):
        config = RunConfig(
            search={"proposals_per_round": agents, **search},
            budget={"rounds": rounds, "patience": 5},
            evaluation={"cv_folds": 2, "check": False}, baseline=BASELINE,
        )
        improver = AutoImprover(self.train, self.val, config, proposer=proposer,
                                space=SPACE)
        return improver, improver.run()

    def test_batched_proposals_all_see_the_results_before_the_round(self):
        proposer = Recording([_mlp(16), _mlp(32), _mlp(64)])
        _, result = self._run(proposer)
        self.assertEqual([seen for _, seen, _ in proposer.seen], [1, 1, 1])
        self.assertEqual([t.trial_id for t in result.trials], [0, 1, 2, 3])
        self.assertEqual([t.agent for t in result.trials[1:]],
                         ["agent0", "agent1", "agent2"])

    def test_sequential_proposals_see_each_trial_as_it_is_scored(self):
        proposer = Recording([_mlp(16), _mlp(32), _mlp(64)])
        self._run(proposer, mode="sequential")
        self.assertEqual([seen for _, seen, _ in proposer.seen], [1, 2, 3])

    def test_a_repeat_inside_a_round_is_retried(self):
        proposer = Recording([_mlp(16), _mlp(16), _mlp(32)])
        _, result = self._run(proposer, agents=2)
        dims = [t.config.model_params.get("hidden_dim") for t in result.trials[1:]]
        self.assertEqual(dims, [16, 32])

    def test_max_trials_caps_the_batch(self):
        proposer = Recording([_mlp(16), _mlp(32), _mlp(64)])
        config = RunConfig(search={"proposals_per_round": 3},
                           budget={"rounds": 1, "max_trials": 2},
                           evaluation={"cv_folds": 2, "check": False}, baseline=BASELINE)
        result = AutoImprover(self.train, self.val, config, proposer=proposer,
                              space=SPACE).run()
        self.assertEqual(len(result.trials), 2)

    def test_instructions_reach_every_prompt(self):
        proposer = Recording([_mlp(16), _mlp(32)])
        self._run(proposer, agents=2, instructions="Prefer small models.")
        self.assertEqual({text for _, _, text in proposer.seen}, {"Prefer small models."})
        from pyhealth.autoimprove.search.prompts import build_prompt
        from pyhealth.autoimprove.search.proposers import ProposalContext

        prompt = build_prompt(ProposalContext("agent0", "explore", SPACE, "a task",
                                              "pr_auc", True,
                                              instructions="Prefer small models."))
        self.assertIn("Instructions from the person running this search", prompt)
        self.assertIn("Prefer small models.", prompt)

    def test_parallel_workers_give_the_same_trials(self):
        _, one = self._run(RandomProposer(seed=3), rounds=2, workers=1)
        with mock.patch.object(training, "run_sandboxed",
                               side_effect=training.run_sandboxed) as spy:
            _, two = self._run(RandomProposer(seed=3), rounds=2, workers=2)
        # every fold of every proposed trial trained in its own process (the
        # baseline is scored before the first round, in this process)
        self.assertEqual(spy.call_count, 2 * (len(two.trials) - 1))
        summary = [(t.trial_id, t.agent, t.config.to_dict(), t.error) for t in one.trials]
        self.assertEqual(summary, [(t.trial_id, t.agent, t.config.to_dict(), t.error)
                                   for t in two.trials])
        for a, b in zip(one.trials, two.trials):
            self.assertIsNone(a.error)
            self.assertAlmostEqual(a.score, b.score, places=6)
            self.assertEqual(len(a.fold_scores), len(b.fold_scores))


if __name__ == "__main__":
    unittest.main()
