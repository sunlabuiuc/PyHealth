"""pyhealth.autoimprove: search space checks, proposers, the loop, test scoring."""

import importlib.util
import json
import math
import os
import random
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pyhealth.autoimprove import (
    AutoImprover,
    ClaudeLLM,
    LLMProposer,
    ModelSpec,
    Param,
    PipelineConfig,
    RandomProposer,
    SearchSpace,
    Trial,
    build_report,
    check_models,
    load_run,
    paired_gain,
    select_simplest,
    select_trial,
    train_pipeline,
    unverified_numbers,
    with_xgboost,
)
from pyhealth.autoimprove.evaluate import patient_folds
from pyhealth.autoimprove.explain import attributions, summarize
from pyhealth.autoimprove.proposers import ProposalContext, parse_json
from pyhealth.datasets import create_sample_dataset
from pyhealth.models import LogisticRegression
from pyhealth.processors import BinaryLabelProcessor
from pyhealth.tasks import FeatureInfo, TaskCard

SPACE = SearchSpace(
    train_params={
        "epochs": Param("int", 1, 2, default=1),
        "lr": Param("float", 1e-4, 1e-1, log=True, default=1e-2),
        "batch_size": Param("choice", choices=(16, 32), default=16),
        "pos_weight": Param("choice", choices=("none", "balanced"), default="none"),
    }
)
SPACE.models = {k: SPACE.models[k] for k in ("LogisticRegression", "MLP", "RNN")}


def _samples(n, seed, offset=0):
    """Code sequences; the label is 1 when code 'c1' is present."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        codes = rng.sample([f"c{j}" for j in range(8)], 3)
        out.append({
            "patient_id": f"p{offset + i}",
            "visit_id": f"v{offset + i}",
            "conditions": codes,
            "label": int("c1" in codes),
        })
    return out


def _three_classes(codes):
    return int("c1" in codes) + int("c2" in codes)


def _datasets(label_type="binary", label=None):
    """Train, validation and test splits; ``label(codes)`` replaces the label."""
    parts = [_samples(80, 0), _samples(40, 1, 1000), _samples(40, 2, 2000)]
    if label is not None:
        for sample in (s for part in parts for s in part):
            sample["label"] = label(sample["conditions"])
    schema = ({"conditions": "sequence"}, {"label": label_type})
    train = create_sample_dataset(parts[0], *schema)
    kwargs = {
        "input_processors": train.input_processors,
        "output_processors": train.output_processors,
    }
    val = create_sample_dataset(parts[1], *schema, **kwargs)
    test = create_sample_dataset(parts[2], *schema, **kwargs)
    return train, val, test


def _reply(config):
    return f"Analysis.\n```json\n{json.dumps(config)}\n```"


class ScriptedLLM:
    """Returns queued replies for proposals and a fixed lesson for reflections."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, system, prompt):
        self.prompts.append(prompt)
        if "Rewrite the lessons" in prompt:
            return _reply({"lessons": ["RNN beats LogisticRegression here"]})
        return self.replies.pop(0)


class UnreadableInputs:
    """A model that cannot read the task's inputs, for the model check."""

    def __init__(self, dataset, **kwargs):
        raise ValueError("cannot read 'conditions'")


class TestSearchSpace(unittest.TestCase):
    def test_validate_clips_drops_and_fills_defaults(self):
        config, notes = SearchSpace().validate(
            PipelineConfig("RNN", {"hidden_dim": 999, "colour": "red"},
                           {"lr": 5.0, "batch_size": 7})
        )
        self.assertEqual(config.model_params["hidden_dim"], 256)
        self.assertNotIn("colour", config.model_params)
        self.assertEqual(config.model_params["rnn_type"], "GRU")  # default filled
        self.assertEqual(config.train_params["lr"], 1e-1)
        self.assertEqual(config.train_params["batch_size"], 64)
        self.assertEqual(len(notes), 4)

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            SearchSpace().validate(PipelineConfig("os.system"))

    def test_samples_are_valid(self):
        rng = random.Random(0)
        for _ in range(20):
            config = SearchSpace().sample(rng)
            _, notes = SearchSpace().validate(config)
            self.assertEqual(notes, [])

    def test_transformer_samples_fit_their_heads(self):
        # embedding_dim must divide by heads, or attention asserts at fit time.
        rng = random.Random(0)
        space = SearchSpace({"Transformer": SearchSpace().models["Transformer"]})
        for _ in range(50):
            params = space.sample(rng).model_params
            self.assertEqual(params["embedding_dim"] % params["heads"], 0)

    def test_key_ignores_rationale_and_order(self):
        a = PipelineConfig("RNN", {"a": 1, "b": 2}, {}, "x")
        b = PipelineConfig("RNN", {"b": 2, "a": 1}, {}, "y")
        self.assertEqual(a.key(), b.key())


class TestParseJson(unittest.TestCase):
    def test_fenced_and_bare(self):
        self.assertEqual(parse_json('x ```json\n{"a": 1}\n``` y'), {"a": 1})
        self.assertEqual(parse_json('answer: {"a": 2}'), {"a": 2})

    def test_last_block_wins(self):
        text = '```json\n{"a": 1}\n```\nrevised:\n```json\n{"a": 3}\n```'
        self.assertEqual(parse_json(text), {"a": 3})

    def test_no_json(self):
        with self.assertRaises(ValueError):
            parse_json("no config here")


class TestLLMProposer(unittest.TestCase):
    def test_malformed_replies_are_asked_again(self):
        llm = ScriptedLLM([
            _reply({"model": "MLP", "model_params": 5}),
            _reply({"model": "MLP", "model_params": {"hidden_dim": 16}}),
        ])
        context = ProposalContext("a0", "explore", SPACE, "binary task", "pr_auc", True)
        config = LLMProposer(llm).propose(context)
        self.assertEqual(config.model_params, {"hidden_dim": 16})
        self.assertIn("model_params must be an object, not int", llm.prompts[1])

    def test_malformed_lessons_keep_the_old_ones(self):
        proposer = LLMProposer(lambda system, prompt: _reply({"lessons": "use an RNN"}))
        self.assertEqual(proposer.reflect([], ["keep this"], "pr_auc"), ["keep this"])


class TestAutoImprover(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.val, cls.test = _datasets()

    def test_llm_loop_baseline_retries_lessons_and_resume(self):
        llm = ScriptedLLM([
            "I am not sure.",  # unparseable: asked again
            _reply({"model": "RNN", "model_params": {"hidden_dim": 16,
                    "embedding_dim": 16}, "train_params": {"epochs": 2}}),
            _reply({"model": "NotAModel"}),  # rejected: the turn retries
            _reply({"model": "MLP", "model_params": {"hidden_dim": 16,
                    "embedding_dim": 16}}),
        ])
        baseline = PipelineConfig("LogisticRegression", {"embedding_dim": 16})
        with tempfile.TemporaryDirectory() as tmp:
            lessons_path = os.path.join(tmp, "lessons.json")
            improver = AutoImprover(
                self.train, self.val, LLMProposer(llm), baseline=baseline,
                space=SPACE, report_metrics=["roc_auc"], n_agents=2, n_cycles=1,
                run_dir=os.path.join(tmp, "run"), lessons_path=lessons_path,
            )
            result = improver.run()

            self.assertEqual([t.agent for t in result.trials],
                             ["baseline", "agent0", "agent1"])
            self.assertTrue(all(t.error is None for t in result.trials))
            self.assertEqual(result.trials[1].config.train_params["epochs"], 2)
            self.assertIn("roc_auc", result.trials[1].metrics)
            self.assertEqual(result.lessons, ["RNN beats LogisticRegression here"])
            with open(lessons_path) as f:
                stored = json.load(f)
            self.assertEqual(len(stored), 1)  # one search fingerprint
            self.assertEqual(next(iter(stored.values()))["lessons"], result.lessons)
            self.assertIn("Models available in this search: LogisticRegression, MLP, RNN",
                          llm.prompts[-1])
            self.assertIn("rejected", llm.prompts[3])
            # Prompts carry aggregates only, never sample contents.
            self.assertIn("Training positives", llm.prompts[0])
            self.assertNotIn("p1", llm.prompts[0].split())

            # A second run with the same directory resumes at cycle 1.
            llm.replies = [_reply({"model": "RNN", "model_params": {
                "hidden_dim": 32, "embedding_dim": 16}})]
            again = AutoImprover(
                self.train, self.val, LLMProposer(llm), baseline=baseline,
                space=SPACE, n_agents=1, n_cycles=2,
                run_dir=os.path.join(tmp, "run"),
            )
            result2 = again.run()
            self.assertEqual(len(result2.trials), 4)
            self.assertEqual(result2.trials[-1].cycle, 1)
            self.assertIn("RNN beats LogisticRegression here", llm.prompts[-2])

    def test_repeated_proposal_is_not_rerun(self):
        same = _reply({"model": "LogisticRegression",
                       "model_params": {"embedding_dim": 16}})
        llm = ScriptedLLM([same, same])
        improver = AutoImprover(
            self.train, self.val, LLMProposer(llm),
            baseline=PipelineConfig("LogisticRegression", {"embedding_dim": 16}),
            space=SPACE, n_agents=1, n_cycles=1,
        )
        result = improver.run()
        self.assertEqual(len(result.trials), 1)
        self.assertIn("repeated", llm.prompts[1])

    def test_failed_trial_is_recorded_as_feedback(self):
        llm = ScriptedLLM([_reply({"model": "RNN", "train_params": {
            "pos_weight": "balanced"}, "model_params": {"embedding_dim": 16}})])
        improver = AutoImprover(
            self.train, self.val, LLMProposer(llm), space=SPACE,
            metric="no_such_metric", n_agents=1, n_cycles=1,
        )
        result = improver.run()
        self.assertIsNotNone(result.trials[0].error)
        self.assertRegex(result.trials[0].error, r" at \w+\.py:\d+")
        self.assertIsNone(result.best)

    def test_patient_folds_keep_patients_together(self):
        samples = _samples(30, 3)
        for i, s in enumerate(samples):
            s["patient_id"] = f"p{i // 3}"  # 10 patients, 3 samples each
        data = create_sample_dataset(
            samples, {"conditions": "sequence"}, {"label": "binary"}
        )
        folds = patient_folds(data, 4, seed=0)
        self.assertEqual(sorted(i for f in folds for i in f), list(range(30)))
        for f in folds:
            self.assertIn(len(f), (6, 9))
            patients = {data[int(i)]["patient_id"] for i in f}
            self.assertEqual(sum(len(data.patient_to_index[p]) for p in patients),
                             len(f))
        with self.assertRaises(ValueError):
            patient_folds(data, 11)

    def test_cross_validated_scores(self):
        llm = ScriptedLLM([_reply({"model": "MLP", "model_params": {
            "hidden_dim": 16, "embedding_dim": 16}})])
        improver = AutoImprover(
            self.train, self.val, LLMProposer(llm),
            baseline=PipelineConfig("LogisticRegression", {"embedding_dim": 16}),
            space=SPACE, n_agents=1, n_cycles=1, cv_folds=3,
        )
        result = improver.run()
        self.assertTrue(all(t.error is None for t in result.trials))
        self.assertIn("pr_auc_sd", result.trials[0].metrics)
        self.assertEqual(len(result.trials[0].fold_scores), 3)
        self.assertIn("cross-validation folds", llm.prompts[0])
        self.assertIn("(sd ", llm.prompts[0])
        self.assertIn("Current selection: #0", llm.prompts[0])
        self.assertIn("standard error", llm.prompts[0])
        # the selection follows the one-SE rule over the paired candidates
        self.assertIs(result.best,
                      select_simplest(result.trials, True, 1.0, SPACE.rung)[0])

    def test_clipped_repeat_names_the_clipping(self):
        llm = ScriptedLLM([
            _reply({"model": "RNN", "model_params": {"hidden_dim": 999,
                    "embedding_dim": 16}}),  # clipped to 256: a repeat
            _reply({"model": "MLP", "model_params": {"hidden_dim": 16,
                    "embedding_dim": 16}}),
        ])
        improver = AutoImprover(
            self.train, self.val, LLMProposer(llm),
            baseline=PipelineConfig("RNN", {"hidden_dim": 256, "embedding_dim": 16}),
            space=SPACE, n_agents=1, n_cycles=1,
        )
        result = improver.run()
        self.assertEqual([t.config.model for t in result.trials], ["RNN", "MLP"])
        self.assertIn("hidden_dim: clipped to 256", llm.prompts[1])
        self.assertIn("repeated an earlier trial", llm.prompts[1])

    def test_lessons_are_scoped_to_the_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.json")
            with open(path, "w") as f:
                json.dump(["an unscoped lesson"], f)  # older format: set aside
            first = ScriptedLLM([_reply({"model": "MLP", "model_params": {
                "hidden_dim": 16, "embedding_dim": 16}})])
            AutoImprover(self.train, self.val, LLMProposer(first), space=SPACE,
                         n_agents=1, n_cycles=1, lessons_path=path).run()
            self.assertNotIn("an unscoped lesson", first.prompts[0])

            linear = SearchSpace({"LogisticRegression": SPACE.models["LogisticRegression"]},
                                 SPACE.train_params)
            second = ScriptedLLM([_reply({"model": "LogisticRegression",
                                          "model_params": {"embedding_dim": 16}})])
            AutoImprover(self.train, self.val, LLMProposer(second), space=linear,
                         n_agents=1, n_cycles=1, lessons_path=path).run()
            # lessons learned with RNN/MLP available do not reach a linear-only search
            self.assertNotIn("RNN beats LogisticRegression", second.prompts[0])
            with open(path) as f:
                stored = json.load(f)
            self.assertEqual(len(stored), 2)
            self.assertTrue(all("models:" in v["search"] for v in stored.values()))

            third = ScriptedLLM([_reply({"model": "RNN", "model_params": {
                "hidden_dim": 32, "embedding_dim": 16}})])
            AutoImprover(self.train, self.val, LLMProposer(third), space=SPACE,
                         n_agents=1, n_cycles=1, lessons_path=path).run()
            self.assertIn("RNN beats LogisticRegression", third.prompts[0])

    def test_patience_counts_new_best_scores(self):
        # Every proposal scores higher than the last, but its fold scores are
        # too noisy to pass the lower-bound rule, so the selection never
        # changes. The search goes on while the best score improves and stops
        # after one cycle (patience=1) without a new best.
        def fake_cv(config, space, train, val, metric, folds, *args, **kwargs):
            mean = config.model_params["hidden_dim"] / 1000
            spread = 0.0 if config.model_params["hidden_dim"] == 16 else 0.2
            fold_scores = [mean + spread, mean - spread, mean]
            return {"metrics": {metric: mean}, "fold_scores": fold_scores,
                    "seconds": 0.0, "n_params": 0}

        dims = [32, 64, 128, 100, 200]
        llm = ScriptedLLM([_reply({"model": "MLP", "model_params": {
            "hidden_dim": d, "embedding_dim": 16}}) for d in dims])
        improver = AutoImprover(
            self.train, self.val, LLMProposer(llm),
            baseline=PipelineConfig("MLP", {"hidden_dim": 16, "embedding_dim": 16}),
            space=SPACE, n_agents=1, n_cycles=5, patience=1, cv_folds=3,
        )
        with mock.patch("pyhealth.autoimprove.improver.run_config_cv", fake_cv):
            result = improver.run()
        self.assertEqual(result.best.agent, "baseline")
        # baseline, 32, 64, 128, then 100 makes no progress and the run stops
        self.assertEqual(
            [t.config.model_params["hidden_dim"] for t in result.trials],
            [16, 32, 64, 128, 100],
        )

    def test_cross_validation_without_a_baseline(self):
        scores = {16: [0.50, 0.50, 0.50], 32: [0.40, 0.70, 0.66], 64: [0.55, 0.55, 0.55]}

        def fake_cv(config, space, train, val, metric, folds, *args, **kwargs):
            fold_scores = scores[config.model_params["hidden_dim"]]
            return {"metrics": {metric: sum(fold_scores) / 3}, "fold_scores": fold_scores,
                    "seconds": 0.0, "n_params": 0}

        llm = ScriptedLLM([_reply({"model": "MLP", "model_params": {
            "hidden_dim": d, "embedding_dim": 16}}) for d in (16, 32, 64)])
        with self.assertLogs("pyhealth.autoimprove.improver", "WARNING") as logs:
            improver = AutoImprover(self.train, self.val, LLMProposer(llm), space=SPACE,
                                    n_agents=3, n_cycles=1, cv_folds=3, check=False)
        self.assertIn("no baseline", logs.output[0])
        with mock.patch("pyhealth.autoimprove.improver.run_config_cv", fake_cv):
            result = improver.run()
        # the best mean, not the trial with the steadiest gain over the first one
        self.assertEqual(result.best.config.model_params["hidden_dim"], 32)
        self.assertIn("no starting pipeline to compare with", llm.prompts[1])

    def test_resuming_with_changed_settings_is_refused(self):
        baseline = PipelineConfig("LogisticRegression", {"embedding_dim": 16})
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")

            def improver(**kwargs):
                settings = {"baseline": baseline, "space": SPACE, "n_agents": 1,
                            "n_cycles": 1, "run_dir": run, "check": False, **kwargs}
                return AutoImprover(self.train, self.val, RandomProposer(seed=0),
                                    **settings)

            improver().run()
            with open(os.path.join(run, "meta.json")) as f:
                self.assertEqual(json.load(f)["metric"], "pr_auc")
            with self.assertRaisesRegex(ValueError, "different metric"):
                improver(metric="roc_auc").run()
            with self.assertRaisesRegex(ValueError, "different baseline"):
                improver(baseline=PipelineConfig("MLP", {"embedding_dim": 16})).run()
            with self.assertRaisesRegex(ValueError, "different cv_folds, promote_se"):
                improver(cv_folds=3).run()
            # a larger budget is not a different setting
            self.assertEqual(len(improver(n_cycles=2).run().trials), 3)
            # a run saved before meta.json existed is resumed, with a warning
            os.remove(os.path.join(run, "meta.json"))
            with self.assertLogs("pyhealth.autoimprove.improver", "WARNING"):
                improver(n_cycles=2).run()
            self.assertTrue(os.path.exists(os.path.join(run, "meta.json")))

    def test_error_messages_stay_out_of_prompts(self):
        secret = "raw-value-4711"

        def failing(*args, **kwargs):
            raise ValueError(f"could not parse {secret!r}")

        llm = ScriptedLLM([_reply({"model": "MLP", "model_params": {
            "hidden_dim": d, "embedding_dim": 16}}) for d in (16, 32)])
        improver = AutoImprover(self.train, self.val, LLMProposer(llm), space=SPACE,
                                n_agents=1, n_cycles=2, check=False)
        with mock.patch("pyhealth.autoimprove.improver.run_config", failing):
            result = improver.run()
        self.assertIn(secret, result.trials[0].error)  # kept for the user
        self.assertIn("error: ValueError at", llm.prompts[1])
        self.assertNotIn(secret, "".join(llm.prompts))

    def test_proposer_errors_skip_the_turn(self):
        class FlakyLLM(ScriptedLLM):
            failures = 1

            def __call__(self, system, prompt):
                if self.failures:
                    self.failures -= 1
                    raise RuntimeError("Claude declined the request")
                return super().__call__(system, prompt)

        llm = FlakyLLM([_reply({"model": "MLP", "model_params": {
            "hidden_dim": 16, "embedding_dim": 16}})])
        improver = AutoImprover(self.train, self.val, LLMProposer(llm), space=SPACE,
                                n_agents=2, n_cycles=1, check=False)
        with self.assertLogs("pyhealth.autoimprove.improver", "WARNING") as logs:
            result = improver.run()
        self.assertEqual([t.agent for t in result.trials], ["agent1"])
        self.assertTrue(any("turn skipped" in line for line in logs.output))

        llm.failures = 10  # a broken backend stops the run
        improver = AutoImprover(self.train, self.val, LLMProposer(llm), space=SPACE,
                                n_agents=3, n_cycles=1, check=False)
        with self.assertRaisesRegex(RuntimeError, "failed 3 turns in a row"):
            improver.run()

    def test_patient_folds_are_stratified(self):
        samples = _samples(40, 4)
        for i, s in enumerate(samples):
            s["label"] = int(i % 8 == 0)  # 5 positive patients of 40
        data = create_sample_dataset(
            samples, {"conditions": "sequence"}, {"label": "binary"}
        )
        for seed in range(3):
            folds = patient_folds(data, 5, seed=seed)
            self.assertEqual([sum(int(data[int(i)]["label"]) for i in f) for f in folds],
                             [1] * 5)
            self.assertEqual([len(f) for f in folds], [8] * 5)
        with self.assertLogs("pyhealth.autoimprove.evaluate", "WARNING"):
            patient_folds(data, 8)

        with (mock.patch("pyhealth.autoimprove.evaluate.MAX_LABEL_SCAN", 10),
              self.assertLogs("pyhealth.autoimprove.evaluate", "INFO") as logs):
            folds = patient_folds(data, 5)
        self.assertIn("not stratified", logs.output[0])
        self.assertEqual(sorted(int(i) for f in folds for i in f), list(range(40)))

    def test_models_that_fail_the_check_are_removed(self):
        space = SearchSpace(
            {**SPACE.models, "Unreadable": ModelSpec(f"{__name__}.UnreadableInputs")},
            SPACE.train_params,
        )
        improver = AutoImprover(
            self.train, self.val, RandomProposer(seed=0),
            baseline=PipelineConfig("LogisticRegression", {"embedding_dim": 16}),
            space=space, n_agents=2, n_cycles=1,
        )
        result = improver.run()
        self.assertEqual(list(result.dropped_models), ["Unreadable"])
        self.assertIn("cannot read 'conditions'", result.dropped_models["Unreadable"])
        self.assertNotIn("Unreadable", improver.space.models)
        self.assertTrue(all(t.error is None for t in result.trials))
        self.assertIn("Not searched: Unreadable", result.summary())

        with self.assertRaisesRegex(ValueError, "baseline model Unreadable"):
            AutoImprover(self.train, self.val, RandomProposer(), space=space,
                         baseline=PipelineConfig("Unreadable")).run()
        only = SearchSpace({"Unreadable": space.models["Unreadable"]}, SPACE.train_params)
        with self.assertRaisesRegex(ValueError, "No model in the search space runs"):
            AutoImprover(self.train, self.val, RandomProposer(), space=only).run()

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "needs xgboost")
    def test_models_that_cannot_read_the_inputs_are_removed(self):
        # XGBoost cannot turn code sequences into fixed-width columns.
        space = with_xgboost(SPACE)
        problems = check_models(space, self.train)
        self.assertEqual([k for k, v in problems.items() if v], ["XGBoost"])
        self.assertIn("bag_of_codes", problems["XGBoost"])

        improver = AutoImprover(
            self.train, self.val, RandomProposer(seed=0),
            baseline=PipelineConfig("LogisticRegression", {"embedding_dim": 16}),
            space=space, n_agents=3, n_cycles=1,
        )
        result = improver.run()
        self.assertEqual(list(result.dropped_models), ["XGBoost"])
        self.assertNotIn("XGBoost", improver.space.models)
        self.assertTrue(all(t.error is None for t in result.trials))
        self.assertIn("Not searched: XGBoost", result.summary())

        with self.assertRaisesRegex(ValueError, "baseline model XGBoost"):
            AutoImprover(self.train, self.val, RandomProposer(), space=space,
                         baseline=PipelineConfig("XGBoost")).run()

    def test_balanced_pos_weight_uses_the_training_labels(self):
        config, _ = SPACE.validate(PipelineConfig(
            "LogisticRegression", {"embedding_dim": 16}, {"pos_weight": "balanced"}))
        model = train_pipeline(config, SPACE, self.train, self.val, "pr_auc")
        positives = sum(s["label"] for s in _samples(80, 0))
        self.assertAlmostEqual(float(model.pos_weight), (80 - positives) / positives,
                               places=5)

    def test_random_search_budget_and_test_comparison(self):
        improver = AutoImprover(
            self.train, self.val, RandomProposer(seed=1),
            baseline=PipelineConfig("LogisticRegression", {"embedding_dim": 16}),
            space=SPACE, n_agents=2, n_cycles=5, max_trials=3,
        )
        result = improver.run()
        self.assertEqual(len(result.trials), 3)
        self.assertEqual(len({t.config.key() for t in result.trials}), 3)
        self.assertIn("| # |", result.summary())

        test = improver.evaluate_on_test(self.test, n_boot=50)
        self.assertIn("best", test)
        self.assertIn("baseline", test)
        self.assertIn("difference", test)
        self.assertLessEqual(test["best"]["lower"], test["best"]["upper"])
        self.assertNotIn("predictions", test)


class TestLabelTypes(unittest.TestCase):
    def test_label_type_sets_the_metric_and_the_space(self):
        cases = [
            ("binary", lambda codes: int("c1" in codes), "pr_auc", True),
            (BinaryLabelProcessor, lambda codes: int("c1" in codes), "pr_auc", True),
            ("multiclass", _three_classes, "f1_macro", False),
            ("multilabel", lambda codes: sorted({"c1", "c2"} & set(codes)) or ["none"],
             "pr_auc_samples", True),
            ("regression", lambda codes: float(_three_classes(codes)), "mse", False),
        ]
        for label_type, label, metric, weighted in cases:
            with self.subTest(label_type=label_type):
                samples = _samples(20, 0)
                for s in samples:
                    s["label"] = label(s["conditions"])
                data = create_sample_dataset(
                    samples, {"conditions": "sequence"}, {"label": label_type}
                )
                improver = AutoImprover(data, data, RandomProposer(), space=SPACE,
                                        check=False)
                self.assertEqual(improver.metric, metric)
                self.assertEqual("pos_weight" in improver.space.train_params, weighted)

    def test_multiclass_search_and_test_scores(self):
        train, val, test = _datasets("multiclass", _three_classes)
        improver = AutoImprover(
            train, val, RandomProposer(seed=0),
            baseline=PipelineConfig("LogisticRegression", {"embedding_dim": 16},
                                    {"pos_weight": "balanced"}),
            space=SPACE, n_agents=2, n_cycles=1,
        )
        result = improver.run()
        self.assertTrue(all(t.error is None for t in result.trials))
        self.assertFalse(any("pos_weight" in t.config.train_params for t in result.trials))

        scores = improver.evaluate_on_test(test, return_predictions=True)
        self.assertNotIn("difference", scores)
        self.assertIn("estimate", scores["best"])
        self.assertIn("estimate", scores["baseline"])
        self.assertEqual(scores["predictions"]["best"].shape, (40, 3))


class TestSelection(unittest.TestCase):
    @staticmethod
    def _trial(i, score, folds, error=None):
        return Trial(i, 0, f"a{i}", "explore", PipelineConfig("MLP", {"x": i}),
                     metric_name="pr_auc", score=score, fold_scores=folds, error=error)

    def test_paired_rule(self):
        base = self._trial(0, 0.50, [0.48, 0.50, 0.52])
        base.agent = "baseline"
        noisy = self._trial(1, 0.52, [0.40, 0.58, 0.58])        # higher mean, not reliable
        steady = self._trial(2, 0.515, [0.495, 0.515, 0.535])   # +0.015 on every fold
        failed = self._trial(3, None, [], error="boom")
        trials = [base, noisy, steady, failed]
        self.assertIs(select_trial(trials, True), noisy)
        self.assertIs(select_trial(trials, True, promote_se=1.0), steady)
        self.assertIs(select_trial(trials[:2], True, promote_se=1.0), base)
        self.assertEqual(paired_gain([1.0, 2.0], [2.0, 3.0], maximize=False), (1.0, 0.0))
        self.assertIsNone(select_trial([failed], True, 1.0))

    def test_without_a_baseline_the_best_mean_is_selected(self):
        # No trial to compare with: the first finished trial must not act as one.
        first = self._trial(0, 0.50, [0.50, 0.50, 0.50])
        noisy = self._trial(1, 0.587, [0.40, 0.70, 0.66])
        steady = self._trial(2, 0.55, [0.55, 0.55, 0.55])
        tie = self._trial(3, 0.587, [0.60, 0.58, 0.58])
        for order in ([first, noisy, steady, tie], [tie, steady, noisy, first]):
            self.assertIs(select_trial(order, True, promote_se=1.0), noisy)
        failed_base = self._trial(4, None, [], error="boom")
        failed_base.agent = "baseline"
        self.assertIs(select_trial([failed_base, first, steady], True, 1.0), steady)

    def test_selection_ignores_trial_order(self):
        base = self._trial(0, 0.50, [0.50, 0.50, 0.50, 0.50])
        base.agent = "baseline"
        early = self._trial(1, 0.52, [0.49, 0.55, 0.50, 0.54])  # +0.02, noisy
        steady = self._trial(2, 0.519, [0.518, 0.52, 0.519, 0.519])  # +0.019, steady
        later_noise = self._trial(3, 0.525, [0.45, 0.60, 0.50, 0.55])
        trials = [base, early, steady, later_noise]
        self.assertIs(select_trial(trials, True, 1.0), steady)
        for order in ([base, steady, early, later_noise], [later_noise, early, steady, base]):
            self.assertIs(select_trial(order, True, 1.0), steady)
        self.assertIs(select_trial([base, early, later_noise], True, 1.0), early)


class TestOneSERule(unittest.TestCase):
    @staticmethod
    def _trial(i, model, score, folds, agent=None, params=None):
        return Trial(i, 0, agent or f"a{i}", "explore",
                     PipelineConfig(model, params or {}), metric_name="pr_auc",
                     score=score, fold_scores=folds)

    def test_simpler_model_within_one_se_wins(self):
        space = with_xgboost(SearchSpace())
        base = self._trial(0, "XGBoost", 0.50, [0.48, 0.50, 0.52], "baseline",
                           {"max_depth": 4})
        lr = self._trial(1, "LogisticRegression", 0.55, [0.52, 0.57, 0.56])
        deep = self._trial(2, "XGBoost", 0.56, [0.55, 0.555, 0.575], params={
            "max_depth": 6})
        noisy = self._trial(3, "MLP", 0.60, [0.40, 0.80, 0.60])  # not a candidate
        selected, top = select_simplest([base, lr, deep, noisy], True, 1.0, space.rung)
        self.assertEqual((selected.trial_id, top.trial_id), (1, 2))
        # without a ladder the top candidate is selected
        self.assertEqual(select_simplest([base, lr, deep], True, 1.0)[0].trial_id, 2)

    def test_top_kept_when_simpler_is_clearly_worse(self):
        space = with_xgboost(SearchSpace())
        base = self._trial(0, "XGBoost", 0.50, [0.50, 0.50, 0.50], "baseline")
        lr = self._trial(1, "LogisticRegression", 0.52, [0.519, 0.52, 0.521])
        deep = self._trial(2, "XGBoost", 0.60, [0.599, 0.60, 0.601])
        selected, top = select_simplest([base, lr, deep], True, 1.0, space.rung)
        self.assertIs(selected, deep)
        self.assertIs(top, deep)

    def test_baseline_stays_without_consistent_gains(self):
        base = self._trial(0, "MLP", 0.50, [0.48, 0.50, 0.52], "baseline")
        noisy = self._trial(1, "LogisticRegression", 0.52, [0.40, 0.58, 0.58])
        selected, top = select_simplest([base, noisy], True, 1.0, SearchSpace().rung)
        self.assertIs(selected, base)
        self.assertIs(top, base)

    def test_without_a_baseline_every_trial_is_a_candidate(self):
        # The first trial must not act as a baseline that others have to beat.
        space = with_xgboost(SearchSpace())
        first = self._trial(0, "XGBoost", 0.50, [0.50, 0.50, 0.50])
        lr = self._trial(1, "LogisticRegression", 0.55, [0.45, 0.65, 0.55])
        deep = self._trial(2, "XGBoost", 0.56, [0.50, 0.66, 0.52])
        for order in ([first, lr, deep], [deep, lr, first]):
            selected, top = select_simplest(order, True, 1.0, space.rung)
            self.assertEqual((selected.trial_id, top.trial_id), (1, 2))
        report = build_report([first, lr, deep], metric="pr_auc", promote_se=1.0,
                              space=space)
        why = report.to_markdown()
        self.assertIn("no starting pipeline to compare with", why)
        self.assertNotIn("starting pipeline;", why)  # no trial is labelled as one

    def test_ladder_and_plain_words(self):
        space = with_xgboost(SearchSpace())
        rungs = [space.rung(PipelineConfig("LogisticRegression")),
                 space.rung(PipelineConfig("XGBoost", {"max_depth": 1})),
                 space.rung(PipelineConfig("XGBoost", {"max_depth": 3})),
                 space.rung(PipelineConfig("MLP")),
                 space.rung(PipelineConfig("RNN"))]
        self.assertEqual(rungs, [1, 2, 3, 3, 4])
        words = space.explain(PipelineConfig(
            "XGBoost", {"max_depth": 1, "scale_pos_weight": None}))
        self.assertIn("additive", words[1])
        self.assertIn("observed frequencies", words[2])


class TestExplainedRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.val, cls.test = _datasets()

    def test_decision_log_records_hypothesis_and_change(self):
        llm = ScriptedLLM([
            _reply({"model": "MLP", "model_params": {"hidden_dim": 16,
                    "embedding_dim": 16}, "rationale": "a small MLP",
                    "expected": "PR-AUC up", "kind": "technical"}),
            _reply({"model": "MLP", "model_params": {"hidden_dim": 32,
                    "embedding_dim": 16}, "based_on": 1,
                    "rationale": "wider"}),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            improver = AutoImprover(
                self.train, self.val, LLMProposer(llm),
                baseline=PipelineConfig("LogisticRegression", {"embedding_dim": 16}),
                space=SPACE, n_agents=1, n_cycles=2, patience=5, run_dir=tmp,
            )
            result = improver.run()
            first, second = result.trials[1], result.trials[2]
            self.assertEqual(first.parent, 0)  # nearest earlier trial
            self.assertEqual(first.change["model"], "LogisticRegression → MLP")
            self.assertEqual(first.config.hypothesis,
                             {"expected": "PR-AUC up", "kind": "technical"})
            self.assertEqual((second.parent, second.change),
                             (1, {"hidden_dim": "16 → 32"}))
            # the log round-trips, and older records without the fields load
            with open(os.path.join(tmp, "trials.jsonl")) as f:
                saved = [json.loads(line) for line in f]
            self.assertEqual(Trial.from_dict(saved[2]).change, second.change)
            old = {k: v for k, v in saved[1].items() if k not in ("parent", "change")}
            old["unknown_future_field"] = 1
            self.assertIsNone(Trial.from_dict(old).parent)
            self.assertTrue(os.path.exists(os.path.join(tmp, "run.json")))

    def test_attribution_reverts_one_setting_at_a_time(self):
        # depth 4 -> 2 is worth 0.04 and n_estimators 300 -> 600 nothing
        def fake_cv(config, space, train, val, metric, folds, *args, **kwargs):
            p = config.model_params
            mean = 0.50 + (0.04 if p["max_depth"] == 2 else 0.0)
            return {"metrics": {metric: mean},
                    "fold_scores": [mean - 0.01, mean, mean + 0.01],
                    "seconds": 0.0, "n_params": 0}

        target = {"max_depth": 2, "n_estimators": 600}
        llm = ScriptedLLM([_reply({"model": "XGBoost", "model_params": target})])
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "pyhealth.autoimprove.improver.run_config_cv", fake_cv
        ):
            improver = AutoImprover(
                self.train, self.val, LLMProposer(llm),
                baseline=PipelineConfig("XGBoost", {"max_depth": 4}),
                space=with_xgboost(SPACE), n_agents=1, n_cycles=1, cv_folds=3,
                run_dir=tmp, check=False,
            )
            result = improver.run()
            self.assertEqual(result.best.trial_id, 1)
            out = improver.attribute()
            rows = {r["setting"]: r for r in out["rows"]}
            self.assertEqual(set(rows), {"max_depth", "n_estimators"})
            self.assertAlmostEqual(rows["max_depth"]["contribution"], 0.04)
            self.assertAlmostEqual(rows["n_estimators"]["contribution"], 0.0)
            self.assertEqual(out["reference"], "the starting pipeline (#0)")
            self.assertTrue(os.path.exists(os.path.join(tmp, "attribution.json")))

            report = improver.report()
            text = report.to_markdown()
            self.assertIn("What each change contributed", text)
            self.assertIn("max_depth", text)
            for name in ("report.html", "report.md", "report.json"):
                self.assertTrue(os.path.exists(os.path.join(tmp, name)))
            # rebuilt from the run directory alone, the report is the same
            self.assertEqual(load_run(tmp).to_markdown(), text)

    def test_report_from_a_real_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            improver = AutoImprover(
                self.train, self.val, RandomProposer(seed=1),
                baseline=PipelineConfig("LogisticRegression", {"embedding_dim": 16}),
                space=SPACE, n_agents=2, n_cycles=2, cv_folds=3, run_dir=tmp,
            )
            result = improver.run()
            test = improver.evaluate_on_test(self.test, n_boot=50, return_predictions=True)
            with open(os.path.join(tmp, "test.json")) as f:
                saved = json.load(f)
            self.assertEqual(saved["best"]["trial_id"], result.best.trial_id)
            self.assertIn("predictions", test)
            self.assertNotIn("predictions", saved)  # aggregates only on disk
            self.assertEqual(test["metric"], "pr_auc")
            report = improver.report()
            titles = [s.title for s in report.sections]
            self.assertEqual(titles, [
                "Summary", "Task and data", "The model and why",
                "Performance in clinical terms", "Reasoning trace",
                "Clinician review", "Reporting checklist", "How to read this report",
            ])
            html_text = report.to_html()
            self.assertIn("<table>", html_text)
            self.assertIn("Why this model", html_text)
            verdicts = report.data["verdicts"]
            self.assertEqual(verdicts[result.best.trial_id], "selected")
            self.assertEqual(len(verdicts), len(result.trials))

    def test_number_checker(self):
        trials = [Trial(0, -1, "baseline", "baseline", PipelineConfig("MLP"),
                        score=0.2164),
                  Trial(1, 0, "a0", "explore",
                        PipelineConfig("MLP", {}, {"weight_decay": 0.01}),
                        score=0.2950)]
        text = ("#1 scored 0.295 vs 0.216 (+0.079) with weight decay 0.01; "
                "batch 32 dropped to 0.256")
        self.assertEqual(unverified_numbers(text, trials), ["0.256"])


def _tabular(n, seed, offset=0):
    """Three numeric columns: one raises risk, one lowers it, one is noise."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        x = [rng.gauss(0, 1) for _ in range(3)]
        risk = 1 / (1 + math.exp(-(2.0 * x[0] - 1.5 * x[1])))
        out.append({"patient_id": f"q{offset + i}", "tab": x,
                    "label": int(rng.random() < risk)})
    return out


def _tabular_datasets():
    schema = ({"tab": "tensor"}, {"label": "binary"})
    train = create_sample_dataset(_tabular(300, 0), *schema)
    kwargs = {"input_processors": train.input_processors,
              "output_processors": train.output_processors}
    val = create_sample_dataset(_tabular(80, 1, 1000), *schema, **kwargs)
    test = create_sample_dataset(_tabular(200, 2, 2000), *schema, **kwargs)
    return train, val, test


# The card expects "protect" to raise risk; the data say it lowers it.
CARD = TaskCard(
    population=("synthetic patients", "a test of the explanation"),
    outcome="a planted binary label",
    features={
        "risk_up": FeatureInfo("planted risk factor", "labs", "+"),
        "protect": FeatureInfo("planted protective factor", "labs", "+"),
        "noise": FeatureInfo("unrelated value", "other", ""),
    },
    columns={"tab": ["risk_up", "protect", "noise"]},
    notes="No missing values.",
)


class TestClinicalReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.val, cls.test = _tabular_datasets()

    def test_explain_flags_review_and_privacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            improver = AutoImprover(
                self.train, self.val, RandomProposer(seed=0),
                baseline=PipelineConfig("LogisticRegression", {"embedding_dim": 16},
                                        {"epochs": 2}),
                space=SPACE, n_agents=1, n_cycles=1, cv_folds=3, run_dir=tmp,
                task_card=CARD,
            )
            improver.run()
            explanation = improver.explain(top_k=3)
            by_name = {f["name"]: f for f in explanation["features"]}
            self.assertEqual(by_name["risk_up"]["learned"], "+")
            self.assertEqual(by_name["risk_up"]["check"], "as expected")
            self.assertEqual(by_name["protect"]["learned"], "-")
            self.assertEqual(by_name["protect"]["check"], "opposite")
            self.assertEqual(explanation["groups"][0]["group"], "labs")
            self.assertEqual(explanation["stability"]["models"], 3)

            ids = [f["id"] for f in improver.flags()]
            self.assertIn("direction:protect", ids)
            self.assertNotIn("direction:risk_up", ids)
            improver.approve("direction:protect", "Dr. Test", "planted in the data")
            flag = next(f for f in improver.flags() if f["id"] == "direction:protect")
            self.assertEqual(flag["review"]["decision"], "approved")
            with self.assertRaises(ValueError):
                improver.approve("no_such_flag", "Dr. Test")

            halves = ["early" if i < 100 else "late" for i in range(len(self.test))]
            test = improver.evaluate_on_test(self.test, n_boot=50,
                                             subgroups={"half": halves})
            clinical = test["best"]["clinical"]
            self.assertIn("calibration_slope", clinical["calibration"])
            self.assertIn("half", clinical["subgroups"])

            report = improver.report()
            titles = [s.title for s in report.sections]
            self.assertIn("How it decides", titles)
            self.assertIn("Performance in clinical terms", titles)
            text = report.to_markdown()
            self.assertIn("approved by Dr. Test", text)
            self.assertIn("planted protective factor", text)
            self.assertIn("TRIPOD+AI", text)
            self.assertEqual(load_run(tmp, space=SPACE).to_markdown(), text)

            # Privacy: no patient ids, and every published cell has 11+ patients.
            blob = report.to_json() + report.to_html()
            self.assertIsNone(re.search(r"\bq\d+\b", blob))
            for f in report.data["explanation"]["features"]:
                self.assertTrue(all(b["n"] >= 11 for b in f["bins"]))
            for sub in clinical["subgroups"].values():
                for row in sub["groups"].values():
                    self.assertTrue(row["n"] is None or row["n"] >= 11)
            self.assertTrue(all(b["n"] >= 11 for b in clinical["calibration_bins"]))

    def test_logistic_attributions_are_exact(self):
        model = LogisticRegression(self.train, embedding_dim=8)
        out = attributions(model, self.train, max_samples=100)
        values = out["fields"]["tab"]["values"]
        contribs = out["fields"]["tab"]["contributions"]
        layer = model.embedding_model.embedding_layers["tab"]
        weights = (model.fc.weight @ layer.weight).detach().numpy().ravel()
        # integrated gradients start from the mean of the explained samples
        expected = (values - values.mean(axis=0)) * weights
        np.testing.assert_allclose(contribs, expected, atol=1e-4)

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "needs xgboost")
    def test_tree_attributions(self):
        from pyhealth.models import XGBoostModel

        model = XGBoostModel(self.train, n_estimators=50, max_depth=2).fit(self.train)
        out = summarize(attributions(model, self.train), CARD, top_k=3)
        self.assertTrue(out["method"].startswith("TreeSHAP"))
        by_name = {f["name"]: f for f in out["features"]}
        self.assertEqual(by_name["risk_up"]["learned"], "+")
        self.assertEqual(by_name["protect"]["check"], "opposite")


class TestClaudeLLM(unittest.TestCase):
    def _client(self, stop_reason="end_turn"):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                stop_reason=stop_reason,
                content=[SimpleNamespace(type="thinking"),
                         SimpleNamespace(type="text", text="hello")],
            )

        messages = SimpleNamespace(create=create)
        client = SimpleNamespace(messages=messages,
                                 beta=SimpleNamespace(messages=messages))
        return client, calls

    def test_request_shape(self):
        client, calls = self._client()
        self.assertEqual(ClaudeLLM(client=client)("sys", "hi"), "hello")
        self.assertEqual(calls[0]["model"], "claude-opus-5-5")
        self.assertEqual(calls[0]["fallbacks"], "default")
        self.assertEqual(calls[0]["system"], "sys")

        ClaudeLLM(client=client, fallback=False)("sys", "hi")
        self.assertNotIn("fallbacks", calls[1])

    def test_refusal_raises(self):
        client, _ = self._client(stop_reason="refusal")
        with self.assertRaises(RuntimeError):
            ClaudeLLM(client=client)("sys", "hi")

    def test_old_sdk_is_refused(self):
        old = SimpleNamespace(__version__="0.40.0", Anthropic=lambda: None)
        with (mock.patch.dict(sys.modules, {"anthropic": old}),
              self.assertRaisesRegex(ImportError, "anthropic>=1.15")):
            ClaudeLLM()

    @unittest.skipUnless(
        os.environ.get("ANTHROPIC_API_KEY") and importlib.util.find_spec("anthropic"),
        "needs ANTHROPIC_API_KEY and the anthropic package",
    )
    def test_live_request(self):
        # The SDK and the API accept the request shape: the fallback beta
        # header, fallbacks and output_config.
        reply = ClaudeLLM(effort="low", max_tokens=1000)(
            "Answer in one word.", "What colour is the sky on a clear day?")
        self.assertTrue(reply.strip())


if __name__ == "__main__":
    unittest.main()
