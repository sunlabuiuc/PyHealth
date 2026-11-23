"""pyhealth.autoimprove evaluation turns: adoption checks and guardrails."""

import json
import os
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

from pyhealth.autoimprove import (
    AutoImprover,
    EvaluationSpec,
    Param,
    PipelineConfig,
    RunConfig,
    SearchSpace,
    clearly_worse,
    load_run,
)
from pyhealth.autoimprove.evaluation.guardrails import (
    decide,
    metrics_menu,
    perfect_predictions,
)
from pyhealth.autoimprove.search.proposers import Proposer
from pyhealth.autoimprove.search.prompts import build_evaluation_prompt
from pyhealth.datasets import create_sample_dataset

SPACE = SearchSpace(
    train_params={
        "epochs": Param("int", 1, 2, default=1),
        "lr": Param("float", 1e-4, 1e-1, log=True, default=1e-2),
        "batch_size": Param("choice", choices=(16, 32), default=16),
    }
)
SPACE.models = {k: SPACE.models[k] for k in ("LogisticRegression", "MLP")}

SHIFT = """\
import numpy as np


def evaluate(y_true, y_prob, info):
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    return {"abs_shift": float(abs(p.mean() - y.mean()))}
"""

BRIER = """\
import numpy as np
from pyhealth.metrics import binary_metrics_fn


def evaluate(y_true, y_prob, info):
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    return {"brier": float(binary_metrics_fn(y, p, metrics=["brier"])["brier"])}
"""


def _samples(n, seed, offset=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        codes = rng.sample([f"c{j}" for j in range(8)], 3)
        out.append({"patient_id": f"p{offset + i}", "visit_id": f"v{offset + i}",
                    "conditions": codes, "label": int("c1" in codes)})
    return out


def _spec(name, metrics, code=SHIFT, needs=()):
    return EvaluationSpec.from_dict({"name": name, "metrics": metrics,
                                     "needs": list(needs)}, code)


def _result(values, deterministic=True, perfect=0.0, constant=0.2):
    return {"values": values, "deterministic": deterministic,
            "perfect": {"m": perfect}, "constant": {"m": constant}}


class TestSpecAndRules(unittest.TestCase):
    def test_spec_validation(self):
        ok = _spec("calibration", {"abs_shift": "lower"})
        self.assertEqual(ok.keys(), ["calibration.abs_shift"])
        self.assertEqual(_spec("c", {"m": {"direction": "higher"}}).metrics,
                         {"m": "higher"})
        for meta, code, error in (
            ({"name": "Bad Name", "metrics": {"m": "lower"}}, SHIFT, "identifier"),
            ({"name": "c", "metrics": {}}, SHIFT, "higher"),
            ({"name": "c", "metrics": {"m": "smaller"}}, SHIFT, "direction"),
            ({"name": "c", "metrics": {"m": "lower"}, "needs": "age"}, SHIFT, "list"),
            ({"name": "c", "metrics": {"m": "lower"}},
             "import os\ndef evaluate(y, p, i):\n    return {}\n", "import of os"),
            ({"name": "c", "metrics": {"m": "lower"}}, "x = 1\n", "evaluate"),
        ):
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    EvaluationSpec.from_dict(meta, code)

    def test_clearly_worse_and_perfect_predictions(self):
        self.assertTrue(clearly_worse([0.6, 0.62, 0.61], [0.7, 0.71, 0.72], "higher"))
        self.assertFalse(clearly_worse([0.7, 0.62, 0.75], [0.7, 0.71, 0.72], "higher"))
        self.assertFalse(clearly_worse([0.9], [0.1], "lower"))  # one fold: unclear
        self.assertFalse(clearly_worse([float("nan")] * 3, [0.1] * 3, "lower"))
        self.assertEqual(perfect_predictions(np.array([1, 0]), (2,)).tolist(),
                         [1.0, 0.0])

    def test_decide(self):
        spec = _spec("g", {"m": "lower"})
        base = [{"m": 0.10}, {"m": 0.11}, {"m": 0.12}]
        worse = [{"m": 0.30}, {"m": 0.31}, {"m": 0.32}]
        fine = [{"m": 0.09}, {"m": 0.12}, {"m": 0.10}]
        nan = [{"m": float("nan")}] * 3

        def run(values, **kwargs):
            args = dict(trial_ids=list(range(len(values))), baseline_id=0,
                        candidates=list(range(1, len(values))), n_active=0,
                        max_guardrails=3)
            args.update(kwargs.pop("args", {}))
            return decide(spec, _result(values, **kwargs), **args)

        self.assertEqual(run([base, worse, fine, fine]), ("active", "passed every check"))
        status, why = run([base, worse, fine], deterministic=False)
        self.assertEqual((status, "different values" in why), ("rejected", True))
        status, why = run([nan, worse, fine])
        self.assertEqual((status, "baseline" in why), ("rejected", True))
        status, why = run([base, nan, nan, fine, fine])
        self.assertEqual((status, "only 60%" in why), ("rejected", True))
        status, why = run([base, worse, fine], perfect=0.5, constant=0.1)
        self.assertEqual((status, "backwards" in why), ("rejected", True))
        status, why = run([base, base, base])
        self.assertEqual((status, "same value" in why), ("report_only", True))
        status, why = run([base, worse, worse, fine])
        self.assertEqual((status, "remove 2 of 3" in why), ("report_only", True))
        status, why = run([base, worse, fine, fine], args={"n_active": 3})
        self.assertEqual((status, "already active" in why), ("report_only", True))
        status, why = run([base, worse], args={"baseline_id": 9})
        self.assertEqual(status, "rejected")

    def test_metrics_menu_lists_pyhealth_metrics(self):
        menu = metrics_menu("binary")
        for name in ("binary_metrics_fn(", "subgroup_metrics(", "net_benefit("):
            self.assertIn(name, menu)


class GuardrailProposer(Proposer):
    """Two MLP configs for model turns; queued evaluations for evaluation turns."""

    def __init__(self, evaluations):
        self.configs = [32, 64]
        self.evaluations = list(evaluations)
        self.contexts = []

    def propose(self, context):
        if not self.configs:
            raise ValueError("nothing left")
        return PipelineConfig("MLP", {"embedding_dim": 16,
                                      "hidden_dim": self.configs.pop(0)})

    def propose_evaluation(self, context):
        self.contexts.append(context)
        return self.evaluations.pop(0)


class TestGuardrailsInARun(unittest.TestCase):
    """#1 has the best score but is miscalibrated; #2 is good on both."""

    @classmethod
    def setUpClass(cls):
        schema = ({"conditions": "sequence"}, {"label": "binary"})
        cls.samples = _samples(90, 0)
        cls.train = create_sample_dataset(cls.samples, *schema)
        kwargs = {"input_processors": cls.train.input_processors,
                  "output_processors": cls.train.output_processors}
        cls.val = create_sample_dataset(_samples(40, 1, 1000), *schema, **kwargs)
        cls.test = create_sample_dataset(_samples(60, 2, 2000), *schema, **kwargs)
        cls.labels = np.array([s["label"] for s in cls.samples], dtype=float)

    def fake_cv(self, config, space, train, val, metric, folds, *args, **kwargs):
        hidden = config.model_params["hidden_dim"]
        y = self.labels
        prev = y.mean()
        prob = {16: 0.5 * y + 0.5 * prev,      # baseline, calibrated
                32: 0.5 * y + 0.45,            # best PR-AUC, shifted up
                64: 0.6 * y + 0.4 * prev}[hidden]
        scores = {16: [0.50, 0.51, 0.49], 32: [0.70, 0.72, 0.71],
                  64: [0.60, 0.62, 0.61]}[hidden]
        fold = np.zeros(len(y), dtype=np.int16)
        for k, held in enumerate(folds):
            fold[held] = k
        oof = {"index": np.arange(len(y)), "fold": fold, "y_true": y[:, None],
               "y_prob": prob[:, None].astype(np.float32), "split": "train"}
        return {"metrics": {metric: float(np.mean(scores))}, "fold_scores": scores,
                "seconds": 0.0, "n_params": 0, "stopped_early": False, "oof": oof}

    def improver(self, tmp, proposer):
        return AutoImprover(
            self.train, self.val, RunConfig(
                # one model round, then two evaluations proposed after it
                search={"proposals_per_round": 2},
                budget={"rounds": 1, "patience": 5},
                evaluation={"cv_folds": 3, "check": False},
                guardrails={"enabled": True, "turns": 2},
                output={"run_dir": tmp},
                baseline=PipelineConfig("MLP", {"embedding_dim": 16, "hidden_dim": 16}),
            ),
            proposer=proposer, space=SPACE,
        )

    def test_guardrail_changes_the_selection(self):
        shift = _spec("calibration", {"abs_shift": "lower"})
        backwards = _spec("accuracy_like", {"brier": "higher"}, code=BRIER)
        proposer = GuardrailProposer([shift, backwards])
        with tempfile.TemporaryDirectory() as tmp:
            improver = self.improver(tmp, proposer)
            with mock.patch("pyhealth.autoimprove.state.run_config_cv",
                            self.fake_cv):
                result = improver.run()
            by_hidden = {t.config.model_params["hidden_dim"]: t.trial_id
                         for t in result.trials}
            status = {e["spec"].name: (e["status"], e["reason"])
                      for e in improver.evaluations}
            self.assertEqual(status["calibration"][0], "active")
            self.assertEqual(status["accuracy_like"][0], "rejected")
            self.assertIn("backwards", status["accuracy_like"][1])
            # without the guardrail #1 (hidden 32) would win; it is removed
            self.assertEqual(improver.guardrail_failures(),
                             {by_hidden[32]: ["calibration.abs_shift"]})
            self.assertEqual(result.best.trial_id, by_hidden[64])
            # every trial was scored, old ones included
            scores = improver.evaluations[0]["scores"]
            self.assertEqual(sorted(scores), sorted(by_hidden.values()))
            self.assertEqual(len(scores[by_hidden[16]]), 3)
            # the prompt listed pyhealth.metrics and the columns
            prompt = build_evaluation_prompt(proposer.contexts[0])
            self.assertIn("binary_metrics_fn(", prompt)
            self.assertIn("n_conditions", prompt)
            # later prompts show the guardrail and who fails it
            lines = improver.state._context("agent0", "explore", "").evaluations
            self.assertIn(f"fails: #{by_hidden[32]}", lines[0])
            # saved, reported, and restored on resume
            with open(os.path.join(tmp, "evaluations.json")) as f:
                saved = json.load(f)
            self.assertEqual(saved[0]["fails"], {str(by_hidden[32]):
                                                 ["calibration.abs_shift"]})
            self.assertTrue(os.path.exists(
                os.path.join(tmp, "evaluations", "calibration.py")))
            logged = improver.generated_code("evaluation")
            self.assertEqual([(r["name"], r["status"]) for r in logged],
                             [("calibration", "active"), ("accuracy_like", "rejected")])
            self.assertIn("backwards", logged[1]["reason"])
            self.assertTrue(os.path.exists(os.path.join(tmp, "generated",
                                                        "0000_evaluation.json")))
            report = improver.report()
            self.assertEqual(report.data["selected"], by_hidden[64])
            self.assertIn("Guardrails", [s.title for s in report.sections])
            self.assertEqual(load_run(tmp).data["selected"], by_hidden[64])
            resumed = self.improver(tmp, GuardrailProposer([]))
            self.assertEqual(resumed.run().best.trial_id, by_hidden[64])
            self.assertEqual(resumed.evaluations[0]["status"], "active")
            # the test split reports each active guardrail
            out = improver.evaluate_on_test(self.test, n_boot=50)
            self.assertIn("best", out["guardrails"]["calibration.abs_shift"])
            self.assertIn("baseline", out["guardrails"]["calibration.abs_shift"])

    def test_guardrails_need_cross_validation(self):
        with self.assertRaisesRegex(ValueError, "cv_folds"):
            AutoImprover(
                self.train, self.val, RunConfig(
                    guardrails={"enabled": True},
                    evaluation={"check": False},
                ),
                proposer=GuardrailProposer([]), space=SPACE,
            )
        with self.assertRaisesRegex(ValueError, "guardrails.enabled"):
            AutoImprover(
                self.train, self.val, RunConfig(
                    search={"plan": ("model", "evaluation")},
                    evaluation={"cv_folds": 3, "check": False},
                ),
                proposer=GuardrailProposer([]), space=SPACE,
            )
        with self.assertRaisesRegex(ValueError, "kinds"):
            AutoImprover(
                self.train, self.val, RunConfig(
                    search={"plan": ("model", "dance")},
                    evaluation={"cv_folds": 3, "check": False},
                ),
                proposer=GuardrailProposer([]), space=SPACE,
            )

    def test_unknown_columns_are_sent_back(self):
        bad = _spec("by_age", {"abs_shift": "lower"}, needs=["age"])
        good = _spec("calibration", {"abs_shift": "lower"})
        proposer = GuardrailProposer([bad, good])
        with tempfile.TemporaryDirectory() as tmp:
            improver = self.improver(tmp, proposer)
            with mock.patch("pyhealth.autoimprove.state.run_config_cv",
                            self.fake_cv):
                improver.run()
        self.assertIn("unknown columns ['age']", proposer.contexts[1].feedback)
        self.assertEqual([e["spec"].name for e in improver.evaluations],
                         ["calibration"])


if __name__ == "__main__":
    unittest.main()
