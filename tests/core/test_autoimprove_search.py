"""pyhealth.autoimprove feature search: acceptance, confirmation, pruning, review gate."""

import json
import os
import random
import tempfile
import unittest
from datetime import datetime, timedelta

import polars as pl

from pyhealth.autoimprove import (
    DEFAULT_MODELS,
    AutoImprover,
    FeatureSpace,
    LLMProposer,
    Param,
    PipelineConfig,
    QueueProposer,
    SearchSpace,
    Trial,
    build_report,
    with_xgboost,
)
from pyhealth.autoimprove.evaluate import patient_folds, run_config_cv
from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset

T = datetime(2021, 6, 30)
VISITS_90 = {"op": "count", "window": 90, "select": {"event_type": "visits"},
             "rationale": "recent visits mark an active illness", "direction": "+"}
ANY_90 = {"op": "exists", "window": 90, "select": {"event_type": "visits"},
          "rationale": "any recent visit", "direction": "+"}
LABS_90 = {"op": "count", "window": 90, "select": {"event_type": "labs"},
           "rationale": "unrelated by construction", "direction": "+"}
OLD_365 = {"op": "count", "window": 365, "select": {"event_type": "labs"},
           "rationale": "also unrelated", "direction": "+"}


def _world(n, seed, shuffle=False, marker=False):
    """Patients whose outcome depends on visits in the last 90 days (15% noise).

    With ``marker``, patients with the outcome also get a "markers" event
    just before t: a planted leak.
    """
    rng = random.Random(seed)
    rows, samples = [], []
    for i in range(n):
        pid = f"s{seed}_{i}"
        k = rng.choice([0, 0, 1, 1, 2, 3])
        for j in range(k):
            rows.append({"patient_id": pid, "event_type": "visits",
                         "timestamp": T - timedelta(days=5 + 20 * j)})
        rows.append({"patient_id": pid, "event_type": "visits",
                     "timestamp": T - timedelta(days=300)})
        for j in range(rng.randint(0, 3)):
            rows.append({"patient_id": pid, "event_type": "labs",
                         "timestamp": T - timedelta(days=10 + 25 * j)})
        label = int(k >= 2)
        if rng.random() < 0.15:
            label = 1 - label
        samples.append({"patient_id": pid, "t": T, "x": [rng.gauss(0, 1)],
                        "label": label})
    if shuffle:
        labels = [s["label"] for s in samples]
        rng.shuffle(labels)
        for s, y in zip(samples, labels):
            s["label"] = y
    if marker:
        rows += [{"patient_id": s["patient_id"], "event_type": "markers",
                  "timestamp": T - timedelta(days=3)} for s in samples if s["label"]]
    return rows, samples


class World:
    """An event log and train / val / test samples over it."""

    tables = ["visits", "labs", "markers"]

    def __init__(self, shuffle=False, marker=False):
        rows, train = _world(300, 0, shuffle, marker)
        more, val = _world(80, 1, shuffle, marker)
        rest, test = _world(200, 2, shuffle, marker)
        self.df = pl.DataFrame(rows + more + rest).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us")))
        schema = ({"x": "tensor"}, {"label": "binary"})
        self.train = create_sample_dataset(train, *schema)
        kw = {"input_processors": self.train.input_processors,
              "output_processors": self.train.output_processors}
        self.val = create_sample_dataset(val, *schema, **kw)
        self.test = create_sample_dataset(test, *schema, **kw)

    @property
    def global_event_df(self):
        return self.df.lazy()

    def get_patient(self, pid):
        return Patient(pid, self.df.filter(pl.col("patient_id") == pid))


SPACE = SearchSpace(
    {"LogisticRegression": DEFAULT_MODELS["LogisticRegression"]},
    {"epochs": Param("int", 1, 40, default=25),
     "lr": Param("float", 1e-3, 1e-1, log=True, default=0.05),
     "batch_size": Param("choice", choices=(32,), default=32),
     "pos_weight": Param("choice", choices=("none",), default="none"),
     "patience": Param("int", 1, 50, default=50)},
)
BASELINE = PipelineConfig("LogisticRegression", {"embedding_dim": 8})


def _improver(world, proposer, run_dir=None, **kw):
    fs = FeatureSpace(world, time_key="t", min_patients=5)
    kw.setdefault("n_agents", 1)
    kw.setdefault("n_cycles", 6)
    kw.setdefault("patience", 6)
    return AutoImprover(world.train, world.val, proposer, baseline=BASELINE,
                        space=SPACE, cv_folds=5, features=fs, run_dir=run_dir,
                        confirm=True, check=False, **kw)


class TestFeatureSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = World()

    def test_finds_the_planted_feature_and_prunes_the_redundant_one(self):
        # "any visit" helps alone; "visit count" makes it redundant
        queue = QueueProposer([LABS_90, ANY_90, VISITS_90, OLD_365])
        with tempfile.TemporaryDirectory() as tmp:
            improver = _improver(self.world, queue, tmp)
            result = improver.run()
            added = {name: t for t in result.trials for name in t.added}
            self.assertTrue(added["exists_90d__visits"].accepted)
            self.assertTrue(added["count_90d__visits"].accepted)
            # noise groups may pass the lenient search bound by chance;
            # confirmation and pruning must remove them
            final = [s["name"] for s in result.best.features]
            self.assertEqual(final, ["count_90d__visits"])  # pruned to the planted
            with open(os.path.join(tmp, "prune.json")) as f:
                pruning = json.load(f)
            self.assertIn(["exists_90d__visits"], pruning["dropped"])
            contributions = {r["features"][0]: r["contribution"]
                             for r in pruning["ablation"]}
            self.assertGreater(contributions["count_90d__visits"], 0.05)

            # the log records specs and hypotheses, and a run resumes from it
            with open(os.path.join(tmp, "trials.jsonl")) as f:
                saved = [json.loads(line) for line in f]
            planted = next(t for t in saved if t["added"] == ["count_90d__visits"])
            self.assertEqual(planted["features"][-1]["rationale"],
                             "recent visits mark an active illness")
            self.assertTrue(planted["config"]["rationale"].startswith("queued"))
            again = _improver(self.world, QueueProposer([]), tmp)
            resumed = again.run()
            self.assertEqual(len(resumed.trials), len(result.trials))
            self.assertEqual([s["name"] for s in again._accepted()],
                             [s["name"] for s in improver._accepted()])

            test = improver.evaluate_on_test(self.world.test, n_boot=50)
            self.assertGreater(test["difference"]["estimate"], 0.05)
            text = improver.report().to_markdown()
            self.assertIn("## Features", text)
            self.assertIn("## Confirmation", text)
            self.assertIn("the number of visits events in the 90 days", text)

    def test_nothing_survives_confirmation_with_shuffled_labels(self):
        world = World(shuffle=True)
        queue = QueueProposer([VISITS_90, ANY_90, LABS_90, OLD_365])
        result = _improver(world, queue).run()
        self.assertEqual(result.best.agent, "baseline")
        self.assertEqual(result.best.features, [])

    def test_reversed_effect_is_held_back_until_approved(self):
        wrong = {**VISITS_90, "direction": "-"}
        with tempfile.TemporaryDirectory() as tmp:
            improver = _improver(self.world, QueueProposer([wrong]), tmp)
            result = improver.run()
            self.assertEqual([s["name"] for s in result.best.features],
                             ["count_90d__visits"])
            improver.explain(top_k=2, stability=False)
            ids = [f["id"] for f in improver.flags()]
            self.assertIn("direction:count_90d__visits", ids)
            test = improver.evaluate_on_test(self.world.test, n_boot=50)
            self.assertEqual(test["held_back"], ["count_90d__visits"])
            self.assertIn("without_flagged", test)
            self.assertIn("held for review", improver.report().to_markdown())
            improver.approve("direction:count_90d__visits", "Dr. Test",
                             "the card's direction was wrong")
            test = improver.evaluate_on_test(self.world.test, n_boot=50)
            self.assertNotIn("held_back", test)

    def test_llm_proposes_features_from_the_menu(self):
        replies = [
            '```json\n{"features": [' + json.dumps(VISITS_90) + '], '
            '"rationale": "recent utilisation", "expected": "score up"}\n```',
        ]

        class LLM:
            prompts = []

            def __call__(self, system, prompt):
                LLM.prompts.append(prompt)
                if "Rewrite the lessons" in prompt:
                    return '```json\n{"lessons": ["recent visits help"]}\n```'
                if replies:
                    return replies.pop(0)
                return '```json\n{"features": [' + json.dumps(LABS_90) + "]}\n```"

        improver = _improver(self.world, LLMProposer(LLM()), n_cycles=2,
                             feature_cycles=(1, 1))
        result = improver.run()
        planted = next(t for t in result.trials if t.added == ["count_90d__visits"])
        self.assertTrue(planted.accepted)
        self.assertEqual(planted.config.hypothesis["expected"], "score up")
        feature_prompt = next(p for p in LLM.prompts if "A feature is a JSON spec" in p)
        self.assertIn("No features have been accepted yet", feature_prompt)

    def test_a_feature_that_reveals_the_outcome_is_held_as_suspicious(self):
        world = World(marker=True)
        marker = {"op": "count", "window": 30, "select": {"event_type": "markers"},
                  "rationale": "a planted leak", "direction": "+"}
        result = _improver(world, QueueProposer([marker])).run()
        trial = next(t for t in result.trials if t.added)
        self.assertEqual(trial.suspicious, ["count_30d__markers"])
        self.assertFalse(trial.accepted)
        self.assertIn("possible leakage", " ".join(trial.notes))

    def test_cross_validation_stops_early_when_told(self):
        out = run_config_cv(BASELINE, SPACE, self.world.train, self.world.val,
                            "roc_auc", patient_folds(self.world.train, 5),
                            stop_if=lambda scores: len(scores) >= 2)
        self.assertTrue(out["stopped_early"])
        self.assertEqual(len(out["fold_scores"]), 2)


class TestConfirmedReport(unittest.TestCase):
    def test_why_quotes_the_scores_the_selection_used(self):
        # On the search folds the 17-feature trial is clearly best; on the
        # confirmation folds the pruned 4-feature trial is within one SE of
        # it, and that is where the selection was made.
        def trial(i, agent, n_features, search, confirmed_gain):
            t = Trial(i, 0, agent, "explore", PipelineConfig("XGBoost"),
                      metric_name="pr_auc", score=sum(search) / 5,
                      fold_scores=list(search),
                      features=[{"name": f"f{i}_{k}"} for k in range(n_features)])
            return t, [c + g for c, g in zip(base_confirm, confirmed_gain)]

        base_search = [0.47, 0.48, 0.46, 0.47, 0.47]
        base_confirm = [0.46, 0.47, 0.47, 0.48, 0.46]
        rows = [
            trial(0, "baseline", 0, base_search, [0] * 5),
            trial(1, "agent0", 17, [b + 0.04 for b in base_search],
                  [0.035, 0.02, 0.04, 0.02, 0.03]),
            trial(2, "agent1", 10, [b + 0.02 for b in base_search],
                  [0.03, 0.025, 0.02, 0.03, 0.03]),
            trial(3, "pruned", 4, [b + 0.015 for b in base_search],
                  [0.028, 0.024, 0.02, 0.03, 0.028]),
        ]
        trials = [t for t, _ in rows]
        confirmed = {t.trial_id: c for t, c in rows}
        report = build_report(
            trials, metric="pr_auc", promote_se=1.0, space=with_xgboost(SPACE),
            confirmed=confirmed, confirm_se=2.0,
            pruning={"selected": 2, "pruned_trial": 3, "ablation": [], "dropped": [],
                     "final": [f"f3_{k}" for k in range(4)]})
        self.assertEqual(report.data["selected"], 3)
        text = report.to_markdown()
        why = next(line for line in text.splitlines() if "Why this model" in line)
        self.assertIn("on the confirmation folds", why)
        self.assertIn("uses fewer engineered features (4 vs 17)", why)
        self.assertIn("It is #2 after pruning", why)
        self.assertNotIn("+0.025", why)  # the search-fold gap the rule did not use
        self.assertIn("pruned to #3, the selection", text)

        # a trial the confirmation did not re-score is not judged on its folds
        extra = Trial(4, 0, "agent2", "explore", PipelineConfig("XGBoost"),
                      metric_name="pr_auc", score=0.475, fold_scores=[
                          b + 0.005 for b in base_search])
        text = build_report(
            [*trials, extra], metric="pr_auc", promote_se=0.0,
            space=with_xgboost(SPACE), confirmed=confirmed, confirm_se=2.0,
            pruning={"selected": 2, "pruned_trial": 3, "ablation": [],
                     "dropped": [], "final": []}).to_markdown()
        self.assertIn("| #4 | XGBoost | 4 | 0.475 |", text)
        self.assertIn("not re-scored", text)


if __name__ == "__main__":
    unittest.main()
