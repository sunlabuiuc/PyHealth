"""pyhealth.autoimprove code turns: the static check, the sandbox, code trials."""

import json
import os
import random
import tempfile
import time
import unittest

import numpy as np

from pyhealth.autoimprove import (
    AutoImprover,
    Param,
    PipelineConfig,
    SearchSpace,
    config_diff,
    load_run,
    static_check,
)
from pyhealth.autoimprove.search.prompts import TEMPLATE, task_context
from pyhealth.autoimprove.evaluation.training import fit_predict_code
from pyhealth.autoimprove.search.proposers import Proposer
from pyhealth.autoimprove.search.prompts import build_code_prompt, parse_code
from pyhealth.autoimprove.utils.sandbox import SandboxError, run_sandboxed
from pyhealth.datasets import create_sample_dataset

SPACE = SearchSpace(
    train_params={
        "epochs": Param("int", 1, 2, default=1),
        "lr": Param("float", 1e-4, 1e-1, log=True, default=1e-2),
        "batch_size": Param("choice", choices=(16, 32), default=16),
    }
)
SPACE.models = {k: SPACE.models[k] for k in ("LogisticRegression", "MLP")}

FAST = TEMPLATE.replace("epochs=5", "epochs=1")


def _samples(n, seed, offset=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        codes = rng.sample([f"c{j}" for j in range(8)], 3)
        out.append({"patient_id": f"p{offset + i}", "visit_id": f"v{offset + i}",
                    "conditions": codes, "label": int("c1" in codes)})
    return out


def _datasets():
    schema = ({"conditions": "sequence"}, {"label": "binary"})
    train = create_sample_dataset(_samples(80, 0), *schema)
    kwargs = {"input_processors": train.input_processors,
              "output_processors": train.output_processors}
    val = create_sample_dataset(_samples(40, 1, 1000), *schema, **kwargs)
    test = create_sample_dataset(_samples(40, 2, 2000), *schema, **kwargs)
    return train, val, test


class TestStaticCheck(unittest.TestCase):
    def check(self, body, entry="build_and_train"):
        return static_check(body + f"\ndef {entry}(*a, **k):\n    return None\n", entry)

    def test_template_passes(self):
        self.assertEqual(static_check(TEMPLATE, "build_and_train"), [])

    def test_rejects_file_network_and_shell_access(self):
        for code in ("import os", "import subprocess", "import socket",
                     "from pathlib import Path", "import pickle",
                     "x = open('f')", "x = eval('1')", "x = __import__('os')",
                     "import torch\ntorch.save({}, 'f')", "from numpy import load",
                     "import numpy as np\nnp.fromfile('f')",
                     "import torch.hub", "from torch.utils import cpp_extension",
                     "from . import x", "from numpy import *"):
            with self.subTest(code=code):
                self.assertTrue(self.check(code), code)

    def test_rejects_dunder_escapes(self):
        for code in ("x = ().__class__.__subclasses__()",
                     "x = __builtins__", "g = (lambda: 0).__globals__",
                     "x = getattr(object, '__subclasses__')",
                     "name = 'sys'\nx = getattr(object, name)"):
            with self.subTest(code=code):
                self.assertTrue(self.check(code), code)

    def test_init_and_literal_getattr_are_allowed(self):
        code = ("import torch.nn as nn\nclass M(nn.Module):\n"
                "    def __init__(self):\n        super().__init__()\n"
                "        self.w = getattr(nn, 'Linear')(1, 1)\n")
        self.assertEqual(self.check(code), [])

    def test_missing_entry_and_syntax_error(self):
        self.assertIn("must define a function build_and_train",
                      static_check("x = 1\n", "build_and_train")[0])
        self.assertIn("syntax error", static_check("def (:\n", "build_and_train")[0])


class TestSandbox(unittest.TestCase):
    def test_timeout_stops_the_process(self):
        code = "while True:\n    pass\n"
        start = time.perf_counter()
        with self.assertRaises(SandboxError) as ctx:
            run_sandboxed(code, "pyhealth.autoimprove.utils.sandbox:_noop", {}, timeout=3)
        self.assertLess(time.perf_counter() - start, 30)
        self.assertIn("TimeoutError", ctx.exception.brief)

    def test_error_brief_drops_the_message(self):
        code = ("def build_and_train(train, val, **kw):\n"
                "    return {'a': 1}['a raw value 12345']\n")
        train, val, _ = _datasets()
        with self.assertRaises(SandboxError) as ctx:
            fit_predict_code(code, train, val, [val], "pr_auc", timeout=120)
        self.assertIn("KeyError: 'a raw value 12345'", ctx.exception.full)
        # the brief form names the type and the line, never the message
        self.assertTrue(ctx.exception.brief.startswith("KeyError at <trial>:2:"))


class TestIsolation(unittest.TestCase):
    def test_child_has_no_network(self):
        out = run_sandboxed("", "pyhealth.autoimprove.utils.sandbox:_network_probe",
                            {}, timeout=120)
        if not out["isolated"]:
            self.skipTest("unprivileged network namespaces are not allowed here")
        self.assertEqual(out["interfaces"], ["lo"])


class TestCodeTrials(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.val, cls.test = _datasets()

    def test_template_trains_and_predicts(self):
        out = fit_predict_code(FAST, self.train, self.val, [self.val, self.test],
                               "pr_auc", timeout=300)
        self.assertEqual(out["mode"], "binary")
        self.assertEqual(len(out["predictions"]), 2)
        y_true, y_prob, ids = out["predictions"][0]
        self.assertEqual(np.asarray(y_prob).shape, (40, 1))
        self.assertTrue(((y_prob >= 0) & (y_prob <= 1)).all())
        self.assertGreater(out["n_params"], 0)

    def test_contract_checks(self):
        cases = {
            "reads the label": FAST.replace(
                '"y_prob": self.label_processor.activation(logits)',
                '"y_prob": y_true.float()'),
            "same prediction": FAST.replace(
                '"y_prob": self.label_processor.activation(logits)',
                '"y_prob": torch.full_like(y_true.float(), 0.3)'),
            "BaseModel": FAST.replace("    return model\n", "    return 'not a model'\n"),
            "probabilities": FAST.replace(
                '"y_prob": self.label_processor.activation(logits)',
                '"y_prob": logits * 100'),
        }
        for expected, code in cases.items():
            with self.subTest(expected=expected):
                self.assertNotEqual(code, FAST)
                with self.assertRaises(SandboxError) as ctx:
                    fit_predict_code(code, self.train, self.val, [self.val],
                                     "pr_auc", timeout=300)
                self.assertTrue(ctx.exception.brief.startswith("ContractError"),
                                ctx.exception.brief)
                self.assertIn(expected, ctx.exception.brief)

    def test_task_context_describes_shapes_not_values(self):
        text = task_context(None, self.train)
        self.assertIn("conditions: SequenceProcessor", text)
        self.assertIn("tensor int64 (4, 3)", text)
        self.assertIn("mode binary, output_size() = 1", text)
        self.assertNotIn("c1", text)


class TestCodeConfigs(unittest.TestCase):
    def test_validate_checks_the_code(self):
        config, notes = SPACE.validate(PipelineConfig("Code", code=FAST))
        self.assertEqual((config.model, config.model_params, notes), ("Code", {}, []))
        with self.assertRaisesRegex(ValueError, "import of os"):
            SPACE.validate(PipelineConfig("Code", code="import os\n"
                                          "def build_and_train(): pass\n"))
        with self.assertRaisesRegex(ValueError, "needs its code"):
            SPACE.validate(PipelineConfig("Code"))

    def test_key_round_trip_rung_and_diff(self):
        a = PipelineConfig("Code", code=FAST, rationale="r")
        b = PipelineConfig("Code", code=TEMPLATE)
        self.assertIn(a.code_sha, a.key())
        self.assertNotIn("build_and_train", a.key())
        self.assertEqual(PipelineConfig.from_dict(a.to_dict()).code, FAST)
        self.assertEqual(SPACE.rung(a), 4)
        self.assertEqual(config_diff(a, b), {"code": f"{a.code_sha} → {b.code_sha}"})
        # listed configs keep their old keys
        self.assertNotIn("code", PipelineConfig("MLP").to_dict())

    def test_parse_code_and_prompt(self):
        reply = ("Plan.\n```python\ndef build_and_train(*a, **k):\n    pass\n```\n"
                 '```json\n{"rationale": "pooled", "kind": "technical"}\n```')
        code, meta = parse_code(reply)
        self.assertTrue(code.startswith("def build_and_train"))
        self.assertEqual(meta["rationale"], "pooled")
        with self.assertRaises(ValueError):
            parse_code('```json\n{"a": 1}\n```')


class CodeProposer(Proposer):
    """Proposes listed models for model turns and fixed code for code turns."""

    def __init__(self, code):
        self.code = list(code)
        self.contexts = []

    def propose(self, context):
        return PipelineConfig("MLP", {"embedding_dim": 16, "hidden_dim": 16})

    def propose_code(self, context):
        self.contexts.append(context)
        return PipelineConfig("Code", code=self.code.pop(0), rationale="pooled linear")


class TestCodeTurns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.val, cls.test = _datasets()

    def test_code_and_model_turns_share_the_leaderboard(self):
        broken = FAST.replace("    return model\n", "    return None\n")
        proposer = CodeProposer([FAST, broken])
        with tempfile.TemporaryDirectory() as tmp:
            improver = AutoImprover(
                self.train, self.val, proposer,
                baseline=PipelineConfig("LogisticRegression", {"embedding_dim": 16}),
                space=SPACE, n_agents=2, n_cycles=2, patience=5, cv_folds=2,
                cycle_plan=("model", "code"), run_dir=tmp, check=False,
            )
            result = improver.run()
            kinds = [(t.agent, t.config.model) for t in result.trials]
            self.assertEqual(kinds[0], ("baseline", "LogisticRegression"))
            code_trials = [t for t in result.trials if t.config.code]
            self.assertEqual(len(code_trials), 2)
            ok, failed = code_trials
            self.assertIsNone(ok.error)
            self.assertEqual(len(ok.fold_scores), 2)
            self.assertTrue(failed.error_brief.startswith("ContractError"))
            # the code prompt had the task layout and the template
            prompt = build_code_prompt(proposer.contexts[0])
            self.assertIn("build_and_train", prompt)
            self.assertIn("conditions: SequenceProcessor", prompt)
            # artefacts: code, out-of-fold predictions, logs
            self.assertTrue(os.path.exists(os.path.join(
                tmp, "code", f"trial_{ok.trial_id}.py")))
            preds = np.load(os.path.join(tmp, "preds", f"trial_{ok.trial_id}.npz"))
            self.assertEqual(preds["index"].tolist(), list(range(len(self.train))))
            with open(os.path.join(tmp, "trials.jsonl")) as f:
                saved = [json.loads(line) for line in f]
            self.assertEqual(saved[ok.trial_id]["config"]["code"], FAST)
            # resuming keeps every trial and runs nothing new
            again = AutoImprover(
                self.train, self.val, CodeProposer([]),
                baseline=PipelineConfig("LogisticRegression", {"embedding_dim": 16}),
                space=SPACE, n_agents=2, n_cycles=2, cv_folds=2,
                cycle_plan=("model", "code"), run_dir=tmp, check=False,
            ).run()
            self.assertEqual(len(again.trials), len(result.trials))

    def test_code_baseline_through_test_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            improver = AutoImprover(
                self.train, self.val, CodeProposer([]),
                baseline=PipelineConfig("Code", code=FAST, rationale="pooled linear"),
                space=SPACE, n_agents=1, n_cycles=0, cv_folds=2, run_dir=tmp,
                check=False,
            )
            result = improver.run()
            self.assertTrue(result.best.config.code)
            out = improver.evaluate_on_test(self.test, n_boot=50)
            self.assertIn("estimate", out["best"])
            self.assertEqual(out["best"]["config"]["code"], FAST)
            with self.assertRaises(NotImplementedError):
                improver.explain()
            self.assertEqual(improver.attribute()["rows"], [])
            report = improver.report()
            titles = [s.title for s in report.sections]
            self.assertIn("Generated code", titles)
            self.assertIn("def build_and_train", report.to_markdown())
            self.assertIn("Generated code", [s.title for s in load_run(tmp).sections])


if __name__ == "__main__":
    unittest.main()
