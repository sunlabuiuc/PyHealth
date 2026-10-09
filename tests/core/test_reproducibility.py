"""Reproducible training: seeded per-loader shuffles, no global RNG side
effects from cache writes or imports, and best-epoch restore without logging.
"""

import os
import random
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import BaseDataset, create_sample_dataset, get_dataloader
from pyhealth.models import MLP
from pyhealth.tasks import BaseTask
from pyhealth.trainer import Trainer
from pyhealth.utils import preserve_rng_state

N = 40


def _samples(n=N):
    return [
        {"patient_id": f"p{i}", "x": [float(i % 7), float(i % 3)], "y": i % 2}
        for i in range(n)
    ]


def _dataset(in_memory):
    return create_sample_dataset(_samples(), {"x": "tensor"}, {"y": "binary"}, in_memory=in_memory)


def _order(loader):
    return [pid for batch in loader for pid in batch["patient_id"]]


def _rng_states():
    return (
        random.getstate(),
        tuple(np.asarray(s).tobytes() if isinstance(s, np.ndarray) else s for s in np.random.get_state()),
        torch.get_rng_state().numpy().tobytes(),
    )


class EventTask(BaseTask):
    task_name = "repro_task"
    input_schema = {"codes": "sequence", "x": "tensor"}
    output_schema = {"y": "binary"}

    def __call__(self, patient):
        k = int(patient.patient_id[1:])
        codes = [e.code for e in patient.get_events("events")]
        return [{"patient_id": patient.patient_id, "codes": codes,
                 "x": [float(k % 5), float(len(codes))], "y": k % 2}]


def _write_events(root: Path, n_patients=30):
    root.mkdir(parents=True, exist_ok=True)
    rows = ["patient_id,time,code"] + [
        f"p{k},2020-01-0{1 + v} 00:00:00,c{(k + v) % 6}" for k in range(n_patients) for v in range(1 + k % 3)
    ]
    (root / "events.csv").write_text("\n".join(rows) + "\n")
    (root / "config.yaml").write_text(
        'version: "1.0"\ntables:\n  events:\n    file_path: "events.csv"\n'
        '    patient_id: "patient_id"\n    timestamp: "time"\n    attributes:\n      - "code"\n'
    )


class TestSeededShuffle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = {"memory": _dataset(True), "disk": _dataset(False)}

    def test_same_seed_same_orders_over_epochs(self):
        for name, ds in self.datasets.items():
            with self.subTest(backend=name):
                a = get_dataloader(ds, 8, shuffle=True, seed=3)
                b = get_dataloader(ds, 8, shuffle=True, seed=3)
                epochs_a = [_order(a), _order(a)]
                epochs_b = [_order(b), _order(b)]
                self.assertEqual(epochs_a, epochs_b)
                self.assertNotEqual(epochs_a[0], epochs_a[1])  # new order each epoch
                self.assertEqual(sorted(epochs_a[0]), sorted(f"p{i}" for i in range(N)))
                # a new loader with the same seed restarts the same sequence
                c = get_dataloader(ds, 8, shuffle=True, seed=3)
                self.assertEqual([_order(c), _order(c)], epochs_a)

    def test_different_seeds_differ(self):
        for name, ds in self.datasets.items():
            with self.subTest(backend=name):
                self.assertNotEqual(
                    _order(get_dataloader(ds, 8, shuffle=True, seed=1)),
                    _order(get_dataloader(ds, 8, shuffle=True, seed=2)),
                )

    def test_iterating_leaves_global_rng_untouched(self):
        for name, ds in self.datasets.items():
            with self.subTest(backend=name):
                loader = get_dataloader(ds, 8, shuffle=True, seed=5)
                before = _rng_states()
                _order(loader), _order(loader)
                self.assertEqual(_rng_states(), before)

    def test_torch_seed_alone_controls_default_seed(self):
        for name, ds in self.datasets.items():
            with self.subTest(backend=name):
                torch.manual_seed(11)
                first = _order(get_dataloader(ds, 8, shuffle=True))
                torch.manual_seed(11)
                second = _order(get_dataloader(ds, 8, shuffle=True))
                torch.manual_seed(12)
                other = _order(get_dataloader(ds, 8, shuffle=True))
                self.assertEqual(first, second)
                self.assertNotEqual(first, other)

    def test_unshuffled_loader_keeps_dataset_order(self):
        for name, ds in self.datasets.items():
            with self.subTest(backend=name):
                self.assertEqual(_order(get_dataloader(ds, 8)), [f"p{i}" for i in range(N)])

    def test_loaders_on_one_dataset_are_independent(self):
        natural = [f"p{i}" for i in range(N)]
        for name, ds in self.datasets.items():
            with self.subTest(backend=name):
                train = get_dataloader(ds, 8, shuffle=True, seed=4)
                expected = _order(get_dataloader(ds, 8, shuffle=True, seed=4))
                evaluation = get_dataloader(ds, 8, shuffle=False)
                self.assertEqual(_order(evaluation), natural)
                self.assertEqual(_order(train), expected)
                self.assertNotEqual(expected, natural)

    def test_set_shuffle_on_dataset_is_seeded_too(self):
        ds = _dataset(True)
        ds.set_shuffle(True, seed=9)
        first = [s["patient_id"] for s in ds]
        before = _rng_states()
        second = [s["patient_id"] for s in ds]
        self.assertEqual(_rng_states(), before)
        self.assertNotEqual(first, second)
        ds.set_shuffle(True, seed=9)
        self.assertEqual([s["patient_id"] for s in ds], first)


class TestNoGlobalRngSideEffects(unittest.TestCase):
    def test_disk_dataset_creation(self):
        before = _rng_states()
        create_sample_dataset(_samples(), {"x": "tensor"}, {"y": "binary"}, in_memory=False)
        self.assertEqual(_rng_states(), before)

    def test_set_task_cache_write_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            _write_events(root)
            make = lambda: BaseDataset(
                root=str(root), tables=["events"], dataset_name="Repro",
                config_path=str(root / "config.yaml"), cache_dir=str(Path(tmp) / "cache"),
            )
            orders = []
            for _ in range(2):  # first call writes the cache, second reuses it
                torch.manual_seed(21)
                random.seed(21)
                np.random.seed(21)
                before = _rng_states()
                samples = make().set_task(EventTask())
                self.assertEqual(_rng_states(), before)
                orders.append(_order(get_dataloader(samples, 4, shuffle=True)))
            self.assertEqual(orders[0], orders[1])

    def test_preserve_rng_state(self):
        before = _rng_states()
        with preserve_rng_state():
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
        self.assertEqual(_rng_states(), before)

    def test_seeded_interpreter_does_not_reseed_caller(self):
        from pyhealth.interpret.methods import RandomBaseline

        ds = _dataset(True)
        model = MLP(dataset=ds)
        batch = next(iter(get_dataloader(ds, 8)))
        explainer = RandomBaseline(model, random_seed=0)
        torch.manual_seed(1)
        before = _rng_states()
        first = explainer.attribute(**batch)
        self.assertEqual(_rng_states(), before)
        second = explainer.attribute(**batch)
        self.assertTrue(all(torch.equal(first[k], second[k]) for k in first))

    def test_importing_models_does_not_reseed(self):
        script = textwrap.dedent("""
            import numpy as np, torch
            np.random.seed(123); torch.manual_seed(123)
            a = (np.random.get_state()[1].tobytes(), torch.get_rng_state().numpy().tobytes())
            import pyhealth.models
            b = (np.random.get_state()[1].tobytes(), torch.get_rng_state().numpy().tobytes())
            print("unchanged" if a == b else "changed")
        """)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "unchanged")


class TestBestEpochRestore(unittest.TestCase):
    """Validation labels are the inverse of training labels, so validation
    PR-AUC is best early and falls as the model fits the training data."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 4))
        y = (x[:, 0] > 0).astype(int)
        train = [{"patient_id": f"t{i}", "x": x[i].tolist(), "y": int(y[i])} for i in range(200)]
        val = [{"patient_id": f"v{i}", "x": x[i].tolist(), "y": int(1 - y[i])} for i in range(200)]
        cls.train_ds = create_sample_dataset(train, {"x": "tensor"}, {"y": "binary"})
        cls.val_ds = create_sample_dataset(
            val, {"x": "tensor"}, {"y": "binary"},
            input_processors=cls.train_ds.input_processors,
            output_processors=cls.train_ds.output_processors,
        )

    def _train(self, enable_logging, keep_best_in_memory=True, output_path=None):
        torch.manual_seed(0)
        model = MLP(dataset=self.train_ds, hidden_dim=16)
        trainer = Trainer(model=model, metrics=["pr_auc"], enable_logging=enable_logging,
                          output_path=output_path)
        scores = []
        evaluate = trainer.evaluate
        trainer.evaluate = lambda loader: scores.append(evaluate(loader)) or scores[-1]
        trainer.train(
            get_dataloader(self.train_ds, 32, shuffle=True, seed=0),
            get_dataloader(self.val_ds, 64),
            epochs=12, monitor="pr_auc", patience=3,
            optimizer_params={"lr": 0.05}, keep_best_in_memory=keep_best_in_memory,
        )
        trainer.evaluate = evaluate
        epoch_scores = [s["pr_auc"] for s in scores]
        final = trainer.evaluate(get_dataloader(self.val_ds, 64))["pr_auc"]
        return model, epoch_scores, final

    def test_best_epoch_restored_without_logging(self):
        model, epoch_scores, final = self._train(enable_logging=False)
        best = max(epoch_scores)
        self.assertLess(epoch_scores.index(best), len(epoch_scores) - 1, "best epoch should not be the last")
        self.assertEqual(final, best)

        with tempfile.TemporaryDirectory() as tmp:
            logged, logged_scores, logged_final = self._train(enable_logging=True, output_path=tmp)
            self.assertEqual(logged_scores, epoch_scores)
            self.assertEqual(logged_final, best)
            on_disk = logged.state_dict()
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, on_disk[key]), key)

    def test_opt_out_keeps_last_epoch(self):
        with self.assertLogs("pyhealth.trainer", level="WARNING"):
            _, epoch_scores, final = self._train(enable_logging=False, keep_best_in_memory=False)
        self.assertEqual(final, epoch_scores[-1])
        self.assertLess(final, max(epoch_scores))


E2E_SCRIPT = textwrap.dedent("""
    import hashlib, logging, sys, warnings
    from pathlib import Path
    logging.disable(logging.WARNING); warnings.simplefilter("ignore")
    import torch
    sys.path.insert(0, sys.argv[5])
    from test_reproducibility import EventTask, _write_events
    from pyhealth.datasets import BaseDataset, create_sample_dataset, get_dataloader
    from pyhealth.models import MLP, RNN
    from pyhealth.trainer import Trainer

    def main():
        seed, backend, cache = int(sys.argv[1]), sys.argv[2], Path(sys.argv[4])
        torch.manual_seed(seed)
        if backend == "memory":
            samples = [{"patient_id": f"p{i}", "codes": [f"c{(i + j) % 6}" for j in range(1 + i % 3)],
                        "x": [float(i % 5), float(1 + i % 3)], "y": i % 2} for i in range(30)]
            ds = create_sample_dataset(samples, EventTask.input_schema, EventTask.output_schema)
        else:
            _write_events(cache / "data")
            ds = BaseDataset(root=str(cache / "data"), tables=["events"], dataset_name="E2E",
                             config_path=str(cache / "data" / "config.yaml"),
                             cache_dir=str(cache / "cache")).set_task(EventTask())
        hashes = []
        for model_cls in (MLP, RNN):
            trainer = Trainer(model=model_cls(dataset=ds), enable_logging=False)
            trainer.train(get_dataloader(ds, 8, shuffle=True), epochs=3)
            _, y_prob, _ = trainer.inference(get_dataloader(ds, 8))
            hashes.append(hashlib.sha256(y_prob.tobytes()).hexdigest()[:16])
        print(" ".join(hashes))

    if __name__ == "__main__":
        main()
""")


class TestEndToEndAcrossProcesses(unittest.TestCase):
    def _run(self, script, seed, backend, cache):
        """Trains MLP and RNN in a fresh process; returns their prediction hashes."""
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        out = subprocess.run(
            [sys.executable, script, str(seed), backend, "-", str(cache), str(Path(__file__).parent)],
            capture_output=True, text=True, env=env, timeout=600,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        return out.stdout.strip().splitlines()[-1]

    def test_bit_identical_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "e2e.py"
            script.write_text(E2E_SCRIPT)
            for backend in ("memory", "disk"):
                with self.subTest(backend=backend):
                    cache = Path(tmp) / backend
                    first = self._run(script, 7, backend, cache)   # disk: writes the cache
                    second = self._run(script, 7, backend, cache)  # disk: reuses it
                    other = self._run(script, 8, backend, cache)
                    self.assertEqual(first, second)  # MLP and RNN hashes both equal
                    for a, b in zip(first.split(), other.split()):
                        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
