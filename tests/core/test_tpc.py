"""Fast unit tests for the TPC model. All data is small and synthetic."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from examples.eicu_los_tpc import (
    main as clinical_main,
)
from examples.eicu_los_tpc import (
    preflight,
    results_markdown,
    run_experiments,
    train_configuration,
)
from examples.synthetic_icu_los_tpc import (
    build_dataset,
    los_metrics,
    make_synthetic_samples,
    train_and_eval,
)
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import TPC, TPCLayer
from pyhealth.models.tpc import (
    MAX_LOS_DAYS,
    MIN_LOS_DAYS,
    asymmetric_msle_loss,
    compute_decay_indicators,
    huber_log_loss,
    msle_loss,
    prepare_time_series,
)
from pyhealth.tasks import TPCDayOne
from tests.core.test_yaib_eicu import _lab
from tests.core.test_yaib_eicu import _patient as eicu_patient
from tests.core.test_yaib_mimic4 import (
    T0,
    _admission,
)
from tests.core.test_yaib_mimic4 import (
    _patient as mimic_patient,
)

STEPS, FEATS, STATIC, DIAG = 6, 3, 2, 5


def _make_dataset(n: int = 4):
    g = torch.Generator().manual_seed(0)
    samples = []
    for i in range(n):
        samples.append(
            {
                "patient_id": f"p{i}",
                "visit_id": f"v{i}",
                "ts": torch.randn(STEPS, FEATS, generator=g).tolist(),
                "decay": torch.rand(STEPS, FEATS, generator=g).tolist(),
                "static": torch.randn(STATIC, generator=g).tolist(),
                "diag": (torch.rand(DIAG, generator=g) > 0.5).float().tolist(),
                "los": 1.0 + i,
            }
        )
    return create_sample_dataset(
        samples=samples,
        input_schema={
            "ts": "tensor",
            "decay": "tensor",
            "static": "tensor",
            "diag": "tensor",
        },
        output_schema={"los": "regression"},
        dataset_name="tpc_test",
        in_memory=True,
    )


def _small_model(dataset, **kwargs):
    params = {
        "value_key": "ts",
        "decay_key": "decay",
        "temporal_channels": 3,
        "pointwise_channels": 4,
        "num_layers": 2,
        "kernel_size": 3,
        "final_fc_size": 5,
        "main_dropout": 0.0,
        "temporal_dropout": 0.0,
    }
    params.update(kwargs)
    return TPC(dataset, **params)


class TestTPC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=4)))

    def test_instantiation(self):
        model = _small_model(self.dataset)
        self.assertEqual(model.mode, "regression")
        self.assertEqual(len(model.layers), 2)

    def test_no_temporary_directory(self):
        with patch("pyhealth.datasets.sample_dataset.tempfile.mkdtemp") as mkdtemp:
            _make_dataset()
        mkdtemp.assert_not_called()

    def test_forward_shapes_and_keys(self):
        model = _small_model(self.dataset)
        out = model(**self.batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, out)
        self.assertEqual(out["y_prob"].shape, (4, 1))
        self.assertEqual(out["y_true"].shape, (4, 1))
        self.assertEqual(out["loss"].dim(), 0)

    def test_forward_sequence_shape_and_bounds(self):
        model = _small_model(self.dataset)
        seq = model.forward_sequence(**self.batch)
        self.assertEqual(seq.shape, (4, STEPS))
        self.assertTrue(bool((seq >= MIN_LOS_DAYS).all()))
        self.assertTrue(bool((seq <= MAX_LOS_DAYS).all()))

    def test_with_static_and_diagnoses(self):
        model = _small_model(self.dataset, static_key="static", diag_key="diag")
        out = model(**self.batch)
        self.assertEqual(out["y_prob"].shape, (4, 1))

    def test_no_decay_ablation(self):
        model = _small_model(self.dataset, decay_key=None)
        out = model(**self.batch)
        self.assertEqual(out["y_prob"].shape, (4, 1))

    def test_mse_loss_option(self):
        model = _small_model(self.dataset, loss="mse")
        self.assertTrue(torch.isfinite(model(**self.batch)["loss"]))

    def test_gradients_flow(self):
        model = _small_model(self.dataset, static_key="static", diag_key="diag")
        model(**self.batch)["loss"].backward()
        # BaseModel keeps an empty `_dummy_param` only to track the device.
        grads = [
            p.grad
            for name, p in model.named_parameters()
            if p.requires_grad and name != "_dummy_param"
        ]
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(any(bool(g.abs().sum() > 0) for g in grads))

    def test_causality(self):
        """Changing the last hour must not change earlier predictions."""
        model = _small_model(self.dataset).eval()
        batch = {k: v.clone() for k, v in self.batch.items() if torch.is_tensor(v)}
        base = model.forward_sequence(**batch)
        batch["ts"][:, -1, :] += 10.0
        changed = model.forward_sequence(**batch)
        self.assertTrue(torch.allclose(base[:, :-1], changed[:, :-1], atol=1e-5))

    def test_layer_feature_growth(self):
        layer = TPCLayer(2, num_features=3, temporal_channels=4, pointwise_channels=5)
        h = torch.randn(2, 3 + 5, 5, STEPS)  # R^2 = F + Z, C = Y + 1
        x, decay = torch.randn(2, 3, STEPS), torch.rand(2, 3, STEPS)
        prev_temp = torch.randn(2, 3 * 4, STEPS)  # previous layer: R=F, Y=4
        h_next, point, temp_out = layer(
            h, x, decay, None, None, [torch.randn(2, 5, STEPS)], prev_temp
        )
        self.assertEqual(h_next.shape, (2, 3 + 5 * 2, 5, STEPS))
        self.assertEqual(point.shape, (2, 5, STEPS))
        self.assertEqual(temp_out.shape, (2, (3 + 5) * 4, STEPS))

    def test_dilation_schedules(self):
        schedules = {
            "official": [1, 3, 6, 9],
            "paper": [1, 2, 3, 4],
            "constant": [1, 1, 1, 1],
            "linear": [1, 2, 3, 4],
            "exponential": [1, 2, 4, 8],
        }
        for schedule, expected in schedules.items():
            with self.subTest(schedule=schedule):
                model = _small_model(
                    self.dataset,
                    num_layers=4,
                    kernel_size=4,
                    dilation_schedule=schedule,
                ).eval()
                self.assertEqual([layer.dilation for layer in model.layers], expected)
                seq = model.forward_sequence(**self.batch)
                changed_batch = dict(self.batch)
                changed_batch["ts"] = self.batch["ts"].clone()
                changed_batch["ts"][:, -1] += 10
                changed = model.forward_sequence(**changed_batch)
                torch.testing.assert_close(seq[:, :-1], changed[:, :-1])
                self.assertEqual(seq.shape, (4, STEPS))
        with self.assertRaises(ValueError):
            _small_model(self.dataset, dilation_schedule="bad")

    def test_loss_modes_and_gradients(self):
        for mode, helper in (
            ("msle", msle_loss),
            ("mse", torch.nn.functional.mse_loss),
            ("asymmetric_msle", asymmetric_msle_loss),
            ("huber_log", huber_log_loss),
        ):
            with self.subTest(loss=mode):
                model = _small_model(self.dataset, loss=mode)
                out = model(**self.batch)
                torch.testing.assert_close(
                    out["loss"], helper(out["y_prob"], out["y_true"])
                )
                out["loss"].backward()
                self.assertTrue(torch.isfinite(model.final2.weight.grad).all())
                self.assertGreater(model.final2.weight.grad.abs().sum().item(), 0)

    def test_internal_preprocessing(self):
        model = _small_model(
            self.dataset,
            observed_key="decay",
            decay_key=None,
            decay_base=0.5,
            lab_indices=[1, 2],
            forward_fill_labs=False,
        ).eval()
        values, decay = prepare_time_series(
            self.batch["ts"], self.batch["decay"] > 0.5, 0.5, [1, 2], False
        )
        batch = dict(self.batch)
        batch["decay"] = self.batch["decay"] > 0.5
        internal = model.forward_sequence(**batch)
        model.observed_key = None
        model.decay_key = "decay"
        batch.update(ts=values, decay=decay)
        torch.testing.assert_close(internal, model.forward_sequence(**batch))

    def test_extreme_logits_have_finite_gradients(self):
        model = _small_model(self.dataset).eval()
        with torch.no_grad():
            model.final2.weight.zero_()
            model.final2.bias.fill_(1000)
        out = model(**self.batch)
        self.assertTrue((out["y_prob"] == MAX_LOS_DAYS).all())
        out["loss"].backward()
        self.assertTrue(torch.isfinite(model.final2.bias.grad).all())

    def test_invalid_configuration(self):
        for config in (
            {"loss": "bad"},
            {"num_layers": 0},
            {"decay_base": 0},
            {"huber_delta": 0},
            {"underprediction_weight": 0.5},
            {"forward_fill_labs": False},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                _small_model(self.dataset, **config)

    def test_time_features(self):
        model = _small_model(self.dataset, time_key="ts")  # reuse (time, F) input
        out = model(**self.batch)
        self.assertEqual(out["y_prob"].shape, (4, 1))

    def test_wrong_mode_raises(self):
        samples = [
            {
                "patient_id": f"p{i}",
                "visit_id": f"v{i}",
                "ts": torch.randn(STEPS, FEATS).tolist(),
                "decay": torch.rand(STEPS, FEATS).tolist(),
                "label": i % 2,
            }
            for i in range(2)
        ]
        ds = create_sample_dataset(
            samples=samples,
            input_schema={"ts": "tensor", "decay": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="tpc_bad_mode",
            in_memory=True,
        )
        with self.assertRaises(ValueError):
            TPC(ds, value_key="ts", decay_key="decay")

    def test_missing_key_raises(self):
        with self.assertRaises(ValueError):
            TPC(self.dataset, value_key="nope")

    def test_normalized_labels_raise(self):
        ds = create_sample_dataset(
            samples=[
                {"ts": torch.ones(STEPS, FEATS).tolist(), "los": label}
                for label in (1.0, 2.0)
            ],
            input_schema={"ts": "tensor"},
            output_schema={"los": ("regression", {"normalization": "minmax"})},
            in_memory=True,
        )
        with self.assertRaisesRegex(ValueError, "unnormalized"):
            TPC(ds, value_key="ts", decay_key=None)


class TestHelpers(unittest.TestCase):
    def test_preparation_and_decay_bases(self):
        values = torch.tensor(
            [[[float("nan"), 2.0], [4.0, float("nan")], [float("nan"), float("nan")]]]
        )
        observed = torch.isfinite(values)
        for base in (0.5, 0.75, 0.9):
            filled, decay = prepare_time_series(values, observed, base, [1], False)
            torch.testing.assert_close(
                filled, torch.tensor([[[0.0, 2.0], [4.0, 0.0], [4.0, 0.0]]])
            )
            torch.testing.assert_close(
                decay, torch.tensor([[[0.0, 1.0], [1.0, base], [base, base**2]]])
            )
        filled, _ = prepare_time_series(values, observed)
        self.assertEqual(filled[0, 2, 1].item(), 2.0)

    def test_invalid_helper_arguments(self):
        y = torch.ones(2)
        for invoke in (
            lambda: compute_decay_indicators(torch.ones(1, 2, 3), 0),
            lambda: compute_decay_indicators(torch.ones(2, 3)),
            lambda: prepare_time_series(torch.ones(1, 2, 3), torch.ones(1, 2, 2)),
            lambda: prepare_time_series(
                torch.ones(1, 2, 3), torch.ones(1, 2, 3), lab_indices=[3]
            ),
            lambda: prepare_time_series(
                torch.full((1, 2, 3), float("nan")), torch.ones(1, 2, 3)
            ),
            lambda: msle_loss(y, y, eps=0),
            lambda: msle_loss(y, y[:, None]),
            lambda: asymmetric_msle_loss(y, y, float("inf")),
            lambda: huber_log_loss(y, y, delta=0),
        ):
            with self.assertRaises(ValueError):
                invoke()

    def test_asymmetric_proportional_error(self):
        target = torch.tensor([2.0])
        under = asymmetric_msle_loss(torch.tensor([1.0]), target, 3)
        over = asymmetric_msle_loss(torch.tensor([4.0]), target, 3)
        torch.testing.assert_close(under, 3 * over)
        pred = torch.tensor([1.0, 4.0], requires_grad=True)
        truth = target.expand_as(pred)
        torch.testing.assert_close(
            asymmetric_msle_loss(pred, truth, 1), msle_loss(pred, truth)
        )
        asymmetric_msle_loss(pred, truth).backward()
        self.assertLess(pred.grad[0].item(), 0)
        self.assertGreater(pred.grad[1].item(), 0)

    def test_huber_log_transition(self):
        residual = torch.tensor([0.5, 2.0])
        pred = residual.exp().requires_grad_()
        loss = huber_log_loss(pred, torch.ones(2), delta=1)
        torch.testing.assert_close(loss, torch.tensor((0.125 + 1.5) / 2))
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertLess(loss.item(), msle_loss(pred, torch.ones(2)).item())

    def test_log_loss_clamping(self):
        for helper in (msle_loss, asymmetric_msle_loss, huber_log_loss):
            pred = torch.tensor([0.0, 2.0], requires_grad=True)
            loss = helper(pred, torch.tensor([0.0, 1.0]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(pred.grad).all())

    def test_decay_indicators(self):
        obs = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]])
        out = compute_decay_indicators(obs)
        self.assertEqual(out.shape, obs.shape)
        self.assertAlmostEqual(out[0, 0, 0].item(), 1.0)
        self.assertAlmostEqual(out[0, 2, 0].item(), 0.75**2, places=5)
        self.assertEqual(out[0, 0, 1].item(), 0.0)  # never observed yet
        self.assertAlmostEqual(out[0, 2, 1].item(), 1.0)

    def test_msle(self):
        y = torch.tensor([1.0, 2.0])
        self.assertAlmostEqual(msle_loss(y, y).item(), 0.0, places=6)
        self.assertGreater(msle_loss(y, y * 2).item(), 0.0)

    def test_msle_penalises_proportional_error(self):
        short = msle_loss(torch.tensor([3.0]), torch.tensor([1.0]))
        long = msle_loss(torch.tensor([32.0]), torch.tensor([30.0]))
        self.assertGreater(short.item(), long.item())


class TestSyntheticExample(unittest.TestCase):
    def test_preprocessing_changes_only_labs(self):
        filled = make_synthetic_samples(3, 7)
        unfilled = make_synthetic_samples(3, 7, forward_fill_labs=False)
        slower_decay = make_synthetic_samples(3, 7, decay_base=0.9)
        for original, no_fill, slower in zip(filled, unfilled, slower_decay):
            self.assertEqual(original["los"], no_fill["los"])
            self.assertEqual(original["decay"], no_fill["decay"])
            self.assertEqual(original["time_series"], slower["time_series"])
            x = np.array(original["time_series"])
            other = np.array(no_fill["time_series"])
            observed = np.array(original["decay"]) == 1
            np.testing.assert_array_equal(x[:, :4], other[:, :4])
            np.testing.assert_array_equal(x[observed], other[observed])
            self.assertTrue((other[:, 4:][~observed[:, 4:]] == 0).all())

    def test_metrics_bias_and_long_stays(self):
        truth = np.array([1.0, 2.0, 6.0, 8.0])
        pred = truth / 2
        metrics = los_metrics(truth, pred)
        self.assertEqual(metrics["MAPE"], 50)
        self.assertEqual(metrics["Bias"], -2.125)
        self.assertEqual(metrics["LongMAD"], 3.5)
        self.assertEqual(metrics["LongBias"], -3.5)
        self.assertEqual(metrics["LongN"], 2)
        self.assertAlmostEqual(metrics["MSLE"], np.log(2) ** 2)
        for invalid in (np.zeros(4), np.full(4, np.nan)):
            with self.assertRaises(ValueError):
                los_metrics(truth, invalid)
        with self.assertRaises(ValueError):
            los_metrics(truth[:2], pred[:2])

    def test_training_is_reproducible(self):
        samples = make_synthetic_samples(4, 0)
        for sample, label in zip(samples, (1.0, 2.0, 6.0, 8.0)):
            sample["los"] = label
        train = build_dataset(samples, "train")
        test = build_dataset(samples, "test", training=train)
        self.assertIs(train.input_processors["decay"], test.input_processors["decay"])
        params = {
            "num_layers": 1,
            "temporal_channels": 2,
            "pointwise_channels": 2,
            "final_fc_size": 3,
            "main_dropout": 0.2,
        }
        first = train_and_eval(train, test, 1, 0.0023, 0, **params)
        second = train_and_eval(train, test, 1, 0.0023, 0, **params)
        self.assertEqual(first, second)


class TestClinicalPreparation(unittest.TestCase):
    def test_eicu_day_one_window_and_label(self):
        task = TPCDayOne("eicu")
        patient = eicu_patient(
            hours=240,
            extra=(
                _lab(23 * 60 + 59, "creatinine", 1.0),
                _lab(24 * 60 + 1, "creatinine", 10.0),
            ),
        )
        sample = task(patient)[0]
        crea = task.feature_names.index("crea")
        self.assertEqual(sample["time_series"].shape, (24, 48))
        self.assertEqual(sample["time_series"][23, crea], 1)
        self.assertTrue(sample["observed"][23, crea])
        self.assertFalse(sample["observed"][0, crea])
        self.assertEqual(sample["los"], 9)
        self.assertIn(crea, task.lab_indices)
        self.assertNotIn(task.feature_names.index("hr"), task.lab_indices)

    def test_mimic_day_one_window(self):
        task = TPCDayOne("mimic4")
        patient = mimic_patient(
            stays=[(10, 1, T0, 48, range(48))], admissions=[_admission()]
        )
        sample = task(patient)[0]
        self.assertEqual(sample["time_series"].shape, (24, 48))
        self.assertEqual(sample["los"], 1)
        self.assertEqual(sample["visit_id"], "10")

    def test_short_stay_and_invalid_source(self):
        self.assertEqual(TPCDayOne("eicu")(eicu_patient(hours=24)), [])
        task = TPCDayOne("eicu")
        for hours, expected in ((48.5, 24.5 / 24), (24.25, 1 / 48)):
            with self.subTest(hours=hours):
                self.assertAlmostEqual(
                    task(eicu_patient(hours=hours))[0]["los"], expected
                )
        with self.assertRaises(ValueError):
            TPCDayOne("unsupported")

    def test_preparation_does_not_load_or_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "plan"
            with (
                patch("examples.eicu_los_tpc.eICUDataset") as dataset,
                patch("examples.eicu_los_tpc.TPC") as model,
            ):
                clinical_main(
                    ["--prepare-only", "--all-configs", "--output-dir", str(output)]
                )
            dataset.assert_not_called()
            model.assert_not_called()
            plan = json.loads((output / "plan.json").read_text())
            self.assertEqual(plan["clinical_experiments"], "NOT_RUN")
            self.assertEqual(plan["clinical_results"], "PENDING")
            self.assertEqual(plan["results"], {})
            self.assertEqual(len(plan["configurations"]), 45)
            table = (output / "results.md").read_text()
            self.assertIn("| baseline | PENDING", table)
            self.assertNotIn("+/-", table)
            with self.assertRaises(FileExistsError):
                clinical_main(["--output-dir", str(output)])
        self.assertFalse(output.exists())

    def test_run_failure_is_explicit_and_persistent(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed"
            with (
                patch("examples.eicu_los_tpc.preflight"),
                patch(
                    "examples.eicu_los_tpc.run_experiments",
                    side_effect=RuntimeError("synthetic unit-test failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "synthetic unit-test failure"),
            ):
                clinical_main(
                    [
                        "--run",
                        "--root",
                        temporary,
                        "--data-version",
                        "unit-fixture",
                        "--output-dir",
                        str(output),
                    ]
                )
            status = json.loads((output / "results.json").read_text())
            self.assertEqual(status["status"], "FAILED")
            self.assertEqual(status["clinical_experiments"], "INCOMPLETE")
            self.assertEqual(status["clinical_results"], "PENDING")

    def test_preflight_missing_files_and_join(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, "no download"):
                preflight(root, "eicu")
            for name in (
                "hosp/patients",
                "hosp/admissions",
                "icu/icustays",
                "hosp/labevents",
                "icu/chartevents",
                "icu/outputevents",
            ):
                path = root / f"{name}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with self.assertRaisesRegex(FileNotFoundError, "d_labitems"):
                preflight(root, "mimic4")
            (root / "hosp/d_labitems.csv").touch()
            preflight(root, "mimic4")

    def test_incomplete_results_stay_pending(self):
        runs = [
            {
                "seed": seed,
                "test": {
                    **{
                        metric: float(seed + 1)
                        for metric in (
                            "MAD",
                            "MAPE",
                            "MSLE",
                            "R2",
                            "Bias",
                            "LongMAD",
                            "LongBias",
                        )
                    },
                    "LongN": 2,
                },
            }
            for seed in range(3)
        ]
        incomplete = results_markdown({"base": runs[:2]}, ["base"], [0, 1, 2])
        self.assertIn("| base | PENDING", incomplete)
        complete = results_markdown({"base": runs}, ["base"], [0, 1, 2])
        self.assertIn("2.000 +/- 1.000", complete)

    def test_synthetic_checkpoint_round_trip(self):
        task = TPCDayOne("eicu")
        samples = []
        for i, label in enumerate((1.0, 2.0, 6.0, 8.0)):
            values = np.full((24, 48), float(i + 1), dtype=np.float32)
            values[0, 0] = np.nan
            samples.append(
                {
                    "patient_id": f"unit{i}",
                    "visit_id": str(i),
                    "time_series": values,
                    "observed": np.isfinite(values),
                    "static": [60 + i, 1, 180, 80],
                    "los": label,
                }
            )
        train = create_sample_dataset(samples, task.input_schema, task.output_schema)
        heldout = [
            {
                **sample,
                "patient_id": f"heldout{i}",
                "visit_id": f"heldout{i}",
                "time_series": sample["time_series"] + 1000,
            }
            for i, sample in enumerate(samples)
        ]
        test = create_sample_dataset(
            heldout,
            task.input_schema,
            task.output_schema,
            input_processors=train.input_processors,
            output_processors=train.output_processors,
        )
        self.assertAlmostEqual(test.input_processors["time_series"].center[1], 2.5)
        self.assertGreater(test[0]["time_series"][1, 1], 500)
        self.assertTrue(torch.isnan(test[0]["time_series"][0, 0]))
        self.assertFalse(test[0]["observed"][0, 0])
        args = Namespace(epochs=1, batch_size=4, device="cpu")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "unit.pt"
            result = train_configuration(
                (train, test, test), task, {"num_layers": 1}, 0, args, checkpoint
            )
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["samples"]["test"], 4)
            self.assertTrue(np.isfinite(result["test"]["MSLE"]))
            args = Namespace(
                cpu_threads=1,
                dataset="eicu",
                root=Path(temporary),
                cache_dir=Path(temporary) / "cache",
                dev=False,
                num_workers=1,
                seeds=[0, 1, 2],
                output_dir=Path(temporary),
            )
            plan = {"configurations": {"baseline": {}}, "results": {}}
            with (
                patch("examples.eicu_los_tpc.eICUDataset") as source,
                patch(
                    "examples.eicu_los_tpc.train_configuration",
                    side_effect=[{**result, "seed": seed} for seed in args.seeds],
                ),
            ):
                source.return_value.set_task.return_value = (train, test, test)
                run_experiments(args, plan)
                self.assertEqual(source.return_value.set_task.call_count, 3)
                split = source.return_value.set_task.call_args.kwargs["split"]
                self.assertEqual(split.seed, 2)
            saved = json.loads((args.output_dir / "results.json").read_text())
            self.assertEqual(saved["status"], "COMPLETED")
            self.assertEqual(len(saved["results"]["baseline"]), 3)
            self.assertIn("+/-", (args.output_dir / "results.md").read_text())
            normalization = json.loads(
                (args.output_dir / "preprocessing_seed2.json").read_text()
            )
            self.assertAlmostEqual(
                normalization["inputs"]["time_series"]["center"][1], 2.5
            )
        self.assertFalse(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
