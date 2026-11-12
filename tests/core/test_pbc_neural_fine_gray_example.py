"""Fast runner contract tests; multi-epoch smoke checks are run separately.

Contributors: Amey Deotale (ameysd2), Anushka Shinde (ashinde4),
Abid Hossain (ahossa5).
Paper: Neural Fine-Gray: Monotonic neural networks for competing risks.
https://proceedings.mlr.press/v209/jeanselme23a.html
"""

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from pyhealth.trainer import Trainer


def load_example() -> ModuleType:
    """Load the checked-out runner without executing its command-line entry."""
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "pbc_neural_fine_gray"
        / "pbc_competing_risks_neural_fine_gray.py"
    )
    specification = importlib.util.spec_from_file_location("pbc_example", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load the checked-out example: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class TestPBCExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.example = load_example()

    def test_corruption_is_nested_and_native_missingness_preserved(self) -> None:
        example = self.example
        samples = [{"features": [1.0] * 25} for _ in range(4)]
        samples[0]["features"][example.PBC_MASK_INDICES[0]] = np.nan
        low = example.corrupt(samples, 0.1, 7)
        high = example.corrupt(samples, 0.3, 7)
        for original, left, right in zip(samples, low, high):
            native, low_mask, high_mask = (
                np.isnan(row["features"]) for row in (original, left, right)
            )
            self.assertTrue((~low_mask | high_mask).all())
            self.assertTrue((~native | low_mask).all())
            changed = np.flatnonzero(high_mask & ~native)
            self.assertTrue(set(changed).issubset(example.PBC_MASK_INDICES))

    def test_metric_grid_and_matched_reporting(self) -> None:
        example = self.example
        rows = [
            {
                "fold": 0,
                "mode": mode,
                "missing_rate": rate,
                "horizon": horizon,
                "risk": risk,
                "metric": metric,
                "lane": "corrected",
                "status": "ok",
                "value": 0.1 if mode == "augmented" else 0.2,
            }
            for mode in (*example.MODES, *example.COMPARATORS)
            for rate in example.RATES
            for horizon in example.HORIZONS
            for risk in (1, 2)
            for metric in ("brier", "concordance")
        ]
        self.assertEqual(len(rows), 288)
        example.validate_rows(rows, [0], False)
        with self.assertRaises(ValueError):
            example.validate_rows(rows[:-1], [0], False)
        with self.assertRaises(ValueError):
            example.validate_rows(rows + [rows[0]], [0], False)
        with tempfile.TemporaryDirectory() as directory:
            example.report(Path(directory), rows)
            h2 = json.loads((Path(directory) / "h2.json").read_text())
            self.assertTrue(h2["directional_criteria_met"])

    def test_wholly_unsupported_reports_keep_score_and_empty_pair_schemas(self) -> None:
        example = self.example
        train, test = [{"time": 1.0, "event": 1}], [{"time": 4.0, "event": 0}]
        with tempfile.TemporaryDirectory() as directory:
            for name, h2, horizons in (
                ("primary", True, example.HORIZONS),
                ("paper", False, example.PAPER_HORIZONS),
            ):
                with self.subTest(name=name):
                    rows = []
                    for mode in (*example.MODES, *example.COMPARATORS):
                        for rate in example.RATES:
                            rows.extend(
                                example.evaluate(
                                    np.zeros((1, len(horizons), 2)),
                                    train,
                                    test,
                                    0,
                                    mode,
                                    rate,
                                    horizons=horizons,
                                )
                            )
                    example.validate_rows(rows, [0], False, horizons=horizons)
                    path = Path(directory) / name
                    example.report(path, rows, h2=h2)
                    frame = pd.read_csv(path / "metrics.csv")
                    self.assertEqual(len(frame), len(rows))
                    self.assertTrue((frame.status == "unsupported").all())
                    self.assertTrue(frame.value.isna().all())
                    self.assertTrue(frame.reason.notna().all())
                    summary = pd.read_csv(path / "summary.csv")
                    self.assertTrue(summary.empty)
                    self.assertIn("mean", summary.columns)
                    if h2:
                        pairs = pd.read_csv(path / "paired_effects.csv")
                        self.assertTrue(pairs.empty)
                        self.assertEqual(
                            list(pairs.columns),
                            [
                                "fold",
                                "rate",
                                "control",
                                "augmented_minus_control",
                            ],
                        )
                        result = json.loads((path / "h2.json").read_text())
                        self.assertFalse(result["all_available_folds_supported"])
                        self.assertFalse(result["directional_criteria_met"])
                    else:
                        self.assertFalse((path / "h2.json").exists())

    def test_invalid_scores_are_not_support_fallbacks(self) -> None:
        example = self.example
        train = [{"time": 10.0, "event": 1}]
        test = [{"time": 1.0, "event": 2}]
        prediction = np.zeros((1, 4, 2))
        with (
            patch.object(
                example, "competing_risk_brier", side_effect=ValueError("bug")
            ),
            self.assertRaisesRegex(ValueError, "bug"),
        ):
            example.evaluate(prediction, train, test, 0, "original", 0.0)

    def test_historical_metrics_preserve_reference_inputs_and_rows(self) -> None:
        example = self.example
        train = [
            {"time": 2.0, "event": 1},
            {"time": 6.0, "event": 2},
            {"time": 10.0, "event": 0},
        ]
        test = [
            {"time": 1.0, "event": 1},
            {"time": 3.0, "event": 2},
            {"time": 12.0, "event": 1},
            {"time": 12.0, "event": 2},
            {"time": 12.0, "event": 0},
        ]
        predictions = np.arange(40).reshape(5, 4, 2) / 100
        metrics = ModuleType("sksurv.metrics")
        metrics.brier_score = Mock(return_value=(np.array([0.0]), np.array([0.125])))
        metrics.concordance_index_ipcw = Mock(return_value=(0.75, 1, 0, 0, 0))
        with patch.dict(
            "sys.modules",
            {"sksurv": ModuleType("sksurv"), "sksurv.metrics": metrics},
        ):
            rows = example.historical_metrics(predictions, train, test, fold=3)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["status"] == "ok" for row in rows))
        self.assertTrue(all(row["lane"] == "paper_binary_cause" for row in rows))
        self.assertTrue(all(row["fold"] == 3 for row in rows))
        self.assertEqual({row["horizon"] for row in rows}, {3.19, 4.95, 7.45})
        for row in rows:
            self.assertEqual(row["value"], 0.125 if row["metric"] == "brier" else 0.75)
        selections = {1: [0, 1, 3, 4], 2: [0, 1, 2, 4]}
        for index, call in enumerate(metrics.brier_score.call_args_list):
            risk = 1 + index // 3
            horizon = (3.19, 4.95, 7.45)[index % 3]
            selected = selections[risk]
            train_labels, test_labels, survival, times = call.args
            np.testing.assert_array_equal(train_labels["event"], [risk == 1, risk == 2, False])
            np.testing.assert_array_equal(train_labels["time"], [2.0, 6.0, 10.0])
            np.testing.assert_array_equal(
                test_labels["event"], [test[i]["event"] == risk for i in selected]
            )
            np.testing.assert_array_equal(
                test_labels["time"], [test[i]["time"] for i in selected]
            )
            cif = predictions[selected, example.HORIZONS.index(horizon), risk - 1]
            np.testing.assert_array_equal(survival, (1 - cif)[:, None])
            self.assertEqual(times, [horizon])
            concordance = metrics.concordance_index_ipcw.call_args_list[index]
            np.testing.assert_array_equal(concordance.args[0], train_labels)
            np.testing.assert_array_equal(concordance.args[1], test_labels)
            np.testing.assert_array_equal(concordance.args[2], cif)
            self.assertEqual(concordance.kwargs, {"tau": horizon})

    def test_historical_metrics_report_known_support_errors_independently(self) -> None:
        metrics = ModuleType("sksurv.metrics")
        metrics.brier_score = Mock(return_value=(np.array([0.0]), np.array([0.125])))
        metrics.concordance_index_ipcw = Mock(return_value=(0.75, 1, 0, 0, 0))
        train = [{"time": 10.0, "event": 1}]
        test = [{"time": 1.0, "event": 2}]
        with patch.dict(
            "sys.modules",
            {"sksurv": ModuleType("sksurv"), "sksurv.metrics": metrics},
        ):
            for name, metric in (
                ("brier_score", "brier"),
                ("concordance_index_ipcw", "concordance"),
            ):
                function = getattr(metrics, name)
                for message in (
                    "time must be smaller than largest observed time point",
                    "censoring survival function is zero at one or more time points",
                    "all samples are censored",
                    "all times must be within follow-up time of test data",
                    "No comparable pairs, cannot estimate concordant index.",
                ):
                    with self.subTest(metric=metric, message=message):
                        function.side_effect = ValueError(message)
                        rows = self.example.historical_metrics(
                            np.zeros((1, 4, 2)), train, test, fold=0
                        )
                        unsupported = [row for row in rows if row["metric"] == metric]
                        self.assertEqual(len(unsupported), 6)
                        self.assertTrue(
                            all(row["status"] == "unsupported" for row in unsupported)
                        )
                        self.assertTrue(
                            all(row["reason"] == message for row in unsupported)
                        )
                        self.assertTrue(all("value" not in row for row in unsupported))
                        supported = [row for row in rows if row["metric"] != metric]
                        self.assertEqual(len(supported), 6)
                        self.assertTrue(all(row["status"] == "ok" for row in supported))
                function.side_effect = None

    def test_historical_metrics_propagate_unexpected_errors_and_nonfinite_scores(
        self,
    ) -> None:
        metrics = ModuleType("sksurv.metrics")
        metrics.brier_score = Mock(return_value=(np.array([0.0]), np.array([0.125])))
        metrics.concordance_index_ipcw = Mock(return_value=(0.75, 1, 0, 0, 0))
        train = [{"time": 10.0, "event": 1}]
        test = [{"time": 1.0, "event": 2}]
        with patch.dict(
            "sys.modules",
            {"sksurv": ModuleType("sksurv"), "sksurv.metrics": metrics},
        ):
            for name in ("brier_score", "concordance_index_ipcw"):
                function = getattr(metrics, name)
                with self.subTest(function=name, failure="unexpected ValueError"):
                    error = ValueError("unexpected prediction shape")
                    function.side_effect = error
                    with self.assertRaises(ValueError) as raised:
                        self.example.historical_metrics(
                            np.zeros((1, 4, 2)), train, test, fold=0
                        )
                    self.assertIs(raised.exception, error)
                    function.side_effect = None
                original = function.return_value
                for value in (np.nan, np.inf):
                    with self.subTest(function=name, nonfinite=value):
                        function.return_value = (
                            (np.array([0.0]), np.array([value]))
                            if name == "brier_score"
                            else (value, 1, 0, 0, 0)
                        )
                        with self.assertRaisesRegex(
                            FloatingPointError, "Nonfinite historical score"
                        ):
                            self.example.historical_metrics(
                                np.zeros((1, 4, 2)), train, test, fold=0
                            )
                function.return_value = original

    def test_checkpoint_resume_validates_identity_and_content(self) -> None:
        example = self.example
        samples = [
            {
                "patient_id": f"fake-{i}",
                "features": [0.0] * 25,
                "missing_mask": [0.0] * 25,
                "time": 0.2 + i / 10,
                "event": i,
            }
            for i in range(3)
        ]
        config = {
            "layers": [4],
            "layers_surv": [4],
            "dropout": 0.0,
            "batch": 2,
            "learning_rate": 0.001,
        }
        args = Namespace(seed=7, device="cpu", epochs=1, patience=1)

        def fake_train(
            trainer: Trainer, *positional: object, **keywords: object
        ) -> None:
            """Set predictable weights without running any training epochs."""
            with torch.no_grad():
                for parameter in trainer.model.parameters():
                    parameter.zero_()

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                example.Trainer,
                "train",
                autospec=True,
                side_effect=fake_train,
            ) as train,
        ):
            path = Path(directory)
            model, first = example.fit_model(
                samples,
                samples,
                config,
                "original",
                path,
                args,
            )
            restored, second = example.fit_model(
                samples,
                samples,
                config,
                "original",
                path,
                args,
            )
            self.assertEqual(train.call_count, 1)
            self.assertEqual(first, second)
            _, seeded = example.fit_model(
                samples,
                samples,
                config,
                "original",
                path / "fresh-seed",
                args,
                training_seed=8,
            )
            self.assertEqual(seeded["identity"]["seed"], 8)
            self.assertEqual(args.seed, 7)
            with self.assertRaisesRegex(ValueError, "identity/hash mismatch"):
                example.fit_model(
                    samples,
                    samples,
                    config,
                    "original",
                    path / "fresh-seed",
                    args,
                    training_seed=9,
                )
            torch.testing.assert_close(
                model.predict_cif([[0.0] * 25], [1.0]),
                restored.predict_cif([[0.0] * 25], [1.0]),
            )
            with self.assertRaisesRegex(ValueError, "identity/hash mismatch"):
                example.fit_model(samples, samples, config, "constant", path, args)
            metadata = json.loads((path / "fit.json").read_text())
            metadata["checkpoint_sha256"] = "wrong"
            (path / "fit.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "identity/hash mismatch"):
                example.fit_model(samples, samples, config, "original", path, args)

    def test_budget_validation(self) -> None:
        with self.assertRaises(SystemExit):
            self.example.parse_args(
                [
                    "--synthetic",
                    "--output-dir",
                    "unused",
                    "--trials",
                    "0",
                ]
            )
        with self.assertRaises(SystemExit):
            self.example.parse_args(["--output-dir", "unused"])
        for options in (
            ["--sensitivity-seeds", "-1"],
            ["--seed", "-1"],
            ["--cox-penalties", "0"],
            ["--cox-penalties", "nan"],
            ["--cox-penalties", "1", "1"],
        ):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.example.parse_args(
                    ["--synthetic", "--output-dir", "unused", *options]
                )

    def test_cox_selection_uses_validation_only_and_keeps_training_fit(self) -> None:
        example = self.example
        train = [{"features": [1.0] * 25, "time": 0.5, "event": 1}]
        validation = [
            {"features": [9.0] * 25, "time": 0.75, "event": risk} for risk in (1, 2)
        ]
        scores = {0.1: -3.0, 1.0: -1.0, 10.0: -4.0}
        scored = []

        class StubCox:
            def __init__(self, penalty):
                self.penalty = penalty

            def score(self, frame, scoring_method):
                scored.append((frame.copy(), scoring_method))
                return scores[self.penalty]

        def fake_fit(samples, penalty):
            self.assertIs(samples, train)
            return [StubCox(penalty), StubCox(penalty)], [example.PBC_FEATURES[0]]

        with patch.object(example, "fit_cox", side_effect=fake_fit) as fitted:
            models, columns, selection = example.select_cox(
                train, validation, list(scores)
            )
        self.assertEqual(fitted.call_count, 3)
        self.assertEqual([model.penalty for model in models], [1.0, 1.0])
        self.assertEqual(selection["selected_penalty"], 1.0)
        self.assertFalse(selection["refit_on_validation"])
        self.assertEqual(columns, [example.PBC_FEATURES[0]])
        self.assertEqual(len(scored), 6)
        for frame, method in scored:
            self.assertEqual(method, "log_likelihood")
            np.testing.assert_array_equal(frame[columns].to_numpy(), [[9.0], [9.0]])
            self.assertEqual(frame.duration.tolist(), [0.75, 0.75])
            self.assertEqual(frame.observed.sum(), 1)
        for penalties in ([], [0.1, 0.1], [0.0], [float("nan")]):
            with self.subTest(penalties=penalties), self.assertRaises(ValueError):
                example.select_cox(train, validation, penalties)
        scores[1.0] = float("nan")
        with (
            patch.object(example, "fit_cox", side_effect=fake_fit),
            self.assertRaises(FloatingPointError),
        ):
            example.select_cox(train, validation, [1.0])

    def test_architecture_cases_freeze_other_selected_settings(self) -> None:
        selected = {
            "layers": [25, 25],
            "layers_surv": [50],
            "dropout": 0.25,
            "batch": 100,
            "learning_rate": 0.001,
        }
        original = json.loads(json.dumps(selected))
        cases = self.example.architecture_configurations(selected)
        self.assertEqual(len(cases), 7)
        self.assertEqual(selected, original)
        for label, config in cases.items():
            self.assertEqual(config["batch"], selected["batch"])
            self.assertEqual(config["learning_rate"], selected["learning_rate"])
            self.assertEqual(len(config["layers"]), 2)
            self.assertEqual(len(config["layers_surv"]), 1)
            if label.startswith("width"):
                width = int(label.split("-")[1])
                self.assertEqual(config["layers"], [width, width])
                self.assertEqual(config["layers_surv"], [width])
                self.assertEqual(config["dropout"], selected["dropout"])
            else:
                self.assertEqual(config["layers"], selected["layers"])
                self.assertEqual(config["layers_surv"], selected["layers_surv"])

    def test_neural_scoring_separates_paper_times_and_shares_corruption(self) -> None:
        example = self.example
        samples = [
            {"features": [float(i)] * 25, "time": time, "event": event}
            for i, (time, event) in enumerate(((2, 1), (4, 2), (8, 0), (10, 1)))
        ]
        processor = example.CompetingRisksPreprocessor().fit(samples)

        class StubNeural:
            def __init__(self):
                self.masks = []

            def predict_cif(self, features, horizons, masks):
                self.masks.append(masks)
                return torch.zeros(
                    (len(features), len(horizons), 2), dtype=torch.float64
                )

        first, second = StubNeural(), StubNeural()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, paper = example.score_neural(
                first,
                processor,
                samples,
                samples,
                0,
                "original",
                root / "first",
                123,
                paper_time=True,
            )
            example.score_neural(
                second,
                processor,
                samples,
                samples,
                0,
                "constant",
                root / "second",
                123,
            )
            self.assertEqual(first.masks, second.masks)
            example.validate_rows(primary, [0], False, modes=("original",))
            example.validate_rows(
                paper,
                [0],
                False,
                modes=("original",),
                horizons=example.PAPER_HORIZONS,
            )
            self.assertEqual(len(primary), 48)
            self.assertEqual(len(paper), 36)
            with np.load(root / "first" / "private_predictions_0.0.npz") as saved:
                self.assertEqual(saved["cif"].shape, (4, 4, 2))
                self.assertEqual(saved["paper_cif"].shape, (4, 3, 2))
            example.report(root / "paper", paper, h2=False)
            self.assertFalse((root / "paper" / "h2.json").exists())

    def test_wholly_unsupported_sensitivity_reports_keep_score_schema(self) -> None:
        example = self.example
        train, test = [{"time": 1.0, "event": 1}], [{"time": 4.0, "event": 0}]
        rows = [
            {**row, "training_seed": seed}
            for seed in (8, 9)
            for fold in (0, 1)
            for mode in example.MODES
            for rate in example.RATES
            for row in example.evaluate(
                np.zeros((1, len(example.HORIZONS), 2)),
                train,
                test,
                fold,
                mode,
                rate,
            )
        ]
        self.assertTrue(all("value" not in row for row in rows))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example.report_sensitivity(root, rows, [0, 1], [8, 9])
            aggregate = pd.read_csv(root / "metrics.csv")
            self.assertEqual(len(aggregate), 768)
            self.assertTrue(aggregate.value.isna().all())
            self.assertTrue((aggregate.status == "unsupported").all())
            self.assertEqual(aggregate.reason.tolist(), [row["reason"] for row in rows])
            self.assertEqual(set(aggregate.training_seed), {8, 9})
            for seed in (8, 9):
                with self.subTest(seed=seed):
                    frame = pd.read_csv(root / f"seed-{seed}" / "metrics.csv")
                    self.assertEqual(list(frame.columns), list(aggregate.columns))
                    self.assertEqual(len(frame), 384)
                    self.assertTrue(frame.value.isna().all())
                    self.assertTrue((frame.status == "unsupported").all())
                    self.assertTrue(frame.reason.notna().all())
                    pairs = pd.read_csv(root / f"seed-{seed}" / "paired_effects.csv")
                    self.assertTrue(pairs.empty)
                    self.assertEqual(len(pairs.columns), 4)
            decision = json.loads((root / "h2.json").read_text())
            self.assertEqual(decision["seed_count"], 2)
            self.assertEqual(decision["directional_criteria_met_count"], 0)
            self.assertTrue(decision["primary_analysis_unchanged"])
            self.assertTrue(
                all(
                    not seed["all_available_folds_supported"]
                    for seed in decision["seeds"]
                )
            )
            self.assertTrue(all("value" not in row for row in rows))

    def test_sensitivity_grid_keeps_seed_decisions_separate(self) -> None:
        example = self.example
        rows = [
            {
                "training_seed": seed,
                "fold": fold,
                "mode": mode,
                "missing_rate": rate,
                "horizon": horizon,
                "risk": risk,
                "metric": metric,
                "lane": "corrected",
                "status": "ok",
                "value": 0.1 if mode == "augmented" and seed == 8 else 0.2,
            }
            for seed in (8, 9)
            for fold in (0, 1)
            for mode in example.MODES
            for rate in example.RATES
            for horizon in example.HORIZONS
            for risk in (1, 2)
            for metric in ("brier", "concordance")
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example.report_sensitivity(root, rows, [0, 1], [8, 9])
            decision = json.loads((root / "h2.json").read_text())
            self.assertEqual(decision["directional_criteria_met_count"], 1)
            self.assertTrue(decision["primary_analysis_unchanged"])
            pd.testing.assert_frame_equal(
                pd.read_csv(root / "metrics.csv"), pd.DataFrame(rows), check_exact=True
            )
            with self.assertRaisesRegex(ValueError, "seed grid"):
                example.report_sensitivity(root, rows, [0, 1], [8, 9, 10])
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                example.report_sensitivity(root, rows[:-1], [0, 1], [8, 9])

    def test_pipeline_hashes_cover_shared_and_new_library_sources(self) -> None:
        example = self.example
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = (
                "pyhealth/datasets/utils.py",
                "pyhealth/datasets/sample_dataset.py",
                "pyhealth/processors/base_processor.py",
                "pyhealth/datasets/configs/pbc.yaml",
                "pyproject.toml",
                "examples/pbc_neural_fine_gray/pbc_competing_risks_neural_fine_gray.py",
                "examples/pbc_neural_fine_gray/pbc_neural_fine_gray_replication.py",
                "examples/pbc_neural_fine_gray/requirements-neural-fine-gray.txt",
            )
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            original = example.pipeline_source_hashes(root)
            self.assertEqual(set(original), set(names))
            for name in names[:3]:
                (root / name).write_text("changed")
                self.assertNotEqual(
                    original[name], example.pipeline_source_hashes(root)[name]
                )
            added = root / "pyhealth/processors/new_processor.py"
            added.write_text("new")
            self.assertIn(
                "pyhealth/processors/new_processor.py",
                example.pipeline_source_hashes(root),
            )
            added.with_suffix(".pyc").write_bytes(b"cache")
            self.assertEqual(len(example.pipeline_source_hashes(root)), len(names) + 1)

    def test_rejected_cohort_preserves_audit_even_in_audit_only_mode(self) -> None:
        example = self.example
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "cohort_audit.json"
            audit.write_bytes(b'{"sha256": "original"}\n')
            example.write_json(
                root / "manifest.json", {"audit": {"sha256": "original"}}
            )
            for extra in ([], ["--audit-only"]):
                with (
                    patch.object(
                        example,
                        "load_samples",
                        return_value=([], {"sha256": "changed"}),
                    ),
                    self.assertRaisesRegex(ValueError, "Changed source data"),
                ):
                    example.main(["--synthetic", "--output-dir", str(root), *extra])
                self.assertEqual(audit.read_bytes(), b'{"sha256": "original"}\n')

    def test_cox_interval_allocation_and_single_cause_limit(self) -> None:
        class StubCox:
            def __init__(self, cumulative: list[float]) -> None:
                self.baseline_cumulative_hazard_ = pd.DataFrame(
                    cumulative, index=[0.0, 1.0, 2.0]
                )

            def predict_cumulative_hazard(self, frame: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(
                    np.repeat(
                        self.baseline_cumulative_hazard_.to_numpy(), len(frame), axis=1
                    ),
                    index=self.baseline_cumulative_hazard_.index,
                )

        horizons = np.array([0.0, 0.5, 1.0, 2.0])
        samples = [{"features": [0.0] * 25}]
        first = StubCox([0.0, np.log(2), np.log(4)])
        second = StubCox([0.0, np.log(2), np.log(2)])
        columns = [self.example.PBC_FEATURES[0]]
        curves = self.example.predict_cox([first, second], columns, samples, horizons)
        np.testing.assert_allclose(
            curves[0], [[0, 0], [0, 0], [0.375, 0.375], [0.5, 0.375]]
        )
        self.assertTrue(all(self.example.curve_checks(curves, origin=True).values()))
        single = self.example.predict_cox(
            [first, StubCox([0.0, 0.0, 0.0])], columns, samples, horizons
        )
        np.testing.assert_allclose(single[0], [[0, 0], [0, 0], [0.5, 0], [0.75, 0]])

    def test_cuda_memory_uses_selected_device(self) -> None:
        samples = [
            {
                "features": [0.0] * 25,
                "missing_mask": [0.0] * 25,
                "time": 0.2 + i / 10,
                "event": i,
            }
            for i in range(3)
        ]
        config = {
            "layers": [4],
            "layers_surv": [4],
            "dropout": 0.0,
            "batch": 2,
            "learning_rate": 0.001,
        }
        original_to = torch.nn.Module.to

        def cpu_only_to(
            module: torch.nn.Module, *args: object, **kwargs: object
        ) -> torch.nn.Module:
            if args and str(args[0]).startswith("cuda"):
                return module
            return original_to(module, *args, **kwargs)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(torch.nn.Module, "to", autospec=True, side_effect=cpu_only_to),
            patch.object(self.example.Trainer, "train"),
            patch.object(torch.cuda, "reset_peak_memory_stats") as reset,
            patch.object(
                torch.cuda, "max_memory_allocated", return_value=7 * 2**20
            ) as peak,
        ):
            _, info = self.example.fit_model(
                samples,
                samples,
                config,
                "original",
                Path(directory),
                Namespace(seed=7, device="cuda:1", epochs=1, patience=1),
            )
            reset.assert_called_once_with("cuda:1")
            peak.assert_called_once_with("cuda:1")
            self.assertEqual(info["peak_gpu_allocated_mib"], 7.0)

    def test_completion_counts_finite_search_space_not_requested_trials(self) -> None:
        example = self.example
        samples = [
            {
                "patient_id": f"fake-{i}",
                "features": [1.0] * 25,
                "time": 0.2 + 2 * i,
                "event": i % 3,
            }
            for i in range(6)
        ]

        class StubNeural:
            def predict_cif(
                self, features: list, horizons: np.ndarray, masks: list
            ) -> torch.Tensor:
                return torch.zeros(
                    (len(features), len(horizons), 2), dtype=torch.float64
                )

        class StubAJ:
            def __init__(self, **kwargs: object) -> None:
                self.cumulative_density_ = pd.DataFrame([0.0], index=[0.0])

            def fit(self, *args: object, **kwargs: object) -> None:
                pass

        def fake_fit(train, validation, config, mode, directory, args):
            info = {
                "identity": {"config": config},
                "validation_nll_scaled": 0.1,
                "checkpoint_sha256": "stub",
                "parameters": 0,
            }
            if directory.name == "trial-0":
                directory.mkdir(exist_ok=True)
                example.write_json(directory / "fit.json", info)
            return StubNeural(), info

        lifelines = ModuleType("lifelines")
        lifelines.AalenJohansenFitter = StubAJ
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("sys.modules", {"lifelines": lifelines}),
            patch.object(
                example, "load_samples", return_value=(samples, {"sha256": "synthetic"})
            ),
            patch.object(
                example,
                "train_test_split",
                side_effect=lambda indices, **kwargs: (indices, indices),
            ),
            patch.object(example, "fit_model", side_effect=fake_fit),
            patch.object(
                example, "pipeline_source_hashes", return_value={"stub.py": "synthetic"}
            ),
            patch.object(example, "version", return_value="test"),
            patch.object(example, "select_cox", return_value=([], [], {"stub": True})),
            patch.object(
                example,
                "predict_cox",
                side_effect=lambda models, columns, samples, horizons: np.zeros(
                    (len(samples), len(horizons), 2)
                ),
            ),
            self.assertWarnsRegex(UserWarning, "1024"),
        ):
            example.main(
                [
                    "--synthetic",
                    "--output-dir",
                    directory,
                    "--folds",
                    "2",
                    "--trials",
                    "1025",
                    "--epochs",
                    "1",
                ]
            )
            completion = json.loads((Path(directory) / "completion.json").read_text())
            self.assertEqual(completion["training_fits"], 2 * (1024 + 3))
