"""Check experiment integrity using small synthetic inputs only."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from clip_stress import build_parser, run
from lifesnaps_stress_clip_stress import (
    execute,
    experiment_identity,
    experiment_jobs,
    job_path,
    parser,
    random_folds,
    subject_folds,
    summarize,
)
from support2_mortality_clip_stress import (
    prepare_records,
    synthetic_table,
)


class TestRecordedExperiments(unittest.TestCase):
    def test_reference_downstream_filters_after_mixed_batching(self):
        from clip_stress import prepare_datasets, synthetic_data
        from clip_stress_experiment_utils import ReferenceLabeledBatches

        train, *_ = prepare_datasets(synthetic_data(7), 7, 1)
        batches = list(ReferenceLabeledBatches(train, 200, 16, 0))
        self.assertEqual(sum(len(batch["label"]) for batch in batches), len(train))
        self.assertGreater(len(batches), int(np.ceil(len(train) / 16)))
        self.assertTrue(all(len(batch["label"]) <= 16 for batch in batches))

    def test_reference_folds_train_on_four_groups_and_never_select_on_test(self):
        from clip_stress import synthetic_data

        data = synthetic_data(7)
        splits = subject_folds(data["patient_id"], 5, 0, "reference_user")
        args = build_parser().parse_args(
            [
                "--ablation",
                "supervised",
                "--epochs",
                "2",
                "--checkpoint-selection",
                "final",
            ]
        )
        split = next(
            s
            for s in splits
            if len(np.unique(data["label"][np.isin(data["patient_id"], s["test"])]))
            == 2
        )
        result = run(args, data=data, subject_splits=split)
        detail = result["details"]["supervised/finetune"]
        self.assertEqual(result["sample_counts"]["validation"], 0)
        self.assertEqual(detail["validation_history"], [])
        self.assertEqual(detail["selected_epoch"], 2)
        self.assertEqual(detail["test_evaluations"], 1)
        self.assertEqual(len(split["train"]) + len(split["test"]), 12)

    def test_finetune_only_records_one_test_pass_and_best_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            args = build_parser().parse_args(
                [
                    "--downstream-stage",
                    "finetune",
                    "--epochs",
                    "2",
                    "--pretrain-epochs",
                    "1",
                    "--artifacts-dir",
                    temp,
                ]
            )
            result = run(args)
            self.assertEqual(list(result["results"]), ["multimodal/finetune"])
            detail = result["details"]["multimodal/finetune"]
            predictions = detail["predictions"]
            metrics = result["results"]["multimodal/finetune"]
            self.assertEqual(detail["test_evaluations"], 1)
            self.assertTrue(Path(detail["checkpoint"]["path"]).exists())
            self.assertEqual(len(predictions["source_row"]), len(predictions["y_true"]))
            self.assertEqual(
                metrics["roc_auc"],
                roc_auc_score(
                    predictions["y_true"],
                    predictions["y_prob"],
                ),
            )
            self.assertEqual(
                metrics["average_precision"],
                average_precision_score(
                    predictions["y_true"],
                    predictions["y_prob"],
                ),
            )
            history = detail["validation_history"]
            self.assertEqual(
                detail["selected_epoch"],
                1 + np.argmax([epoch["roc_auc"] for epoch in history]),
            )

    def test_shards_cover_matrix_and_resume_identity_rejects_changed_budget(self):
        args = parser().parse_args(
            [
                "--output-dir",
                "/tmp/example",
                "--folds",
                "5",
                "--seeds",
                "0",
                "1",
                "2",
                "3",
            ]
        )
        jobs = experiment_jobs(args)
        self.assertEqual(len(jobs), 40)
        self.assertEqual(len(set(jobs)), 40)
        self.assertEqual({tau for *_, tau in jobs}, {None, args.temperature})
        self.assertEqual({seed for seed, *_ in jobs}, {0, 1, 2, 3})
        shards = [jobs[i::4] for i in range(4)]
        self.assertEqual(sorted(job for shard in shards for job in shard), sorted(jobs))
        self.assertEqual(
            [sum(tau is not None for *_, tau in shard) for shard in shards], [5] * 4
        )
        before, _ = experiment_identity(args)
        args.device, args.worker_index = "cuda:0", 1
        self.assertEqual(before, experiment_identity(args)[0])
        args.epochs += 1
        self.assertNotEqual(before, experiment_identity(args)[0])

    def test_summarize_reports_mean_sd_and_paired_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parser().parse_args(
                ["--output-dir", tmp, "--folds", "2", "--seeds", "0", "1"]
            )
            digest, _ = experiment_identity(args)
            pretrain = f"tau-{args.temperature:g}"
            # roc_auc per (seed, fold): pretraining beats supervised by 0.1 + 0.01*k
            scores = {}
            for k, (seed, fold) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                scores[seed, fold, "supervised"] = (0.70 + 0.02 * k, 0.30 + 0.01 * k)
                scores[seed, fold, pretrain] = (0.80 + 0.03 * k, 0.40)

            def write(seed, fold, name, **changes):
                roc, pr = scores[seed, fold, name]
                record = {
                    "complete": True, "experiment_id": digest, "recipe": name,
                    "seed": seed, "fold": fold + 1,
                    "results": {"test": {"roc_auc": roc, "pr_auc": pr}},
                    **changes,
                }
                job_path(args, seed, fold, name).write_text(json.dumps(record))

            jobs = experiment_jobs(args)
            self.assertEqual(len(jobs), 8)
            for seed, fold, name, _ in jobs:
                write(seed, fold, name)
            report = summarize(args)

            self.assertEqual(report["runs"], 8)
            for name in ("supervised", pretrain):
                roc = [scores[s, f, name][0] for s in (0, 1) for f in (0, 1)]
                stats = report["summary"][name]["roc_auc"]
                self.assertAlmostEqual(stats["mean"], np.mean(roc))
                self.assertAlmostEqual(stats["std"], np.std(roc, ddof=1))
                self.assertEqual(stats["runs"], 4)
            self.assertAlmostEqual(report["summary"][pretrain]["pr_auc"]["std"], 0.0)
            paired = report["paired_against_supervised"]
            self.assertEqual(list(paired), [pretrain])
            np.testing.assert_allclose(
                paired[pretrain]["roc_auc_differences"], [0.10, 0.11, 0.12, 0.13]
            )
            self.assertAlmostEqual(paired[pretrain]["mean"], 0.115)
            saved = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertEqual(saved["experiment_id"], digest)

            # a record from another configuration, or an unfinished one, is refused
            for changes in ({"experiment_id": "other"}, {"complete": False}):
                with self.subTest(changes=changes):
                    write(1, 1, "supervised", **changes)
                    with self.assertRaisesRegex(ValueError, "Incompatible or"):
                        summarize(args)
            write(1, 1, "supervised")
            job_path(args, 1, 1, pretrain).unlink()
            with self.assertRaises(FileNotFoundError):
                summarize(args)

    def test_resume_skips_finished_jobs_and_refuses_foreign_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parser().parse_args(
                ["--output-dir", tmp, "--folds", "2", "--seeds", "0"]
            )
            digest, _ = experiment_identity(args)
            done = {"complete": True, "experiment_id": digest}
            for seed, fold, name, _ in experiment_jobs(args):
                job_path(args, seed, fold, name).write_text(json.dumps(done))
            execute(args)  # every job is finished, so nothing is retrained
            foreign = job_path(args, 0, 1, "supervised")
            foreign.write_text(json.dumps({**done, "experiment_id": "other"}))
            with self.assertRaisesRegex(ValueError, "Refusing to reuse"):
                execute(args)

    def test_subject_fold_protocol_matches_every_subject_once(self):
        splits = subject_folds(np.arange(20), 5, 7)
        tested = []
        for split in splits:
            groups = [set(split[name]) for name in ("train", "validation", "test")]
            self.assertFalse(
                groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2]
            )
            tested.extend(split["test"])
        self.assertEqual(len(tested), len(set(tested)))
        self.assertEqual(set(tested), set(map(str, range(20))))

    def test_paper_random_folds_are_stratified_and_cover_every_row_once(self):
        labels = np.array([-1] * 50 + [0] * 30 + [1] * 20)
        splits = random_folds(labels, 5)
        tested = np.concatenate([split["test"] for split in splits])
        self.assertEqual(sorted(tested.tolist()), list(range(100)))
        for split in splits:
            self.assertFalse(set(split["train"]) & set(split["test"]))
            self.assertEqual(len(split["validation"]), 0)
            np.testing.assert_array_equal(
                np.bincount(labels[split["test"]] + 1), [10, 6, 4]
            )
        again = random_folds(labels, 5)
        for a, b in zip(splits, again):
            np.testing.assert_array_equal(a["test"], b["test"])

    def test_row_splits_let_a_subject_appear_in_train_and_test(self):
        from clip_stress import synthetic_data

        data = synthetic_data(7)
        split = random_folds(data["label"], 3)[0]
        args = build_parser().parse_args(
            [
                "--ablation",
                "supervised",
                "--epochs",
                "1",
                "--checkpoint-selection",
                "final",
            ]
        )
        result = run(args, data=data, sample_splits=split)
        self.assertTrue(
            set(result["subjects"]["train"]) & set(result["subjects"]["test"])
        )
        self.assertEqual(result["sample_counts"]["test"], len(split["test"]))
        predictions = result["details"]["supervised/finetune"]["predictions"]
        self.assertEqual(predictions["source_row"], sorted(split["test"].tolist()))
        np.testing.assert_array_equal(
            predictions["y_true"], data["label"][predictions["source_row"]]
        )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            run(
                args, data=data, sample_splits=dict(split, validation=split["test"][:1])
            )

    def test_support2_outcomes_and_held_out_values_do_not_change_training_features(
        self,
    ):
        records = (
            synthetic_table()
            .rename(columns={"sno": "patient_id", "hospdead": "label"})
            .to_dict("records")
        )
        for row in records:
            row["patient_id"] = str(row["patient_id"])
        first, metadata = prepare_records(records)
        changed = json.loads(json.dumps(records))
        heldout = set(metadata["subjects"]["validation"] + metadata["subjects"]["test"])
        for row in changed:
            row["label"] = 1 - row["label"]
            row["surv2m"] = row["label"]
            if row["patient_id"] in heldout:
                row["age"] += 10000
        second, other_metadata = prepare_records(changed)
        for a, b in zip(first["train"], second["train"]):
            np.testing.assert_array_equal(a["tabular"], b["tabular"])
        self.assertEqual(metadata["train_mean"], other_metadata["train_mean"])


if __name__ == "__main__":
    unittest.main()
