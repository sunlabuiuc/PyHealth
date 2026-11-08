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
    experiment_identity,
    experiment_jobs,
    parser,
    subject_folds,
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
        args = parser().parse_args(["--output-dir", "/tmp/example", "--folds", "5"])
        jobs = experiment_jobs(args)
        self.assertEqual(len(jobs), 20)
        self.assertEqual(len(set(jobs)), 20)
        shards = [jobs[i::2] for i in range(2)]
        self.assertEqual(sorted(job for shard in shards for job in shard), sorted(jobs))
        self.assertEqual(
            [sum(tau is not None for _, _, tau in shard) for shard in shards], [8, 7]
        )
        before, _ = experiment_identity(args)
        args.device, args.worker_index = "cuda:0", 1
        self.assertEqual(before, experiment_identity(args)[0])
        args.epochs += 1
        self.assertNotEqual(before, experiment_identity(args)[0])

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
