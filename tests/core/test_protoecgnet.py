"""Tests for ProtoECGNet using tiny synthetic ECG tensors (no real data)."""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import (
    PTBXL_BRANCH_KWARGS,
    PTBXL_BRANCH_LABELS,
    ProtoECGNet,
    PrototypeBranch,
    ResNet1dBackbone,
    ResNet2dBackbone,
)

LABELS = ["A", "B", "C", "D", "E", "F"]
TINY = {"proto_dim": 8, "base_width": 4, "layers": (1, 1, 1, 1)}


def make_dataset(num_samples: int = 12, length: int = 192):
    """Creates a small multilabel ECG dataset with random 12-lead signals."""
    torch.manual_seed(0)
    samples = []
    for i in range(num_samples):
        labels = [LABELS[i % 6]]
        if i % 3 == 0:
            labels.append(LABELS[(i + 2) % 6])
        samples.append(
            {
                "patient_id": f"p{i}",
                "signal": torch.randn(12, length).tolist(),
                "labels": labels,
            }
        )
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"labels": "multilabel"},
    )


class TestPaperConstants(unittest.TestCase):
    """The physician-defined PTB-XL label groups of the paper."""

    def test_group_sizes_and_disjointness(self):
        sizes = {k: len(v) for k, v in PTBXL_BRANCH_LABELS.items()}
        self.assertEqual(sizes, {"rhythm": 16, "morphology": 52, "global": 3})
        flat = [c for codes in PTBXL_BRANCH_LABELS.values() for c in codes]
        self.assertEqual(len(flat), 71)
        self.assertEqual(len(set(flat)), 71)

    def test_branch_kwargs_cover_groups(self):
        self.assertEqual(set(PTBXL_BRANCH_KWARGS), set(PTBXL_BRANCH_LABELS))
        self.assertEqual(PTBXL_BRANCH_KWARGS["morphology"]["proto_time_len"], 3)

    def test_branch_kwargs_match_paper(self):
        per_label = {
            k: v["prototypes_per_label"] for k, v in PTBXL_BRANCH_KWARGS.items()
        }
        self.assertEqual(per_label, {"rhythm": 5, "morphology": 18, "global": 7})
        self.assertEqual(PTBXL_BRANCH_KWARGS["morphology"]["topk"], 5)
        self.assertEqual(PTBXL_BRANCH_KWARGS["global"]["proto_time_len"], 32)


class TestPaperShapes(unittest.TestCase):
    """Prototype shapes of the paper's branches on a 12 x 1000 ECG."""

    def test_global_2d_prototype_spans_full_time_axis(self):
        branch = PrototypeBranch(
            12, 2, proto_dim=8, prototypes_per_label=1, proto_time_len=32,
            backbone="resnet2d",
        )
        self.assertEqual(tuple(branch.prototype_vectors.shape[1:]), (8, 32))
        logits, sim, best = branch(torch.randn(1, 12, 1000))
        self.assertEqual(logits.shape, (1, 2))
        self.assertEqual(sim.shape, (1, 2))
        self.assertTrue(torch.all(best == 0))  # proto_time_len=32 spans the full 32-step map for a 1000-sample input, so unfold yields exactly one window position


class TestPrototypeTerms(unittest.TestCase):
    """Clustering and separation follow Eqs. 4-5 of the paper."""

    def make_branch(self):
        return PrototypeBranch(
            12, 3, proto_dim=4, prototypes_per_label=1, base_width=4,
            layers=(1, 1, 1, 1),
        )

    def test_clustering_and_separation_use_per_sample_prototype_sets(self):
        branch = self.make_branch()  # 3 labels, 1 prototype each → prototype index equals label index
        sim = torch.tensor([[5.0, 3.0, 1.0]])
        y = torch.tensor([[1.0, 1.0, 0.0]])  # labels 0 and 1 present
        terms = branch.prototype_terms(sim, y)
        self.assertAlmostEqual(float(terms["clst"]), -5.0)  # max over {0, 1}
        # Separation only sees prototype 2; prototypes of the other present
        # label must not be penalized.
        self.assertAlmostEqual(float(terms["sep"]), 1.0)

    def test_empty_sets_contribute_zero(self):
        branch = self.make_branch()
        sim = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
        y = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        terms = branch.prototype_terms(sim, y)
        self.assertAlmostEqual(float(terms["clst"]), -1.0)  # (-2 + 0) / 2
        self.assertAlmostEqual(float(terms["sep"]), 1.0)  # (0 + 2) / 2


class TestBackbones(unittest.TestCase):
    """Feature-map shapes of the two backbones."""

    def test_backbone_1d_output_shape(self):
        backbone = ResNet1dBackbone(base_width=4, layers=(1, 1, 1, 1))
        out = backbone(torch.randn(2, 12, 256))
        self.assertEqual(out.shape[:2], (2, backbone.out_channels))
        self.assertEqual(out.shape[2], 8)  # 256 / (2 * 2 * 2 * 2 * 2)

    def test_backbone_2d_keeps_time_axis(self):
        backbone = ResNet2dBackbone()
        out = backbone(torch.randn(2, 12, 256))
        self.assertEqual(out.shape, (2, 512, 8))


class TestPrototypeBranch(unittest.TestCase):
    """Prototype assignment, similarity types and pooling."""

    def kwargs(self, **extra):
        base = {"proto_dim": 8, "base_width": 4, "layers": (1, 1, 1, 1)}
        base.update(extra)
        return base

    def test_prototype_assignment_counts(self):
        branch = PrototypeBranch(
            12, 4, prototypes_per_label=3, joint_prototypes_per_pair=2,
            **self.kwargs(),
        )
        self.assertEqual(branch.num_prototypes, 4 * 3 + 6 * 2)
        identity = branch.prototype_class_identity
        self.assertEqual(identity.shape, (24, 4))
        self.assertTrue(torch.all(identity[:12].sum(dim=1) == 1))
        self.assertTrue(torch.all(identity[12:].sum(dim=1) == 2))

    def test_last_layer_initialization(self):
        branch = PrototypeBranch(12, 3, prototypes_per_label=2, **self.kwargs())
        own = branch.prototype_class_identity.t()
        weight = branch.classifier.weight
        self.assertTrue(torch.all(weight[own == 1] == 1.0))
        self.assertTrue(torch.all(weight[own == 0] == -0.5))

    def test_cosine_similarity_is_bounded_by_scale(self):
        branch = PrototypeBranch(
            12, 3, prototypes_per_label=2, proto_time_len=2, **self.kwargs()
        )
        x = torch.randn(3, 12, 128)
        sims = branch.similarity_windows(branch.windows(branch.feature_map(x)))
        self.assertLessEqual(float(sims.max()), branch.scale + 1e-4)
        self.assertGreaterEqual(float(sims.min()), -branch.scale - 1e-4)
        self.assertAlmostEqual(branch.scale, (8 * 2) ** 0.5)

    def test_l2_similarity_is_finite(self):
        branch = PrototypeBranch(
            12, 3, prototypes_per_label=1, similarity="l2", **self.kwargs()
        )
        logits, sim, window = branch(torch.randn(2, 12, 128))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(sim).all())
        self.assertEqual(window.shape, (2, 3))

    def test_topk_pooling_is_mean_of_best_windows(self):
        torch.manual_seed(1)
        branch = PrototypeBranch(
            12, 2, prototypes_per_label=1, proto_time_len=2, topk=2, **self.kwargs()
        ).eval()
        x = torch.randn(2, 12, 192)
        windows = branch.windows(branch.feature_map(x))
        sims = branch.similarity_windows(windows)
        expected = sims.topk(2, dim=1).values.mean(dim=1)
        _, pooled, _ = branch(x)
        torch.testing.assert_close(pooled, expected)
        single = PrototypeBranch(
            12, 2, prototypes_per_label=1, proto_time_len=2, topk=1, **self.kwargs()
        )
        self.assertEqual(single.topk, 1)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            PrototypeBranch(12, 2, similarity="dot", **self.kwargs())
        with self.assertRaises(ValueError):
            PrototypeBranch(12, 2, backbone="vit", **self.kwargs())
        with self.assertRaises(ValueError):
            PrototypeBranch(12, 2, topk=0, **self.kwargs())
        with self.assertRaises(ValueError):
            PrototypeBranch(12, 2, proto_time_len=0, **self.kwargs())

    def test_2d_branch_forward(self):
        branch = PrototypeBranch(
            12, 3, proto_dim=8, prototypes_per_label=2, proto_time_len=3,
            backbone="resnet2d", topk=2,
        )
        logits, sim, _ = branch(torch.randn(2, 12, 256))
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(sim.shape, (2, 6))


class TestProtoECGNet(unittest.TestCase):
    """Instantiation, forward, loss, stages, projection and fusion."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = make_dataset()
        cls.loader = get_dataloader(cls.dataset, batch_size=4, shuffle=False)
        cls.batch = next(iter(cls.loader))

    def test_forward_shapes_and_loss(self):
        model = ProtoECGNet(
            self.dataset, prototypes_per_label=2, proto_time_len=2, **TINY
        )
        out = model(**self.batch)
        self.assertEqual(out["logit"].shape, (4, 6))
        self.assertEqual(out["y_prob"].shape, (4, 6))
        self.assertEqual(out["y_true"].shape, (4, 6))
        self.assertEqual(out["similarity"].shape, (4, 12))
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all())

    def test_l2_model_variant_runs(self):
        model = ProtoECGNet(
            self.dataset, prototypes_per_label=1, similarity="l2", **TINY
        )
        self.assertTrue(torch.isfinite(model(**self.batch)["loss"]))

    def test_forward_without_labels_has_no_loss(self):
        model = ProtoECGNet(self.dataset, prototypes_per_label=1, **TINY)
        out = model(signal=self.batch["signal"])
        self.assertNotIn("loss", out)

    def test_gradients_reach_backbone_and_prototypes(self):
        model = ProtoECGNet(self.dataset, prototypes_per_label=2, **TINY)
        model(**self.batch)["loss"].backward()
        branch = model.branches["all"]
        self.assertGreater(float(branch.prototype_vectors.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(branch.backbone.stem[0].weight.grad).all())

    def test_contrastive_term_activates_with_cooccurrence(self):
        model = ProtoECGNet(
            self.dataset, prototypes_per_label=2, lam_cnrst=300.0, **TINY
        ).eval()
        base = model(**self.batch)["loss"].item()
        labels = torch.stack([self.dataset[i]["labels"] for i in range(12)])
        _, cooc = ProtoECGNet.compute_label_statistics(labels)
        model.set_label_statistics(label_cooccurrence=cooc)
        with_cnrst = model(**self.batch)["loss"].item()
        self.assertNotEqual(base, with_cnrst)
        self.assertTrue(torch.isfinite(torch.tensor(with_cnrst)))

    def test_label_statistics_values(self):
        labels = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0.0]])
        weights, cooc = ProtoECGNet.compute_label_statistics(labels)
        counts = torch.tensor([3.0, 3.0, 1.0])
        torch.testing.assert_close(weights, (counts.sum() - counts) / counts)
        self.assertAlmostEqual(float(cooc[0, 1]), 2 / 4)  # Jaccard(A, B)
        self.assertAlmostEqual(float(cooc[0, 0]), 1.0)

    def test_stage_freezing(self):
        model = ProtoECGNet(self.dataset, prototypes_per_label=2, **TINY)
        branch = model.branches["all"]
        model.set_stage("joint")
        self.assertTrue(branch.prototype_vectors.requires_grad)
        self.assertTrue(branch.backbone.stem[0].weight.requires_grad)
        self.assertFalse(branch.classifier.weight.requires_grad)
        model.set_stage("classifier")
        self.assertTrue(branch.classifier.weight.requires_grad)
        self.assertFalse(branch.prototype_vectors.requires_grad)
        self.assertFalse(branch.backbone.stem[0].weight.requires_grad)
        model.set_stage("prototypes")
        self.assertTrue(branch.prototype_vectors.requires_grad)
        self.assertFalse(branch.add_on[0].weight.requires_grad)
        model.set_stage("all")
        self.assertTrue(branch.classifier.weight.requires_grad)

    def test_feature_extractor_stage_trains_head_only_path(self):
        model = ProtoECGNet(self.dataset, prototypes_per_label=1, **TINY)
        model.set_stage("feature_extractor")
        model(**self.batch)["loss"].backward()
        branch = model.branches["all"]
        self.assertIsNotNone(branch.pretrain_head.weight.grad)
        self.assertIsNone(branch.prototype_vectors.grad)

    def test_backbone_bn_stays_in_train_mode_in_feature_extractor_stage(self):
        """Backbone BatchNorm must stay in train mode so running stats update.

        Regression: _frozen_modules previously used recurse=False, which
        mistakenly included the whole PrototypeBranch (because
        prototype_vectors has requires_grad=False), causing train() to call
        .eval() on the branch and freeze BN statistics.
        """
        import torch.nn as nn

        model = ProtoECGNet(self.dataset, prototypes_per_label=1, **TINY)
        model.set_stage("feature_extractor")
        model.train()

        branch = model.branches["all"]
        bn = next(
            (
                m
                for m in branch.backbone.modules()
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
            ),
            None,
        )
        self.assertIsNotNone(bn, "backbone must contain a BatchNorm layer")
        self.assertTrue(bn.training, "BatchNorm must be in train mode")

        mean_before = bn.running_mean.clone()
        model(**self.batch)
        self.assertFalse(
            torch.equal(bn.running_mean, mean_before),
            "BatchNorm running_mean must update when backbone is in train mode",
        )

    def test_invalid_stage_and_fusion(self):
        model = ProtoECGNet(self.dataset, prototypes_per_label=1, **TINY)
        with self.assertRaises(ValueError):
            model.set_stage("warmup")
        with self.assertRaises(ValueError):
            model.set_stage("fusion")  # fusion='average' has no fusion head
        with self.assertRaises(ValueError):
            ProtoECGNet(self.dataset, fusion="max", **TINY)

    def test_label_groups_must_partition(self):
        with self.assertRaises(ValueError):
            ProtoECGNet(self.dataset, label_groups={"x": ["A", "B"]}, **TINY)
        with self.assertRaises(ValueError):
            ProtoECGNet(
                self.dataset, label_groups={"x": LABELS, "y": ["A"]}, **TINY
            )

    def test_unknown_labels_in_groups_are_dropped(self):
        groups = {"r": ["A", "B", "C", "ZZZ"], "m": ["D", "E", "F"], "empty": ["QQ"]}
        model = ProtoECGNet(self.dataset, label_groups=groups, **TINY)
        self.assertEqual(model.branch_names, ["r", "m"])
        self.assertEqual(model.branches["r"].num_labels, 3)

    def test_two_branches_with_learned_fusion(self):
        groups = {"rhythm": ["A", "B", "C"], "morph": ["D", "E", "F"]}
        model = ProtoECGNet(
            self.dataset,
            label_groups=groups,
            branch_kwargs={"morph": {"proto_time_len": 3}},
            prototypes_per_label=2,
            fusion="learned",
            **TINY,
        )
        self.assertEqual(model.branches["rhythm"].prototype_vectors.shape[2], 1)
        self.assertEqual(model.branches["morph"].prototype_vectors.shape[2], 3)
        out = model(**self.batch)
        self.assertEqual(out["logit"].shape, (4, 6))
        self.assertEqual(out["similarity"].shape, (4, 12))
        model.set_stage("fusion")
        model(**self.batch)["loss"].backward()
        self.assertIsNotNone(model.fusion_classifier.weight.grad)
        self.assertIsNone(model.branches["morph"].prototype_vectors.grad)

    def test_mixed_1d_and_2d_branches(self):
        groups = {"rhythm": ["A", "B", "C"], "morph": ["D", "E", "F"]}
        model = ProtoECGNet(
            self.dataset,
            label_groups=groups,
            branch_kwargs={
                "morph": {"backbone": "resnet2d", "proto_time_len": 3, "topk": 2}
            },
            prototypes_per_label=1,
            proto_dim=8,
            base_width=4,
            layers=(1, 1, 1, 1),
        )
        out = model(**self.batch)
        self.assertEqual(out["logit"].shape, (4, 6))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_fusion_mask_matches_assignment(self):
        groups = {"rhythm": ["A", "B", "C"], "morph": ["D", "E", "F"]}
        model = ProtoECGNet(
            self.dataset, label_groups=groups, prototypes_per_label=2,
            fusion="learned", **TINY,
        )
        mask = model.fusion_mask
        self.assertEqual(mask.shape, (6, 12))
        self.assertTrue(torch.all(mask.sum(dim=0) == 1))  # one label per prototype
        self.assertTrue(torch.all(mask.sum(dim=1) == 2))  # 2 prototypes per label

    def test_l1_penalty_only_in_final_stages(self):
        torch.manual_seed(0)
        plain = ProtoECGNet(self.dataset, prototypes_per_label=2, **TINY).eval()
        sparse = ProtoECGNet(
            self.dataset, prototypes_per_label=2, lam_l1=1.0, **TINY
        ).eval()
        sparse.load_state_dict(plain.state_dict())
        for stage in ("joint", "classifier"):
            plain.set_stage(stage)
            sparse.set_stage(stage)
            diff = float(sparse(**self.batch)["loss"] - plain(**self.batch)["loss"])
            if stage == "classifier":
                self.assertGreater(diff, 0.0)
            else:
                self.assertAlmostEqual(diff, 0.0, places=5)

    def test_too_long_prototype_raises(self):
        model = ProtoECGNet(
            self.dataset, prototypes_per_label=1, proto_time_len=99, **TINY
        )
        with self.assertRaises(ValueError):
            model(**self.batch)

    def test_projection_snaps_prototypes_to_training_segments(self):
        model = ProtoECGNet(
            self.dataset, prototypes_per_label=1, proto_time_len=2, **TINY
        )
        info = model.project_prototypes(self.loader)["all"]
        self.assertTrue(all(entry is not None for entry in info))
        branch = model.branches["all"]
        model.eval()
        best = []
        with torch.no_grad():
            for batch in self.loader:
                windows = branch.windows(branch.feature_map(batch["signal"].float()))
                best.append(branch.similarity_windows(windows).amax(dim=(0, 1)))
        # After projection every prototype exactly matches a training window,
        # i.e. attains the maximal cosine similarity (equal to the scale).
        self.assertTrue(torch.all(torch.stack(best).amax(dim=0) > branch.scale - 1e-3))

    def test_projection_respects_labels(self):
        model = ProtoECGNet(self.dataset, prototypes_per_label=1, **TINY)
        info = model.project_prototypes(self.loader)["all"]
        for label_index, entry in enumerate(info):
            sample = self.dataset[entry["sample"]]
            self.assertEqual(float(sample["labels"][label_index]), 1.0)

    def test_prototype_contributions_shape(self):
        model = ProtoECGNet(self.dataset, prototypes_per_label=2, **TINY)
        out = model.prototype_contributions(self.batch["signal"])
        self.assertEqual(out["all"].shape, (4, 6, 12))

    def test_frozen_bn_stats_unchanged_in_classifier_stage(self):
        model = ProtoECGNet(self.dataset, prototypes_per_label=2, **TINY)
        model.set_stage("classifier")
        branch = model.branches["all"]
        bn = next(
            m for m in branch.backbone.modules()
            if isinstance(m, torch.nn.BatchNorm1d)
        )
        mean_before = bn.running_mean.clone()
        var_before = bn.running_var.clone()
        model.train()
        model(**self.batch)
        torch.testing.assert_close(bn.running_mean, mean_before)
        torch.testing.assert_close(bn.running_var, var_before)

    def test_frozen_bn_stats_unchanged_in_fusion_stage(self):
        groups = {"rhythm": ["A", "B", "C"], "morph": ["D", "E", "F"]}
        model = ProtoECGNet(
            self.dataset, label_groups=groups, prototypes_per_label=2,
            fusion="learned", **TINY,
        )
        model.set_stage("fusion")
        bn = next(
            m for m in model.branches["rhythm"].backbone.modules()
            if isinstance(m, torch.nn.BatchNorm1d)
        )
        mean_before = bn.running_mean.clone()
        model.train()
        model(**self.batch)
        torch.testing.assert_close(bn.running_mean, mean_before)

    def test_prototype_contributions_sum_to_logit_fusion_stage(self):
        groups = {"rhythm": ["A", "B", "C"], "morph": ["D", "E", "F"]}
        model = ProtoECGNet(
            self.dataset, label_groups=groups, prototypes_per_label=2,
            fusion="learned", **TINY,
        )
        model.set_stage("fusion")
        signal = self.batch["signal"]
        contribs = model.prototype_contributions(signal)
        # Contributions (without bias) must sum to logit - fusion bias.
        total = sum(contribs[n].sum(dim=-1) for n in model.branch_names)
        with torch.no_grad():
            model.eval()
            logit = model(**self.batch)["logit"]
        expected = logit - model.fusion_classifier.bias
        torch.testing.assert_close(total, expected, atol=1e-4, rtol=1e-4)


    def test_set_pos_weight_routes_to_class_weights(self):
        """ProtoECGNet.set_pos_weight must mirror the parsed weight into
        ``class_weights`` (regression: previously a silent no-op)."""
        model = ProtoECGNet(self.dataset, prototypes_per_label=1, **TINY)

        # Scalar: broadcast to a (1,)-shape tensor, assigned verbatim.
        model.set_pos_weight(4.0)
        torch.testing.assert_close(
            model.class_weights.detach().cpu(), torch.tensor([4.0])
        )

        # Per-label sequence: length-6 vector covering self.dataset's labels.
        weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        model.set_pos_weight(weights)
        torch.testing.assert_close(
            model.class_weights.detach().cpu(), torch.tensor(weights)
        )

        # None resets to the ones default (identity for BCE pos_weight).
        model.set_pos_weight(None)
        torch.testing.assert_close(
            model.class_weights.detach().cpu(),
            torch.ones(model.num_labels),
        )

    def test_project_prototypes_raises_on_shuffled_loader(self):
        model = ProtoECGNet(self.dataset, prototypes_per_label=1, **TINY)
        shuffled = get_dataloader(self.dataset, batch_size=4, shuffle=True)
        with self.assertRaises(ValueError, msg="shuffled loader must raise"):
            model.project_prototypes(shuffled)

    def test_state_dict_round_trip_preserves_forward_output(self):
        """state_dict → load_state_dict must reproduce forward outputs exactly.

        Covers prototypes, classifier weights, fusion weights and the buffers
        populated by ``set_label_statistics`` (``class_weights``,
        ``label_cooccurrence``). ``class_weights`` and ``label_cooccurrence``
        were previously registered as ``None`` buffers and promoted to tensors
        only when supplied; strict ``load_state_dict`` then failed because the
        saved checkpoint carried keys the fresh model lacked. This test guards
        against that regression.
        """
        torch.manual_seed(0)
        groups = {"rhythm": ["A", "B", "C"], "morph": ["D", "E", "F"]}
        model = ProtoECGNet(
            self.dataset, label_groups=groups, prototypes_per_label=2,
            fusion="learned", **TINY,
        )
        # Set non-trivial label statistics so the buffers carry real tensors
        # and the contrastive term is active in forward.
        weights, cooc = ProtoECGNet.compute_label_statistics(
            torch.stack([self.dataset[i]["labels"] for i in range(12)])
        )
        model.set_label_statistics(class_weights=weights, label_cooccurrence=cooc)
        # Nudge weights off their initializer so random re-init would be caught.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1 * torch.randn_like(p))
        model.eval()
        with torch.no_grad():
            expected = model(**self.batch)["logit"]

        clone = ProtoECGNet(
            self.dataset, label_groups=groups, prototypes_per_label=2,
            fusion="learned", **TINY,
        ).eval()
        # strict=True by default; fails if keys mismatch.
        clone.load_state_dict(model.state_dict())
        with torch.no_grad():
            got = clone(**self.batch)["logit"]
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(clone.class_weights, model.class_weights)
        torch.testing.assert_close(clone.label_cooccurrence, model.label_cooccurrence)


if __name__ == "__main__":
    unittest.main()
