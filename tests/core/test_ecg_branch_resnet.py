"""Tests for ECGBranchResNet using tiny synthetic ECG tensors (no real data).

Author: Hanho Ryu (NetID: hanhor2)
"""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import ECGBranchResNet, ResNetBranch

LABELS = ["A", "B", "C", "D", "E"]
GROUPS = {"g0": ["A", "B"], "g1": ["C", "D", "E"]}
TINY = {"base_width": 4, "layers": (1, 1, 1, 1)}
LENGTH = 128


def make_dataset(num_samples: int = 6, output: object = "multilabel"):
    """Creates a small multilabel dataset with random 12-lead signals."""
    torch.manual_seed(0)
    samples = []
    for i in range(num_samples):
        labels = [LABELS[i % 5]]
        if i % 2 == 0:
            labels.append(LABELS[(i + 3) % 5])
        samples.append(
            {
                "patient_id": f"p{i}",
                "signal": torch.randn(12, LENGTH).tolist(),
                "labels": labels,
            }
        )
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"labels": output},
    )


DATASET = make_dataset()


def first_batch(dataset=DATASET, batch_size: int = 4) -> dict:
    """Returns the first batch of a deterministic dataloader."""
    return next(iter(get_dataloader(dataset, batch_size=batch_size, shuffle=False)))


class TestResNetBranch(unittest.TestCase):
    """The single black-box branch."""

    def test_1d_shapes(self):
        branch = ResNetBranch(12, 3, **TINY)
        logits, feats = branch(torch.randn(2, 12, LENGTH))
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(feats.shape, (2, branch.feature_dim))
        self.assertEqual(branch.feature_dim, 32)  # 4 * 2**3

    def test_2d_shapes(self):
        branch = ResNetBranch(12, 2, backbone="resnet2d")
        logits, feats = branch(torch.randn(1, 12, LENGTH))
        self.assertEqual(logits.shape, (1, 2))
        self.assertEqual(feats.shape, (1, 512))

    def test_invalid_options(self):
        with self.assertRaises(ValueError):
            ResNetBranch(12, 2, backbone="vit")
        with self.assertRaises(ValueError):
            ResNetBranch(12, 2, dropout=1.0)


class TestInstantiation(unittest.TestCase):
    """Construction, grouping and validation."""

    def test_default_is_single_full_label_branch(self):
        model = ECGBranchResNet(DATASET, **TINY)
        self.assertEqual(model.branch_names, ["all"])
        self.assertEqual(model.num_labels, 5)
        self.assertIsNone(model.fusion_classifier)

    def test_groups_map_to_label_indices(self):
        model = ECGBranchResNet(DATASET, label_groups=GROUPS, **TINY)
        names = model.label_names
        for branch, members in GROUPS.items():
            index = model.branch_index(branch).tolist()
            self.assertEqual([names[i] for i in index], members)

    def test_unknown_labels_are_dropped(self):
        groups = {"g0": ["A", "B", "ZZZ"], "g1": ["C", "D", "E"], "g2": ["YYY"]}
        model = ECGBranchResNet(DATASET, label_groups=groups, **TINY)
        self.assertEqual(model.branch_names, ["g0", "g1"])

    def test_invalid_groups(self):
        for groups in (
            {"g0": ["A", "B"]},  # missing labels
            {"g0": ["A", "B", "C"], "g1": ["C", "D", "E"]},  # duplicate
            {"g.0": ["A", "B"], "g1": ["C", "D", "E"]},  # bad name
        ):
            with self.assertRaises(ValueError):
                ECGBranchResNet(DATASET, label_groups=groups, **TINY)

    def test_invalid_options(self):
        with self.assertRaises(ValueError):
            ECGBranchResNet(DATASET, fusion="attention", **TINY)
        with self.assertRaises(ValueError):
            ECGBranchResNet(DATASET, lam_l1=-1.0, **TINY)
        with self.assertRaises(ValueError):
            ECGBranchResNet(DATASET, branch_kwargs={"nope": {}}, **TINY)
        with self.assertRaises(ValueError):
            ECGBranchResNet(DATASET, class_weights=torch.ones(3), **TINY)

    def test_requires_multilabel(self):
        dataset = create_sample_dataset(
            samples=[
                {"patient_id": "a", "signal": [[0.0] * 16] * 12, "y": 0},
                {"patient_id": "b", "signal": [[1.0] * 16] * 12, "y": 1},
            ],
            input_schema={"signal": "tensor"},
            output_schema={"y": "binary"},
        )
        with self.assertRaises(ValueError):
            ECGBranchResNet(dataset, **TINY)

    def test_branch_kwargs_override(self):
        model = ECGBranchResNet(
            DATASET,
            label_groups=GROUPS,
            branch_kwargs={"g1": {"base_width": 8, "dropout": 0.0}},
            **TINY,
        )
        self.assertEqual(model.branches["g0"].feature_dim, 32)
        self.assertEqual(model.branches["g1"].feature_dim, 64)
        self.assertEqual(model.branches["g1"].dropout.p, 0.0)

    def test_class_weights_default_and_override(self):
        model = ECGBranchResNet(DATASET, **TINY)
        torch.testing.assert_close(model.class_weights, torch.ones(5))
        model = ECGBranchResNet(DATASET, class_weights=torch.tensor(2.0), **TINY)
        torch.testing.assert_close(model.class_weights, torch.full((5,), 2.0))


class TestForward(unittest.TestCase):
    """Forward pass, output shapes and loss."""

    def test_output_keys_and_shapes(self):
        for fusion in ("none", "logits", "features"):
            with self.subTest(fusion=fusion):
                model = ECGBranchResNet(
                    DATASET, label_groups=GROUPS, fusion=fusion, **TINY
                )
                out = model(**first_batch(), embed=True)
                self.assertEqual(out["logit"].shape, (4, 5))
                self.assertEqual(out["y_prob"].shape, (4, 5))
                self.assertEqual(out["y_true"].shape, (4, 5))
                self.assertEqual(out["branch_logit"].shape, (4, 5))
                self.assertEqual(out["embed"].shape, (4, 64))
                self.assertEqual(out["loss"].dim(), 0)
                self.assertTrue(torch.isfinite(out["loss"]))
                self.assertTrue(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all())

    def test_without_labels(self):
        model = ECGBranchResNet(DATASET, **TINY)
        out = model(signal=torch.randn(3, 12, LENGTH))
        self.assertEqual(out["logit"].shape, (3, 5))
        self.assertNotIn("loss", out)

    def test_scatter_places_branch_logits(self):
        model = ECGBranchResNet(DATASET, label_groups=GROUPS, **TINY).eval()
        x = torch.randn(2, 12, LENGTH)
        out = model(signal=x)
        for name in model.branch_names:
            logits, _ = model.branches[name](x)
            torch.testing.assert_close(
                out["logit"][:, model.branch_index(name)], logits
            )

    def test_logit_fusion_starts_at_branch_prediction(self):
        model = ECGBranchResNet(
            DATASET, label_groups=GROUPS, fusion="logits", **TINY
        ).eval()
        out = model(signal=torch.randn(2, 12, LENGTH))
        torch.testing.assert_close(out["logit"], out["branch_logit"])

    def test_fusion_stage_warm_starts_from_branch_heads(self):
        for fusion in ("logits", "features"):
            with self.subTest(fusion=fusion):
                model = ECGBranchResNet(
                    DATASET, label_groups=GROUPS, fusion=fusion, **TINY
                ).eval()
                model.set_stage("branches")
                with torch.no_grad():
                    model.fusion_classifier.weight.normal_()
                model.set_stage("fusion")
                out = model(signal=torch.randn(2, 12, LENGTH))
                torch.testing.assert_close(out["logit"], out["branch_logit"])
                self.assertEqual(model._l1_penalty().item(), 0.0)

    def test_random_fusion_init_is_kept(self):
        for fusion in ("logits", "features"):
            with self.subTest(fusion=fusion):
                model = ECGBranchResNet(
                    DATASET,
                    label_groups=GROUPS,
                    fusion=fusion,
                    fusion_init="random",
                    **TINY,
                )
                before = model.fusion_classifier.weight.clone()
                # PyTorch's default init, not the identity / own-group mask.
                self.assertFalse(torch.equal(before, model.fusion_mask))
                model.set_stage("branches")
                model.set_stage("fusion")
                torch.testing.assert_close(model.fusion_classifier.weight, before)
        with self.assertRaises(ValueError):
            ECGBranchResNet(DATASET, fusion="logits", fusion_init="zeros", **TINY)
        with self.assertRaises(ValueError):
            ECGBranchResNet(DATASET, **TINY).init_fusion_from_branches()

    def test_branches_stage_bypasses_fusion(self):
        model = ECGBranchResNet(
            DATASET, label_groups=GROUPS, fusion="features", **TINY
        ).eval()
        model.set_stage("branches")
        out = model(signal=torch.randn(2, 12, LENGTH))
        torch.testing.assert_close(out["logit"], out["branch_logit"])


class TestTraining(unittest.TestCase):
    """Gradients, stages and the L1 penalty."""

    def test_gradients_reach_every_parameter(self):
        for fusion in ("none", "logits"):
            with self.subTest(fusion=fusion):
                model = ECGBranchResNet(
                    DATASET, label_groups=GROUPS, fusion=fusion, **TINY
                )
                model(**first_batch())["loss"].backward()
                for name, param in model.named_parameters():
                    if name != "_dummy_param":
                        self.assertIsNotNone(param.grad, name)

    def test_feature_fusion_bypasses_branch_heads(self):
        model = ECGBranchResNet(DATASET, label_groups=GROUPS, fusion="features", **TINY)
        model(**first_batch())["loss"].backward()
        for name, param in model.named_parameters():
            if name == "_dummy_param":
                continue
            if ".head." in name:
                self.assertIsNone(param.grad, name)
            else:
                self.assertIsNotNone(param.grad, name)

    def test_fusion_stage_freezes_branches(self):
        model = ECGBranchResNet(DATASET, label_groups=GROUPS, fusion="features", **TINY)
        model.set_stage("fusion")
        model.train()
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(
            trainable, {"fusion_classifier.weight", "fusion_classifier.bias"}
        )
        self.assertFalse(model.branches.training)
        self.assertTrue(model.fusion_classifier.training)
        bn = model.branches["g0"].backbone.stem[1]
        before = bn.running_mean.clone()
        model(**first_batch())["loss"].backward()
        torch.testing.assert_close(bn.running_mean, before)
        self.assertIsNone(model.branches["g0"].head.weight.grad)
        self.assertIsNotNone(model.fusion_classifier.weight.grad)

    def test_branches_stage_freezes_fusion(self):
        model = ECGBranchResNet(DATASET, label_groups=GROUPS, fusion="logits", **TINY)
        model.set_stage("branches")
        self.assertFalse(model.fusion_classifier.weight.requires_grad)
        self.assertTrue(model.branches["g1"].head.weight.requires_grad)
        model.set_stage("all")
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_invalid_stage(self):
        model = ECGBranchResNet(DATASET, **TINY)
        with self.assertRaises(ValueError):
            model.set_stage("joint")
        with self.assertRaises(ValueError):
            model.set_stage("fusion")  # no fusion head

    def test_l1_penalty_only_on_cross_group_weights(self):
        model = ECGBranchResNet(
            DATASET, label_groups=GROUPS, fusion="logits", lam_l1=1.0, **TINY
        )
        model.set_stage("fusion")
        # Identity initialization: no cross-group weight, no penalty.
        self.assertEqual(model._l1_penalty().item(), 0.0)
        with torch.no_grad():
            model.fusion_classifier.weight.fill_(1.0)
        # 5 labels x 5 inputs, of which 5 are own-group (identity) entries.
        self.assertEqual(model._l1_penalty().item(), 20.0)
        batch = first_batch()
        with_l1 = model(**batch)["loss"]
        model.lam_l1 = 0.0
        self.assertAlmostEqual((with_l1 - model(**batch)["loss"]).item(), 20.0, 4)

    def test_feature_fusion_mask_blocks(self):
        model = ECGBranchResNet(DATASET, label_groups=GROUPS, fusion="features", **TINY)
        mask = model.fusion_mask
        self.assertEqual(mask.shape, (5, 64))
        g0 = model.branch_index("g0")
        self.assertTrue(mask[g0, :32].eq(1).all())
        self.assertTrue(mask[g0, 32:].eq(0).all())

    def test_one_optimizer_step_lowers_loss(self):
        torch.manual_seed(0)
        model = ECGBranchResNet(DATASET, dropout=0.0, **TINY)
        batch = first_batch(batch_size=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.eval()  # fixed BatchNorm statistics make the check deterministic
        before = model(**batch)["loss"]
        before.backward()
        optimizer.step()
        self.assertLess(model(**batch)["loss"].item(), before.item())

    def test_stage_round_trips_through_state_dict(self):
        model = ECGBranchResNet(DATASET, label_groups=GROUPS, fusion="logits", **TINY)
        model.set_stage("fusion")
        clone = ECGBranchResNet(DATASET, label_groups=GROUPS, fusion="logits", **TINY)
        clone.load_state_dict(model.state_dict())
        self.assertEqual(clone.stage, "fusion")
        self.assertFalse(clone.branches["g0"].head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
