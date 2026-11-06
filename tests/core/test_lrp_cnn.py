"""Tests for LRP on continuous (image) inputs with use_embeddings=False.

Covers the CNN layer path, residual blocks (torchvision BasicBlock with and
without a projection shortcut), logit extraction fallbacks, and the shape /
feature-splitting utilities on LayerwiseRelevancePropagation.
"""

import unittest

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock

from pyhealth.interpret.methods import LayerwiseRelevancePropagation


class TinyCNN(nn.Module):
    """Small CNN touching every supported image layer type."""

    def __init__(self, num_classes=3, return_logit=True):
        super().__init__()
        self.return_logit = return_logit
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Dropout(0.1),
        )
        self.head = nn.Linear(16, num_classes)

    def forward(self, image, **kwargs):
        logit = self.head(self.features(image))
        y_prob = torch.softmax(logit, dim=-1)
        if self.return_logit:
            return {"logit": logit, "y_prob": y_prob}
        return {"y_prob": y_prob}


class TinyResNet(nn.Module):
    """Two residual blocks: identity shortcut, then projection shortcut."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.layer1 = BasicBlock(4, 4)
        self.layer2 = BasicBlock(
            4, 8, stride=2,
            downsample=nn.Sequential(nn.Conv2d(4, 8, 1, stride=2), nn.BatchNorm2d(8)),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8, num_classes)

    def forward(self, image, **kwargs):
        x = self.layer2(self.layer1(self.stem(image)))
        return {"logit": self.fc(self.flatten(self.pool(x)))}


class NoLinearCNN(nn.Module):
    """Model with no Linear layer that only reports probabilities."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, image, **kwargs):
        return {"y_prob": torch.softmax(self.flatten(self.pool(self.conv(image))), dim=-1)}


class BinaryCNN(nn.Module):
    """Single-logit binary classifier."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 1)

    def forward(self, image, **kwargs):
        return {"logit": self.fc(self.flatten(image))}


class BadOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, image, **kwargs):
        return {"embedding": self.conv(image)}


def _lrp(model, **kwargs):
    model.eval()
    return LayerwiseRelevancePropagation(model, use_embeddings=False, **kwargs)


class TestLRPInputValidation(unittest.TestCase):

    def test_use_embeddings_requires_embedding_interface(self):
        with self.assertRaises(ValueError):
            LayerwiseRelevancePropagation(TinyCNN(), use_embeddings=True)


class TestLRPCNN(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.model = TinyCNN()
        self.image = torch.randn(2, 1, 8, 8)

    def test_attribution_shape_both_rules(self):
        for rule in ("epsilon", "alphabeta"):
            attr = _lrp(self.model, rule=rule).attribute(image=self.image)
            self.assertEqual(set(attr), {"image"})
            self.assertEqual(attr["image"].shape, self.image.shape)
            self.assertTrue(torch.isfinite(attr["image"]).all())

    def test_target_class_changes_attribution(self):
        lrp = _lrp(self.model)
        a0 = lrp.attribute(target_class_idx=0, image=self.image)["image"]
        a1 = lrp.attribute(target_class_idx=1, image=self.image)["image"]
        self.assertFalse(torch.allclose(a0, a1))

    def test_hooks_are_removed_after_attribute(self):
        _lrp(self.model).attribute(image=self.image)
        for module in self.model.modules():
            self.assertEqual(len(module._forward_hooks), 0)

    def test_extra_feature_keys_and_list_labels(self):
        # Declared feature keys missing from the batch are skipped, and
        # non-tensor label values are converted before being forwarded.
        self.model.feature_keys = ["image", "absent"]
        self.model.label_keys = ["label"]
        attr = _lrp(self.model).attribute(image=self.image, label=[0, 1])
        self.assertEqual(set(attr), {"image"})

    def test_logits_from_last_linear_when_missing_logit_key(self):
        model = TinyCNN(return_logit=False)
        attr = _lrp(model).attribute(image=self.image)["image"]
        self.assertEqual(attr.shape, self.image.shape)
        self.assertGreater(attr.abs().sum().item(), 0.0)

    def test_y_prob_fallback_without_linear(self):
        attr = _lrp(NoLinearCNN()).attribute(image=self.image)["image"]
        self.assertEqual(attr.shape, self.image.shape)

    def test_missing_logit_and_y_prob_raises(self):
        with self.assertRaises(KeyError):
            _lrp(BadOutputModel()).attribute(image=self.image)

    def test_visualize(self):
        lrp = _lrp(self.model)
        attr = lrp.attribute(image=self.image)["image"]
        lrp.visualize(plt, self.image[0], attr[0], show=False)
        self.assertIn("epsilon", plt.gca().get_title())
        plt.close("all")


class TestLRPBinaryTarget(unittest.TestCase):

    def test_negative_class_negates_epsilon_relevance(self):
        torch.manual_seed(0)
        lrp = _lrp(BinaryCNN())
        image = torch.randn(2, 1, 4, 4)
        pos = lrp.attribute(image=image)["image"]
        neg = lrp.attribute(target_class_idx=0, image=image)["image"]
        self.assertTrue(torch.allclose(neg, -pos, atol=1e-6))


class TestLRPResidualBlocks(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.model = TinyResNet()
        self.image = torch.randn(2, 1, 8, 8)

    def test_both_rules_produce_finite_input_relevance(self):
        for rule in ("epsilon", "alphabeta"):
            attr = _lrp(self.model, rule=rule, beta=0.5, alpha=1.5).attribute(image=self.image)
            self.assertEqual(attr["image"].shape, self.image.shape)
            self.assertTrue(torch.isfinite(attr["image"]).all(), rule)
            self.assertGreater(attr["image"].abs().sum().item(), 0.0, rule)

    def test_residual_blocks_are_tracked(self):
        from pyhealth.interpret.methods.lrp import _LRPHookContext

        with _LRPHookContext(self.model) as ctx:
            with torch.no_grad():
                self.model(image=self.image)
            self.assertEqual(set(ctx.residual_blocks), {"layer1", "layer2"})
            self.assertFalse(ctx.residual_blocks["layer1"]["has_downsample"])
            self.assertTrue(ctx.residual_blocks["layer2"]["has_downsample"])
            lrp = _lrp(self.model)
            self.assertEqual(lrp._get_block_for_layer("layer2.conv1", ctx), "layer2")
            self.assertIsNone(lrp._get_block_for_layer("fc", ctx))
        self.assertEqual(ctx.residual_blocks, {})


class TestLRPShapeUtilities(unittest.TestCase):

    def setUp(self):
        self.lrp = _lrp(TinyCNN())

    def test_match_shapes(self):
        m = self.lrp._match_shapes
        r = torch.ones(2, 3, 4, 4)
        self.assertIs(m(r, r.shape), r)
        # 2D -> 4D: exact reshape, or spread the total uniformly
        self.assertEqual(m(torch.ones(2, 48), (2, 3, 4, 4)).shape, (2, 3, 4, 4))
        spread = m(torch.ones(2, 5), (2, 3, 4, 4))
        self.assertEqual(spread.shape, (2, 3, 4, 4))
        self.assertAlmostEqual(spread[0].sum().item(), 5.0, places=5)
        # 4D -> 4D: channel and spatial mismatch
        self.assertEqual(m(torch.ones(2, 1, 2, 2), (2, 3, 4, 4)).shape, (2, 3, 4, 4))
        # 3D -> 4D
        self.assertEqual(m(torch.ones(2, 4, 4), (2, 3, 4, 4)).shape, (2, 3, 4, 4))
        self.assertEqual(m(torch.ones(2, 2, 2), (2, 3, 4, 4)).shape, (2, 3, 4, 4))
        # Unhandled combination is returned unchanged
        odd = torch.ones(2, 3)
        self.assertIs(m(odd, (2, 3, 4)), odd)

    def test_unhandled_layer_passes_relevance_through(self):
        r = torch.ones(1, 3)
        self.assertIs(self.lrp._propagate_through_layer(nn.Tanh(), {}, r), r)

    def test_split_pre_split_relevance(self):
        emb = {"a": torch.zeros(2, 3, 4)}
        # Extra trailing columns (e.g. StageNet's time interval) are dropped and
        # pooled relevance is spread evenly across timesteps.
        out = self.lrp._split_relevance_to_features(
            {"a": torch.ones(2, 5), "not_an_input": torch.ones(2, 1)}, emb
        )
        self.assertEqual(set(out), {"a"})
        self.assertEqual(out["a"].shape, (2, 3, 4))
        self.assertTrue(torch.allclose(out["a"].sum(dim=(1, 2)), torch.full((2,), 4.0)))

    def test_split_concatenated_relevance(self):
        self.lrp.model.feature_keys = ["a", "b", "missing"]
        emb = {"a": torch.zeros(2, 3, 4), "b": torch.zeros(2, 2)}
        relevance = torch.arange(12.0).view(2, 6)
        out = self.lrp._split_relevance_to_features(relevance, emb)
        self.assertEqual(out["a"].shape, (2, 3, 4))
        self.assertTrue(torch.equal(out["a"][:, 0], relevance[:, :4]))
        self.assertTrue(torch.equal(out["b"], relevance[:, 4:]))

    def test_split_falls_back_to_even_distribution(self):
        emb = {"a": torch.zeros(2, 4), "b": torch.zeros(2, 1, 2, 2)}
        out = self.lrp._split_relevance_to_features(torch.ones(2, 3), emb)
        for key in emb:
            self.assertTrue(torch.allclose(out[key], torch.full((2, 3), 0.5)))


if __name__ == "__main__":
    unittest.main()
