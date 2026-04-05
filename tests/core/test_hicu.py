import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.hicu import (
    AsymmetricLoss,
    HiCu,
    build_icd10_hierarchy,
)


class TestHiCu(unittest.TestCase):
    """Test cases for the HiCu model."""

    def setUp(self):
        """Create a synthetic multilabel dataset with ICD-10 codes."""
        self.samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "text": ["patient", "has", "fever", "and", "cough"],
                "icd_codes": ["E11.321", "I10", "J44.1"],
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "text": ["chest", "pain", "shortness", "of", "breath"],
                "icd_codes": ["I21.09", "I11.0"],
            },
            {
                "patient_id": "p2",
                "visit_id": "v2",
                "text": ["abdominal", "pain", "nausea"],
                "icd_codes": ["K21.0", "E11.65"],
            },
        ]
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"text": "sequence"},
            output_schema={"icd_codes": "multilabel"},
            dataset_name="test_hicu",
        )
        self.model = HiCu(
            self.dataset,
            num_filter_maps=8,
            embedding_dim=16,
        )

    def test_model_initialization(self):
        """HiCu instantiates with correct attributes."""
        self.assertIsInstance(self.model, HiCu)
        self.assertEqual(self.model.num_filter_maps, 8)
        self.assertEqual(self.model.embedding_dim, 16)
        self.assertEqual(self.model.text_key, "text")
        self.assertEqual(self.model.label_key, "icd_codes")
        self.assertEqual(len(self.model.depth_sizes), 3)
        # Depth 0 = chapters, depth 2 = full codes
        self.assertLessEqual(self.model.depth_sizes[0], self.model.depth_sizes[2])

    def test_forward_pass(self):
        """Forward pass returns all required output keys."""
        loader = get_dataloader(self.dataset, batch_size=2)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, ret, f"Missing key: {key}")
        self.assertEqual(ret["loss"].dim(), 0)  # scalar
        self.assertEqual(ret["y_prob"].shape[0], 2)  # batch size

    def test_backward_pass(self):
        """Gradients flow after loss.backward()."""
        loader = get_dataloader(self.dataset, batch_size=2)
        batch = next(iter(loader))

        ret = self.model(**batch)
        ret["loss"].backward()

        has_grad = any(
            p.requires_grad and p.grad is not None for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No gradients after backward pass")

    def test_output_shapes(self):
        """Output shapes match num_codes at the current depth."""
        loader = get_dataloader(self.dataset, batch_size=2)
        batch = next(iter(loader))

        for depth in range(3):
            self.model.set_depth(depth)
            with torch.no_grad():
                ret = self.model(**batch)
            expected_codes = self.model.depth_sizes[depth]
            self.assertEqual(
                ret["y_prob"].shape[1],
                expected_codes,
                f"y_prob width mismatch at depth {depth}",
            )
            self.assertEqual(
                ret["logit"].shape[1],
                expected_codes,
                f"logit width mismatch at depth {depth}",
            )

    def test_depth_change(self):
        """set_depth changes active decoder and output size."""
        loader = get_dataloader(self.dataset, batch_size=2)
        batch = next(iter(loader))

        self.model.set_depth(0)
        with torch.no_grad():
            ret0 = self.model(**batch)

        self.model.set_depth(2)
        with torch.no_grad():
            ret2 = self.model(**batch)

        # Depth 0 has fewer codes than depth 2.
        self.assertLess(ret0["y_prob"].shape[1], ret2["y_prob"].shape[1])

    def test_weight_transfer(self):
        """After set_depth, child weights equal parent weights at mapped positions."""
        c2p = self.model.hierarchy["child_to_parent"].get(1, {})
        if not c2p:
            self.skipTest("No child-to-parent mapping at depth 1")

        self.model.set_depth(1)

        parent_w = self.model.decoder.attention[0].weight.data
        child_w = self.model.decoder.attention[1].weight.data

        for child_idx, parent_idx in c2p.items():
            torch.testing.assert_close(
                child_w[child_idx],
                parent_w[parent_idx],
                msg=f"Weight mismatch: child {child_idx} != parent {parent_idx}",
            )

    def test_asymmetric_loss(self):
        """AsymmetricLoss produces reasonable scalar values."""
        loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        targets[:, :3] = 1.0

        loss = loss_fn(logits, targets)

        self.assertEqual(loss.dim(), 0)
        self.assertTrue(loss.item() > 0, "Loss should be positive")
        self.assertTrue(torch.isfinite(loss), "Loss should be finite")

    def test_icd10_hierarchy(self):
        """Hierarchy builder maps E11.321 to correct ancestors."""
        codes = ["E11.321", "I10", "I11.0", "J44.1"]
        h = build_icd10_hierarchy(codes)

        # Depth 0: chapters
        self.assertIn("E00-E89", h["depth_to_codes"][0])
        self.assertIn("I00-I99", h["depth_to_codes"][0])
        self.assertIn("J00-J99", h["depth_to_codes"][0])

        # Depth 1: categories (3-char)
        self.assertIn("E11", h["depth_to_codes"][1])
        self.assertIn("I10", h["depth_to_codes"][1])
        self.assertIn("I11", h["depth_to_codes"][1])
        self.assertIn("J44", h["depth_to_codes"][1])

        # Depth 2: full codes
        self.assertEqual(h["depth_to_codes"][2], codes)

        # Check parent-child: E11 (cat) should be child of E00-E89 (chapter)
        cat_idx = h["code_to_index"][1]["E11"]
        chapter_idx = h["code_to_index"][0]["E00-E89"]
        self.assertEqual(h["child_to_parent"][1][cat_idx], chapter_idx)


if __name__ == "__main__":
    unittest.main()
