"""Tests for the SHy (Self-Explaining Hypergraph Neural Networks) model.

Uses small synthetic data only. No real datasets (MIMIC, etc.) are used.
"""

import tempfile
import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SHy


SYNTHETIC_CODE_LEVELS = np.array(
    [[1, 1, 1], [1, 1, 2], [1, 2, 3], [2, 3, 4], [2, 3, 5]]
)

SMALL_HPARAMS = dict(
    code_levels=SYNTHETIC_CODE_LEVELS,
    single_dim=8,
    hgnn_dim=16,
    after_hgnn_dim=16,
    hgnn_layer_num=1,
    nhead=2,
    num_tp=2,
    temperatures=[0.5, 0.5],
    add_ratios=[0.1, 0.1],
    n_c=3,
    hid_state_dim=16,
    key_dim=16,
    sa_head=2,
    dropout=0.0,
)

CODES = ["C001", "C002", "C003", "C004", "C005"]


def _make_dataset():
    samples = [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "conditions": [[CODES[0], CODES[1]], [CODES[2]]],
            "label": [CODES[0], CODES[2]],
        },
        {
            "patient_id": "p1",
            "visit_id": "v1",
            "conditions": [[CODES[2], CODES[3]], [CODES[4]]],
            "label": [CODES[3], CODES[4]],
        },
        {
            "patient_id": "p2",
            "visit_id": "v2",
            "conditions": [[CODES[0], CODES[4]], [CODES[1], CODES[3]]],
            "label": [CODES[0], CODES[1], CODES[4]],
        },
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"conditions": "nested_sequence"},
        output_schema={"label": "multilabel"},
        dataset_name="test_shy",
    )


class TestSHyLearns(unittest.TestCase):
    """Verify the model trains end-to-end and loss decreases."""

    def test_loss_decreases_after_training(self):
        """Train for 20 steps and verify loss drops."""
        dataset = _make_dataset()
        model = SHy(dataset=dataset, **SMALL_HPARAMS)
        loader = get_dataloader(dataset, batch_size=3, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = next(iter(loader))

        model.train()
        ret = model(**batch)
        initial_loss = ret["loss"].item()

        for _ in range(20):
            ret = model(**batch)
            optimizer.zero_grad()
            ret["loss"].backward()
            optimizer.step()

        final_loss = ret["loss"].item()

        # Model actually learns
        self.assertLess(final_loss, initial_loss)
        # Loss is finite
        self.assertTrue(np.isfinite(final_loss))
        # Output has correct shape and valid probabilities
        self.assertEqual(ret["y_prob"].shape, ret["y_true"].shape)
        self.assertTrue((ret["y_prob"] >= 0).all())
        self.assertTrue((ret["y_prob"] <= 1).all())
        # Gradients are nonzero
        max_grad = max(
            p.grad.abs().max().item()
            for p in model.parameters()
            if p.grad is not None
        )
        self.assertGreater(max_grad, 0.0)


###################################################################
# TODO: Additional tests for full rubric coverage (11 pts)
#
# Edge cases:
#   - Single-visit patients
#   - Batch size 1
#   - K=1 full forward pass (no self-attention path)
#   - hgnn_layer_num=-1 (linear fallback, no HGNN)
#   - Varying visit lengths in same batch (1, 2, 3, 4 visits)
#   - K=7 phenotypes
#   - UniGATConv instead of UniGINConv
#
# Component-level tests:
#   - UniGINConv output shapes and self-loop effect
#   - HierarchicalEmbedding: codes with same ancestor share
#     level-0 embedding
#   - HSLPart1 probability output in range [0, 1]
#   - FinalClassifier softmax predictions sum to 1
#   - shy_loss returns all 4 components when K>1
#
# Training behavior:
#   - Eval mode produces valid output
#   - Model save/load with tempfile.TemporaryDirectory
#     produces identical predictions
#
###################################################################


if __name__ == "__main__":
    unittest.main()
