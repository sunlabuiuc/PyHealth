import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn

from pyhealth.models.mvcl_model import MultiViewContrastiveModel

class TestMultiViewContrastiveModel(unittest.TestCase):
    def setUp(self):
        """Set up dummy hyperparameters and dynamic modules for testing."""
        self.hidden_dim = 32
        self.batch_size = 4
        self.seq_len = 10
        self.num_classes = 3
        
        self.view_names = ["view_A", "view_B"]
        
        self.projectors = nn.ModuleDict({
            view: nn.Linear(1, self.hidden_dim) for view in self.view_names
        })
        
        self.encoders = nn.ModuleDict({
            view: nn.Linear(self.hidden_dim, self.hidden_dim) for view in self.view_names
        })
        
        self.augmentations = {
            view: lambda x: x + torch.randn_like(x) * 0.01 for view in self.view_names
        }

        self.mock_dataset = MagicMock()
        self.mock_dataset.input_schema = {view: "tensor" for view in self.view_names}
        self.mock_dataset.output_schema = {"label": "multiclass"}

    def test_pretrain_forward(self):
        """Tests the representation learning stage and NT-Xent loss."""
        model = MultiViewContrastiveModel(
            dataset=self.mock_dataset, 
            encoders=self.encoders,
            projectors=self.projectors,
            augmentations=self.augmentations,
            hidden_dim=self.hidden_dim,
            training_stage="pretrain"
        )
        
        kwargs = {
            "view_A": torch.randn(self.batch_size, self.seq_len, 1),
            "view_B": torch.randn(self.batch_size, self.seq_len, 1)
        }
        
        outputs = model(**kwargs)
        
        self.assertIn("loss", outputs)
        self.assertIn("z_view_A", outputs)
        self.assertIn("z_view_B", outputs)
        self.assertEqual(outputs["z_view_A"].shape, (self.batch_size, self.hidden_dim))
        self.assertTrue(torch.is_tensor(outputs["loss"]))
        self.assertFalse(torch.isnan(outputs["loss"]))

    def test_finetune_forward(self):
        """Tests the classification finetuning stage and cross-entropy loss."""
        model = MultiViewContrastiveModel(
            dataset=self.mock_dataset,
            encoders=self.encoders,
            projectors=self.projectors,
            augmentations=self.augmentations,
            hidden_dim=self.hidden_dim,
            training_stage="finetune",
            num_classes=self.num_classes
        )
        
        kwargs = {
            "view_A": torch.randn(self.batch_size, self.seq_len, 1),
            "view_B": torch.randn(self.batch_size, self.seq_len, 1),
            "label": torch.randint(0, self.num_classes, (self.batch_size,))
        }
        
        outputs = model(**kwargs)
        
        self.assertIn("loss", outputs)
        self.assertIn("logit", outputs)
        self.assertIn("y_prob", outputs)
        self.assertIn("y_true", outputs)
        self.assertEqual(outputs["logit"].shape, (self.batch_size, self.num_classes))
        self.assertEqual(outputs["y_prob"].shape, (self.batch_size, self.num_classes))
        self.assertTrue(torch.allclose(outputs["y_prob"].sum(dim=-1), torch.ones(self.batch_size)))

if __name__ == '__main__':
    unittest.main()