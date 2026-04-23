import unittest
import torch
import os
import tempfile
import shutil
from typing import Dict
import random

from pyhealth.models import ResNetLSTM, CNNLSTM
from pyhealth.datasets import create_sample_dataset


class TestConv2dResNetLSTM(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

        # TUH-like fake structure 
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = os.path.join(
            self.temp_dir,
            "data/tuh_eeg/tuh_eeg_seizure/v2.0.5/edf/dev"
        )
        os.makedirs(self.base_path, exist_ok=True)

        subject = "aaaaaajy"
        session = "s001_2002"
        montage = "02_tcp_le"

        self.sample_dir = os.path.join(
            self.base_path, subject, session, montage
        )
        os.makedirs(self.sample_dir, exist_ok=True)

        # Fake EEG data 
        self.batch_size = 2
        self.seq_len = 120
        self.in_channel = 8
        self.output_dim = 3
        self.device = "cpu"

        self.samples = [self.__generate_random_samples(i) for i in range(self.batch_size)]
        task_name    : str            = "tusz_task"
        input_schema : Dict[str, str] = { "signal": "tensor" }
        output_schema: Dict[str, str] = {
            "label"        : "tensor",
            "label_bitgt_1": "tensor",
            "label_bitgt_2": "tensor",
            "label_name"   : "text",
        }

        self.dataset = create_sample_dataset(
            samples       = self.samples,
            input_schema  = input_schema,
            output_schema = output_schema,
            dataset_name  = "tusz",
            task_name     = task_name
        )

        self.model = CNNLSTM(
            dataset=self.dataset,
            encoder    = None,
            num_layers = 1,
            output_dim = self.output_dim,
            batch_size = self.batch_size,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def __generate_random_samples(self, i):
        num = random.randint(1, 3)
        if num == 1:
            return {
                "patient_id"   : f"p{i}",
                "signal"       : torch.randn((20, 6000)) * 50 - 20,
                "label"        : torch.zeros(1500, dtype=torch.uint8),
                "label_bitgt_1": torch.zeros(1500, dtype=torch.uint8),
                "label_bitgt_2": torch.zeros(1500, dtype=torch.uint8),
                "label_name"   : '0_patF'
            }

        if num == 2:
            return {
                "patient_id"   : f"p{i}",
                "signal"       : torch.randn((20, 6000)) * 50 - 20,
                "label"        : torch.ones(1500, dtype=torch.uint8),
                "label_bitgt_1": torch.ones(1500, dtype=torch.uint8),
                "label_bitgt_2": torch.ones(1500, dtype=torch.uint8),
                "label_name"   : '1_middle'
            }

        return {
            "patient_id"   : f"p{i}",
            "signal"       : torch.randn((20, 6000)) * 50 - 20,
            "label"        : torch.full((1500,), 5, dtype=torch.uint8),
            "label_bitgt_1": torch.ones(1500, dtype=torch.uint8),
            "label_bitgt_2": torch.ones(1500, dtype=torch.uint8),
            "label_name"   : '5_middle'
        }


    # BASIC FUNCTIONAL TESTS

    def test_forward(self):
        x = torch.randn(self.batch_size, self.seq_len, self.in_channel)

        with torch.no_grad():
            output, hidden = self.model(x)

        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        self.assertEqual(len(hidden), 2)

    def test_hidden_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.in_channel)

        _, (h, c) = self.model(x)

        self.assertEqual(h.shape, (1, self.batch_size, 256))
        self.assertEqual(c.shape, (1, self.batch_size, 256))

    # GRADIENT TEST

    def test_backward(self):
        x = torch.randn(self.batch_size, self.seq_len, self.in_channel)
        target = torch.randint(0, self.output_dim, (self.batch_size,))

        output, _ = self.model(x)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, target)
        loss.backward()

        has_grad = any(
            p.grad is not None for p in self.model.parameters() if p.requires_grad
        )

        self.assertTrue(has_grad)

    # MINI TRAINING STEP

    def test_training_step(self):
        x = torch.randn(self.batch_size, self.seq_len, self.in_channel)
        target = torch.randint(0, self.output_dim, (self.batch_size,))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        self.model.train()

        output1, _ = self.model(x)
        loss1 = loss_fn(output1, target)

        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        output2, _ = self.model(x)
        loss2 = loss_fn(output2, target)

        # Loss should change after update (not necessarily decrease, but different)
        self.assertNotEqual(loss1.item(), loss2.item())

    # NUMERICAL STABILITY

    def test_output_is_finite(self):
        x = torch.randn(self.batch_size, self.seq_len, self.in_channel)

        output, _ = self.model(x)

        self.assertTrue(torch.isfinite(output).all())

    # CONSISTENCY TEST

    def test_deterministic_forward(self):
        x = torch.randn(self.batch_size, self.seq_len, self.in_channel)

        torch.manual_seed(0)
        out1, _ = self.model(x)

        torch.manual_seed(0)
        out2, _ = self.model(x)

        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()