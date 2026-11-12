"""Small synthetic tests for MAS and Cox loss."""

import unittest
import torch
from pyhealth.models.mas import MAS, cox_time_varying_loss


class TestMAS(unittest.TestCase):
    def test_cox_loss_known_result(self):
        scores = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        intervals = torch.tensor([
            [0.0, 1.0, 1.0],  # 3 at risk: log(3)
            [0.0, 2.0, 1.0],  # 2 at risk: log(2)
            [0.0, 3.0, 0.0],
        ])
        loss = cox_time_varying_loss(scores, intervals)
        target = (torch.log(torch.tensor(3.0)) + torch.log(torch.tensor(2.0))) / 2
        self.assertTrue(torch.allclose(loss, target, atol=1e-6))
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_start_stop_risk_membership(self):
        scores = torch.zeros(3)
        intervals = torch.tensor([
            [0.0, 1.0, 1.0],
            [1.0, 2.0, 0.0],  # starts at first event time: excluded at t=1
            [0.0, 3.0, 0.0],
        ])
        loss = cox_time_varying_loss(scores, intervals)
        self.assertTrue(torch.allclose(loss, torch.log(torch.tensor(2.0))))

    def test_tied_events_use_breslow_denominator(self):
        scores = torch.tensor([0.0, 1.0, 2.0])
        intervals = torch.tensor(
            [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 2.0, 0.0]]
        )
        expected = torch.logsumexp(scores, dim=0) - scores[:2].mean()
        self.assertTrue(torch.allclose(cox_time_varying_loss(scores, intervals), expected))

    def test_invalid_interval(self):
        with self.assertRaises(ValueError):
            cox_time_varying_loss(torch.zeros(1), torch.tensor([[1., 1., 1.]]))

    def test_model_forward_backward(self):
        class DatasetStub:
            input_schema = {"features": "standardized_tensor"}
            output_schema = {"survival_interval": "survival_interval"}
            class FeatureProcessor:
                def size(self):
                    return 4
            input_processors = {"features": FeatureProcessor()}

        model = MAS(DatasetStub())
        x = torch.tensor([[0.2, 0.1, 0., 1.], [0.5, -0.2, 1., 0.], [0., 0.5, -1., 0.]])
        intervals = torch.tensor([[0., 1., 1.], [0., 2., 1.], [0., 3., 0.]])
        output = model(features=x, survival_interval=intervals)
        self.assertEqual(output["y_prob"].shape, (3, 1))
        self.assertEqual(output["y_true"].shape, (3, 3))
        output["loss"].backward()
        self.assertTrue(torch.isfinite(model.linear.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
