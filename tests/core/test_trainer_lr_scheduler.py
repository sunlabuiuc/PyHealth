import unittest

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MLP
from pyhealth.trainer import Trainer


class TestTrainerLRScheduler(unittest.TestCase):
    def _tiny_dataset(self):
        samples = [
            {
                "patient_id": f"p{i}",
                "visit_id": "v0",
                "feature": [float(i)],
                "label": i % 2,
            }
            for i in range(8)
        ]
        return create_sample_dataset(
            samples,
            {"feature": "tensor"},
            {"label": "binary"},
            dataset_name="test_trainer_sched",
        )

    def test_default_optimizer_unchanged(self):
        dataset = self._tiny_dataset()
        model = MLP(dataset=dataset, hidden_dim=4)
        trainer = Trainer(model=model, device="cpu")
        loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        trainer.train(train_dataloader=loader, epochs=1, steps_per_epoch=2)

    def test_scheduler_steps_per_iteration(self):
        dataset = self._tiny_dataset()
        model = MLP(dataset=dataset, hidden_dim=4)
        final_lrs: list[float] = []

        def factory(optimizer, total_steps):
            self.assertEqual(total_steps, 4)
            sched = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

            class Wrapper:
                def step(self_inner):
                    sched.step()
                    final_lrs.append(sched.get_last_lr()[0])

            return Wrapper()

        trainer = Trainer(model=model, device="cpu")
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        trainer.train(
            train_dataloader=loader,
            epochs=2,
            steps_per_epoch=2,
            optimizer_class=AdamW,
            optimizer_params={"lr": 1e-2},
            lr_scheduler=factory,
        )
        self.assertEqual(len(final_lrs), 4)
        self.assertLess(final_lrs[-1], final_lrs[0])

    def test_custom_param_groups_used(self):
        dataset = self._tiny_dataset()
        model = MLP(dataset=dataset, hidden_dim=4)
        groups = [
            {
                "params": list(model.parameters()),
                "lr": 2e-3,
                "weight_decay": 0.05,
            }
        ]
        trainer = Trainer(model=model, device="cpu")
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        trainer.train(
            train_dataloader=loader,
            epochs=1,
            steps_per_epoch=1,
            optimizer_class=AdamW,
            optimizer_params={"lr": 1e-3},
            param_groups=groups,
        )


if __name__ == "__main__":
    unittest.main()
