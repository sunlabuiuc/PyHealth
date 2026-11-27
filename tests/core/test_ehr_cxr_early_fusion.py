import unittest

import torch
from torch import nn

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import EHRCXREarlyFusion

HIDDEN = 8
EHR_DIM = 6


def _dataset(n: int = 4):
    generator = torch.Generator().manual_seed(7)
    samples = [
        {
            "patient_id": f"p{i}",
            "visit_id": f"v{i}",
            "ehr": torch.randn(5, EHR_DIM, generator=generator).tolist(),
            "cxr": torch.randn(3, 32, 32, generator=generator).tolist(),
            "label": i % 2,
        }
        for i in range(n)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"ehr": "tensor", "cxr": "tensor"},
        output_schema={"label": "binary"},
        dataset_name="early_fusion_test",
    )


def _model(dataset, **kwargs):
    options = {
        "ehr_hidden_dim": HIDDEN,
        "ehr_num_layers": 2,
        "cxr_backbone": "resnet18",
    }
    options.update(kwargs)
    return EHRCXREarlyFusion(dataset, **options)


class TestEHRCXREarlyFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = _dataset()
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=4)))

    def test_forward_outputs(self) -> None:
        output = _model(self.dataset)(**self.batch)
        self.assertEqual(set(output), {"logit", "y_prob", "loss", "y_true"})
        self.assertEqual(output["logit"].shape, (4, 1))
        self.assertTrue(((output["y_prob"] >= 0) & (output["y_prob"] <= 1)).all())
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_forward_without_label_has_no_loss(self) -> None:
        batch = {k: v for k, v in self.batch.items() if k != "label"}
        output = _model(self.dataset)(**batch)
        self.assertNotIn("loss", output)

    def test_trainable_parameters_when_frozen(self) -> None:
        model = _model(self.dataset)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        projection = 512 * HIDDEN + HIDDEN
        classifier = 2 * HIDDEN + 1
        self.assertEqual(trainable, projection + classifier)

    def test_frozen_encoders_get_no_gradient(self) -> None:
        model = _model(self.dataset)
        model.train()
        model(**self.batch)["loss"].backward()
        for encoder in (model.ehr_encoder, model.cxr_encoder):
            self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        self.assertIsNotNone(model.projection.weight.grad)
        self.assertIsNotNone(model.classifier.weight.grad)

    def test_frozen_encoders_stay_in_eval_mode(self) -> None:
        model = _model(self.dataset)
        model.train()
        self.assertTrue(model.training)
        self.assertFalse(model.ehr_encoder.training)
        self.assertFalse(model.cxr_encoder.training)

    def test_frozen_batchnorm_statistics_do_not_change(self) -> None:
        model = _model(self.dataset)
        before = model.cxr_encoder.bn1.running_mean.clone()
        model.train()
        model(**self.batch)
        self.assertTrue(torch.equal(before, model.cxr_encoder.bn1.running_mean))

    def test_fresh_model_keeps_frozen_encoders_in_eval_mode(self) -> None:
        # No train() or eval() call before the forward passes.
        model = _model(self.dataset)
        self.assertTrue(model.training)
        self.assertFalse(model.ehr_encoder.training)
        self.assertFalse(model.cxr_encoder.training)
        before = model.cxr_encoder.bn1.running_mean.clone()
        first = model(**self.batch)["logit"]
        second = model(**self.batch)["logit"]
        self.assertTrue(torch.equal(before, model.cxr_encoder.bn1.running_mean))
        # The 2-layer LSTM's dropout would make the two passes differ.
        self.assertTrue(torch.equal(first, second))

    def test_unfrozen_encoders_are_trained(self) -> None:
        model = _model(self.dataset, freeze_encoders=False)
        model.train()
        self.assertTrue(model.cxr_encoder.training)
        model(**self.batch)["loss"].backward()
        self.assertIsNotNone(model.ehr_encoder.weight_ih_l0.grad)
        self.assertIsNotNone(model.cxr_encoder.conv1.weight.grad)

    def test_state_dicts_are_loaded(self) -> None:
        source = _model(self.dataset, freeze_encoders=False)
        target = _model(
            self.dataset,
            ehr_state_dict=source.ehr_encoder.state_dict(),
            cxr_state_dict=source.cxr_encoder.state_dict(),
        )
        for name, value in source.ehr_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, target.ehr_encoder.state_dict()[name]))
        for name, value in source.cxr_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, target.cxr_encoder.state_dict()[name]))

    def test_mismatched_state_dict_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            _model(self.dataset, cxr_state_dict={"conv1.weight": torch.zeros(1)})

    def test_ehr_only_model(self) -> None:
        with self.assertRaises(ValueError):
            _model(self.dataset, cxr_feature_key=None)
        model = _model(self.dataset, cxr_feature_key=None, freeze_encoders=False)
        batch = {k: v for k, v in self.batch.items() if k != "cxr"}
        output = model(**batch)
        self.assertEqual(output["logit"].shape, (4, 1))
        self.assertEqual(model.classifier.in_features, HIDDEN)

    def test_module_backbone(self) -> None:
        backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        model = _model(self.dataset, cxr_backbone=backbone, cxr_feature_dim=10)
        self.assertEqual(model(**self.batch)["logit"].shape, (4, 1))
        with self.assertRaises(ValueError):
            _model(self.dataset, cxr_backbone=backbone)
        with self.assertRaises(ValueError):
            _model(self.dataset, cxr_backbone="resnet18", cxr_feature_dim=10)
        with self.assertRaises(ValueError):
            _model(
                self.dataset,
                cxr_backbone=nn.Identity(),
                cxr_feature_dim=10,
                augment_cxr=True,
            )

    def test_augmentation_only_in_training_mode(self) -> None:
        model = _model(self.dataset, augment_cxr=True)
        model.train()
        self.assertEqual(model(**self.batch)["logit"].shape, (4, 1))
        model.eval()
        first = model(**self.batch)["logit"]
        second = model(**self.batch)["logit"]
        self.assertTrue(torch.equal(first, second))

    def test_images_added_to_the_batch_by_the_training_loop(self) -> None:
        dataset = create_sample_dataset(
            samples=[
                {
                    "patient_id": f"p{i}",
                    "ehr": torch.randn(5, EHR_DIM).tolist(),
                    "label": i % 2,
                }
                for i in range(4)
            ],
            input_schema={"ehr": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="early_fusion_no_image_input",
        )
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        batch["cxr"] = torch.randn(4, 3, 32, 32)
        self.assertEqual(_model(dataset)(**batch)["logit"].shape, (4, 1))

    def test_unknown_backbone_and_keys_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _model(self.dataset, cxr_backbone="vgg")
        with self.assertRaises(ValueError):
            _model(self.dataset, ehr_feature_key="missing")


if __name__ == "__main__":
    unittest.main()
