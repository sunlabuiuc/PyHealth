import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DiPro
from pyhealth.models.dipro import (
    _EHRIntervalEncoder,
    _GroupEmbedding,
    _OrderAwareProjection,
)

DISEASES = ("a", "b")
STATUSES = ("worsened", "improved", "no change")
LABELS = [f"{disease}: {status}" for disease in DISEASES for status in STATUSES]
SIZE = 16
REGIONS = 3
EHR_DIM = 4
BOX = [0, 0, 8, 8]


def _ehr(rows, seed):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(rows, EHR_DIM, generator=generator) + 0.1).tolist()


def _sample(index, path, regions=False, demographics=False):
    sample = {
        "patient_id": f"p{index}",
        "ehr": _ehr(10 + index, index),
        "first_ehr_hour": 1,
        "cxr": ([path, path], [1.5, 9.5]),
        "progression": ["a: worsened", "b: no change"],
        "region_progression": [[0, 3], [1, 2], [3, 3]],
    }
    if regions:
        sample["cxr"] = (*sample["cxr"], [[BOX] * REGIONS] * 2)
    if demographics:
        sample["demographics"] = [float(index), 1.0]
    return sample


def _dataset(path, regions=False, demographics=False, n=3):
    cxr = ("time_image", {"image_size": SIZE, "mode": "RGB"})
    if regions:
        cxr = ("time_region_image", {"num_regions": REGIONS, "image_size": SIZE})
    schema = {"ehr": "tensor", "cxr": cxr}
    if demographics:
        schema["demographics"] = "tensor"
    return create_sample_dataset(
        samples=[_sample(i, path, regions, demographics) for i in range(n)],
        input_schema=schema,
        output_schema={"progression": ("multilabel", {"labels": LABELS})},
        in_memory=True,
    )


def _model(dataset, **kwargs):
    torch.manual_seed(0)
    return DiPro(
        dataset,
        num_diseases=len(DISEASES),
        d_model=16,
        group_num=4,
        backbone="resnet18",
        pretrained=False,
        **kwargs,
    )


def _batch(dataset, size=3):
    return next(iter(get_dataloader(dataset, batch_size=size)))


class DiProCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.path = str(Path(cls.folder.name) / "cxr.png")
        pixels = np.random.default_rng(0).integers(0, 255, (24, 24), dtype=np.uint8)
        Image.fromarray(pixels).save(cls.path)
        cls.images = _dataset(cls.path)
        cls.regions = _dataset(cls.path, regions=True, demographics=True)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()


class TestForward(DiProCase):
    def test_whole_images(self):
        output = _model(self.images)(**_batch(self.images))
        self.assertEqual(output["logit"].shape, (3, 6))
        self.assertEqual(output["y_true"].shape, (3, 6))
        self.assertTrue(output["loss"].isfinite())

    def test_probabilities_sum_to_one_per_disease(self):
        output = _model(self.images)(**_batch(self.images))
        totals = output["y_prob"].view(3, 2, 3).sum(dim=-1)
        self.assertTrue(torch.allclose(totals, torch.ones(3, 2)))

    def test_regions_and_demographics(self):
        model = _model(self.regions)
        self.assertEqual(model.num_regions, REGIONS)
        output = model(**_batch(self.regions))
        self.assertEqual(output["logit"].shape, (3, 6))
        self.assertGreater(output["dynamic_order_loss"], 0)

    def test_no_labels(self):
        batch = _batch(self.images)
        del batch["progression"]
        output = _model(self.images)(**batch)
        self.assertEqual(set(output), {"logit", "y_prob"})

    def test_labels_do_not_change_predictions(self):
        model = _model(self.regions).eval()
        batch = _batch(self.regions)
        with torch.no_grad():
            with_labels = model(**batch)["logit"]
            batch.pop("progression")
            batch.pop("region_progression")
            without_labels = model(**batch)["logit"]
        self.assertTrue(torch.equal(with_labels, without_labels))

    def test_single_sample_with_one_region_trains(self):
        # BatchNorm cannot train on the one row of this batch.
        model = _model(self.images).train()
        batch = {
            key: value[:1] for key, value in _batch(self.images).items()
            if key in ("ehr", "cxr", "first_ehr_hour", "progression")
        }
        self.assertTrue(model(**batch)["loss"].isfinite())

    def test_three_images_use_the_consistency_loss(self):
        model = _model(self.images)
        batch = _batch(self.images)
        batch["cxr"] = [
            (torch.rand(3, 3, SIZE, SIZE), torch.tensor([1.5, 5.5, 9.5]), "image")
            for _ in range(3)
        ]
        output = model(**batch)
        self.assertGreater(output["consistency_loss"], 0)

    def test_rejects_wrong_setup(self):
        with self.assertRaisesRegex(ValueError, "labels"):
            DiPro(self.images, num_diseases=3, pretrained=False)
        with self.assertRaisesRegex(ValueError, "backbone"):
            DiPro(self.images, num_diseases=2, backbone="vgg", pretrained=False)
        with self.assertRaisesRegex(ValueError, "multiple"):
            DiPro(self.images, num_diseases=2, d_model=12, pretrained=False)


class TestGradients(DiProCase):
    def test_every_parameter_gets_a_gradient(self):
        model = _model(self.regions, num_transformer_layers=2)
        model(**_batch(self.regions))["loss"].backward()
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and parameter.numel() > 0
            and parameter.grad is None
        ]
        self.assertEqual(missing, [])

    def test_frozen_backbone(self):
        model = _model(self.images, freeze_backbone=True)
        model(**_batch(self.images))["loss"].backward()
        self.assertTrue(
            all(p.grad is None for p in model.cxr_encoder.resnet.parameters())
        )

    def test_padding_learns_only_from_missing_images(self):
        model = _model(self.images)
        batch = _batch(self.images)
        batch["cxr"][0] = None
        model(**batch)["loss"].backward()
        self.assertGreater(model.region_padding.grad.abs().sum(), 0)


class TestSaveLoad(DiProCase):
    def test_round_trip_keeps_predictions_and_weights(self):
        model = _model(self.images).eval()
        model.set_class_weights(self.images)
        loaded = _model(self.images).eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(model.state_dict(), path)
            loaded.load_state_dict(torch.load(path))
        batch = _batch(self.images)
        with torch.no_grad():
            self.assertTrue(
                torch.equal(model(**batch)["logit"], loaded(**batch)["logit"])
            )
        self.assertTrue(torch.equal(model.class_weights, loaded.class_weights))


class TestMissingModalities(DiProCase):
    def _finite(self, model, batch):
        output = model(**batch)
        self.assertTrue(output["logit"].isfinite().all())
        return output

    def test_sample_without_images(self):
        batch = _batch(self.images)
        batch["cxr"][1] = None
        self._finite(_model(self.images), batch)

    def test_sample_with_one_image(self):
        batch = _batch(self.images)
        images, hours, tag = batch["cxr"][1]
        batch["cxr"][1] = (images[:1], hours[:1], tag)
        self._finite(_model(self.images), batch)

    def test_no_images_at_all(self):
        batch = _batch(self.images)
        del batch["cxr"]
        self._finite(_model(self.images), batch)

    def test_sample_without_ehr(self):
        batch = _batch(self.images)
        batch["ehr"][1] = 0
        self._finite(_model(self.images), batch)

    def test_no_ehr_at_all(self):
        batch = _batch(self.images)
        del batch["ehr"], batch["first_ehr_hour"]
        self._finite(_model(self.images), batch)

    def test_no_ehr_inside_the_interval(self):
        batch = _batch(self.images)
        batch["first_ehr_hour"] = [100, 100, 100]
        self._finite(_model(self.images), batch)

    def _same_logits(self, alone, batched):
        model = _model(self.images).eval()
        with torch.no_grad():
            expected = model(**alone)["logit"]
            got = model(**batched)["logit"]
        self.assertTrue(torch.isfinite(got).all())
        self.assertTrue(torch.allclose(expected, got[:1], atol=1e-5))

    def test_no_ehr_inside_the_interval_is_batch_independent(self):
        # Sample 0 (10 rows) has no row inside its interval; sample 2 has 12.
        batch = _batch(self.images)
        length = int((batch["ehr"][0] != 0).any(-1).sum())
        alone = {
            "ehr": batch["ehr"][:1, :length],
            "cxr": batch["cxr"][:1],
            "first_ehr_hour": [100],
        }
        batched = {
            "ehr": batch["ehr"][[0, 2]],
            "cxr": [batch["cxr"][0], batch["cxr"][2]],
            "first_ehr_hour": [100, 1],
        }
        self.assertLess(length, batched["ehr"].size(1))
        self._same_logits(alone, batched)

    def test_sample_without_ehr_is_batch_independent(self):
        batch = _batch(self.images)
        empty = torch.zeros_like(batch["ehr"][:1, :3])
        alone = {"ehr": empty, "cxr": batch["cxr"][:1], "first_ehr_hour": [1]}
        batched = {
            "ehr": torch.cat([F.pad(empty, (0, 0, 0, batch["ehr"].size(1) - 3)),
                              batch["ehr"][2:]]),
            "cxr": [batch["cxr"][0], batch["cxr"][2]],
            "first_ehr_hour": [1, 1],
        }
        self._same_logits(alone, batched)
        # Leaving ``ehr`` out of the batch is the same as all-zero rows.
        self._same_logits({"cxr": batch["cxr"][:1]}, batched)

    def test_nothing_to_predict_from(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            _model(self.images)()

    def test_missing_demographics_input(self):
        batch = _batch(self.regions)
        del batch["demographics"]
        self._finite(_model(self.regions), batch)

    def test_missing_region_labels_skip_their_loss(self):
        batch = _batch(self.regions)
        del batch["region_progression"]
        self.assertEqual(_model(self.regions)(**batch)["dynamic_order_loss"], 0)

    def test_padding_does_not_change_a_sample(self):
        model = _model(self.images).eval()
        batch = _batch(self.images)
        length = int((batch["ehr"][0] != 0).any(-1).sum())
        alone = {
            "ehr": batch["ehr"][:1, :length],
            "cxr": batch["cxr"][:1],
            "first_ehr_hour": batch["first_ehr_hour"][:1],
        }
        padded = {**alone, "ehr": batch["ehr"][:1]}
        self.assertLess(length, batch["ehr"].size(1))
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(
                    model(**alone)["logit"], model(**padded)["logit"], atol=1e-5
                )
            )

    def _longer(self, entry):
        """The same sample with a third, different, image 10 hours later."""
        images, hours, *rest = entry
        generator = torch.Generator().manual_seed(1)
        extra = torch.rand(images[:1].shape, generator=generator)
        return (
            torch.cat([images, extra]),
            torch.cat([hours, hours[-1:] + 10]),
            *rest,
        )

    def _assert_image_padding_is_ignored(self, dataset, keys):
        model = _model(dataset).eval()
        batch = _batch(dataset)
        alone = {key: batch[key][:1] for key in keys}
        mixed = {key: batch[key][:2] for key in keys}
        mixed["cxr"] = [batch["cxr"][0], self._longer(batch["cxr"][1])]
        with torch.no_grad():
            expected = model(**alone)["logit"]
            got = model(**mixed)["logit"][:1]
        self.assertTrue(torch.allclose(expected, got, atol=1e-5))

    def test_image_padding_does_not_change_a_sample(self):
        self._assert_image_padding_is_ignored(
            self.images, ("ehr", "cxr", "first_ehr_hour")
        )

    def test_image_padding_does_not_change_a_sample_with_regions(self):
        self._assert_image_padding_is_ignored(
            self.regions, ("ehr", "cxr", "first_ehr_hour", "demographics")
        )

    def test_sample_without_images_is_finite_and_batch_independent(self):
        model = _model(self.images).eval()
        batch = _batch(self.images)
        keys = ("ehr", "first_ehr_hour")
        alone = {**{key: batch[key][:1] for key in keys}, "cxr": [None]}
        mixed = {
            **{key: batch[key][:2] for key in keys},
            "cxr": [None, self._longer(batch["cxr"][1])],
        }
        with torch.no_grad():
            expected = model(**alone)["logit"]
            got = model(**mixed)["logit"]
        self.assertTrue(torch.isfinite(got).all())
        self.assertTrue(torch.allclose(expected, got[:1], atol=1e-5))

    def test_padded_pairs_do_not_enter_the_losses(self):
        model = _model(self.images).eval()
        batch = _batch(self.images, size=1)
        longer = {**batch, "cxr": [self._longer(batch["cxr"][0])]}
        image_mask = torch.tensor([[True, True, False]])
        regions, _, _ = model._stack_cxr(longer["cxr"], 1)
        with torch.no_grad():
            masked = model.cxr_encoder(regions, image_mask[:, 1:])
            two = model.cxr_encoder(regions[:, :2])
        for loss in ("orthogonal", "static_order", "consistency"):
            self.assertTrue(torch.allclose(masked[loss], two[loss], atol=1e-6), loss)

    def test_padding_inside_the_interval_does_not_change_a_sample(self):
        # A short series: the padded rows' hours fall inside [1.5, 9.5].
        model = _model(self.images).eval()
        batch = _batch(self.images)
        alone = {
            "ehr": batch["ehr"][:1, :4],
            "cxr": batch["cxr"][:1],
            "first_ehr_hour": batch["first_ehr_hour"][:1],
        }
        padded = {**alone, "ehr": F.pad(alone["ehr"], (0, 0, 0, 4))}
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(
                    model(**alone)["logit"], model(**padded)["logit"], atol=1e-5
                )
            )

    def test_zero_row_before_real_rows_is_real(self):
        ehr = torch.ones(1, 3, EHR_DIM)
        ehr[0, 1] = 0
        _, mask, times = _model(self.images)._ehr_inputs(ehr, [4], 1)
        self.assertEqual(mask.tolist(), [[True, True, True]])
        self.assertEqual(times.tolist(), [[4.0, 5.0, 6.0]])


class TestLosses(DiProCase):
    def _batch(self, labels):
        batch = _batch(self.images)
        batch["progression"] = torch.tensor(labels, dtype=torch.float32)
        return batch

    def test_total_uses_the_weights(self):
        output = _model(self.images)(**_batch(self.images))
        total = (
            2.0 * output["prediction_loss"]
            + 1.0 * output["orthogonal_loss"]
            + 0.5 * output["static_order_loss"]
            + 2.0 * output["dynamic_order_loss"]
        )
        self.assertAlmostEqual(output["loss"].item(), total.item(), places=4)

    def test_paper_defaults(self):
        model = _model(self.images)
        self.assertEqual(
            (
                model.lambda_pred,
                model.lambda_orthogonal,
                model.lambda_consistency,
                model.lambda_dynamic_order,
            ),
            (2.0, 1.0, 0.0, 2.0),
        )
        self.assertEqual(model.lambda_static_order, 0.5)

    def test_not_mentioned_is_ignored(self):
        output = _model(self.images)(**self._batch([[0] * 6] * 3))
        self.assertEqual(output["prediction_loss"], 0)
        self.assertEqual(output["dynamic_order_loss"], 0)

    def test_prediction_loss_sums_the_diseases(self):
        model = _model(self.images)
        logits = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        target = torch.tensor([[0, 1]])
        expected = F.cross_entropy(logits[:, 0], target[:, 0]) + F.cross_entropy(
            logits[:, 1], target[:, 1]
        )
        self.assertAlmostEqual(
            model._disease_loss(logits, target).item(), expected.item(), places=5
        )
        masked = torch.tensor([[0, 3]])
        self.assertAlmostEqual(
            model._disease_loss(logits, masked).item(),
            F.cross_entropy(logits[:, 0], target[:, 0]).item(),
            places=5,
        )

    def test_reversal_swaps_worsened_and_improved(self):
        model = _model(self.regions)
        # Logits that favor each region's own class, in both directions.
        labels = torch.tensor([[[0, 1], [1, 2], [2, 3]]])
        forward = F.one_hot(labels.clamp(max=2), 3).float()[:, None] * 9
        swapped = torch.where(labels < 2, 1 - labels, labels)
        backward = F.one_hot(swapped.clamp(max=2), 3).float()[:, None] * 9
        encoded = {"forward_logits": forward, "reversed_logits": backward}
        target = torch.zeros(1, 2, dtype=torch.long)
        mask = torch.ones(1, 1, dtype=torch.bool)
        right = model._region_loss(encoded, target, labels, mask)
        encoded["reversed_logits"] = forward
        wrong = model._region_loss(encoded, target, labels, mask)
        self.assertLess(right, 0.01)
        self.assertGreater(wrong, 1)

    def test_missing_pair_has_no_region_labels(self):
        model = _model(self.regions)
        logits = torch.zeros(1, 1, REGIONS, 2, 3)
        encoded = {"forward_logits": logits, "reversed_logits": logits}
        labels = torch.zeros(1, REGIONS, 2, dtype=torch.long)
        loss = model._region_loss(
            encoded,
            torch.zeros(1, 2, dtype=torch.long),
            labels,
            torch.zeros(1, 1, dtype=torch.bool),
        )
        self.assertEqual(loss, 0)

    def test_class_weights_formula(self):
        model = _model(self.images)
        worsened, no_change = [1, 0, 0], [0, 0, 1]
        model.set_class_weights(
            [
                {"progression": worsened + no_change},
                {"progression": worsened + [0, 0, 0]},
                {"progression": no_change + no_change},
                {"progression": worsened + no_change},
            ]
        )
        # a: 3 worsened, 1 no change, of 4. b: 3 no change of 3.
        expected = [[4 / 9, 1.0, 4 / 3], [1.0, 1.0, 1 / 3]]
        self.assertTrue(
            torch.allclose(model.class_weights, torch.tensor(expected))
        )

    def test_class_weights_need_samples(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            _model(self.images).set_class_weights([])


class TestComponents(unittest.TestCase):
    def _layouts(self, concat):
        projection = _OrderAwareProjection(2, 0.0, concat)
        seen = []
        projection._project = lambda pairs: seen.append(pairs) or pairs
        projection(torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]]))
        return seen[1].flatten().tolist()

    def test_reversed_pair_of_the_authors_code_interleaves(self):
        self.assertEqual(self._layouts(False), [3.0, 1.0, 4.0, 2.0])

    def test_reversed_pair_can_swap_the_two_images(self):
        self.assertEqual(self._layouts(True), [3.0, 4.0, 1.0, 2.0])

    def test_interval_attention_ignores_rows_outside(self):
        torch.manual_seed(0)
        encoder = _EHRIntervalEncoder(16).eval()
        ehr = torch.randn(1, 6, 16)
        times = torch.arange(6.0)[None]
        mask = torch.ones(1, 6, dtype=torch.bool)
        intervals = torch.tensor([[[1.0, 3.0]]])
        output = encoder(ehr, times, mask, intervals)

        changed = ehr.clone()
        changed[0, [0, 4, 5]] = 99.0  # before and after the interval
        same = encoder(changed, times, mask, intervals)
        self.assertTrue(torch.allclose(output, same))

        changed = ehr.clone()
        changed[0, 2] = 99.0  # inside
        self.assertFalse(
            torch.allclose(output, encoder(changed, times, mask, intervals))
        )

    def test_interval_attention_ignores_padding_inside_the_interval(self):
        torch.manual_seed(0)
        encoder = _EHRIntervalEncoder(16).eval()
        ehr = torch.randn(1, 6, 16)
        times = torch.arange(6.0)[None]
        mask = torch.tensor([[True, True, True, False, False, False]])
        intervals = torch.tensor([[[1.0, 5.0]]])  # rows 3 to 5 are inside it
        output = encoder(ehr, times, mask, intervals)

        changed = ehr.clone()
        changed[0, 3:] = 99.0
        same = encoder(changed, times, mask, intervals)
        self.assertTrue(torch.allclose(output, same))

    def test_interval_attention_shape_for_two_pairs(self):
        torch.manual_seed(0)
        output = _EHRIntervalEncoder(16)(
            torch.randn(2, 5, 16),
            torch.arange(5.0).repeat(2, 1),
            torch.ones(2, 5, dtype=torch.bool),
            torch.tensor([[[0.0, 2.0], [2.0, 4.0]]] * 2),
        )
        self.assertEqual(output.shape, (2, 2, 5, 16))

    def test_group_embedding_lies_in_the_span_of_the_groups(self):
        torch.manual_seed(0)
        embedding = _GroupEmbedding(8, 3)
        output = embedding(torch.randn(2, 4, 8))
        self.assertEqual(output.shape, (2, 4, 8))
        self.assertLessEqual(torch.linalg.matrix_rank(output.flatten(0, 1)), 3)


if __name__ == "__main__":
    unittest.main()
