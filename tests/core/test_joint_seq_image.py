"""Tests for PhysioNet2012Mortality, the time-series images and JointSeqImage."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pyhealth.datasets import (
    PhysioNet2012Dataset,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.models import JointSeqImage
from pyhealth.models.joint_seq_image import (
    focal_loss,
    margin_contrastive,
    nt_xent,
    reverse_padded,
    time_lags,
)
from pyhealth.models.time_series_images import (
    IMAGE_TYPES,
    field_cells,
    grid_shape,
    line_graph_cells,
    render_images,
    resample,
    variable_colors,
)
from pyhealth.tasks import (
    PhysioNet2012Mortality,
    physionet2012_mortality_statistics,
    split_by_record_ids,
)
from pyhealth.trainer import Trainer

HEADER = "Time,Parameter,Value\n"
DESCRIPTORS = "00:00,Age,54\n00:00,Gender,0\n00:00,Height,-1\n00:00,ICUType,4\n00:00,Weight,80\n"


def _record(record_id, lines):
    return HEADER + f"00:00,RecordID,{record_id}\n" + DESCRIPTORS + "".join(lines)


def _write_fake_root(root: Path) -> None:
    folder = root / "set-a"
    folder.mkdir(parents=True)
    # 132539: HR twice at 00:10 (averaged), Temp at 01:05, HR after 48 h (dropped).
    (folder / "132539.txt").write_text(_record(132539, [
        "00:10,HR,70\n", "00:10,HR,80\n", "01:05,Temp,37\n", "01:05,HR,90\n",
        "47:59,HR,100\n", "48:00,HR,110\n", "02:00,Weight,81\n",
    ]))
    # 132540: one observation, died.
    (folder / "132540.txt").write_text(_record(132540, ["03:00,HR,60\n"]))
    # 132541: no time series (like the 12 records Raindrop removes).
    (folder / "132541.txt").write_text(_record(132541, []))
    (root / "Outcomes-a.txt").write_text(
        "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death\n"
        "132539,6,1,5,-1,0\n132540,16,8,8,3,1\n132541,10,3,4,-1,0\n"
    )


class TestPhysioNet2012Mortality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "p12"
        _write_fake_root(cls.root)
        cls.dataset = PhysioNet2012Dataset(
            root=str(cls.root), cache_dir=Path(cls.tmp.name) / "cache"
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _samples(self, task):
        return {s["patient_id"]: s for p in self.dataset.iter_patients() for s in task(p)}

    def test_irregular_times_values_and_labels(self):
        samples = self._samples(PhysioNet2012Mortality(variables=["HR", "Temp"], max_length=5))
        self.assertEqual(sorted(samples), ["132539", "132540"])  # 132541: no series
        s = samples["132539"]
        self.assertEqual(s["x"].shape, (5, 2))
        np.testing.assert_allclose(s["x_time"][:3], [10 / 60, 65 / 60, 47 + 59 / 60], rtol=1e-6)
        self.assertTrue(np.isnan(s["x_time"][3:]).all())  # padding; 48:00 dropped
        np.testing.assert_array_equal(
            s["x"][:3], np.array([[75, np.nan], [90, 37], [100, np.nan]], dtype="f4")
        )
        self.assertTrue(np.isnan(s["x"][3:]).all())
        self.assertEqual(s["mortality"], 0)
        self.assertEqual(samples["132540"]["mortality"], 1)

    def test_truncates_to_max_length(self):
        (s,) = PhysioNet2012Mortality(variables=["HR", "Temp"], max_length=2)(
            self.dataset.get_patient("132539")
        )
        np.testing.assert_allclose(s["x_time"], [10 / 60, 65 / 60], rtol=1e-6)
        self.assertEqual(np.isfinite(s["x"]).sum(), 3)

    def test_normalization_and_outliers(self):
        mean, std = {"HR": 80.0, "Temp": 37.0}, {"HR": 5.0, "Temp": 1.0}
        (s,) = PhysioNet2012Mortality(mean, std, variables=["HR", "Temp"])(
            self.dataset.get_patient("132539")
        )
        np.testing.assert_allclose(s["x"][:3, 0], [-1, 2, 4])
        (s,) = PhysioNet2012Mortality(mean, std, variables=["HR", "Temp"], outlier_sd=3)(
            self.dataset.get_patient("132539")
        )
        self.assertTrue(np.isnan(s["x"][2, 0]))  # +4 SD dropped
        # Every value of 132540 is an outlier, so it gives no sample.
        self.assertEqual(
            PhysioNet2012Mortality({"HR": 0.0}, {"HR": 1.0}, ["HR"], outlier_sd=3)(
                self.dataset.get_patient("132540")
            ),
            [],
        )
        with self.assertRaises(ValueError):
            PhysioNet2012Mortality(mean=mean)

    def test_statistics(self):
        mean, std = physionet2012_mortality_statistics(self.dataset, variables=["HR"])
        observed = [75, 90, 100, 60]  # per minute; 48:00 excluded
        self.assertAlmostEqual(mean["HR"], np.mean(observed), places=5)
        self.assertAlmostEqual(std["HR"], np.std(observed), places=5)
        only = physionet2012_mortality_statistics(self.dataset, ["132540"], ["HR"])
        self.assertEqual((only[0]["HR"], only[1]["HR"]), (60.0, 1.0))  # std 0 -> 1
        with self.assertRaises(ValueError):
            physionet2012_mortality_statistics(self.dataset, variables=["Lactate"])

    def test_set_task_and_split_by_record_ids(self):
        samples = self.dataset.set_task(PhysioNet2012Mortality(variables=["HR", "Temp"]))
        self.assertEqual(len(samples), 2)
        train, val, test = split_by_record_ids(samples, ["132540", "999"], [], ["132539"])
        self.assertEqual((len(train), len(val), len(test)), (1, 0, 1))
        self.assertEqual(test[0]["patient_id"], "132539")
        with self.assertRaises(ValueError):
            split_by_record_ids(samples, ["132539"], ["132539"], [])


class TestTimeSeriesImages(unittest.TestCase):
    def test_grid_and_colors(self):
        self.assertEqual(grid_shape(36), (6, 6))
        self.assertEqual(grid_shape(1), (1, 1))
        colors = variable_colors(30)
        self.assertTrue(torch.all(colors[12] < colors[0] + 1e-6))  # second cycle darker

    def test_line_graph_draws_points_and_segments(self):
        values = torch.tensor([[[1.0], [0.0], [-1.0]]])
        times = torch.tensor([[0.0, 1.0, 2.0]])
        mask = torch.tensor([[[True], [False], [True]]])
        cell = line_graph_cells(
            values, times, mask, 9, (0.0, 2.0), (-1.0, 1.0),
            torch.zeros(1, 3), marker_radius=0,
        )[0, 0, 0]
        # A black diagonal from the top left to the bottom right, nothing else.
        np.testing.assert_array_equal(cell.numpy(), 1 - np.eye(9))

    def test_unobserved_and_padding_are_not_drawn(self):
        values = torch.randn(1, 4, 2)
        times = torch.tensor([[0.0, 1.0, float("nan"), float("nan")]])
        mask = torch.tensor([[[True, False], [True, False], [True, False], [True, False]]])
        image = render_images(values, times, mask, cell_size=8, time_range=(0, 1))
        self.assertEqual(image.shape, (1, 3, 8, 16))
        self.assertTrue(torch.all(image[..., 8:] == 1))  # variable 2 never observed
        self.assertTrue(torch.any(image[..., :8] < 1))

    def test_resample_interpolates_and_holds(self):
        values = torch.tensor([[[2.0], [0.0], [4.0]]])
        times = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[[True], [False], [True]]])
        series, observed = resample(values, times, mask, 5, (0.0, 4.0))
        np.testing.assert_allclose(series[0, 0].numpy(), [2, 2, 3, 4, 4])
        self.assertTrue(observed.all())
        series, observed = resample(values, times, torch.zeros_like(mask), 5, (0.0, 4.0))
        self.assertFalse(observed.any())
        self.assertTrue(torch.all(series == 0))

    def test_fields(self):
        x = torch.tensor([[[-1.0, 0.0, 1.0]]])
        gasf = field_cells(x, "gasf", (-1, 1))[0, 0, 0]
        phi = torch.arccos(x[0, 0])
        np.testing.assert_allclose(gasf, (torch.cos(phi[:, None] + phi[None]) + 1) / 2, atol=1e-6)
        gadf = field_cells(x, "gadf", (-1, 1))[0, 0, 0]
        np.testing.assert_allclose(gadf, (torch.sin(phi[:, None] - phi[None]) + 1) / 2, atol=1e-6)
        rp = field_cells(x, "recurrence", (-1, 1))[0, 0, 0]
        np.testing.assert_allclose(rp, (x[0, 0, :, None] - x[0, 0, None]).abs() / 2)
        # Bins 0, 1, 1 (of 2): transitions 0->1 and 1->1, so W = [[0, 1], [0, 1]].
        mtf = field_cells(torch.tensor([[[-0.9, 0.1, 0.9]]]), "mtf", (-1, 1), num_bins=2)
        np.testing.assert_allclose(mtf[0, 0, 0], [[0, 1, 1], [0, 1, 1], [0, 1, 1]])
        with self.assertRaises(ValueError):
            field_cells(x, "nope", (-1, 1))

    def test_every_image_type(self):
        values = torch.randn(2, 6, 3)
        times = torch.linspace(0, 48, 6).expand(2, 6)
        mask = torch.rand(2, 6, 3) > 0.3
        for kind in IMAGE_TYPES:
            image = render_images(values, times, mask, kind, cell_size=8)
            self.assertEqual(image.shape, (2, 3, 16, 16), kind)
            self.assertTrue(torch.all((image >= 0) & (image <= 1)), kind)
        with self.assertRaises(ValueError):
            render_images(values, times, mask, "photo")


class TestJointSeqImageParts(unittest.TestCase):
    def test_time_lags(self):
        times = torch.tensor([[0.0, 1.0, 3.0, 4.0]])
        mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]])
        np.testing.assert_allclose(
            time_lags(times, mask)[0].numpy(), [[0, 0], [1, 1], [3, 3], [1, 1]]
        )

    def test_reverse_padded(self):
        x = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]).unsqueeze(-1)
        out = reverse_padded(x, torch.tensor([3, 2]))
        np.testing.assert_array_equal(out[..., 0], [[3, 2, 1, 0], [5, 4, 0, 0]])

    def test_nt_xent_matches_definition(self):
        torch.manual_seed(0)
        a, b = torch.randn(3, 4), torch.randn(3, 4)
        z = F.normalize(torch.cat([a, b]), dim=-1)
        sim = torch.exp(z @ z.T / 0.5)
        expected = np.mean([
            -torch.log(sim[i, (i + 3) % 6] / (sim[i].sum() - sim[i, i])).item()
            for i in range(6)
        ])
        self.assertAlmostEqual(nt_xent(a, b, 0.5).item(), expected, places=5)

    def test_margin_contrastive_matches_eq9(self):
        torch.manual_seed(0)
        s, v = torch.randn(4, 5), torch.randn(4, 5)
        labels = torch.tensor([0, 0, 1, 1])
        sim = F.normalize(v, dim=-1) @ F.normalize(s, dim=-1).T
        tau, m = 0.3, 0.2

        def term(scores, i):
            other = labels != labels[i]
            denominator = torch.exp((scores + m * other) / tau).sum()
            return torch.log(torch.exp(scores[i] / tau) / denominator)

        expected = -np.mean([
            (term(sim[i], i) + term(sim[:, i], i)).item() for i in range(4)
        ])
        self.assertAlmostEqual(
            margin_contrastive(s, v, labels, tau, m).item(), expected, places=5
        )
        # The margin makes pairs of other classes harder, so the loss grows.
        self.assertGreater(
            margin_contrastive(s, v, labels, tau, 0.5).item(),
            margin_contrastive(s, v, labels, tau, 0.0).item(),
        )

    def test_focal_loss(self):
        logits = torch.tensor([[0.5], [-1.0]])
        y = torch.tensor([[1.0], [0.0]])
        ce = F.binary_cross_entropy_with_logits(logits, y)
        self.assertAlmostEqual(focal_loss(logits, y, gamma=0).item(), ce.item(), places=6)
        p = torch.sigmoid(logits)
        p_true = torch.where(y == 1, p, 1 - p)
        expected = (-(1 - p_true) ** 2 * torch.log(p_true)).mean()
        self.assertAlmostEqual(focal_loss(logits, y).item(), expected.item(), places=6)
        multi = torch.randn(3, 4)
        target = torch.tensor([0, 3, 1])
        self.assertAlmostEqual(
            focal_loss(multi, target, 0, binary=False).item(),
            F.cross_entropy(multi, target).item(), places=6,
        )


TINY_SWIN = {"embed_dim": 8, "depths": [1, 1], "num_heads": [1, 1], "window_size": 4}


def _samples(n=8, length=6, n_vars=4, multiclass=False, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        x = rng.normal(size=(length, n_vars)).astype("float32")
        x[rng.random((length, n_vars)) < 0.5] = np.nan
        steps = rng.integers(3, length + 1)
        t = np.full(length, np.nan, dtype="float32")
        t[:steps] = np.sort(rng.uniform(0, 48, steps))
        x[steps:] = np.nan
        samples.append({
            "patient_id": f"p{i}", "x": x, "x_time": t,
            "label": i % 3 if multiclass else i % 2,
        })
    return samples


def _model(dataset, **kwargs):
    return JointSeqImage(
        dataset, image_size=32, image_model=None, swin_config=TINY_SWIN,
        hidden_dim=8, num_gru_layers=2, projection_dims=(16, 8), num_clusters=2,
        **kwargs,
    )


class TestJointSeqImage(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dataset = create_sample_dataset(
            _samples(), {"x": "tensor", "x_time": "tensor"}, {"label": "binary"}
        )
        self.model = _model(self.dataset)
        self.batch = next(iter(get_dataloader(self.dataset, batch_size=8)))

    def test_forward(self):
        out = self.model(**self.batch)
        self.assertEqual(out["y_prob"].shape, (8, 1))
        self.assertEqual(out["embed"].shape, (8, 16))
        self.assertTrue(torch.isfinite(out["loss"]))
        self.model.eval()
        with torch.no_grad():
            out = self.model(x=self.batch["x"], x_time=self.batch["x_time"])
        self.assertNotIn("loss", out)

    def test_imputation_keeps_observed_values(self):
        values, times, mask, valid = self.model._inputs(self.batch)
        imputed, forward_hat, backward_hat = self.model.impute(values, times, mask, valid)
        observed = mask.bool()
        torch.testing.assert_close(imputed[observed], values[observed])
        missing = ~observed & valid.unsqueeze(-1)
        torch.testing.assert_close(
            imputed[missing], ((forward_hat + backward_hat) / 2)[missing]
        )
        self.assertTrue(torch.all(imputed[~valid] == 0))
        self.assertEqual(values.shape[1], int(valid.sum(1).max()))  # padding trimmed

    def test_trimming_padding_changes_nothing(self):
        self.model.eval()
        padded = {
            "x": torch.cat([self.batch["x"], torch.full((8, 3, 4), float("nan"))], 1),
            "x_time": torch.cat([self.batch["x_time"], torch.full((8, 3), float("nan"))], 1),
        }
        with torch.no_grad():
            short = self.model.encode(**self.batch)
            values = self.model._inputs(padded)[0]
        self.assertEqual(values.shape[1], int(torch.isfinite(self.batch["x_time"]).sum(1).max()))
        with torch.no_grad():
            again = self.model.encode(**padded)
        torch.testing.assert_close(short, again)

    def test_backward_direction_sees_the_future(self):
        values, times, mask, valid = self.model._inputs(self.batch)
        later = values.clone()
        later[0, -1] += 10.0
        later_mask = mask.clone()
        later_mask[0, -1] = 1.0
        valid = valid.clone()
        valid[0] = True
        times = torch.where(torch.isfinite(times), times, torch.full_like(times, 48.0))
        with torch.no_grad():
            _, f1, b1 = self.model.impute(values, times, mask, valid)
            _, f2, b2 = self.model.impute(later, times, later_mask, valid)
        torch.testing.assert_close(f1[0, :-1], f2[0, :-1])  # forward: past only
        self.assertFalse(torch.allclose(b1[0, 0], b2[0, 0]))

    def test_losses_and_both_train_steps(self):
        for loss in (self.model.discriminator_loss(**self.batch),
                     self.model.generator_loss(**self.batch)):
            self.assertTrue(torch.isfinite(loss))
        mse = _model(self.dataset, reconstruction="mse")
        self.assertTrue(torch.isfinite(mse.generator_loss(**self.batch)))

        optimizer = torch.optim.Adam(
            [p for n, p in self.model.named_parameters() if not n.startswith("discriminator.")],
            lr=1e-3,
        )
        d_opt = torch.optim.Adam(self.model.discriminator.parameters(), lr=1e-3)
        g_opt = torch.optim.Adam(self.model.generator_parameters(), lr=1e-3)
        before = {n: p.clone() for n, p in self.model.named_parameters()}
        for generator_optimizer in (g_opt, None):
            losses = self.model.train_step(self.batch, optimizer, d_opt, generator_optimizer)
            self.assertEqual(set(losses), {"discriminator", "generator", "joint"})
        for prefix in ("discriminator.", "forward_imputer.", "backward_imputer.",
                       "image_encoder.", "classifier."):
            changed = any(
                not torch.equal(p, before[n])
                for n, p in self.model.named_parameters() if n.startswith(prefix)
            )
            self.assertTrue(changed, prefix)

    def _optimizers(self):
        optimizer = torch.optim.Adam(
            [p for n, p in self.model.named_parameters() if not n.startswith("discriminator.")],
            lr=1e-3,
        )
        return optimizer, torch.optim.Adam(self.model.discriminator.parameters(), lr=1e-3)

    def test_clusters(self):
        out = self.model(**self.batch)
        self.assertEqual(self.model.cluster_loss(out["embed"]).item(), 0.0)
        self.model.train_step(self.batch, *self._optimizers())
        features = torch.cat(self.model._epoch_features)
        self.model.update_clusters()
        self.assertEqual(self.model.cluster_centers.shape, (2, 16))
        self.assertEqual(self.model._epoch_features, [])
        # Each centre is one of the features (a medoid-like centre, Eq. 12).
        for center in self.model.cluster_centers:
            self.assertTrue(any(torch.allclose(center, f) for f in features))
        loss = self.model.cluster_loss(self.model(**self.batch)["embed"])
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.model.update_clusters()  # nothing stored since: centres kept
        self.assertEqual(self.model.cluster_centers.shape, (2, 16))
        self.model.train_step(self.batch, *self._optimizers())
        self.model.update_clusters()  # restarts from the current centres
        self.assertEqual(self.model.cluster_centers.shape, (2, 16))

    def test_forward_stores_no_features(self):
        # The Trainer path never calls update_clusters, so nothing may pile up.
        self.model.train()
        self.model(**self.batch)["loss"].backward()
        self.assertEqual(self.model._epoch_features, [])

    def test_checkpoint_loads_into_a_new_model(self):
        self.model.train_step(self.batch, *self._optimizers())
        self.model.update_clusters()
        fresh = _model(self.dataset)
        fresh.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(fresh.cluster_centers, self.model.cluster_centers)
        self.model.eval()
        fresh.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                fresh(**self.batch)["y_prob"], self.model(**self.batch)["y_prob"]
            )

    def test_multiclass_and_image_types(self):
        dataset = create_sample_dataset(
            _samples(multiclass=True), {"x": "tensor", "x_time": "tensor"},
            {"label": "multiclass"},
        )
        batch = next(iter(get_dataloader(dataset, batch_size=8)))
        for kind in ("line", "gasf"):
            model = _model(dataset, image_type=kind)
            out = model(**batch)
            self.assertEqual(out["y_prob"].shape, (8, 3))
            torch.testing.assert_close(out["y_prob"].sum(1), torch.ones(8))

    def test_trainer(self):
        trainer = Trainer(model=self.model, metrics=["roc_auc"])
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=True, seed=0)
        trainer.train(train_dataloader=loader, epochs=1)
        self.assertIn("roc_auc", trainer.evaluate(loader))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            _model(self.dataset, image_type="photo")
        with self.assertRaises(ValueError):
            _model(self.dataset, reconstruction="mae")
        with self.assertRaises(ValueError):
            JointSeqImage(self.dataset, image_size=1, image_model=None, swin_config=TINY_SWIN)
        regression = create_sample_dataset(
            [{**s, "label": float(i)} for i, s in enumerate(_samples())],
            {"x": "tensor", "x_time": "tensor"}, {"label": "regression"},
        )
        with self.assertRaises(ValueError):
            _model(regression)
        longer = {
            "x": torch.full((2, 13, 4), 0.5),
            "x_time": torch.arange(13.0).repeat(2, 1),
        }
        with self.assertRaises(ValueError):
            self.model(**longer)


if __name__ == "__main__":
    unittest.main()
