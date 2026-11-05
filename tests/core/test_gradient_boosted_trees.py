"""GradientBoostedTreeModel with a scikit-learn backend, so it runs without xgboost.

The base class does not depend on the tree library: subclasses only build,
fit, predict with and serialize one estimator. ``LinearBackend`` fills those
hooks with linear models, which keeps the feature layout, fitting, prediction,
attribution and checkpoint code under test in environments without xgboost
(``tests/core/test_xgboost_model.py`` covers XGBoostModel itself).
"""

import os
import pickle
import tempfile
import unittest

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import GradientBoostedTreeModel
from pyhealth.trainer import Trainer

LOGGER = "pyhealth.models.gradient_boosted_trees"
MIXED = {"labs": "tensor", "flags": "multi_hot", "codes": "sequence", "visits": "nested_sequence"}


class LinearBackend(GradientBoostedTreeModel):
    """Linear models in place of trees.

    A linear model's contributions (coefficient times value, with the
    intercept as the bias) add up to its margin exactly, as TreeSHAP values
    add up to a tree ensemble's.
    """

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.eval_sets = []

    def _make_estimator(self, scale_pos_weight):
        if self.mode == "regression":
            return LinearRegression()
        weight = None if scale_pos_weight is None else {0: 1.0, 1: scale_pos_weight}
        return LogisticRegression(class_weight=weight, max_iter=2000)

    def _fit_estimator(self, estimator, X, y, eval_set):
        self.eval_sets.append(eval_set)
        estimator.fit(X, y)

    def _predict_margin(self, estimator, X):
        if self.mode == "regression":
            return estimator.predict(X)
        return estimator.decision_function(X)

    def _predict_value(self, estimator, X):
        if self.mode == "regression":
            return estimator.predict(X)
        if self.mode == "multilabel":  # one estimator per label
            return estimator.predict_proba(X)[:, 1]
        return estimator.predict_proba(X)

    def _contributions(self, estimator, X):
        coef = np.atleast_2d(estimator.coef_)  # (K, d)
        bias = np.broadcast_to(
            np.atleast_1d(estimator.intercept_)[None, :, None], (len(X), len(coef), 1)
        )
        values = np.concatenate([X[:, None, :] * coef[None], bias], axis=2)
        return values[:, 0] if len(coef) == 1 else values

    def _to_bytes(self, estimator):
        return pickle.dumps(estimator)

    def _from_bytes(self, raw):
        return pickle.loads(raw)  # bytes this test wrote


def _samples(n=90, label="binary"):
    labels = {
        "binary": lambda i: int(i % 5 > 2),
        "multiclass": lambda i: i % 3,
        "multilabel": lambda i: [["a"], ["b"], ["a", "c"], ["b", "c"]][i % 4],
        "regression": lambda i: float(i % 7) / 2,
    }[label]
    return [
        {
            "patient_id": f"p{i}",
            "labs": [float(i % 7), float((i * 3) % 5), float(i % 2)],
            "flags": ["x", "y", "z"][: i % 4],
            "codes": [f"c{(i + j) % 11}" for j in range(1 + i % 6)],
            "visits": [[f"d{(i + v + k) % 5}" for k in range(1 + v % 3)] for v in range(1 + i % 3)],
            "label": labels(i),
        }
        for i in range(n)
    ]


def _plain_loader(dataset, data=None, **kwargs):
    """A torch DataLoader without get_dataloader's shuffle_samples attribute."""
    collate = get_dataloader(dataset, batch_size=1).collate_fn
    return torch.utils.data.DataLoader(
        dataset if data is None else data, batch_size=32, collate_fn=collate, **kwargs
    )


class TestEveryMode(unittest.TestCase):
    def test_fit_predict_explain_and_checkpoint(self):
        width = {"binary": 1, "multiclass": 3, "multilabel": 3, "regression": 1}
        for mode in ("binary", "multiclass", "multilabel", "regression"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                ds = create_sample_dataset(_samples(label=mode), MIXED, {"label": mode})
                loader = get_dataloader(ds, batch_size=32)
                model = LinearBackend(ds, bag_of_codes=True).fit(loader, loader)
                self.assertTrue(model.is_fitted)
                self.assertTrue(all(e is not None for e in model.eval_sets))

                trainer = Trainer(model=model, enable_logging=False)
                _, y_prob, loss = trainer.inference(loader)
                self.assertEqual(y_prob.shape, (len(ds), width[mode]))
                self.assertTrue(np.isfinite(loss))
                with self.assertRaisesRegex(TypeError, "model.fit"):
                    trainer.train(loader, epochs=1)

                batch = next(iter(loader))
                inputs = {k: v for k, v in batch.items() if k != "label"}
                self.assertEqual(set(model(**inputs)), {"logit", "y_prob"})

                out = model.explain(**batch)
                total = sum(a.sum(1) for a in out["attributions"].values()) + out["bias"]
                logit = out["logit"] if width[mode] > 1 else out["logit"][:, 0]
                torch.testing.assert_close(total, logit, atol=1e-4, rtol=1e-4)
                self.assertEqual(
                    {k: len(v) for k, v in out["feature_names"].items()},
                    {k: v.shape[1] for k, v in out["attributions"].items()},
                )
                ranking = model.mean_abs_shap(ds)
                self.assertEqual(sorted(ranking), sorted(model.feature_names))
                self.assertEqual(list(ranking.values()), sorted(ranking.values(), reverse=True))

                path = os.path.join(tmp, "model.ckpt")
                trainer.save_ckpt(path)
                fresh = LinearBackend(ds, bag_of_codes=True)
                Trainer(model=fresh, enable_logging=False).load_ckpt(path)
                self.assertEqual(fresh.feature_layout, model.feature_layout)
                a, b = model(**batch), fresh(**batch)
                self.assertTrue(torch.equal(a["y_prob"], b["y_prob"]))
                self.assertTrue(torch.equal(a["logit"], b["logit"]))


class TestInputs(unittest.TestCase):
    def test_fixed_width_processors(self):
        samples = [
            {
                "patient_id": f"p{i}",
                "vec": [float(i), float(i % 3)],
                "mh": ["u", "v"][: i % 3],
                "nmh": [["a", "b"][: 1 + v % 2] for v in range(1 + i % 3)],
                "flag": i % 2,
                "score": float(i) / 10,
                "label": int(i % 4 == 0),
            }
            for i in range(40)
        ]
        schema = {"vec": "tensor", "mh": "multi_hot", "nmh": "nested_multihot",
                  "flag": "binary", "score": "regression"}
        ds = create_sample_dataset(samples, schema, {"label": "binary"})
        model = LinearBackend(ds, feature_names={"vec": ["day", "phase"]}).fit(ds)
        self.assertEqual(
            model.feature_names,
            ["day", "phase", "mh=u", "mh=v", "nmh=a", "nmh=b", "flag", "score"],
        )
        X = model.build_feature_matrix(**next(iter(get_dataloader(ds, batch_size=40))))
        # nested_multihot: number of visits containing each code
        np.testing.assert_array_equal(X[:, 4:6], [[1 + i % 3, (1 + i % 3) // 2] for i in range(40)])
        ranges = {f["key"]: (f["start"], f["stop"]) for f in model.feature_layout}
        self.assertEqual(ranges, {"vec": (0, 2), "mh": (2, 4), "nmh": (4, 6), "flag": (6, 7),
                                  "score": (7, 8)})

        with self.assertRaisesRegex(ValueError, r"feature_names\['vec'\] has 1 names"):
            LinearBackend(ds, feature_names={"vec": ["day"]}).fit(ds)
        batch = next(iter(get_dataloader(ds, batch_size=4)))
        with self.assertRaisesRegex(ValueError, "'nmh' has a vocabulary of 1 entries"):
            model.build_feature_matrix(**dict(batch, nmh=torch.zeros(4, 2, 1)))

    def test_bag_of_codes_width_is_fixed(self):
        ds = create_sample_dataset(_samples(), MIXED, {"label": "binary"})
        model = LinearBackend(ds, bag_of_codes=True).fit(ds)
        # batches of 1 and 32 pad to different lengths; the width must not change
        X1 = np.concatenate([model.build_feature_matrix(**b) for b in get_dataloader(ds, batch_size=1)])
        X32 = np.concatenate([model.build_feature_matrix(**b) for b in get_dataloader(ds, batch_size=32)])
        np.testing.assert_array_equal(X1, X32)
        self.assertEqual(X1.shape[1], len(model.feature_names))

    def test_inputs_that_cannot_become_columns_raise(self):
        ds = create_sample_dataset(_samples(), MIXED, {"label": "binary"})
        with self.assertRaisesRegex(ValueError, "'codes'.*bag_of_codes=True"):
            LinearBackend(ds)

        floats = [{"patient_id": f"p{i}", "v": [[float(i)]], "label": i % 2} for i in range(6)]
        ds = create_sample_dataset(floats, {"v": "nested_sequence_floats"}, {"label": "binary"})
        with self.assertRaisesRegex(ValueError, "'v' uses NestedFloatsProcessor"):
            LinearBackend(ds)

        varying = [{"patient_id": f"p{i}", "v": [1.0] * (1 + i // 4), "label": i % 2}
                   for i in range(8)]
        ds = create_sample_dataset(varying, {"v": "tensor"}, {"label": "binary"})
        with self.assertRaisesRegex(ValueError, "'v' has per-sample shape"):
            LinearBackend(ds).fit(get_dataloader(ds, batch_size=4))

    def test_batches_that_do_not_match_the_layout_raise(self):
        ds = create_sample_dataset(_samples(), MIXED, {"label": "binary"})
        model = LinearBackend(ds, bag_of_codes=True).fit(ds)
        batch = next(iter(get_dataloader(ds, batch_size=4)))
        cases = {
            "missing input field 'labs'": (KeyError, {k: v for k, v in batch.items() if k != "labs"}),
            "'labs' is not a tensor": (TypeError, dict(batch, labs=[[1.0, 2.0, 3.0]] * 4)),
            r"'flags' has shape \(2,\)": (ValueError, dict(batch, flags=torch.zeros(4, 2))),
        }
        for message, (error, bad) in cases.items():
            with self.subTest(message), self.assertRaisesRegex(error, message):
                model.build_feature_matrix(**bad)


class TestFitting(unittest.TestCase):
    def test_constructor_checks(self):
        ds = create_sample_dataset(_samples(), {"labs": "tensor"}, {"label": "binary"})
        with self.assertRaisesRegex(ValueError, "'balanced', 'auto' or None"):
            LinearBackend(ds, scale_pos_weight="heavy")
        multiclass = create_sample_dataset(_samples(label="multiclass"), {"labs": "tensor"},
                                           {"label": "multiclass"})
        with self.assertRaisesRegex(ValueError, "binary or multilabel"):
            LinearBackend(multiclass, scale_pos_weight=2.0)
        two = [dict(s, other=s["label"]) for s in _samples()]
        ds = create_sample_dataset(two, {"labs": "tensor"}, {"label": "binary", "other": "binary"})
        with self.assertRaisesRegex(ValueError, "exactly one label"):
            LinearBackend(ds)

    def test_scale_pos_weight(self):
        ds = create_sample_dataset(_samples(), {"labs": "tensor"}, {"label": "binary"})
        y = np.array([s["label"] for s in _samples()])
        model = LinearBackend(ds, scale_pos_weight="balanced").fit(ds)
        np.testing.assert_allclose(model.fitted_scale_pos_weight,
                                   [(y == 0).sum() / (y == 1).sum()])
        self.assertEqual(LinearBackend(ds, scale_pos_weight=3).fit(ds).fitted_scale_pos_weight, [3.0])
        self.assertIsNone(LinearBackend(ds).fit(ds).fitted_scale_pos_weight)

        ml = create_sample_dataset(_samples(label="multilabel"), {"labs": "tensor"},
                                   {"label": "multilabel"})
        model = LinearBackend(ml, scale_pos_weight=[1.0, 2.0, 3.0]).fit(ml)
        self.assertEqual(len(model.estimators_), 3)
        self.assertEqual(model.fitted_scale_pos_weight, [1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "2 values for 3 labels"):
            LinearBackend(ml, scale_pos_weight=[1.0, 2.0]).fit(ml)
        with self.assertLogs(LOGGER, "WARNING") as logs:
            weights = LinearBackend(ml, scale_pos_weight="auto")._resolve_pos_weight(
                np.array([[1, 0, 1], [0, 0, 1], [0, 0, 0]])
            )
        self.assertEqual(weights, [2.0, 1.0, 0.5])
        self.assertIn("no positive training examples", logs.output[0])

    def test_training_sets_that_cannot_be_fit_raise(self):
        ds = create_sample_dataset(_samples(label="multiclass"), {"labs": "tensor"},
                                   {"label": "multiclass"})
        two_classes = ds.subset([i for i in range(len(ds)) if i % 3 != 2])
        with self.assertRaisesRegex(ValueError, r"Classes \[2\] have no training samples"):
            LinearBackend(ds).fit(two_classes)
        with self.assertRaisesRegex(ValueError, "train_data is empty"):
            LinearBackend(ds).fit(torch.utils.data.DataLoader([], batch_size=4))
        with self.assertRaisesRegex(ValueError, "drop_last"):
            LinearBackend(ds).fit(_plain_loader(ds, drop_last=True))

    def test_shuffle_warning_follows_the_fit_loader(self):
        # Disk-backed, so the dataset's flag is litdata's ``shuffle``.
        ds = create_sample_dataset(_samples(), {"labs": "tensor"}, {"label": "binary"},
                                   in_memory=False)
        # A shuffled loader on the same dataset, e.g. another model's training,
        # leaves the flag set; get_dataloader's loaders set it again each pass.
        for _ in get_dataloader(ds, batch_size=32, shuffle=True):
            pass
        self.assertTrue(ds.shuffle)
        with self.assertNoLogs(LOGGER, "WARNING"):
            LinearBackend(ds).fit(get_dataloader(ds, batch_size=32))
        with self.assertLogs(LOGGER, "WARNING") as logs:
            LinearBackend(ds).fit(get_dataloader(ds, 32, shuffle=True))
        self.assertIn("shuffles", logs.output[0])

    def test_shuffle_warning_with_a_plain_dataloader(self):
        # Without shuffle_samples, the sampler and the dataset's flag decide.
        ds = create_sample_dataset(_samples(), {"labs": "tensor"}, {"label": "binary"})
        rows = [ds[i] for i in range(len(ds))]  # map-style, so it can be shuffled
        with self.assertNoLogs(LOGGER, "WARNING"):
            LinearBackend(ds).fit(_plain_loader(ds, rows))
        with self.assertLogs(LOGGER, "WARNING"):
            LinearBackend(ds).fit(_plain_loader(ds, rows, shuffle=True))
        # Iterating a SampleDataset shuffles when its flag is set; this one is
        # in memory, which keeps the flag in _shuffle.
        ds.set_shuffle(False)
        with self.assertNoLogs(LOGGER, "WARNING"):
            LinearBackend(ds).fit(_plain_loader(ds))
        ds.set_shuffle(True)
        with self.assertLogs(LOGGER, "WARNING") as logs:
            LinearBackend(ds).fit(_plain_loader(ds))
        self.assertIn("shuffles", logs.output[0])
        ds.set_shuffle(False)


class TestCheckpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = create_sample_dataset(_samples(), MIXED, {"label": "binary"})
        cls.model = LinearBackend(cls.ds, bag_of_codes=True).fit(cls.ds)

    def test_forward_before_fit_raises(self):
        batch = next(iter(get_dataloader(self.ds, batch_size=4)))
        with self.assertRaisesRegex(RuntimeError, "model.fit"):
            LinearBackend(self.ds, bag_of_codes=True)(**batch)

    def test_an_unfitted_checkpoint_loads_unfitted(self):
        fresh = LinearBackend(self.ds, bag_of_codes=True)
        fresh.load_state_dict(LinearBackend(self.ds, bag_of_codes=True).state_dict())
        self.assertFalse(fresh.is_fitted)

    def test_mismatched_checkpoints_raise(self):
        state = self.model.get_extra_state()
        fresh = LinearBackend(self.ds, bag_of_codes=True)
        with self.assertRaisesRegex(ValueError, "state version 99"):
            fresh.set_extra_state(dict(state, version=99))
        with self.assertRaisesRegex(ValueError, "fit for mode 'regression'"):
            fresh.set_extra_state(dict(state, mode="regression"))

        other = create_sample_dataset(_samples(), {"labs": "tensor"}, {"label": "binary"})
        with self.assertRaisesRegex(ValueError, "feature layout"):
            LinearBackend(other).set_extra_state(state)
        fewer_codes = [dict(s, codes=["c1"]) for s in _samples()]
        smaller = create_sample_dataset(fewer_codes, MIXED, {"label": "binary"})
        with self.assertRaisesRegex(ValueError, "'codes' had"):
            LinearBackend(smaller, bag_of_codes=True).set_extra_state(state)


if __name__ == "__main__":
    unittest.main()
