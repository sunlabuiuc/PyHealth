"""XGBoostModel: parity with plain XGBoost, modes, inputs, persistence, SHAP."""

import ctypes
import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader, split_by_patient
from pyhealth.processors.base_processor import FeatureProcessor
from pyhealth.trainer import Trainer


def _xgboost_available():
    if importlib.util.find_spec("xgboost") is None:
        return False
    try:
        import xgboost  # noqa: F401
    except Exception:  # e.g. libomp missing on macOS
        return False
    return True


def _two_openmp_runtimes():
    """True on macOS when XGBoost's libomp is not the copy torch bundles.

    torch wheels bundle libomp; XGBoost on macOS loads Homebrew's. Two copies
    in one process can crash multithreaded fits, so those tests then skip.
    (scikit-learn's own bundled copy is ignored: it is not used by XGBoost.)
    """
    if sys.platform != "darwin":
        return False
    dyld = ctypes.CDLL(None)
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    names = (dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count()))
    loaded = {os.path.realpath(n) for n in names if n.endswith("/libomp.dylib")}
    return len({p for p in loaded if "/sklearn/" not in p}) > 1


HAS_XGB = _xgboost_available()
HAS_SHAP = HAS_XGB and importlib.util.find_spec("shap") is not None
if HAS_XGB:
    import xgboost as xgb

    from pyhealth.interpret.methods import TreeSHAP
    from pyhealth.models import XGBoostModel

N_FEATURES = 92
PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, tree_method="hist", random_state=0,
)


def _tabular_samples(n=1500, seed=0):
    """92 features with ~10% NaN, one all-NaN and one constant column, ~15% positives."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_FEATURES))
    score = X[:, 0] + 0.8 * X[:, 1] - 0.5 * X[:, 2] + 0.3 * rng.normal(size=n)
    y = (score > np.quantile(score, 0.85)).astype(int)
    X[rng.random(X.shape) < 0.10] = np.nan
    X[:, 5] = np.nan
    X[:, 6] = 3.0
    # 1-3 samples per patient, so the split is by patient, not by row.
    return [
        {"patient_id": f"p{i // 2}", "visit_id": f"v{i}", "x": X[i].tolist(), "label": int(y[i])}
        for i in range(n)
    ]


def _matrix(dataset):
    """Rows of the raw feature, in dataset order, built independently of the model."""
    return np.stack([np.asarray(dataset[i]["x"], dtype=np.float32) for i in range(len(dataset))])


def _labels(dataset):
    return np.array([int(dataset[i]["label"]) for i in range(len(dataset))])


class MedianZScore(FeatureProcessor):
    """Median impute then z-score, statistics in float64, cast to float32 once.

    Mirrors SimpleImputer(median, keep_empty_features=True) + StandardScaler:
    all-NaN columns impute to 0 and zero-variance columns are scaled by 1.
    """

    def fit(self, samples, field):
        X = np.array([s[field] for s in samples], dtype=np.float64)
        med = np.nanmedian(np.where(np.isnan(X).all(0), 0.0, X), axis=0)
        X = np.where(np.isnan(X), med, X)
        self.median, self.mean = med, X.mean(axis=0)
        std = X.std(axis=0)
        self.scale = np.where(std == 0, 1.0, std)

    def process(self, value):
        x = np.asarray(value, dtype=np.float64)
        x = np.where(np.isnan(x), self.median, x)
        return torch.from_numpy(((x - self.mean) / self.scale).astype(np.float32))

    def size(self):
        return None


@unittest.skipUnless(HAS_XGB, "xgboost is not installed")
class TestParityWithXGBoost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = _tabular_samples()
        cls.dataset = create_sample_dataset(cls.samples, {"x": "tensor"}, {"label": "binary"})
        cls.train, cls.val, cls.test = split_by_patient(cls.dataset, [0.7, 0.15, 0.15], seed=0)

    def _reference(self, train_X, train_y, **extra):
        w = (train_y == 0).sum() / (train_y == 1).sum()
        ref = xgb.XGBClassifier(**PARAMS, scale_pos_weight=w, **extra)
        return ref, w

    def test_identical_probabilities_with_native_missing(self):
        model = XGBoostModel(self.dataset, scale_pos_weight="auto", **PARAMS)
        model.fit(get_dataloader(self.train, batch_size=128))
        X, y = _matrix(self.train), _labels(self.train)
        ref, w = self._reference(X, y)
        ref.fit(X, y)
        self.assertEqual(model.fitted_scale_pos_weight, [w])
        self.assertTrue(np.isnan(X).any())
        y_true, y_prob, _ = Trainer(model=model, enable_logging=False).inference(
            get_dataloader(self.test, batch_size=100)
        )
        expected = ref.predict_proba(_matrix(self.test))[:, 1]
        np.testing.assert_array_equal(y_prob[:, 0], expected)
        self.assertEqual(y_prob.shape, (len(self.test), 1))

    def test_identical_after_median_impute_and_zscore(self):
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        train_ids = {self.train[i]["visit_id"] for i in range(len(self.train))}
        proc = MedianZScore()
        proc.fit([s for s in self.samples if s["visit_id"] in train_ids], "x")
        dataset = create_sample_dataset(
            self.samples, {"x": "tensor"}, {"label": "binary"}, input_processors={"x": proc}
        )
        train, _, test = split_by_patient(dataset, [0.7, 0.15, 0.15], seed=0)
        model = XGBoostModel(dataset, scale_pos_weight="auto", **PARAMS).fit(train)

        raw = {s["visit_id"]: s["x"] for s in self.samples}
        rows = lambda ds: np.array([raw[ds[i]["visit_id"]] for i in range(len(ds))], dtype=np.float64)
        pipe = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
        ).fit(rows(train))
        X = pipe.transform(rows(train)).astype(np.float32)
        y = _labels(train)
        ref, _ = self._reference(X, y)
        ref.fit(X, y)
        np.testing.assert_array_equal(model.build_feature_matrix(x=torch.stack([test[i]["x"] for i in range(len(test))])),
                                      pipe.transform(rows(test)).astype(np.float32))
        _, y_prob, _ = Trainer(model=model, enable_logging=False).inference(
            get_dataloader(test, batch_size=64)
        )
        expected = ref.predict_proba(pipe.transform(rows(test)).astype(np.float32))[:, 1]
        np.testing.assert_array_equal(y_prob[:, 0], expected)

    def test_deterministic_single_and_multi_thread(self):
        for n_jobs in (1, 4):
            with self.subTest(n_jobs=n_jobs):
                if n_jobs > 1 and _two_openmp_runtimes():
                    self.skipTest("two OpenMP runtimes loaded (macOS torch + Homebrew libomp)")
                preds = []
                for _ in range(2):
                    m = XGBoostModel(self.dataset, scale_pos_weight="auto", n_jobs=n_jobs, **PARAMS)
                    m.fit(self.train)
                    preds.append(m.predict_numpy(_matrix(self.test))[1])
                np.testing.assert_array_equal(preds[0], preds[1])

    def test_early_stopping_uses_best_iteration(self):
        params = dict(PARAMS, n_estimators=600, learning_rate=0.3)
        model = XGBoostModel(
            self.dataset, early_stopping_rounds=20, eval_metric="aucpr", **params
        ).fit(self.train, self.val)
        best = model.estimators_[0].best_iteration
        self.assertLess(best, 580)
        ref = xgb.XGBClassifier(early_stopping_rounds=20, eval_metric="aucpr", **params)
        ref.fit(_matrix(self.train), _labels(self.train),
                eval_set=[(_matrix(self.val), _labels(self.val))], verbose=False)
        self.assertEqual(best, ref.best_iteration)
        X_test = _matrix(self.test)
        np.testing.assert_array_equal(model.predict_numpy(X_test)[1][:, 0], ref.predict_proba(X_test)[:, 1])
        # predictions stop at best_iteration, not at the last tree
        all_trees = ref.predict_proba(X_test, iteration_range=(0, ref.get_booster().num_boosted_rounds()))
        self.assertFalse(np.array_equal(all_trees[:, 1], ref.predict_proba(X_test)[:, 1]))

    def test_logit_is_the_booster_margin(self):
        model = XGBoostModel(self.dataset, n_estimators=50, max_depth=3).fit(self.train)
        batch = next(iter(get_dataloader(self.test, batch_size=64)))
        out = model(**batch)
        margin = model.estimators_[0].predict(model.build_feature_matrix(**batch), output_margin=True)
        np.testing.assert_array_equal(out["logit"][:, 0].numpy(), margin)
        expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logit"], batch["label"])
        self.assertTrue(torch.allclose(out["loss"], expected_loss))


def _mixed_samples(n=90, label="binary"):
    labels = {
        "binary": lambda i: int(i % 5 > 2),
        "multiclass": lambda i: i % 3,
        "multilabel": lambda i: ["a", "b", "c", "rare"][: 1 + i % 3],
        "regression": lambda i: float(i % 7) / 2,
    }[label]
    return [
        {
            "patient_id": f"p{i}",
            "labs": [float(i % 7), float((i * 3) % 5), float("nan") if i % 4 == 0 else 1.0],
            "flags": ["x", "y", "z"][: i % 4],
            "codes": [f"c{(i + j) % 11}" for j in range(1 + i % 6)],
            "visits": [[f"d{(i + v + k) % 5}" for k in range(1 + v % 3)] for v in range(1 + i % 3)],
            "label": labels(i),
        }
        for i in range(n)
    ]


MIXED = {"labs": "tensor", "flags": "multi_hot", "codes": "sequence", "visits": "nested_sequence"}


@unittest.skipUnless(HAS_XGB, "xgboost is not installed")
class TestModesAndInputs(unittest.TestCase):
    def test_every_mode_through_trainer(self):
        expected_width = {"binary": 1, "multiclass": 3, "multilabel": 3, "regression": 1}
        for mode in ("binary", "multiclass", "multilabel", "regression"):
            with self.subTest(mode=mode):
                ds = create_sample_dataset(_mixed_samples(label=mode), MIXED, {"label": mode})
                model = XGBoostModel(ds, n_estimators=30, max_depth=3, bag_of_codes=True)
                loader = get_dataloader(ds, batch_size=32)
                model.fit(loader)
                trainer = Trainer(model=model, enable_logging=False)
                y_true, y_prob, loss = trainer.inference(loader)
                self.assertEqual(y_prob.shape, (len(ds), expected_width[mode]))
                scores = trainer.evaluate(loader)
                self.assertTrue(np.isfinite(scores["loss"]))
                if mode == "binary":
                    self.assertGreater(scores["roc_auc"], 0.9)
                elif mode == "multiclass":
                    self.assertGreater(scores["accuracy"], 0.9)
                    np.testing.assert_allclose(y_prob.sum(1), 1.0, rtol=1e-5)
                elif mode == "regression":
                    self.assertLess(scores["mse"], 1.0)
                with self.assertRaisesRegex(TypeError, "model.fit"):
                    trainer.train(loader, epochs=1)

    def test_multilabel_uses_one_booster_and_weight_per_label(self):
        ds = create_sample_dataset(_mixed_samples(label="multilabel"), {"labs": "tensor"}, {"label": "multilabel"})
        model = XGBoostModel(ds, n_estimators=10, scale_pos_weight="balanced").fit(ds)
        self.assertEqual(len(model.estimators_), 3)
        y = np.stack([ds[i]["label"].numpy() for i in range(len(ds))])
        expected = [(len(y) - y[:, j].sum()) / y[:, j].sum() for j in range(3)]
        np.testing.assert_allclose(model.fitted_scale_pos_weight, expected)

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
        model = XGBoostModel(ds, n_estimators=5).fit(ds)
        self.assertEqual(
            model.feature_names,
            ["vec[0]", "vec[1]", "mh=u", "mh=v", "nmh=a", "nmh=b", "flag", "score"],
        )
        X = model.build_feature_matrix(**next(iter(get_dataloader(ds, batch_size=40))))
        # nested_multihot: number of visits containing each code
        np.testing.assert_array_equal(X[:, 4:6], [[1 + i % 3, (1 + i % 3) // 2] for i in range(40)])
        ranges = {f["key"]: (f["start"], f["stop"]) for f in model.feature_layout}
        self.assertEqual(ranges, {"vec": (0, 2), "mh": (2, 4), "nmh": (4, 6), "flag": (6, 7), "score": (7, 8)})

    def test_padded_sequence_raises_with_field_name(self):
        ds = create_sample_dataset(_mixed_samples(), MIXED, {"label": "binary"})
        with self.assertRaisesRegex(ValueError, "'codes'.*bag_of_codes=True"):
            XGBoostModel(ds)

    def test_unsupported_processor_raises(self):
        samples = [{"patient_id": f"p{i}", "v": [[float(i)]], "label": i % 2} for i in range(6)]
        ds = create_sample_dataset(samples, {"v": "nested_sequence_floats"}, {"label": "binary"})
        with self.assertRaisesRegex(ValueError, "'v' uses NestedFloatsProcessor"):
            XGBoostModel(ds)

    def test_varying_tensor_width_raises(self):
        samples = [{"patient_id": f"p{i}", "v": [1.0] * (1 + i // 4), "label": i % 2} for i in range(8)]
        ds = create_sample_dataset(samples, {"v": "tensor"}, {"label": "binary"})
        with self.assertRaisesRegex(ValueError, "'v' has per-sample shape"):
            XGBoostModel(ds, n_estimators=2).fit(get_dataloader(ds, batch_size=4))

    def test_bag_of_codes_width_is_fixed_and_counts_match(self):
        samples = _mixed_samples()
        ds = create_sample_dataset(samples, MIXED, {"label": "binary"})
        model = XGBoostModel(ds, n_estimators=5, bag_of_codes=True).fit(ds)
        codes = ds.input_processors["codes"].code_vocab
        visits = ds.input_processors["visits"].code_vocab
        # batches of 1 and 32 pad to different lengths; the width must not change
        X1 = np.concatenate([model.build_feature_matrix(**b) for b in get_dataloader(ds, batch_size=1)])
        X32 = np.concatenate([model.build_feature_matrix(**b) for b in get_dataloader(ds, batch_size=32)])
        np.testing.assert_array_equal(X1, X32)
        self.assertEqual(X1.shape[1], 3 + 3 + (len(codes) - 2) + (len(visits) - 2))
        names = model.feature_names
        for i, s in enumerate(samples):
            expected = np.zeros(len(names))
            for c in s["codes"]:
                expected[names.index(f"codes={c}")] += 1
            for visit in s["visits"]:
                for c in visit:
                    expected[names.index(f"visits={c}")] += 1
            start = 6
            np.testing.assert_array_equal(X1[i, start:], expected[start:])

    def test_fit_rejects_drop_last(self):
        ds = create_sample_dataset(_mixed_samples(), {"labs": "tensor"}, {"label": "binary"})
        loader = torch.utils.data.DataLoader(ds, batch_size=32, drop_last=True,
                                             collate_fn=get_dataloader(ds, 1).collate_fn)
        with self.assertRaisesRegex(ValueError, "drop_last"):
            XGBoostModel(ds).fit(loader)

    def test_forward_before_fit_raises(self):
        ds = create_sample_dataset(_mixed_samples(), {"labs": "tensor"}, {"label": "binary"})
        with self.assertRaisesRegex(RuntimeError, "model.fit"):
            XGBoostModel(ds)(**next(iter(get_dataloader(ds, batch_size=4))))


@unittest.skipUnless(HAS_XGB, "xgboost is not installed")
class TestPersistence(unittest.TestCase):
    def test_checkpoint_round_trip(self):
        for mode in ("binary", "multiclass", "multilabel", "regression"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                ds = create_sample_dataset(_mixed_samples(label=mode), MIXED, {"label": mode})
                loader = get_dataloader(ds, batch_size=32)
                model = XGBoostModel(ds, n_estimators=20, bag_of_codes=True,
                                     early_stopping_rounds=5 if mode == "binary" else None,
                                     eval_metric="logloss" if mode == "binary" else None)
                model.fit(loader, loader if mode == "binary" else None)
                path = os.path.join(tmp, "model.ckpt")
                Trainer(model=model, enable_logging=False).save_ckpt(path)
                fresh = XGBoostModel(ds, bag_of_codes=True)
                Trainer(model=fresh, enable_logging=False).load_ckpt(path)  # weights_only=True
                for batch in loader:
                    a, b = model(**batch), fresh(**batch)
                    self.assertTrue(torch.equal(a["y_prob"], b["y_prob"]))
                    self.assertTrue(torch.equal(a["logit"], b["logit"]))
                self.assertEqual(fresh.feature_layout, model.feature_layout)

    def test_mismatched_layout_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = create_sample_dataset(_mixed_samples(), MIXED, {"label": "binary"})
            model = XGBoostModel(ds, n_estimators=5, bag_of_codes=True).fit(ds)
            path = os.path.join(tmp, "model.ckpt")
            Trainer(model=model, enable_logging=False).save_ckpt(path)
            other = create_sample_dataset(_mixed_samples(), {"labs": "tensor"}, {"label": "binary"})
            with self.assertRaisesRegex(ValueError, "feature layout"):
                Trainer(model=XGBoostModel(other), enable_logging=False).load_ckpt(path)
            fewer_codes = [dict(s, codes=["c1"]) for s in _mixed_samples()]
            smaller = create_sample_dataset(fewer_codes, MIXED, {"label": "binary"})
            with self.assertRaisesRegex(ValueError, "'codes' had"):
                Trainer(model=XGBoostModel(smaller, bag_of_codes=True), enable_logging=False).load_ckpt(path)


@unittest.skipUnless(HAS_XGB, "xgboost is not installed")
class TestTreeSHAP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = create_sample_dataset(_tabular_samples(n=600), {"x": "tensor"}, {"label": "binary"})
        cls.model = XGBoostModel(cls.dataset, n_estimators=100, max_depth=4).fit(cls.dataset)

    def test_contributions_add_up_to_the_margin(self):
        for mode in ("binary", "multiclass", "multilabel", "regression"):
            with self.subTest(mode=mode):
                ds = create_sample_dataset(_mixed_samples(label=mode), MIXED, {"label": mode})
                model = XGBoostModel(ds, n_estimators=30, bag_of_codes=True).fit(ds)
                batch = next(iter(get_dataloader(ds, batch_size=64)))
                out = model.explain(**batch)
                total = sum(a.sum(1) for a in out["attributions"].values()) + out["bias"]
                logit = out["logit"] if mode in ("multiclass", "multilabel") else out["logit"][:, 0]
                torch.testing.assert_close(total, logit, atol=1e-5, rtol=1e-5)
                self.assertEqual(
                    {k: len(v) for k, v in out["feature_names"].items()},
                    {k: v.shape[1] for k, v in out["attributions"].items()},
                )

    @unittest.skipUnless(HAS_SHAP, "shap is not installed")
    def test_global_ranking_matches_shap_tree_explainer(self):
        import shap

        X = _matrix(self.dataset)
        ranking = self.model.mean_abs_shap(self.dataset)
        values = shap.TreeExplainer(self.model.estimators_[0]).shap_values(X)
        reference = dict(zip(self.model.feature_names, np.abs(values).mean(0)))
        np.testing.assert_allclose([ranking[k] for k in reference], list(reference.values()), rtol=1e-4, atol=1e-6)
        top = [k for k, _ in sorted(reference.items(), key=lambda p: -p[1])][:10]
        self.assertEqual(list(ranking)[:10], top)

    def test_interpreter_matches_input_shapes(self):
        ds = create_sample_dataset(_mixed_samples(label="multiclass"), MIXED, {"label": "multiclass"})
        model = XGBoostModel(ds, n_estimators=30, bag_of_codes=True).fit(ds)
        batch = next(iter(get_dataloader(ds, batch_size=16)))
        attributions = TreeSHAP(model).attribute(target_class_idx=2, **batch)
        self.assertEqual(set(attributions), set(MIXED))
        for key in MIXED:
            self.assertEqual(attributions[key].shape, batch[key].shape)
        # each present code's contribution is spread over its positions
        columns = model.explain(**batch)["attributions"]["codes"][..., 2]
        present = model.build_feature_matrix(**batch)[:, 6:6 + columns.shape[1]] > 0
        expected = (columns * torch.as_tensor(present)).sum(1)
        torch.testing.assert_close(attributions["codes"].sum(1), expected, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.all(attributions["codes"][batch["codes"] == 0] == 0))

    def test_interpreter_rejects_other_models(self):
        from pyhealth.models import LogisticRegression

        with self.assertRaises(TypeError):
            TreeSHAP(LogisticRegression(dataset=self.dataset))


class TestWithoutXGBoost(unittest.TestCase):
    def test_missing_xgboost_gives_install_hint(self):
        from unittest import mock

        from pyhealth.models import xgboost_model

        ds = create_sample_dataset(_mixed_samples(), {"labs": "tensor"}, {"label": "binary"})
        with mock.patch.object(xgboost_model.importlib, "import_module", side_effect=ImportError("no")):
            with self.assertRaisesRegex(ImportError, r"pyhealth\[xgboost\]"):
                xgboost_model.XGBoostModel(ds)


if __name__ == "__main__":
    unittest.main()
