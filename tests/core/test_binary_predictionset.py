"""Binary conformal sets must agree with an equivalent two-class model."""

import numpy as np
import pytest
import torch

from pyhealth.calib.predictionset import (
    LABEL,
    RC3P,
    SCRIB,
    BaseConformal,
    ClusterLabel,
    CovariateLabel,
    LabelShift,
    NeighborhoodLabel,
)
from pyhealth.calib.utils import binary_to_two_columns
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
from pyhealth.models import MLP
from pyhealth.trainer import get_metrics_fn


class _ProbabilityModel(torch.nn.Module):
    """Read known probabilities instead of coupling the tests to model training."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.register_buffer("_device", torch.empty(0))

    @property
    def device(self):
        return self._device.device

    def forward(self, p, p0, p1, label=None, embed=False):
        positive = torch.as_tensor(p, dtype=torch.float64, device=self.device)
        if self.mode == "binary":
            probabilities = positive[:, None]
        else:
            # The reference reads both supplied columns independently of the
            # binary conversion helper under test.
            probabilities = torch.stack(
                [
                    torch.as_tensor(p0, dtype=torch.float64, device=self.device),
                    torch.as_tensor(p1, dtype=torch.float64, device=self.device),
                ],
                dim=1,
            )

        result = {"y_prob": probabilities}
        if label is not None:
            truth = torch.as_tensor(label, device=self.device)
            result["y_true"] = (
                truth.float()[:, None] if self.mode == "binary" else truth.long()
            )
        if embed:
            result["embed"] = torch.stack([positive, positive.square()], dim=1)

        return result


def _samples(seed, count):
    rng = np.random.default_rng(seed)
    probabilities = rng.uniform(0.02, 0.98, count)
    labels = rng.binomial(1, probabilities)

    return [
        {"p": float(p), "p0": float(1 - p), "p1": float(p), "label": int(y)}
        for p, y in zip(probabilities, labels)
    ]


def _predictor(method, mode, score_type="threshold", alpha=0.3):
    model = _ProbabilityModel(mode)
    if method is SCRIB:
        return method(model, risk=0.3)

    params = {"alpha": alpha, "score_type": score_type, "random_state": 42}
    if method is ClusterLabel:
        params["n_clusters"] = 2
    if method is NeighborhoodLabel:
        params.update(k_neighbors=16, lambda_L=1.0)

    return method(model, **params)


def _calibrate(predictor, samples, rank_samples=None):
    positive = np.array([row["p"] for row in samples])
    embeddings = np.column_stack([positive, positive**2])
    if isinstance(predictor, ClusterLabel):
        predictor.calibrate(
            samples, train_embeddings=embeddings[:16], cal_embeddings=embeddings
        )
    elif isinstance(predictor, NeighborhoodLabel):
        predictor.calibrate(samples, cal_embeddings=embeddings)
    elif isinstance(predictor, CovariateLabel):
        predictor.calibrate(samples, cal_weights=np.ones(len(samples)))
    elif isinstance(predictor, LabelShift):
        predictor.calibrate(
            samples, target_label_dist=[0.25, 0.75], source_label_dist=[0.5, 0.5]
        )
    elif isinstance(predictor, RC3P):
        predictor.calibrate(samples, rank_cal_dataset=rank_samples)
    else:
        predictor.calibrate(samples)


def _assert_equivalent(method, score_type, alpha=0.3, rank_samples=None):
    calibration = _samples(5, 80)
    test = _samples(19, 48)
    binary = _predictor(method, "binary", score_type, alpha)
    reference = _predictor(method, "multiclass", score_type, alpha)
    _calibrate(binary, calibration, rank_samples)
    _calibrate(reference, calibration, rank_samples)
    batch = next(iter(get_dataloader(test, batch_size=len(test), shuffle=False)))

    with torch.no_grad():
        actual = binary(**batch)
        expected = reference(**batch)
        native = binary.model.model(**batch)

    assert binary.mode == "binary"
    assert actual["y_predset"].dtype == torch.bool
    assert actual["y_predset"].shape == (len(test), 2)
    torch.testing.assert_close(actual["y_predset"], expected["y_predset"])
    torch.testing.assert_close(actual["y_prob"], native["y_prob"], rtol=0, atol=0)
    torch.testing.assert_close(actual["y_true"], native["y_true"], rtol=0, atol=0)
    assert actual["y_prob"].shape == actual["y_true"].shape == (len(test), 1)
    assert actual["y_true"].dtype == torch.float32


@pytest.mark.parametrize(
    "method",
    [
        LABEL,
        BaseConformal,
        ClusterLabel,
        CovariateLabel,
        NeighborhoodLabel,
        RC3P,
        LabelShift,
    ],
)
@pytest.mark.parametrize("score_type", ["threshold", "aps"])
def test_binary_sets_match_two_class_sets(method, score_type):
    _assert_equivalent(method, score_type)


@pytest.mark.parametrize("method", [LABEL, BaseConformal, ClusterLabel, CovariateLabel])
@pytest.mark.parametrize("score_type", ["threshold", "aps"])
def test_binary_class_specific_thresholds(method, score_type):
    _assert_equivalent(method, score_type, alpha=np.array([0.2, 0.4]))


@pytest.mark.parametrize("score_type", ["threshold", "aps"])
def test_binary_rc3p_with_separate_rank_calibration(score_type):
    _assert_equivalent(RC3P, score_type, rank_samples=_samples(29, 40))


def test_binary_scrib_sets_match_two_class_sets():
    _assert_equivalent(SCRIB, "threshold")


@pytest.mark.parametrize(
    "method",
    [
        LABEL,
        BaseConformal,
        ClusterLabel,
        CovariateLabel,
        NeighborhoodLabel,
        RC3P,
        LabelShift,
        SCRIB,
    ],
)
def test_other_modes_remain_unsupported(method):
    with pytest.raises(NotImplementedError):
        _predictor(method, "multilabel")


@pytest.mark.parametrize("method", [SCRIB, NeighborhoodLabel])
def test_empty_set_fallback_uses_both_binary_classes(method, monkeypatch):
    predictor = _predictor(method, "binary")
    if method is SCRIB:
        predictor.t = torch.nn.Parameter(torch.tensor([2.0, 2.0]))
    else:
        _calibrate(predictor, _samples(5, 80))
        monkeypatch.setattr(
            "pyhealth.calib.predictionset.cluster.neighborhood_label._query_weighted_quantile",
            lambda *args, **kwargs: np.inf,
        )
    batch = {"p": [0.2, 0.8], "p0": [0.8, 0.2], "p1": [0.2, 0.8]}
    with torch.no_grad():
        output = predictor(**batch)
    torch.testing.assert_close(
        output["y_predset"], torch.tensor([[True, False], [False, True]])
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("column", [False, True])
def test_binary_expansion_preserves_values_and_numpy_dtype(dtype, column):
    values = np.array([0.0, 0.25, 0.75, 1.0], dtype=dtype)
    result = binary_to_two_columns(values[:, None] if column else values)
    np.testing.assert_array_equal(result, [[1, 0], [0.75, 0.25], [0.25, 0.75], [0, 1]])
    assert result.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_binary_expansion_preserves_torch_dtype_device_and_gradient(dtype):
    values = torch.tensor([0.25, 0.75], dtype=dtype, requires_grad=True)
    result = binary_to_two_columns(values)
    assert result.dtype == values.dtype
    assert result.device == values.device
    torch.testing.assert_close(
        result, torch.tensor([[0.75, 0.25], [0.25, 0.75]], dtype=dtype)
    )
    result[:, 1].sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


@pytest.mark.parametrize("shape", [(), (2, 2), (2, 1, 1)])
def test_binary_expansion_rejects_other_shapes(shape):
    with pytest.raises(ValueError, match="shape"):
        binary_to_two_columns(np.zeros(shape))


SET_METRICS = [
    "set_size",
    "rejection_rate",
    "miscoverage_ps",
    "miscoverage_mean_ps",
    "miscoverage_overall_ps",
    "error_ps",
    "error_mean_ps",
    "error_overall_ps",
]


@pytest.mark.parametrize("column", [False, True])
def test_binary_set_metrics_match_multiclass_and_keep_native_metrics(column):
    labels = np.array([0, 1, 1, 0])
    probabilities = np.array([0.2, 0.7, 0.4, 0.9])
    sets = np.array([[1, 0], [0, 1], [1, 1], [0, 1]])
    native_labels = labels.astype(np.float32)
    if column:
        native_labels, probabilities = native_labels[:, None], probabilities[:, None]
    before = [values.copy() for values in (native_labels, probabilities, sets)]
    result = binary_metrics_fn(
        native_labels,
        probabilities,
        metrics=SET_METRICS + ["accuracy", "roc_auc"],
        y_predset=sets,
    )
    reference = multiclass_metrics_fn(
        labels,
        np.column_stack([1 - probabilities.reshape(-1), probabilities.reshape(-1)]),
        metrics=SET_METRICS,
        y_predset=sets,
    )
    for name in SET_METRICS:
        np.testing.assert_allclose(result[name], reference[name], equal_nan=True)
    native = binary_metrics_fn(
        native_labels, probabilities, metrics=["accuracy", "roc_auc"]
    )
    assert result["accuracy"] == native["accuracy"]
    assert result["roc_auc"] == native["roc_auc"]
    for actual, expected in zip((native_labels, probabilities, sets), before):
        np.testing.assert_array_equal(actual, expected)


def test_binary_metrics_exclude_undefined_classes_and_return_nan_for_empty_input():
    result = binary_metrics_fn(
        np.array([[0.0], [0.0]]),
        np.zeros((2, 1)),
        metrics=SET_METRICS,
        y_predset=np.array([[0, 1], [0, 1]]),
    )
    np.testing.assert_allclose(result["miscoverage_ps"], [1, np.nan], equal_nan=True)
    assert result["miscoverage_mean_ps"] == result["error_mean_ps"] == 1.0
    empty = binary_metrics_fn(
        np.empty((0, 1)),
        np.empty((0, 1)),
        metrics=SET_METRICS,
        y_predset=np.empty((0, 2)),
    )
    for value in empty.values():
        assert np.isnan(value).all()
    assert (
        binary_metrics_fn(np.array([0, 1]), np.array([0.2, 0.8]), metrics=SET_METRICS)
        == {}
    )


@pytest.mark.parametrize(
    "truth,sets",
    [
        (np.array([0.5, 1]), np.eye(2)),
        (np.array([0, 2]), np.eye(2)),
        (np.array([[0, 1]]), np.eye(2)),
        (np.array([0, 1]), np.ones((2, 1))),
        (np.array([0, 1]), np.full((2, 2), 0.5)),
    ],
)
def test_binary_set_metrics_validate_labels_and_membership(truth, sets):
    with pytest.raises(ValueError):
        binary_metrics_fn(
            truth, np.array([0.2, 0.8]), metrics=["set_size"], y_predset=sets
        )


def test_binary_label_processor_forwards_prediction_sets():
    result = get_metrics_fn("binary")(
        np.array([[0.0], [1.0]]),
        np.array([[0.2], [0.8]]),
        metrics=["set_size"],
        y_predset=np.array([[1, 0], [1, 1]]),
    )
    assert result["set_size"] == 1.5


@pytest.mark.parametrize("score_type", ["threshold", "aps"])
def test_binary_label_with_real_model_and_label_processor(score_type):
    with torch.random.fork_rng():
        torch.manual_seed(4)
        samples = [
            {"patient_id": str(i), "x": [i / 16, 1.0], "label": i % 2}
            for i in range(16)
        ]
        dataset = create_sample_dataset(samples, {"x": "tensor"}, {"label": "binary"})
        try:
            model = MLP(dataset=dataset, hidden_dim=8)
            predictor = LABEL(model, alpha=0.3, score_type=score_type, random_state=42)
            predictor.calibrate(dataset.subset(list(range(12))))
            batch = next(
                iter(get_dataloader(dataset.subset([12, 13, 14, 15]), 4, shuffle=False))
            )
            with torch.no_grad():
                output = predictor(**batch)
            assert output["y_prob"].shape == output["y_true"].shape == (4, 1)
            assert output["y_predset"].shape == (4, 2)
            result = binary_metrics_fn(
                output["y_true"].numpy(),
                output["y_prob"].numpy(),
                metrics=SET_METRICS,
                y_predset=output["y_predset"].numpy(),
            )
            assert set(result) == set(SET_METRICS)
        finally:
            dataset.close()


@pytest.mark.parametrize("mode", ["binary", "multiclass"])
@pytest.mark.parametrize("with_labels", [False, True])
def test_shared_prediction_boundary_preserves_native_outputs(mode, with_labels):
    from pyhealth.calib.base_classes import SetPredictor

    class Model(_ProbabilityModel):
        def __init__(self):
            super().__init__(mode)
            self.calls = 0
            self.feature_keys = ["p", "p0", "p1"]

        def forward(self, **kwargs):
            self.calls += 1
            output = super().forward(**kwargs)
            output["loss"] = torch.tensor(0.125)
            output["logit"] = torch.logit(output["y_prob"])
            self.last_output = output
            return output

    class Predictor(SetPredictor):
        def _predict(self, **kwargs):
            output = self.model(**kwargs)
            assert self.model.mode == "multiclass"
            assert output["y_prob"].shape == (2, 2)
            if "y_true" in output:
                assert output["y_true"].shape == (2,)
                assert output["y_true"].dtype == torch.int64
            output["y_predset"] = output["y_prob"] >= 0.5
            return output

    model = Model()
    predictor = Predictor(model)
    batch = {"p": [0.25, 0.75], "p0": [0.75, 0.25], "p1": [0.25, 0.75]}
    if with_labels:
        batch["label"] = [0, 1]
    output = predictor.forward(**batch)

    assert model.calls == 1
    assert predictor.mode == mode
    assert predictor.model.feature_keys == model.feature_keys
    assert set(output) == set(model.last_output) | {"y_predset"}
    assert "_binary_outputs" not in model.last_output
    for key in model.last_output:
        assert output[key] is model.last_output[key]
    torch.testing.assert_close(output["y_predset"], torch.eye(2, dtype=torch.bool))


def test_shared_forward_preserves_favmac_multilabel_behavior():
    from pyhealth.calib.predictionset import FavMac

    class Model(torch.nn.Module):
        mode = "multilabel"
        device = torch.device("cpu")

        def forward(self, x, label=None):
            logits = torch.as_tensor(x, dtype=torch.float64)
            output = {"logit": logits, "y_prob": logits.sigmoid()}
            if label is not None:
                output["y_true"] = torch.as_tensor(label, dtype=torch.float32)
            return output

    rng = np.random.default_rng(18)
    logits = rng.normal(size=(40, 3))
    labels = rng.integers(0, 2, size=(40, 3))
    samples = [{"x": x.tolist(), "label": y.tolist()} for x, y in zip(logits, labels)]
    model = Model()
    predictor = FavMac(model, target_cost=1.0)
    predictor.calibrate(samples[:32])
    batch = next(iter(get_dataloader(samples[32:], 8, shuffle=False)))
    with torch.no_grad():
        output = predictor(**batch)
        native = model(**batch)

    assert predictor.model is model
    assert predictor.mode == "multilabel"
    assert output["y_predset"].shape == (8, 3)
    # FavMac returns integer membership masks; the shared boundary preserves them.
    assert output["y_predset"].dtype == torch.int64
    assert torch.isin(output["y_predset"], torch.tensor([0, 1])).all()
    assert set(output) == set(native) | {"y_predset"}
    for key in native:
        torch.testing.assert_close(output[key], native[key])
