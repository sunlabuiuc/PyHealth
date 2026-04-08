from datetime import datetime, timedelta

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader, split_by_patient
from pyhealth.models.td_lstm_mortality import TDLSTMMortality


def _make_hourly_timestamps(num_steps: int, start: datetime | None = None):
    if start is None:
        start = datetime(2024, 1, 1, 0, 0, 0)
    return [start + i * timedelta(hours=1) for i in range(num_steps)]


def _make_samples():
    return [
        {
            "patient_id": "p1",
            "visit_id": "v1",
            "x": [
                _make_hourly_timestamps(3, datetime(2024, 1, 1, 0, 0, 0)),
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.2, 0.1, 0.0, 0.5],
                    [0.4, 0.3, 0.2, 0.1],
                ],
            ],
            "label": 0,
        },
        {
            "patient_id": "p2",
            "visit_id": "v2",
            "x": [
                _make_hourly_timestamps(3, datetime(2024, 1, 2, 0, 0, 0)),
                [
                    [0.5, 0.4, 0.3, 0.2],
                    [0.6, 0.5, 0.4, 0.3],
                    [0.7, 0.6, 0.5, 0.4],
                ],
            ],
            "label": 1,
        },
        {
            "patient_id": "p3",
            "visit_id": "v3",
            "x": [
                _make_hourly_timestamps(3, datetime(2024, 1, 3, 0, 0, 0)),
                [
                    [0.9, 0.8, 0.7, 0.6],
                    [0.8, 0.7, 0.6, 0.5],
                    [0.7, 0.6, 0.5, 0.4],
                ],
            ],
            "label": 0,
        },
    ]


def _make_dataset():
    samples = _make_samples()
    return create_sample_dataset(
        samples=samples,
        input_schema={"x": "timeseries"},
        output_schema={"label": "binary"},
        dataset_name="test_td_lstm_mortality",
    )


def _make_batch():
    dataset = _make_dataset()
    loader = get_dataloader(dataset, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    return dataset, batch


def test_model_instantiation_supervised():
    dataset = _make_dataset()
    model = TDLSTMMortality(
        dataset=dataset,
        feature_key="x",
        label_key="label",
        mode="binary",
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        gamma=0.95,
        alpha_terminal=0.10,
        n_step=1,
        training_mode="supervised",
    )

    assert model.feature_key == "x"
    assert model.label_key == "label"
    assert model.hidden_dim == 16
    assert model.training_mode == "supervised"


def test_model_instantiation_td():
    dataset = _make_dataset()
    model = TDLSTMMortality(
        dataset=dataset,
        feature_key="x",
        label_key="label",
        mode="binary",
        hidden_dim=16,
        gamma=0.95,
        alpha_terminal=0.10,
        n_step=1,
        training_mode="td",
    )

    assert model.feature_key == "x"
    assert model.label_key == "label"
    assert model.training_mode == "td"


def test_forward_output_shapes_supervised():
    dataset, batch = _make_batch()

    model = TDLSTMMortality(
        dataset=dataset,
        feature_key="x",
        label_key="label",
        mode="binary",
        hidden_dim=16,
        training_mode="supervised",
    )

    out = model(**batch)

    assert "loss" in out
    assert "y_prob" in out
    assert "y_true" in out
    assert "logit" in out
    assert "logits_seq" in out
    assert "probs_seq" in out

    assert out["logits_seq"].shape[0] == 3
    assert out["probs_seq"].shape[0] == 3
    assert out["logit"].shape == (3,)
    assert out["y_prob"].shape == (3,)
    assert out["y_true"].shape == (3,)
    assert out["loss"].ndim == 0


def test_forward_output_shapes_td():
    dataset, batch = _make_batch()

    model = TDLSTMMortality(
        dataset=dataset,
        feature_key="x",
        label_key="label",
        mode="binary",
        hidden_dim=16,
        training_mode="td",
    )
    target_model = TDLSTMMortality(
        dataset=dataset,
        feature_key="x",
        label_key="label",
        mode="binary",
        hidden_dim=16,
        training_mode="td",
    )
    target_model.load_state_dict(model.state_dict())

    out = model(target_model=target_model, **batch)

    assert "loss" in out
    assert "y_prob" in out
    assert "y_true" in out
    assert "logit" in out
    assert "logits_seq" in out
    assert "probs_seq" in out
    assert "td_loss" in out
    assert "terminal_loss" in out

    assert out["logits_seq"].shape[0] == 3
    assert out["probs_seq"].shape[0] == 3
    assert out["logit"].shape == (3,)
    assert out["y_prob"].shape == (3,)
    assert out["y_true"].shape == (3,)
    assert out["loss"].ndim == 0


def test_probability_range():
    dataset, batch = _make_batch()

    model = TDLSTMMortality(
        dataset=dataset,
        feature_key="x",
        label_key="label",
        mode="binary",
        hidden_dim=16,
        training_mode="supervised",
    )

    out = model(**batch)

    assert torch.all(out["probs_seq"] >= 0.0)
    assert torch.all(out["probs_seq"] <= 1.0)
    assert torch.all(out["y_prob"] >= 0.0)
    assert torch.all(out["y_prob"] <= 1.0)


def test_build_n_step_targets_shape():
    dataset, batch = _make_batch()

    model = TDLSTMMortality(
        dataset=dataset,
        feature_key="x",
        label_key="label",
        mode="binary",
        hidden_dim=16,
        gamma=0.95,
        n_step=2,
        training_mode="td",
    )

    x = batch["x"]
    y_true = batch["label"]

    logits_seq = model.forward_logits(x)
    probs_seq = torch.sigmoid(logits_seq)

    td_targets = model.build_n_step_targets(
        target_probs=probs_seq,
        y_true=y_true,
    )

    assert td_targets.shape == probs_seq.shape


def test_td_loss_backward_runs():
    dataset, batch = _make_batch()

    model = TDLSTMMortality(
        dataset=dataset,
        feature_key="x",
        label_key="label",
        mode="binary",
        hidden_dim=16,
        training_mode="td",
    )
    target_model = TDLSTMMortality(
        dataset=dataset,
        feature_key="x",
        label_key="label",
        mode="binary",
        hidden_dim=16,
        training_mode="td",
    )
    target_model.load_state_dict(model.state_dict())

    out = model(target_model=target_model, **batch)
    out["loss"].backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_supervised_loss_backward_runs():
    dataset, batch = _make_batch()

    model = TDLSTMMortality(
        dataset=dataset,
        feature_key="x",
        label_key="label",
        mode="binary",
        hidden_dim=16,
        training_mode="supervised",
    )

    out = model(**batch)
    out["loss"].backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_embed_output_shape():
    dataset, batch = _make_batch()

    model = TDLSTMMortality(
        dataset=dataset,
        feature_key="x",
        label_key="label",
        mode="binary",
        hidden_dim=16,
        training_mode="supervised",
    )

    out = model(embed=True, **batch)

    assert "embedding" in out
    assert out["embedding"].shape == (3, 16)


def test_final_prediction_selection_shape():
    dataset, batch = _make_batch()

    model = TDLSTMMortality(
        dataset=dataset,
        feature_key="x",
        label_key="label",
        mode="binary",
        hidden_dim=16,
        training_mode="supervised",
    )

    logits_seq = model.forward_logits(batch["x"])
    gathered = model._gather_last_valid_step(logits_seq, None)

    assert gathered.shape == (3,)


def test_invalid_training_mode_raises():
    dataset = _make_dataset()

    try:
        _ = TDLSTMMortality(
            dataset=dataset,
            feature_key="x",
            label_key="label",
            mode="binary",
            training_mode="bad_mode",
        )
        assert False, "Expected ValueError for invalid training_mode"
    except ValueError:
        assert True


def test_td_mode_requires_target_model():
    dataset, batch = _make_batch()

    model = TDLSTMMortality(
        dataset=dataset,
        feature_key="x",
        label_key="label",
        mode="binary",
        hidden_dim=16,
        training_mode="td",
    )

    try:
        _ = model(**batch)
        assert False, "Expected ValueError when target_model is missing in TD mode"
    except ValueError:
        assert True


def test_split_by_patient_runs():
    dataset = _make_dataset()
    train_dataset, val_dataset, test_dataset = split_by_patient(
        dataset,
        [0.6, 0.2, 0.2],
        seed=42,
    )

    assert len(train_dataset) + len(val_dataset) + len(test_dataset) == len(dataset)