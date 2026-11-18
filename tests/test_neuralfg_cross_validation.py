"""Fast tests for the paired outer/inner cross-validation orchestrator."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from examples.neuralfg import cross_validation as cv


def toy_dataframe(n=180, seed=9):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "feature1": rng.normal(size=n),
        "feature2": rng.normal(size=n),
        "time": rng.integers(1, 41, size=n),
        "label": np.tile([0, 1, 2], n // 3),
    })
    return df


def test_five_folds_are_disjoint_stratified_and_cover_every_row():
    df = toy_dataframe()
    cfg = cv.CVConfig(n_splits=5, validation_fraction=0.20)
    seen_test = []
    for fold, train_df, val_df, test_df in cv.make_cv_splits(df, cfg):
        assert fold in {1, 2, 3, 4, 5}
        assert set(train_df.index).isdisjoint(val_df.index)
        assert set(train_df.index).isdisjoint(test_df.index)
        assert set(val_df.index).isdisjoint(test_df.index)
        assert len(train_df) + len(val_df) + len(test_df) == len(df)
        assert set(train_df.label) == set(val_df.label) == set(test_df.label) == {0, 1, 2}
        seen_test.extend(test_df.index.tolist())
    assert sorted(seen_test) == sorted(df.index.tolist())


def test_repeatable_folds_with_fixed_seed():
    df = toy_dataframe()
    cfg = cv.CVConfig(n_splits=5, seed=42)
    first = [(x[0], tuple(x[3].index)) for x in cv.make_cv_splits(df, cfg)]
    second = [(x[0], tuple(x[3].index)) for x in cv.make_cv_splits(df, cfg)]
    assert first == second


def test_configuration_changes_are_reflected_in_experiment_name():
    a = cv.CVConfig(dropout=0.0)
    b = cv.CVConfig(dropout=0.25)
    assert cv.experiment_name(a) != cv.experiment_name(b)
    assert "_drop0.25_" in cv.experiment_name(b)


def test_summary_uses_sample_sd_and_quantile_labels():
    records = []
    for fold, c in ((1, 0.6), (2, 0.8)):
        records.append({
            "fold": fold, "model": "NeuralFG", "risk": 1,
            "quantile": "q25", "time": fold * 2.0,
            "c_index": c, "brier_paper": 0.1,
            "brier_competing": 0.2,
        })
    summary = cv.summarize_cv_metrics(pd.DataFrame(records))
    c_row = summary.loc[summary.metric == "c_index"].iloc[0]
    assert c_row["mean"] == pytest.approx(0.7)
    assert c_row["std"] == pytest.approx(np.sqrt(0.02))
    assert c_row["mean_evaluation_time"] == pytest.approx(3.0)
    assert c_row["n_folds"] == 2


def test_end_to_end_orchestration_with_stub_fitters(monkeypatch, tmp_path):
    """Check pairing, output files, and held-out data access without training nets."""
    df = toy_dataframe(n=120)
    cfg = cv.CVConfig(
        n_splits=2, epochs=1, patience=1, output_root=str(tmp_path)
    )
    train_split_ids = {}

    class DummyNeural:
        def state_dict(self):
            return {"dummy": torch.tensor([1.0])}

    def fake_fit(train_df, val_df, evaluation_times, config, *, fold):
        train_split_ids[fold] = set(train_df.index)
        assert set(train_df.index).isdisjoint(val_df.index)
        assert len(evaluation_times) == 3
        return DummyNeural(), pd.DataFrame([{
            "epoch": 1, "train_loss": 2.0, "val_loss": 2.5
        }]), {"best_epoch": 1, "best_val_nll": 2.5, "epochs_run": 1,
             "training_seed": config.seed + fold}

    class DummyFineGray:
        def __init__(self, num_risks, maxiter, tolerance):
            assert num_risks == 2

        def fit(self, train_df):
            self.trained_on = set(train_df.index)
            return self

        def fit_summary(self):
            return pd.DataFrame({"risk": [1, 2], "iterations": [1, 1]})

        def coefficients(self):
            return pd.DataFrame({"feature": ["feature1"], "risk_1": [0.2],
                                 "risk_2": [-0.1]})

    def fake_neural_metrics(model, train_df, eval_df, *, evaluation_times,
                            batch_size):
        assert set(train_df.index).isdisjoint(eval_df.index)
        assert set(train_df.index) in train_split_ids.values()
        return score_frame(len(eval_df), evaluation_times, 0.7)

    def fake_fine_metrics(model, train_df, eval_df, evaluation_times):
        assert model.trained_on == set(train_df.index)
        assert model.trained_on.isdisjoint(eval_df.index)
        return score_frame(len(eval_df), evaluation_times, 0.6)

    def score_frame(n, times, cindex):
        return pd.DataFrame([
            {"risk": risk, "time": float(t), "c_index": cindex,
             "brier_paper": 0.1, "brier_competing": 0.11,
             "mean_predicted_cif": 0.2, "n_test": n}
            for t in times for risk in (1, 2)
        ])

    monkeypatch.setattr(cv, "fit_neuralfg_on_split", fake_fit)
    from examples.neuralfg import finegray_baseline
    from examples.neuralfg import metrics
    monkeypatch.setattr(finegray_baseline, "FineGrayBaseline", DummyFineGray)
    monkeypatch.setattr(finegray_baseline, "evaluate_finegray", fake_fine_metrics)
    monkeypatch.setattr(metrics, "evaluate_neuralfg", fake_neural_metrics)

    all_scores, summary, destination = cv.run_cross_validation(df, cfg)
    assert len(all_scores) == 2 * 2 * 3 * 2  # folds * models * times * risks
    assert (destination / "fold_metrics.csv").exists()
    assert (destination / "cv_summary.csv").exists()
    assert (destination / "fold_1" / "neuralfg" / "best_model.pt").exists()
    assert (destination / "fold_2" / "finegray" / "coefficients.csv").exists()
    assert len(summary) == 2 * 2 * 3 * 3
    assert json.loads((destination / "config.json").read_text())["n_splits"] == 2
