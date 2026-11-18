"""Unit tests for the paper-aligned reproduction split and search grid.
Place at PyHealth/tests/test_paper_reproduction.py.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold
from examples.neuralfg.paper_reproduction import (
    PaperRunConfig, candidate_configs, paper_hyperparameter_grid,
    split_paper_fold, transform_for_network, output_folder, summary_table,
)


def toy(n=300, seed=12):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature1": rng.normal(size=n), "feature2": rng.normal(size=n),
        "time": rng.integers(1, 50, size=n),
        "label": np.tile([0, 1, 2], n // 3),
    })


def test_four_way_split_is_disjoint_and_covers_every_patient():
    df = toy(n=300)
    conf = PaperRunConfig(folds=5)
    all_test = []
    for fold, (dev_pos, test_pos) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=0).split(df, df.label), 1
    ):
        dev, test = df.iloc[dev_pos], df.iloc[test_pos]
        train, tune, stop = split_paper_fold(dev, fold, conf)
        assert (len(train), len(tune), len(stop), len(test)) == (192, 24, 24, 60)
        groups = [set(frame.index) for frame in (train, tune, stop, test)]
        assert len(set.union(*groups)) == 300
        assert sum(map(len, groups)) == 300
        all_test.extend(test.index)
    assert sorted(all_test) == list(range(300))


def test_paper_grid_and_candidate_reproducibility():
    grid = paper_hyperparameter_grid()
    assert {k: len(v) for k, v in grid.items()} == {
        "embedding_shape": 8, "monotonic_shape": 8,
        "learning_rate": 2, "batch_size": 2, "dropout": 4,
    }
    cfg = PaperRunConfig(mode="search", trials=12)
    assert candidate_configs(cfg) == candidate_configs(cfg)
    assert len(candidate_configs(cfg)) == 12
    assert candidate_configs(PaperRunConfig(mode="fixed"))[0]["batch_size"] == 100


def test_training_only_feature_and_time_scaling():
    df = toy(n=300)
    train, tune = df.iloc[:200], df.iloc[200:]
    (t_train, t_tune), metadata = transform_for_network(train, [tune])
    assert np.allclose(t_train[["feature1","feature2"]].mean(), 0, atol=1e-10)
    assert np.allclose(t_train[["feature1","feature2"]].std(ddof=0), 1, atol=1e-10)
    assert metadata["time_scale"] == float(train.time.max() + 1)
    assert np.allclose(t_tune.time, (tune.time + 1) / metadata["time_scale"])
    assert t_tune.index.equals(tune.index)


def test_different_settings_separate_output_folders(tmp_path):
    a = output_folder(PaperRunConfig(output_root=str(tmp_path), mode="search", trials=5))
    b = output_folder(PaperRunConfig(output_root=str(tmp_path), mode="search", trials=100))
    assert a != b
    assert output_folder(PaperRunConfig(output_root=str(tmp_path))) == output_folder(
        PaperRunConfig(output_root=str(tmp_path)))


def test_summary_mean_and_sample_sd():
    df = pd.DataFrame([
        dict(fold=i,model="NeuralFG",risk=1,quantile="q25",
             c_index_reference=x,brier_reference=0.1)
        for i,x in [(1,0.6),(2,0.8)]
    ])
    summary = summary_table(df)
    got = summary[summary.metric == "c_index_reference"].iloc[0]
    assert got["mean"] == pytest.approx(0.7)
    assert got["std"] == pytest.approx(np.sqrt(0.02))
