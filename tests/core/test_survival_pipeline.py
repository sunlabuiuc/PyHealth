"""Authors: Aritra Saha and Ugochukwu Njoku.

Small download-free checks of the accepted model's native training pipeline.
"""

import json

import numpy as np
import pytest
import torch
from examples.survival_pipeline import load_split, restore_pilot, run_pilot
from pyhealth.datasets import create_sample_dataset, get_dataloader


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    """Keep tiny smoke fits cheap without changing later tests' thread count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def parts(request):
    """Seven GBSG-width or 27 SUPPORT-width numeric inputs per patient."""
    width = request.param
    rng = np.random.default_rng(7)
    samples = []
    for i in range(64):
        features = rng.normal(size=width)
        samples.append(
            {
                "patient_id": f"p{i}",
                "features": features.tolist(),
                "duration": [float(np.exp(0.2 * features[0] + rng.normal(0, 0.3)))],
                "event": [float(i % 3 != 0)],
            }
        )
    dataset = create_sample_dataset(
        samples, {"features": "tensor"}, {"duration": "tensor", "event": "tensor"}
    )
    return tuple(
        dataset.subset(indices) for indices in [range(40), range(40, 52), range(52, 64)]
    )


@pytest.mark.parametrize("parts", [7, 27], indirect=True)
@pytest.mark.parametrize("transform", ["softplus", "exp"])
def test_standard_trainer_metrics_and_checkpoint(parts, transform, tmp_path):
    run = tmp_path / "pilot"
    record = run_pilot(
        parts, {"fixture": True}, run, epochs=2, time_transform=transform
    )
    assert record["passed"]
    assert record["checkpoint_max_error"] == 0
    assert record["gradient_checks"] > 0
    assert np.isfinite(record["training_losses"]).all()
    assert record["validation_batch_size"] == len(parts[1])
    assert record["best_validation_nll"] == pytest.approx(
        record["direct_patient_mean_nll"], abs=1e-6
    )
    saved = json.loads((run / "record.json").read_text())
    assert "time_scale" not in saved["model"]
    assert saved["model"]["time_transform"] == transform
    restored = restore_pilot(run, parts[0])
    with np.load(run / "predictions.npz", allow_pickle=False) as arrays:
        batch = next(iter(get_dataloader(parts[2], batch_size=len(parts[2]))))
        assert arrays["patient_id"].tolist() == batch["patient_id"]
        np.testing.assert_array_equal(arrays["time"], batch["duration"][:, 0].numpy())
        np.testing.assert_array_equal(arrays["event"], batch["event"][:, 0].numpy())
        np.testing.assert_array_equal(
            arrays["survival"],
            restored.predict_survival(batch["features"], arrays["grid"]).numpy(),
        )
    with pytest.raises(FileExistsError):
        run_pilot(parts, {}, run, epochs=2)


@pytest.mark.parametrize("split", [0, 11])
def test_invalid_released_partition_is_rejected_before_loading(split, tmp_path):
    with pytest.raises(ValueError, match="1 through 10"):
        load_split("gbsg", tmp_path, tmp_path, split, tmp_path / "cache")
