# Contributor: Xiao Luo, UIUC NetID xiaoluo6.
# Paper: Temporal Pointwise Convolutional Networks for Length of Stay
# Prediction in the Intensive Care Unit. https://arxiv.org/abs/2007.09483
"""Synthetic CSV-to-LSTM checkpoint; no clinical performance claims."""

from pathlib import Path
import json
import tempfile
import numpy as np
import torch
from pyhealth.datasets import eICUDataset, create_sample_dataset, get_dataloader
from pyhealth.models import RNN_V2
from pyhealth.tasks import TPCLengthOfStayEICU
from project.person_c.generate_synthetic import generate, TABLES


def run(root: Path) -> dict:
    """Generate data, check numerical expectations, and train one LSTM step."""
    torch.manual_seed(42)
    torch.set_num_threads(1)
    generate(root / "csv")
    source = eICUDataset(
        root=str(root / "csv"), tables=TABLES, cache_dir=root / "cache"
    )
    task = TPCLengthOfStayEICU(include_decay=True)
    samples = {}
    for sid in (101, 102, 103, 104, 105):
        out = task(source.get_patient(f"synthetic-{sid}"))
        if sid in (102, 104):
            assert out == [], f"Excluded stay {sid} survived"
        else:
            assert len(out) == 1
            samples[sid] = out[0]
    short = samples[101]
    names = task.DYNAMIC_FEATURES

    def value(hour, name):
        return short["dynamic"]["values"][hour, names.index(name)]

    # Hand-calculated oracle: not computed by the task under test.
    for hour, name, expected in [
        (0, "hr", 90),
        (2, "hr", 110),
        (5, "hr", 120),
        (1, "k", 5),
        (4, "k", 3),
        (2, "crea", 1.2),
        (3, "sbp", 130),
        (1, "urine", 150),
        (3, "fio2", 40),
        (4, "temp", 37),
    ]:
        np.testing.assert_allclose(value(hour, name), expected)
    assert np.isnan(value(1, "hr"))
    assert np.isnan(value(5, "k"))  # implausible 99 mmol/L discarded
    assert short["static"]["values"] == [[18.0, 0.0, 165.0, 65.0]]
    assert short["los"]["time"] == list(range(5, 13))
    np.testing.assert_allclose(
        short["los"]["values"], [[7.5], [6.5], [5.5], [4.5], [3.5], [2.5], [1.5], [0.5]]
    )
    np.testing.assert_allclose(
        short["decay"]["values"][:6, names.index("hr")], [1, 0.75, 1, 0.75, 0.375, 1]
    )
    np.testing.assert_allclose(
        short["decay"]["values"][:6, names.index("k")], [0, 1, 0.75, 0.375, 1, 0.75]
    )
    assert samples[103]["los"]["time"][-1] == 336
    assert samples[103]["los"]["values"][-1] == [64.5]
    assert samples[103]["dynamic"]["values"][0, names.index("hr")] == 100
    assert np.isnan(samples[105]["static"]["values"][0][2])
    # Split BEFORE fitting: the high-valued long stay is held out.
    train = create_sample_dataset([samples[101]], task.input_schema, task.output_schema)
    heldout = create_sample_dataset(
        [samples[103], samples[105]],
        task.input_schema,
        task.output_schema,
        input_processors=train.input_processors,
        output_processors=train.output_processors,
    )
    train_before = train[0]["dynamic"][0].clone()
    assert torch.isfinite(heldout[0]["dynamic"][0]).all()
    torch.testing.assert_close(train_before, train[0]["dynamic"][0])
    # Same training processors must be reused, not refitted on held-out patients.
    assert heldout.input_processors["dynamic"] is train.input_processors["dynamic"]
    model = RNN_V2(
        train,
        embedding_dim={
            "dynamic": [{"values": 8}],
            "static": [{"values": 4}],
            "decay": [{"values": 8}],
        },
        hidden_dim=8,
        rnn_type="LSTM",
        mask="concat",
    )
    assert any(isinstance(m, torch.nn.LSTM) for m in model.modules())
    batch = next(iter(get_dataloader(train, batch_size=1, shuffle=False)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    out = model(**batch)
    assert out["logit"].shape[0] == 8
    assert torch.isfinite(out["loss"])
    optimizer.zero_grad()
    out["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) for g in gradients)
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    model.eval()
    with torch.no_grad():
        evaluation = model(
            **next(iter(get_dataloader(heldout, batch_size=2, shuffle=False)))
        )
    assert torch.isfinite(evaluation["loss"])
    assert evaluation["logit"].shape[0] == 332 + 4
    return {
        "status": "passed",
        "synthetic_patients": 5,
        "included_stays": [101, 103, 105],
        "excluded_stays": [102, 104],
        "train_patient": 101,
        "heldout_patients": [103, 105],
        "training_labels": 8,
        "heldout_labels": 336,
        "model": "RNN_V2 / LSTM",
        "decay_enabled": True,
        "optimizer_steps": 1,
        "finite_nonzero_gradients": True,
        "parameters_changed": True,
        "training_loss": float(out["loss"].detach()),
        "heldout_loss": float(evaluation["loss"]),
        "interpretation": "Synthetic integration check only; not model performance evidence.",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="person-c-checkpoint-") as tmp:
        report = run(Path(tmp))
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
