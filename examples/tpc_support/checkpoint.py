# Contributor: Xiao Luo, UIUC NetID xiaoluo6.
# Paper: Temporal Pointwise Convolutional Networks for Length of Stay
# Prediction in the Intensive Care Unit. https://arxiv.org/abs/2007.09483
"""Synthetic CSV-to-task checkpoint for cohort, static features and labels."""

from pathlib import Path
import json
import tempfile
import numpy as np
from pyhealth.datasets import eICUDataset
from pyhealth.tasks import TPCLengthOfStayEICU
from examples.tpc_support.generate_synthetic import generate, TABLES


def run(root: Path) -> dict:
    """Generate data and validate numerical cohort, feature and label expectations."""
    generate(root / "csv")
    source = eICUDataset(
        root=str(root / "csv"), tables=TABLES, cache_dir=root / "cache"
    )
    task = TPCLengthOfStayEICU()
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
    assert samples[103]["los"]["time"][-1] == 336
    assert samples[103]["los"]["values"][-1] == [64.5]
    assert samples[103]["dynamic"]["values"][0, names.index("hr")] == 100
    assert np.isnan(samples[105]["static"]["values"][0][2])
    # TODO(A): downstream model validation belongs in the training
    # example once preprocessing and the agreed input interface are complete.
    return {
        "status": "passed",
        "scope": "CSV loading, cohort, static features, hourly labels",
        "synthetic_patients": 5,
        "included_stays": [101, 103, 105],
        "excluded_stays": [102, 104],
        "model_training": "pending training integration",
        "missingness_and_decay": "pending preprocessing integration",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tpc-checkpoint-") as tmp:
        report = run(Path(tmp))
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
