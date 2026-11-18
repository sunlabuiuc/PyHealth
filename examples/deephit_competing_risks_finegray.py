"""Fit a traditional Fine-Gray baseline without retraining NeuralFG.

Run from PyHealth repo root:
    python examples/deephit_competing_risks_finegray.py
"""

from pathlib import Path
import json

from neuralfg.data import load_data
from neuralfg.train import stratified_split
from neuralfg.finegray_baseline import FineGrayBaseline, evaluate_finegray


# ---------------------------------------------------------
# Fine-Gray experiment configuration
# ---------------------------------------------------------

SEED = 42
MAX_SAMPLES = None   # None = all 30,000 DeepHit observations
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
NUM_RISKS = 2
MAXITER = 250
TOLERANCE = 1e-9

EVALUATE_TEST = False  # Keep False while selecting a model.


def main():
    df = load_data()
    if MAX_SAMPLES is not None:
        df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=SEED).copy()

    # Identical split procedure/seed to NeuralFG's train.py.
    train_df, val_df, test_df = stratified_split(
        df, train_fraction=TRAIN_FRACTION,
        val_fraction=VAL_FRACTION, seed=SEED,
    )
    print(
        f"Split sizes: train={len(train_df)}, "
        f"validation={len(val_df)}, test={len(test_df)}"
    )

    times = (train_df.loc[train_df["label"] > 0, "time"]
             .quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float))
    print("Training-only evaluation times:", times.tolist())


    model = FineGrayBaseline(
        num_risks=NUM_RISKS,
        maxiter=MAXITER,
        tolerance=TOLERANCE,
    )

    model.fit(train_df)
    print("\nFine-Gray fitting summary:")
    print(model.fit_summary().to_string(index=False))


    # ---------------------------------------------------------
    # Create a unique folder for each Fine-Gray experiment
    # ---------------------------------------------------------

    experiment_name = (
        f"n{MAX_SAMPLES if MAX_SAMPLES is not None else 'full'}"
        f"_risk{NUM_RISKS}"
        f"_iter{MAXITER}"
        f"_tol{TOLERANCE}"
        f"_train{int(TRAIN_FRACTION * 100)}"
        f"_val{int(VAL_FRACTION * 100)}"
        f"_seed{SEED}"
    )

    output_dir = (
            Path(__file__).resolve().parent
            / "neuralfg"
            / "results"
            / "finegray"
            / experiment_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)


    val_metrics = evaluate_finegray(
        model, train_df, val_df, evaluation_times=times,
    )
    print("\nFine-Gray validation metrics:")
    print(val_metrics.to_string(index=False))
    val_metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
    model.coefficients().to_csv(output_dir / "coefficients.csv", index=False)
    model.fit_summary().to_csv(output_dir / "fit_summary.csv", index=False)

    settings = {
        "model": "Fine-Gray IPCW Breslow",
        "seed": SEED,
        "max_samples": MAX_SAMPLES,
        "train_fraction": TRAIN_FRACTION,
        "val_fraction": VAL_FRACTION,
        "num_risks": NUM_RISKS,
        "maxiter": MAXITER,
        "tolerance": TOLERANCE,
        "evaluation_times": times.tolist(),
        "evaluate_test": EVALUATE_TEST,
    }
    (output_dir / "config.json").write_text(json.dumps(settings, indent=2))

    if EVALUATE_TEST:
        test_metrics = evaluate_finegray(
            model, train_df, test_df, evaluation_times=times,
        )
        print("\nFine-Gray final test metrics:")
        print(test_metrics.to_string(index=False))
        test_metrics.to_csv(output_dir / "test_metrics.csv", index=False)

    print("\nResults saved to:", output_dir)


if __name__ == "__main__":
    main()
