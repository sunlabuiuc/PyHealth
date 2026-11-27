"""Calculate descriptive cross-validation uncertainty for DCM."""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Summarize model performance and variation across folds."""
    results_dir = Path("results/dcm")

    results = pd.read_csv(results_dir / "fold_results.csv")

    group_columns = [
        "population",
        "model",
        "quantile",
    ]

    records = []

    for keys, group in results.groupby(group_columns):

        population, model, quantile = keys

        for metric in ["ctd", "ece", "brier"]:

            values = group[metric].dropna().to_numpy()

            n = len(values)

            mean = np.mean(values)
            std = np.std(values, ddof=1)
            se = std / np.sqrt(n)

            records.append(
                {
                    "population": population,
                    "model": model,
                    "quantile": quantile,
                    "metric": metric,
                    "n_folds": n,
                    "mean": mean,
                    "std": std,
                    "fold_se": se,
                }
            )

    summary = pd.DataFrame(records)

    summary.to_csv(
        results_dir / "uncertainty_results.csv",
        index=False,
    )

    for population in ["Overall", "Non-White", "White"]:

        print("\n" + "=" * 75)
        print(f"UNCERTAINTY RESULTS: {population}")
        print("=" * 75)

        subset = summary[summary["population"] == population]

        for metric in ["ctd", "ece", "brier"]:

            table = subset[subset["metric"] == metric].copy()

            table["mean +/- SE"] = table.apply(
                lambda row: (f"{row['mean']:.4f} +/- " f"{row['fold_se']:.4f}"),
                axis=1,
            )

            pivot = table.pivot(
                index="model",
                columns="quantile",
                values="mean +/- SE",
            )

            print(f"\n{metric.upper()}")
            print(pivot.to_string())

    print("\nSaved uncertainty_results.csv")


if __name__ == "__main__":
    main()
