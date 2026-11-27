"""Run the proposal's SUPPORT2 ablations A1, A2, and A3.

Author: Shuxian Xu (NetID: sx30).
Paper: `Let the Experts Speak: Improving Survival Prediction & Calibration
via Mixture-of-Experts Heads <https://proceedings.mlr.press/v297/morrill26a.html>`_.

Run all ablations on processed SUPPORT2:
    python -m examples.support2_survival_moe --csv data/support2/support2.csv

Run a short synthetic smoke test:
    python -m examples.support2_survival_moe --synthetic --seeds 11 --max-epochs 1
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from support2_experiment import (
    PATIENT_ID,
    _cluster_stability,
    _evaluation_times,
    _score_predictions,
    _survival_from_pmf,
    _train_neural_model,
    load_support2,
    prepare_split,
)

HIDDEN_DIM = 128
NUM_EXPERTS = 8
CHUNK_SIZE = HIDDEN_DIM // NUM_EXPERTS
STUDY_CONFIGS: dict[str, tuple[tuple[str, float | int | None], ...]] = {
    "A1": (
        ("lambda_lb", 0.0),
        ("lambda_lb", 0.001),
        ("lambda_lb", 0.01),
        ("lambda_lb", 0.1),
    ),
    "A2": (
        ("time_bins", 25),
        ("time_bins", 50),
        ("time_bins", 100),
        ("time_bins", 200),
    ),
    "A3": (
        ("top_k", None),
        ("top_k", 1),
        ("top_k", 2),
    ),
}
METRIC_COLUMNS = (
    "ece",
    "c_index",
    "brier_25",
    "brier_50",
    "brier_75",
    "routing_entropy",
    "num_nonempty_experts",
)


def create_synthetic_support2(path: Path, n_samples: int, seed: int) -> None:
    """Write a small synthetic survival table suitable for a fast smoke test."""
    if n_samples < 100:
        raise ValueError("Synthetic ablation requires at least 100 samples.")
    rng = np.random.default_rng(seed)
    age = rng.normal(65, 12, size=n_samples).clip(18, 95)
    severity = rng.normal(size=n_samples)
    sex = rng.choice(["female", "male"], size=n_samples)
    diagnosis = rng.choice(["COPD", "CHF", "Cancer", "Sepsis"], size=n_samples)
    log_hazard = 0.025 * (age - 65) + 0.7 * severity
    event_time = rng.exponential(scale=220 / np.exp(log_hazard))
    censor_time = rng.uniform(60, 700, size=n_samples)
    event = event_time <= censor_time
    duration = np.minimum(event_time, censor_time).clip(1, 2029)
    pd.DataFrame(
        {
            "sno": [f"synthetic-{index}" for index in range(n_samples)],
            "age": age,
            "severity": severity,
            "sex": sex,
            "dzgroup": diagnosis,
            "d.time": duration,
            "death": event.astype(np.int64),
        }
    ).to_csv(path, index=False)


def _setting_label(study: str, value: float | None) -> str:
    if study == "A1":
        return f"lambda_{value:g}"
    if study == "A2":
        return f"bins_{value}"
    return "dense" if value is None else f"top{value}"


def run_ablation(
    csv_path: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
    max_epochs: int,
    batch_size: int,
    studies: tuple[str, ...] = ("A1", "A2", "A3"),
    device_name: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run proposal ablations with the same split seeds and model backbone."""
    if not seeds:
        raise ValueError("At least one random seed is required.")
    if not studies:
        raise ValueError("Select at least one of the A1, A2, or A3 studies.")
    if max_epochs < 1 or batch_size < 1:
        raise ValueError("max_epochs and batch_size must be positive.")
    unknown_studies = set(studies).difference(STUDY_CONFIGS)
    if unknown_studies:
        raise ValueError(f"Unknown ablation studies: {sorted(unknown_studies)}")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")

    data = load_support2(csv_path)
    metric_records: list[dict[str, float | int | str | None]] = []
    assignment_records: list[dict[str, int | str]] = []
    total_runs = sum(len(STUDY_CONFIGS[study]) for study in studies) * len(seeds)
    progress = tqdm(total=total_runs, desc="SUPPORT2 ablations", unit="model")

    for study in studies:
        for parameter, value in STUDY_CONFIGS[study]:
            setting = _setting_label(study, value)
            for seed in seeds:
                progress.set_postfix(study=study, setting=setting, seed=seed)
                num_time_bins = int(value) if study == "A2" else 100
                load_balance_weight = float(value) if study == "A1" else 0.01
                top_k = value if study == "A3" else None
                split = prepare_split(data, seed, num_time_bins=num_time_bins)
                brier_times, ece_times, all_times = _evaluation_times(split)

                model = _train_neural_model(
                    split=split,
                    expert_type="personalized",
                    seed=seed,
                    device=device,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    learning_rate=5e-4,
                    hidden_dim=HIDDEN_DIM,
                    num_layers=1,
                    num_experts=NUM_EXPERTS,
                    top_k=top_k,
                    load_balance_weight=load_balance_weight,
                )
                model.eval()
                with torch.no_grad():
                    test_output = model(
                        torch.from_numpy(split.x_test).to(device)
                    )
                    test_pmf = test_output["pmf"].cpu().numpy()
                    risk = -(test_pmf * split.time_grid[np.newaxis, :]).sum(axis=1)
                    predicted_survival = _survival_from_pmf(
                        test_pmf, split.time_grid, all_times
                    )
                    routing = model(
                        torch.from_numpy(split.x_all).to(device)
                    )["routing"].cpu().numpy()

                record: dict[str, float | int | str | None] = {
                    "study": study,
                    "setting": setting,
                    "parameter": parameter,
                    "value": (
                        "dense" if study == "A3" and value is None else value
                    ),
                    "seed": seed,
                    "requested_time_bins": num_time_bins,
                    "actual_time_bins": len(split.time_grid),
                    "hidden_dim": HIDDEN_DIM,
                    "num_experts": NUM_EXPERTS,
                    "chunk_size": CHUNK_SIZE,
                    "load_balance_weight": load_balance_weight,
                    "top_k": top_k,
                    **_score_predictions(
                        split,
                        risk,
                        predicted_survival,
                        all_times,
                        brier_times,
                        ece_times,
                    ),
                }
                mean_routing = routing.mean(axis=0)
                record["routing_entropy"] = float(
                    -(mean_routing * np.log(np.maximum(mean_routing, 1e-12))).sum()
                )
                assigned_experts = routing.argmax(axis=1)
                record["num_nonempty_experts"] = int(
                    np.unique(assigned_experts).size
                )
                metric_records.append(record)
                assignment_records.extend(
                    {
                        PATIENT_ID: str(patient_id),
                        "study": study,
                        "setting": setting,
                        "seed": seed,
                        "expert": int(expert),
                    }
                    for patient_id, expert in zip(
                        data[PATIENT_ID], assigned_experts
                    )
                )
                progress.update()
    progress.close()

    metrics_by_seed = pd.DataFrame(metric_records)
    assignments = pd.DataFrame(assignment_records)
    stability_frames: list[pd.DataFrame] = []
    for (study, setting), group in assignments.groupby(["study", "setting"]):
        stability = _cluster_stability(group.drop(columns=["study", "setting"]))
        stability.insert(0, "setting", setting)
        stability.insert(0, "study", study)
        stability_frames.append(stability)
    routing_stability = (
        pd.concat(stability_frames, ignore_index=True)
        if stability_frames
        else pd.DataFrame(columns=["study", "setting", "seed_a", "seed_b", "ari"])
    )

    summary = (
        metrics_by_seed.groupby(["study", "setting", "parameter", "value"])[
            list(METRIC_COLUMNS)
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index()
    stability_summary = (
        routing_stability.groupby(["study", "setting"])["ari"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "ari_mean", "std": "ari_std"})
        .reset_index()
    )
    summary = summary.merge(stability_summary, on=["study", "setting"], how="left")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_seed.to_csv(output_dir / "ablation_by_seed.csv", index=False)
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)
    assignments.to_csv(output_dir / "ablation_assignments.csv", index=False)
    routing_stability.to_csv(
        output_dir / "ablation_routing_stability.csv", index=False
    )
    return metrics_by_seed, summary, routing_stability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/support2/support2.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/support2_ablations"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=tuple(STUDY_CONFIGS),
        default=list(STUDY_CONFIGS),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-samples", type=int, default=600)
    args = parser.parse_args()

    if args.synthetic:
        with tempfile.TemporaryDirectory(prefix="support2_survival_moe_") as folder:
            synthetic_csv = Path(folder) / "support2_synthetic.csv"
            create_synthetic_support2(
                synthetic_csv,
                n_samples=args.synthetic_samples,
                seed=args.seeds[0],
            )
            result = run_ablation(
                synthetic_csv,
                args.output_dir,
                tuple(args.seeds),
                args.max_epochs,
                args.batch_size,
                tuple(args.studies),
                args.device,
            )
    else:
        if not args.csv.exists():
            raise FileNotFoundError(
                f"{args.csv} not found. Run 01_data_processing.ipynb first."
            )
        result = run_ablation(
            args.csv,
            args.output_dir,
            tuple(args.seeds),
            args.max_epochs,
            args.batch_size,
            tuple(args.studies),
            args.device,
        )

    _, summary, _ = result
    print(summary.to_string(index=False))
    print(f"\nAblation results saved under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
